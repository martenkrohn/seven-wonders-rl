import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sevenwonders_rl.utils.checkpoints import load_model
from sevenwonders_rl.utils.checkpoints import file_hash

pytestmark = pytest.mark.integration


def test_short_training_saves_final_partial_rollout_and_repeats(require_engine, tmp_path):
    root = Path(__file__).parents[1]
    states = []
    for name, envs, horizon in (("one", 1, 32), ("two", 2, 16), ("repeat", 2, 16)):
        output = tmp_path / name
        subprocess.run([sys.executable, "scripts/ppo_training/train_unified.py", "--mode", "fixed",
            "--seed", "123", "--total-timesteps", "72", "--n-envs", str(envs),
            "--rollout-horizon", str(horizon), "--eval-every", "1000", "--out-dir", str(output)],
            cwd=root, check=True, capture_output=True, text=True, timeout=60)
        result = json.loads((output / "result.json").read_text())
        updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
        assert result["status"] == "completed" and result["timestep"] == 72
        assert len(updates) == 3 and updates[-1]["transitions"] == 8
        assert not (output / "best.pt").exists()  # no evaluation happened
        assert not Path(json.loads((output / "config.json").read_text())["io_root"]).exists()
        states.append(load_model(output / "latest.pt").state_dict())
    for key in states[1]:
        torch.testing.assert_close(states[1][key], states[2][key], rtol=0, atol=0)


def test_failed_rollout_preserves_saved_update_and_persists_diagnostics(require_engine, tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "failed"
    code = '''
import importlib.util
import sys
from sevenwonders_rl.envs.sevenwonders_env import EngineError
spec = importlib.util.spec_from_file_location("trainer", "scripts/ppo_training/train_unified.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
class FaultInjected(trainer.SevenWondersEnv):
    calls = 0
    def wait_next(self, turn):
        self.calls += 1
        if self.calls == 3:
            raise EngineError("injected_failure", "controlled failed transition")
        return super().wait_next(turn)
trainer.SevenWondersEnv = FaultInjected
trainer.main(["--mode", "fixed", "--total-timesteps", "8", "--n-envs", "1",
              "--rollout-horizon", "2", "--eval-every", "1000", "--out-dir", sys.argv[1]])
'''
    process = subprocess.run([sys.executable, "-c", code, str(output)], cwd=root,
                             capture_output=True, text=True, timeout=60)
    assert process.returncode != 0 and "controlled failed transition" in process.stderr
    failure = json.loads((output / "failure.json").read_text())
    assert failure["pending_rollout_discarded"] and failure["last_completed_update"] == 1
    assert failure["timestep"] == 2 and failure["reason"] == "injected_failure"
    checkpoint = torch.load(output / "latest.pt", weights_only=True)
    assert checkpoint["timestep"] == 2
    assert json.loads((output / "result.json").read_text())["status"] == "failed"
    assert not Path(json.loads((output / "config.json").read_text())["io_root"]).exists()


def test_pure_selfplay_starts_neural_logs_opponents_and_evaluates_neural_only(require_engine, tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "pure"
    code = '''
import importlib.util
import sys
import sevenwonders_rl.evaluation.runner as runner
spec = importlib.util.spec_from_file_location("trainer", "scripts/ppo_training/train_unified.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
def no_bots(*args, **kwargs):
    raise AssertionError("Heuristic/random factory must not run in neural self-play")
trainer.make_policy = runner.make_policy = no_bots
trainer.main(["--mode", "selfplay", "--seed", "7", "--total-timesteps", "512",
              "--n-envs", "2", "--rollout-horizon", "64", "--snapshot-every", "64",
              "--eval-every", "256", "--eval-games-per-matchup", "1", "--out-dir", sys.argv[1]])
'''
    process = subprocess.run([sys.executable, "-c", code, str(output)], cwd=root,
                   capture_output=True, text=True, check=True, timeout=60)
    selections = [json.loads(line) for line in (output / "opponents.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert {row["agent_seat"] for row in episodes} == {0, 1, 2}
    assert selections[0]["selection_step"] == 0
    for row in selections + episodes:
        assert len(row["opponents"]) == 2
        for opponent in row["opponents"].values():
            assert opponent["name"] == "ppo_snapshot"
            assert opponent["sha256"] == file_hash(opponent["checkpoint"])
    assert all(opponent["checkpoint"].endswith("checkpoint_0.pt")
               for opponent in selections[0]["opponents"].values())
    refs = json.loads((output / "references.json").read_text())["references"]
    assert all(ref["sha256"] == file_hash(ref["checkpoint"]) for ref in refs)
    assert not (output / "eval_128.jsonl").exists()
    assert not (output / 'eval_0.jsonl').exists()
    summary = json.loads((output / 'eval_256_summary.json').read_text())
    assert summary['completed'] == 3
    assert [r['step'] for r in summary['references']] == [128]
    assert 'paired_valid' not in summary
    final = json.loads((output / 'eval_512_summary.json').read_text())
    assert final['completed'] == 12
    assert [r['step'] for r in final['references']] == [384, 256]
    assert final['plan_id'] != summary['plan_id']
    assert 'paired_valid' not in final
    assert 'kein direkter Fortschrittsvergleich' in process.stdout
    assert json.loads((output / 'result.json').read_text())['latest_evaluated_step'] == 512
    assert load_model(output / "latest.pt").architecture_version == 2
    assert (output / "evaluated.pt").exists()
    assert not (output / 'best.pt').exists()
    timing = json.loads((output / "timing.json").read_text())["buckets"]
    assert timing["policies.opponents.requests"]["calls"] > timing["policies.opponents.inference"]["calls"]
