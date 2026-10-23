from sevenwonders_rl.features.obs_builder import GLOBAL_DIM
from sevenwonders_rl.features.action_feature_schema import K
import copy
from collections import OrderedDict
import json

import numpy as np
import pytest
import torch

from sevenwonders_rl.evaluation.selfplay import evaluation_plan, paired_progress, reference
from sevenwonders_rl.selfplay.opponent_pool import OpponentPool
from sevenwonders_rl.utils.checkpoints import save_checkpoint, FEATURE_SEMANTICS_VERSION
from sevenwonders_rl.utils.ppo_model import ActorCritic
from test_training_contracts import trainer_module

pytestmark = pytest.mark.component


def test_rolling_references_exclude_initial_current_future_and_refresh(tmp_path):
    from sevenwonders_rl.evaluation.selfplay import rolling_references, plan_identity
    for step in (0, 4096, 8192, 16384, 28672, 32768, 65536):
        (tmp_path / f'checkpoint_{step}.pt').write_bytes(str(step).encode())
    assert rolling_references(tmp_path, 4096) == []
    assert [r['step'] for r in rolling_references(tmp_path, 8192)] == [4096]
    refs = rolling_references(tmp_path, 32768)
    assert [r['step'] for r in refs] == [28672, 16384]
    next_refs = rolling_references(tmp_path, 65536)
    assert [r['step'] for r in next_refs] == [32768, 28672]
    assert plan_identity(evaluation_plan(refs, 42, 1), 'stochastic') != plan_identity(evaluation_plan(next_refs, 42, 1), 'stochastic')
    assert all(0 < r['step'] < 32768 for r in refs)


def test_console_only_compares_identical_evaluation_plans(tmp_path, capsys):
    from sevenwonders_rl.utils.training_console import TrainingConsole
    args = trainer_module().parse_args([])
    console = TrainingConsole(args, tmp_path, tmp_path)
    summary = dict(attempted=3, completed=3, failed=0, truncated=0, win_share=.5,
                   agent_vp_mean=40., vp_diff_mean=2., valid_for_selection=True, plan_id='a')
    for step, identity in [(10, 'a'), (20, 'b')]:
        console.evaluation_start(step, 3, references=[{'step': step - 1}])
        console.evaluation_end({**summary, 'plan_id': identity})
    text = capsys.readouterr().out
    assert 'trainierte Vorgänger' in text
    assert 'kein direkter Fortschrittsvergleich' in text
    assert 'ÄNDERUNG ZUR VORHERIGEN' not in text
    console.evaluation_start(30, 3, references=[{'step': 19}])
    console.evaluation_end({**summary, 'plan_id': 'b'})
    assert 'ÄNDERUNG ZUR VORHERIGEN' in capsys.readouterr().out


def checkpoint(path):
    model = ActorCritic(GLOBAL_DIM, K, hidden=8)
    save_checkpoint(path, model, None, 0, {"global_dim": GLOBAL_DIM, "action_feat_dim": K, "hidden": 8},
                    {"feature_semantics_version": FEATURE_SEMANTICS_VERSION})
    return model


def test_neural_only_start_with_one_snapshot_and_fatal_load_errors(tmp_path, monkeypatch):
    trainer = trainer_module()
    monkeypatch.setattr(trainer, "make_policy", lambda *a, **kw: pytest.fail("No bots in self-play"))
    path = tmp_path / "checkpoint_0.pt"
    model = checkpoint(path)
    original = copy.deepcopy(model.state_dict())
    pool = OpponentPool(str(tmp_path), min_step_lag=99999)
    cache = OrderedDict()
    policies = trainer.sample_selfplay_opponents(pool, np.random.default_rng(0), 0, 1, cache)
    for seat in (0, 2):
        assert policies[seat].name == "ppo_snapshot"
        assert policies[seat].policy.model is not model
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    for key, value in original.items():
        torch.testing.assert_close(policies[0].policy.model.state_dict()[key], value)
    path.unlink()
    with pytest.raises(ValueError, match="initial neural"):
        trainer.sample_selfplay_opponents(pool, np.random.default_rng(0), 0, 1, cache)
    path.write_bytes(b"broken checkpoint")
    with pytest.raises(RuntimeError, match="Cannot load neural opponent.*checkpoint_0"):
        trainer.sample_selfplay_opponents(pool, np.random.default_rng(0), 0, 1)


def test_plan_balances_seats_and_opponent_orders_and_paired_uncertainty(tmp_path):
    refs = [{"checkpoint": "a", "sha256": "A"}, {"checkpoint": "b", "sha256": "B"}]
    plan = evaluation_plan(refs, 42, 3)
    assert len(plan) == 36
    assert {p["engine_seed"] for p in plan}.isdisjoint(p["engine_seed"] for p in evaluation_plan(refs, 9001, 3))
    for seed in {p["engine_seed"] for p in plan}:
        block = [p for p in plan if p["engine_seed"] == seed]
        assert [p["agent_seat"] for p in block].count(0) == 4
        assert len({(p["left"]["checkpoint"], p["right"]["checkpoint"]) for p in block}) == 4
    baseline = [{**p, "status": "completed", "opponents": {"left": p["left"], "right": p["right"]},
                 "vp_diff": 0, "win_share": 0.} for p in plan]
    candidate = [{**row, "vp_diff": 2, "win_share": .5} for row in baseline]
    initial_path, candidate_path = tmp_path / "initial.jsonl", tmp_path / "candidate.jsonl"
    def write(path, records):
        path.write_text("\n".join(json.dumps(row) for row in records))
    write(initial_path, baseline)
    write(candidate_path, candidate)
    progress = paired_progress(candidate_path, initial_path)
    assert progress["seed_blocks"] == 3
    assert progress["vp_diff_gain_ci95"] == [2., 2.]
    assert progress["win_share_gain"] == .5
    candidate[0]["status"] = "failed"
    write(candidate_path, candidate)
    assert not paired_progress(candidate_path, initial_path)["paired_valid"]
    write(candidate_path, candidate[:-1])
    with pytest.raises(ValueError, match="plans differ"):
        paired_progress(candidate_path, initial_path)
