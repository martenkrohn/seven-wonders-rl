"""Committed game thresholds, actual learner evaluation and training isolation."""

import json
from pathlib import Path
import time

import pytest
import torch

from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.evaluation.decision_intermediate import IntermediateEvaluations
from sevenwonders_rl.selfplay.decision_league import (
    LeagueConfig,
    atomic_json,
    file_hash,
)
from sevenwonders_rl.utils.decision_control import request_stop
from sevenwonders_rl.utils.decision_ppo import train


def test_threshold_migration_resume_and_evaluation_commit_recovery(tmp_path):
    old = {"steps": 215232, "completed_games_total": 11762}
    schedule = IntermediateEvaluations(tmp_path, 500, 4, old)
    assert schedule.next_games == 12000
    old["intermediate_evaluation"] = schedule.state_dict()
    resumed = IntermediateEvaluations(tmp_path, 500, 4, old)
    assert resumed.next_games == 12000
    old.update(steps=220000, completed_games_total=12001)
    committed = {
        "interval_games": 500,
        "next_games": 12500,
        "last": {"steps": 220000, "completed_games_total": 12001, "valid": True},
    }
    (tmp_path / "intermediate_evaluation").mkdir()
    atomic_json(tmp_path / "intermediate_evaluation/schedule.json", committed)
    resumed = IntermediateEvaluations(tmp_path, 500, 4, old)
    assert resumed.next_games == 12500 and resumed.last["valid"]
    # A rolled-back learner cannot adopt a future evaluation's commit marker.
    old["steps"] = 219999
    assert IntermediateEvaluations(tmp_path, 500, 4, old).next_games == 12000
    assert IntermediateEvaluations(tmp_path, 0, 4, old).next_games is None


@pytest.mark.parametrize("workers", [1, 2])
def test_real_evaluation_preserves_training_rng_and_league_and_resumes(
    tmp_path, workers
):
    plain = tmp_path / "plain"
    controlled = tmp_path / "controlled"
    env = DecisionEnv()
    try:
        args = dict(
            steps=64,
            seed=137,
            rollout_size=16,
            epochs=1,
            league_config=LeagueConfig(snapshot_every=256),
            workers=workers,
        )
        baseline, _ = train(env, output=plain, **args)
        model, metrics = train(env, output=controlled, evaluation_every_games=2, **args)
        assert all(
            torch.equal(baseline.state_dict()[k], v)
            for k, v in model.state_dict().items()
        )
        left = torch.load(plain / "checkpoint.pt", weights_only=True)
        right = torch.load(controlled / "checkpoint.pt", weights_only=True)
        assert torch.equal(left["torch_rng"], right["torch_rng"])
        assert left["league"] == right["league"]
        assert 2 <= metrics["completed_games_total"] <= 3
        state = metrics["intermediate_evaluation"]
        assert state["last"]["valid"] and state["next_games"] == 4
        assert state["last"]["completed"] == 27
        report = controlled / state["last"]["directory"]
        summary = json.loads((report / "summary.json").read_text())
        # No trained league snapshot exists yet. This must nevertheless test the
        # learner at the just-finished update, rather than snapshot zero.
        assert summary["candidate"]["step"] == state["last"]["steps"] > 0
        assert summary["references"][0]["step"] == 0
        assert [s["step"] for s in right["league"]["snapshots"]] == [0]
        count = len(list((controlled / "intermediate_evaluation").glob("eval-*")))
        _, resumed = train(
            env,
            1,
            137,
            controlled,
            controlled / "checkpoint.pt",
            rollout_size=16,
            epochs=1,
            workers=workers,
            evaluation_every_games=2,
        )
        assert resumed["intermediate_evaluation"]["next_games"] == 4
        assert (
            len(list((controlled / "intermediate_evaluation").glob("eval-*"))) == count
        )
        _, resumed = train(
            env,
            48,
            137,
            controlled,
            controlled / "checkpoint.pt",
            rollout_size=16,
            epochs=1,
            workers=workers,
            evaluation_every_games=2,
        )
        assert resumed["intermediate_evaluation"]["last"]["threshold_games"] == 4
        assert resumed["intermediate_evaluation"]["last"]["valid"]
        latest = controlled / resumed["intermediate_evaluation"]["last"]["directory"]
        summary = json.loads((latest / "summary.json").read_text())
        assert summary["previous"] is not None and summary["paired_change"] is not None
        assert summary["models"]["previous"]["completed"] == 27
    finally:
        env.close()


def test_failed_evaluation_keeps_checkpoint_rng_and_can_stop_cleanly(
    tmp_path, monkeypatch, capsys
):
    import sevenwonders_rl.evaluation.decision_evaluation as evaluator

    seen = []

    def fail(run, *args, **kwargs):
        # A real callback begins only after a complete durable update. Deliberate
        # randomness and delay verify RNG restoration and timing exclusion.
        before = file_hash(Path(run) / "checkpoint.pt")
        seen.append(torch.load(Path(run) / "checkpoint.pt", weights_only=True))
        torch.rand(19)
        time.sleep(0.15)
        assert request_stop(run)
        assert file_hash(Path(run) / "checkpoint.pt") == before
        raise RuntimeError("deliberate evaluation failure")

    monkeypatch.setattr(evaluator, "evaluate_run", fail)
    env = DecisionEnv()
    try:
        started = time.monotonic()
        _, metrics = train(
            env,
            None,
            19,
            tmp_path,
            rollout_size=16,
            epochs=1,
            league_config=LeagueConfig(snapshot_every=256),
            evaluation_every_games=1,
            progress=lambda m: None,
        )
        elapsed = time.monotonic() - started
        assert metrics["status"] == "stopped" and len(seen) == 1
        assert metrics["intermediate_evaluation"]["last"]["valid"] is False
        assert "deliberate evaluation failure" in capsys.readouterr().out
        saved = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
        assert torch.equal(saved["torch_rng"], seen[0]["torch_rng"])
        assert saved["steps"] == seen[0]["steps"] == metrics["steps"]
        assert elapsed - metrics["training_seconds_session"] >= 0.15
    finally:
        env.close()


pytestmark = pytest.mark.integration
