"""Active protocol-3 league integration, immutable opponents and crash recovery."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.features.decision_features import feature_contract
from sevenwonders_rl.selfplay.decision_league import (
    DecisionLeague,
    LeagueConfig,
    file_hash,
    training_directory,
)
from sevenwonders_rl.utils.decision_ppo import (
    DecisionActorCritic,
    evaluate,
    load_checkpoint,
    train,
)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def env():
    torch.set_num_threads(1)
    instance = DecisionEnv()
    try:
        yield instance
    finally:
        instance.close()


def make_league(tmp_path, config=None):
    tmp_path.mkdir(exist_ok=True)
    league = DecisionLeague(
        tmp_path,
        feature_contract("bga-second-edition"),
        config or LeagueConfig(snapshot_every=32),
        12,
    )
    model = DecisionActorCritic()
    league.publish(model, 0)
    return league, model


@pytest.mark.parametrize(
    "changes",
    [
        dict(snapshot_every=0),
        dict(pool_max=0),
        dict(power=float("nan")),
        dict(control_fraction=0),
        dict(half_life=-1),
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        LeagueConfig(**changes)


@pytest.mark.parametrize("seat", [0, 1, 2])
def test_relative_pair_assignment_immutable_cache_and_rng_isolation(
    tmp_path, env, monkeypatch, seat
):
    league, model = make_league(tmp_path)
    league.base_seat = seat
    league.publish(model, 32)
    monkeypatch.setattr(
        league.pool,
        "sample_pair",
        lambda rng, step: (
            "unused",
            "unused",
            dict(pair_left_step=0, pair_right_step=32, pair_probability=0.25),
        ),
    )
    initial_hash = file_hash(tmp_path / "league/checkpoint_0.pt")
    rng = torch.get_rng_state().clone()
    obs, _ = league.start_episode(env, 10, 32)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert env.agent_seat == seat
    assert env.opponents[(seat + 2) % 3].__self__ is league.cache[0]
    assert env.opponents[(seat + 1) % 3].__self__ is league.cache[32]
    assert all(
        not p.requires_grad for m in league.cache.values() for p in m.parameters()
    )
    frozen = {k: v.clone() for k, v in league.cache[0].state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1)
    assert file_hash(tmp_path / "league/checkpoint_0.pt") == initial_hash
    for k, v in frozen.items():
        torch.testing.assert_close(v, league.cache[0].state_dict()[k], rtol=0, atol=0)
    before = [o.__self__ for o in env.opponents.values()]
    env.step(int(np.flatnonzero(obs["action_mask"])[0]))
    assert [o.__self__ for o in env.opponents.values()] == before
    for s in range(3):
        assert len(env.memories[s].decisions) == 1
    with pytest.raises(ValueError, match="active episode"):
        league.start_episode(env, 11, 32)
    with pytest.raises(ValueError, match="immutable"):
        league.publish(model, 0)


def test_snapshot_cap_and_changed_file_rejection(tmp_path):
    league, model = make_league(tmp_path, LeagueConfig(snapshot_every=32, pool_max=2))
    league.publish(model, 32)
    league.publish(model, 64)
    assert [s for s, p in league.pool.eligible(64)] == [32, 64]
    assert len(league.pool.distribution(64)[0]) == 4
    league._policy(0)
    league._policy(32)
    league._policy(64)
    assert len(league.cache) == 2
    assert (tmp_path / "league/checkpoint_0.pt").exists()
    with (tmp_path / "league/checkpoint_32.pt").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        league._policy(32)


def test_real_training_feedback_snapshots_rotating_seats_and_evaluation_isolation(
    tmp_path, env
):
    config = LeagueConfig(snapshot_every=32, pool_max=3)
    model, metrics = train(
        env, 128, 91, tmp_path, rollout_size=32, epochs=1, league_config=config
    )
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    state = checkpoint["league"]
    selections, completed = (
        jsonl(tmp_path / "opponents.jsonl"),
        jsonl(tmp_path / "episodes.jsonl"),
    )
    assert metrics["actually_applied"] == 128 and metrics["completed_games"] >= 5
    assert [r["step"] for r in state["snapshots"]] == [0, 32, 64, 96, 128]
    assert (
        state["completed"]
        == len(completed)
        == sum(r["games"] for r in state["pfsp"]["results"])
    )
    assert [r["agent_seat"] for r in selections] == [
        i % 3 for i in range(len(selections))
    ]
    assert any((r["left_step"], r["right_step"]) != (0, 0) for r in completed)
    for r in selections:
        assert max(r["pair_left_step"], r["pair_right_step"]) <= r["selection_step"]
        for side in ("left", "right"):
            assert file_hash(tmp_path / "league" / r[side]["name"]) == r[side]["sha256"]
    report = json.loads((tmp_path / "league_status.json").read_text())
    assert len(report["active_checkpoints"]) == 3 and len(report["pairs"]) == 9
    assert sum(p["probability"] for p in report["pairs"]) == pytest.approx(1)
    for p in report["pairs"]:
        assert p["probability"] >= config.control_fraction / 9
    hashes = {p.name: file_hash(p) for p in tmp_path.iterdir() if p.is_file()}
    evaluate(env, model, 2, 501)
    assert hashes == {p.name: file_hash(p) for p in tmp_path.iterdir() if p.is_file()}
    loaded, _ = load_checkpoint(tmp_path / "checkpoint.pt", checkpoint["contract"])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, loaded.state_dict()[k], rtol=0, atol=0)


def test_interrupted_update_rolls_back_journals_and_future_snapshot(
    tmp_path, env, monkeypatch
):
    config = LeagueConfig(snapshot_every=32, pool_max=4)
    original = DecisionLeague.publish

    def crash(self, model, step):
        original(self, model, step)
        if step == 64:
            raise RuntimeError("controlled post-snapshot interruption")

    monkeypatch.setattr(DecisionLeague, "publish", crash)
    with pytest.raises(RuntimeError, match="controlled"):
        train(env, 96, 12, tmp_path, rollout_size=32, epochs=1, league_config=config)
    durable = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert durable["steps"] == 32
    assert (tmp_path / "league/checkpoint_64.pt").exists()
    orphan_hash = file_hash(tmp_path / "league/checkpoint_64.pt")
    used_seeds = [r["seed"] for r in jsonl(tmp_path / "opponents.jsonl")]
    # These reports and files must not become authoritative on resume.
    (tmp_path / "league_status.json").write_text('{"timestep":999999}')
    (tmp_path / "league/checkpoint_999999.pt").write_bytes(b"not a checkpoint")
    with (tmp_path / "episodes.jsonl").open("a") as f:
        f.write('{"partial":')
    monkeypatch.setattr(DecisionLeague, "publish", original)
    _, metrics = train(
        env, 64, 12, tmp_path, tmp_path / "checkpoint.pt", rollout_size=32, epochs=1
    )
    assert metrics["steps"] == 96
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=True)["league"]
    assert [r["step"] for r in state["snapshots"]] == [0, 32, 64, 96]
    assert (tmp_path / f"league/orphans/checkpoint_64-{orphan_hash}.pt").exists()
    completed = jsonl(tmp_path / "episodes.jsonl")
    assert (
        state["completed"]
        == len(completed)
        == sum(r["games"] for r in state["pfsp"]["results"])
    )
    selections = jsonl(tmp_path / "opponents.jsonl")
    durable_count = durable["league"]["selections"]
    assert selections[durable_count]["seed"] > max(used_seeds)
    assert len({r["seed"] for r in selections}) == len(selections)
    assert (tmp_path / "episodes.rolled-back").exists()
    assert all(r["pair_left_step"] != 999999 for r in selections)
    report = json.loads((tmp_path / "league_status.json").read_text())
    assert report["abandoned_games"] >= 1


def test_resume_config_and_inventory_validation_and_no_accidental_overwrite(
    tmp_path, env
):
    config = LeagueConfig(snapshot_every=16)
    train(env, 16, 42, tmp_path, rollout_size=16, epochs=1, league_config=config)
    resume = tmp_path / "checkpoint.pt"
    digest = file_hash(resume)
    for changed in (replace(config, power=3), replace(config, pool_max=2)):
        with pytest.raises(ValueError, match="configuration mismatch"):
            train(
                env,
                1,
                42,
                tmp_path,
                resume,
                rollout_size=16,
                epochs=1,
                league_config=changed,
            )
    with pytest.raises(ValueError, match="configuration mismatch"):
        train(env, 1, 42, tmp_path, resume, rollout_size=32, epochs=1)
    with pytest.raises(ValueError, match="without its league"):
        train(
            env, 1, 42, tmp_path, resume, rollout_size=16, epochs=1, league_config=False
        )
    with pytest.raises(ValueError, match="new output directory"):
        train(env, 1, 42, tmp_path, league_config=config)
    assert file_hash(resume) == digest
    snapshot = tmp_path / "league/checkpoint_0.pt"
    snapshot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="snapshot missing or changed"):
        train(env, 1, 42, tmp_path, resume, rollout_size=16, epochs=1)


def test_run_directory_lock(tmp_path):
    with training_directory(tmp_path, None):
        with pytest.raises(ValueError, match="Another trainer"):
            with training_directory(tmp_path, None):
                pass


def test_non_aligned_budgets_preserve_absolute_snapshot_boundaries(tmp_path, env):
    config = LeagueConfig(snapshot_every=16)
    train(env, 45, 17, tmp_path, rollout_size=24, epochs=1, league_config=config)
    first = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert first["steps"] == 45
    assert [r["step"] for r in first["league"]["snapshots"]] == [0, 16, 32]
    _, metrics = train(
        env, 18, 17, tmp_path, tmp_path / "checkpoint.pt", rollout_size=24, epochs=1
    )
    resumed = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert metrics["steps"] == 63 and metrics["actually_applied"] == 18
    assert [r["step"] for r in resumed["league"]["snapshots"]] == [0, 16, 32, 48]


def test_non_league_resume_cannot_silently_create_a_league(tmp_path, env):
    train(env, 4, 17, tmp_path, rollout_size=4, epochs=1)
    path = tmp_path / "checkpoint.pt"
    digest = file_hash(path)
    with pytest.raises(ValueError, match="silently add"):
        train(
            env,
            4,
            17,
            tmp_path,
            path,
            rollout_size=4,
            epochs=1,
            league_config=LeagueConfig(),
        )
    assert file_hash(path) == digest and not (tmp_path / "league").exists()


def test_cli_default_league_and_inherited_resume_configuration(tmp_path):
    root = Path(__file__).parents[1]
    base = [
        sys.executable,
        "scripts/train_decision_ppo.py",
        "--output-dir",
        str(tmp_path),
        "--rollout-size",
        "16",
        "--epochs",
        "1",
        "--seed",
        "4",
        "--json",
    ]
    first = subprocess.run(
        [*base, "--steps", "32", "--snapshot-every", "16"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["league"]["active_snapshots"] == 3
    second = subprocess.run(
        [*base, "--steps", "16", "--resume", str(tmp_path / "checkpoint.pt")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert second.returncode == 0, second.stderr
    result = json.loads(second.stdout)
    assert result["steps"] == 48 and result["league"]["active_snapshots"] == 4
    assert result["evaluation"] == []
    assert result["completed_games_total"] == result["league"]["completed_games"]
    # Old checkpoints remain resumable, but their missing runtime is never invented.
    legacy = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    for key in (
        "training_seconds_session",
        "training_seconds_total",
        "training_duration_complete",
    ):
        legacy["metrics"].pop(key)
    torch.save(legacy, tmp_path / "checkpoint.pt")
    human = subprocess.run(
        [
            *[x for x in base if x != "--json"],
            "--steps",
            "16",
            "--log-seconds",
            "0.001",
            "--resume",
            str(tmp_path / "checkpoint.pt"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert human.returncode == 0, human.stderr
    durable = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    total = durable["metrics"]["completed_games_total"]
    assert total == durable["league"]["completed"] >= result["completed_games_total"]
    assert (
        "RESUME TRAINING" in human.stdout and "TRAININGSFORTSCHRITT" not in human.stdout
    )
    assert "Insgesamt:" not in human.stdout and "Games/min:" in human.stdout
    assert "COMPLETED" in human.stdout
    assert "earlier sessions without timing data are excluded" in human.stdout
    assert not durable["metrics"]["training_duration_complete"]
    assert (
        durable["metrics"]["training_seconds_total"]
        == durable["metrics"]["training_seconds_session"]
    )


pytestmark = pytest.mark.integration
