"""Shared-learner budgets, snapshot boundaries, recovery and owned processes."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.selfplay.decision_league import (
    DecisionLeague,
    LeagueConfig,
    file_hash,
)
from sevenwonders_rl.utils.decision_control import request_stop
from sevenwonders_rl.utils.decision_ppo import train
from sevenwonders_rl.utils.decision_workers import DecisionWorkers

ROOT = Path(__file__).parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_run(directory, step):
    state = torch.load(directory / "checkpoint.pt", weights_only=True)
    league = state["league"]
    episodes = (
        rows(directory / "episodes.jsonl")
        if (directory / "episodes.jsonl").exists()
        else []
    )
    assert state["steps"] == step
    assert (
        league["completed"]
        == len(episodes)
        == sum(r["games"] for r in league["pfsp"]["results"])
    )
    selections = rows(directory / "opponents.jsonl")
    assert len(selections) == league["selections"]
    assert len({r["seed"] for r in selections}) == len(selections)
    assert [r["index"] for r in selections] == list(range(len(selections)))
    assert [r["step"] for r in league["snapshots"]] == list(
        range(0, step + 1, league["config"]["snapshot_every"])
    )
    for row in selections:
        for side in ("left", "right"):
            assert (
                file_hash(directory / "league" / row[side]["name"])
                == row[side]["sha256"]
            )
    assert all(torch.isfinite(value).all() for value in state["model"].values())
    return state


def test_exact_odd_budgets_multi_game_feedback_and_serial_resume(tmp_path):
    env = DecisionEnv()
    directory = tmp_path / "run"
    try:
        _, metrics = train(
            env,
            97,
            23,
            directory,
            rollout_size=16,
            epochs=1,
            league_config=LeagueConfig(snapshot_every=13, pool_max=4),
            workers=3,
        )
        state = check_run(directory, 97)
        assert metrics["completed_games_total"] >= 3
        assert len(state["league"]["active_episodes"]) <= 3
        assert set(r["slot"] for r in rows(directory / "opponents.jsonl")) == {0, 1, 2}
        seeds = state["metrics"]["episode_seeds"]
        prior = len(seeds)
        _, metrics = train(
            env,
            17,
            23,
            directory,
            directory / "checkpoint.pt",
            rollout_size=16,
            epochs=1,
        )
        state = check_run(directory, 114)
        assert metrics["configuration"]["workers"] == 3
        assert min(state["metrics"]["episode_seeds"][prior:]) > max(seeds)
        assert metrics["league"]["abandoned_games"] >= 1
        train(
            env,
            13,
            23,
            directory,
            directory / "checkpoint.pt",
            rollout_size=16,
            epochs=1,
            workers=1,
        )
        assert check_run(directory, 127)["metrics"]["configuration"]["workers"] == 1
    finally:
        env.close()


def test_parallel_crash_rolls_back_and_reserves_tail_seeds(tmp_path, monkeypatch):
    env = DecisionEnv()
    directory = tmp_path / "run"
    try:
        train(
            env,
            32,
            19,
            directory,
            rollout_size=16,
            epochs=1,
            league_config=LeagueConfig(snapshot_every=16),
            workers=2,
        )
        publish = DecisionLeague.publish

        def crash(self, model, step):
            publish(self, model, step)
            if step == 48:
                raise RuntimeError("controlled post-snapshot crash")

        monkeypatch.setattr(DecisionLeague, "publish", crash)
        with pytest.raises(RuntimeError, match="controlled"):
            train(
                env,
                16,
                19,
                directory,
                directory / "checkpoint.pt",
                rollout_size=16,
                epochs=1,
            )
        assert torch.load(directory / "checkpoint.pt", weights_only=True)["steps"] == 32
        orphan = file_hash(directory / "league/checkpoint_48.pt")
        state = torch.load(directory / "checkpoint.pt", weights_only=True)
        with (directory / "opponents.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"index": state["league"]["selections"], "seed": 999999})
                + '\n{"partial":'
            )
        monkeypatch.setattr(DecisionLeague, "publish", publish)
        train(
            env,
            21,
            19,
            directory,
            directory / "checkpoint.pt",
            rollout_size=16,
            epochs=1,
        )
        resumed = check_run(directory, 53)
        assert max(resumed["metrics"]["episode_seeds"]) > 999999
        assert (directory / f"league/orphans/checkpoint_48-{orphan}.pt").exists()
        assert (directory / "opponents.rolled-back").exists()
    finally:
        env.close()


def test_worker_failure_cleans_engines_and_shared_memory():
    env = DecisionEnv()
    pool = DecisionWorkers(2, env)
    memories = [m.name for m in pool.memories]
    workdir = Path(pool.workdir.name)
    try:
        pool.send(0, "reset", {"agent_seat": 0, "seed": 24})
        pool.receive(0)
        worker = pool.processes[0]
        children = (
            Path(f"/proc/{worker.pid}/task/{worker.pid}/children").read_text().split()
        )
        assert children
        worker.kill()
        worker.join(timeout=5)
        with pytest.raises(RuntimeError, match="Game worker"):
            pool.send(0, "step", 0)
            pool.receive(0)
    finally:
        pool.close()
        env.close()
    from multiprocessing.shared_memory import SharedMemory

    for name in memories:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)
    assert not workdir.exists()
    for pid in children:
        path = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 5
        while path.exists() and path.read_text().split()[2] != "Z":
            assert time.monotonic() < deadline
            time.sleep(0.05)


def test_worker_startup_failure_leaves_a_resumable_initial_checkpoint(
    tmp_path, monkeypatch
):
    from sevenwonders_rl.utils import decision_ppo as module

    real = module.DecisionWorkers

    def failure(*args, **kwargs):
        raise RuntimeError("controlled startup failure")

    env = DecisionEnv()
    directory = tmp_path / "run"
    try:
        monkeypatch.setattr(module, "DecisionWorkers", failure)
        with pytest.raises(RuntimeError, match="startup failure"):
            train(
                env,
                8,
                42,
                directory,
                rollout_size=8,
                epochs=1,
                league_config=LeagueConfig(snapshot_every=8),
                workers=2,
            )
        state = torch.load(directory / "checkpoint.pt", weights_only=True)
        assert state["steps"] == 0 and state["league"]["selections"] == 0
        monkeypatch.setattr(module, "DecisionWorkers", real)
        train(
            env, 8, 42, directory, directory / "checkpoint.pt", rollout_size=8, epochs=1
        )
        check_run(directory, 8)
    finally:
        env.close()


def test_parallel_continuous_stop_hard_kill_and_auto_resume(tmp_path):
    directory = tmp_path / "live"
    prior = 0
    command = [
        sys.executable,
        "-u",
        "scripts/train_decision_ppo.py",
        "--continuous",
        "--auto-resume",
        "--output-dir",
        str(directory),
        "--rollout-size",
        "8",
        "--snapshot-every",
        "32",
        "--epochs",
        "1",
    ]
    for index, method in enumerate(("request", "kill", "sigterm")):
        path = tmp_path / f"log-{index}"
        with path.open("w") as stream:
            process = subprocess.Popen(
                command + (["--workers", "2"] if index == 0 else []),
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 60
                while True:
                    assert process.poll() is None, path.read_text()
                    try:
                        step = json.loads(
                            (directory / "training_status.json").read_text()
                        )["steps"]
                    except FileNotFoundError:
                        step = 0
                    if step >= prior + 24:
                        break
                    assert time.monotonic() < deadline, path.read_text()
                    time.sleep(0.05)
                if method == "request":
                    assert request_stop(directory)
                else:
                    process.send_signal(
                        signal.SIGKILL if method == "kill" else signal.SIGTERM
                    )
                process.wait(timeout=30)
                assert process.returncode == (
                    -signal.SIGKILL if method == "kill" else 0
                ), path.read_text()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        state = torch.load(directory / "checkpoint.pt", weights_only=True)
        assert state["metrics"]["configuration"]["workers"] == 2
        if method != "kill":
            assert state["metrics"]["status"] == "stopped"
        prior = state["steps"]
    check_run(directory, prior)


pytestmark = pytest.mark.integration
