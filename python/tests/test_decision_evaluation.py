"""Fixed-panel evaluation, paired block statistics and isolation from live training."""

from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.evaluation.decision_evaluation import (
    durable_run,
    evaluate_run,
    evaluation_lock,
    interval,
    paired,
    prepare_panel,
    summarize,
)
from sevenwonders_rl.selfplay.decision_league import LeagueConfig, file_hash
from sevenwonders_rl.utils.decision_ppo import train

ROOT = Path(__file__).parents[1]


def records(blocks, shift=0):
    return [
        dict(
            block=b,
            left=l,
            right=r,
            seat=s,
            status="completed",
            reason=None,
            margin=b + shift,
            win_share=1 / 3,
            own_points=40 + b + shift,
            wonder_id=s,
            categories={"civilian": 40 + b + shift, "total": 40 + b + shift},
        )
        for b in range(blocks)
        for l in range(3)
        for r in range(3)
        for s in range(3)
    ]


def test_statistics_resample_whole_blocks_and_pair_by_game():
    plan = {"settings": {"blocks": 4, "seed": 1}}
    old = records(4)
    new = records(4, 3)
    summary = summarize(old, plan)
    assert summary["valid"] and summary["attempted"] == 108
    assert summary["margin"] == 1.5
    assert summary["ci95"]["margin"] == interval([0, 1, 2, 3], 1)
    delta = paired(list(reversed(new)), old, plan)
    assert delta["margin"] == {"change": 3.0, "ci95": [3.0, 3.0]}
    assert delta["win_share"]["change"] == 0
    failed = deepcopy(old)
    failed[0].update(status="failed", reason="engine error")
    assert not summarize(failed, plan)["valid"]
    assert summarize(failed, plan)["ci95"] == {}
    assert paired(new, failed, plan) is None
    assert interval([1], 1) is None


def start_run(path, steps=64):
    torch.set_num_threads(1)
    env = DecisionEnv()
    try:
        train(
            env,
            steps,
            31,
            path,
            rollout_size=16,
            epochs=1,
            league_config=LeagueConfig(snapshot_every=16, pool_max=4),
        )
    finally:
        env.close()


def test_real_fixed_panel_repeated_evaluation_and_failures(tmp_path, monkeypatch):
    run = tmp_path / "training"
    start_run(run)
    hashes = {
        str(p.relative_to(run)): file_hash(p) for p in run.rglob("*") if p.is_file()
    }
    first, path = evaluate_run(run, blocks=1, progress=None)
    assert first["valid"] and first["previous"] is None
    assert first["models"]["current"]["completed"] == 27
    assert len(first["models"]["current"]["by_pair"]) == 9
    assert first["paired_change"] is None
    plan = json.loads((run / "evaluation/panel/plan.json").read_text())
    assert [r["step"] for r in plan["references"]] == [16, 32, 48]
    assert first["candidate"]["step"] == 64
    rows = [json.loads(x) for x in (path / "current.jsonl").read_text().splitlines()]
    for row in rows:
        assert (
            sum(v for k, v in row["categories"].items() if k != "total")
            == row["own_points"]
        )
    for name, sha in hashes.items():
        assert file_hash(run / name) == sha
    second, second_path = evaluate_run(run, blocks=1, progress=None)
    assert second["valid"] and second["previous"]["step"] == 64
    assert second["plan_id"] == first["plan_id"]
    for metric in second["paired_change"].values():
        assert metric["change"] == 0
    repeat = [
        json.loads(x) for x in (second_path / "current.jsonl").read_text().splitlines()
    ]
    assert repeat == rows
    latest_before = (run / "evaluation/latest.json").read_bytes()
    from sevenwonders_rl.evaluation import decision_evaluation as module

    real = module.play_game

    def failure(*args):
        row = real(*args)
        row.update(status="failed", reason="controlled failure")
        return row

    monkeypatch.setattr(module, "play_game", failure)
    failed, _ = evaluate_run(run, blocks=1, progress=None)
    assert not failed["valid"] and failed["paired_change"] is None
    assert (run / "evaluation/latest.json").read_bytes() == latest_before
    for name, sha in hashes.items():
        assert file_hash(run / name) == sha
    with pytest.raises(ValueError, match="evaluation plan differs"):
        evaluate_run(run, blocks=2, progress=None)


def test_panel_and_snapshot_integrity_missing_references_and_lock(tmp_path):
    run = tmp_path / "training"
    start_run(run, steps=32)
    with pytest.raises(ValueError, match="Three trained"):
        evaluate_run(run, blocks=1, progress=None)
    env = DecisionEnv()
    try:
        train(env, 32, 31, run, run / "checkpoint.pt", rollout_size=16, epochs=1)
    finally:
        env.close()
    state, trained, _ = durable_run(run)
    output = tmp_path / "eval"
    output.mkdir()
    plan, _ = prepare_panel(output, run, state, trained, 1, 42)
    assert [r["step"] for r in plan["references"]] == [16, 32, 48]
    with evaluation_lock(output):
        with pytest.raises(ValueError, match="already running"):
            with evaluation_lock(output):
                pass
    (output / "panel/reference_0.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="reference opponent"):
        prepare_panel(output, run, state, trained, 1, 42)


def test_evaluation_runs_while_continuous_training_advances(tmp_path):
    run = tmp_path / "live"
    log = tmp_path / "train.log"
    command = [
        sys.executable,
        "-u",
        "scripts/train_decision_ppo.py",
        "--continuous",
        "--auto-resume",
        "--output-dir",
        str(run),
        "--snapshot-every",
        "32",
        "--rollout-size",
        "16",
        "--epochs",
        "1",
    ]

    def current_step():
        try:
            return json.loads((run / "training_status.json").read_text())["steps"]
        except FileNotFoundError:
            return 0

    with log.open("w") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 60
            while current_step() < 96:
                assert process.poll() is None, log.read_text()
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.05)
            before = current_step()
            summary, _ = evaluate_run(run, blocks=1, progress=None)
            assert summary["valid"]
            assert process.poll() is None and current_step() > before
            captured = summary["candidate"]["step"]
            assert captured <= current_step() and captured % 32 == 0
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=30)
            assert process.returncode == 0, log.read_text()
            assert "Evaluation" not in log.read_text()
            saved = torch.load(run / "checkpoint.pt", weights_only=True)
            completed = [
                json.loads(x) for x in (run / "episodes.jsonl").read_text().splitlines()
            ]
            assert len(completed) == saved["league"]["completed"]
            assert len(completed) == sum(
                r["games"] for r in saved["league"]["pfsp"]["results"]
            )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_workspace_evaluation_task_separate_from_continuous_task():
    tasks = json.loads((ROOT / ".vscode/tasks.json").read_text())["tasks"]
    task = next(t for t in tasks if "Snapshot separat evaluieren" in t["label"])
    start = next(t for t in tasks if t["label"].startswith("Training starten"))
    assert any(a.endswith("/eval_decision.py") for a in task["args"])
    assert (
        task["args"][task["args"].index("--run-dir") + 1]
        == start["args"][start["args"].index("--output-dir") + 1]
    )
    assert "--eval-games" not in start["args"]


def test_parallel_matches_serial_including_previous_model_and_canonical_journal(
    tmp_path,
):
    run = tmp_path / "training"
    start_run(run)
    hashes = {
        str(p.relative_to(run)): file_hash(p) for p in run.rglob("*") if p.is_file()
    }
    directory = tmp_path / "evaluation"
    first, path = evaluate_run(
        run, evaluation_dir=directory, blocks=1, progress=None, workers=1
    )
    second, parallel = evaluate_run(
        run, evaluation_dir=directory, blocks=1, progress=None, workers=3
    )
    assert first["valid"] and second["valid"]
    expected = [
        json.loads(x) for x in (path / "current.jsonl").read_text().splitlines()
    ]
    for name in ("current", "previous"):
        actual = [
            json.loads(x) for x in (parallel / f"{name}.jsonl").read_text().splitlines()
        ]
        assert actual == expected
    assert all(row["change"] == 0 for row in second["paired_change"].values())
    assert (
        json.loads((parallel / "manifest.json").read_text())["runtime"]["workers"] == 3
    )
    for name, sha in hashes.items():
        assert file_hash(run / name) == sha


def test_failed_pool_submission_preserves_finished_rows():
    from concurrent.futures import Future
    from sevenwonders_rl.evaluation.decision_evaluation import _parallel_rows

    coordinates = [(0, 0, 0, s) for s in range(3)]

    class Executor:
        def submit(self, function, task):
            if task[1][-1]:
                raise RuntimeError("controlled broken pool")
            future = Future()
            future.set_result(
                dict(block=0, left=0, right=0, seat=0, status="completed", reason=None)
            )
            return future

    actual = list(
        _parallel_rows(Executor(), "candidate", coordinates, {"engine_seeds": [12]})
    )
    assert len(actual) == 3 and actual[0]["status"] == "completed"
    assert all(
        row["status"] == "failed" and "controlled broken pool" in row["reason"]
        for row in actual[1:]
    )


pytestmark = pytest.mark.integration
