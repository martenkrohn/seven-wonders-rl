"""Real subprocess stop, signal, hard-kill recovery and workspace task wiring."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

ROOT = Path(__file__).parents[1]


def wait_for(process, condition, log, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        if process.poll() is not None:
            pytest.fail(f"Trainer exited: {log.read_text()}")
        time.sleep(0.05)
    pytest.fail(f"Trainer timed out: {log.read_text()}")


def status(directory):
    try:
        return json.loads((directory / "training_status.json").read_text())
    except FileNotFoundError:
        return {}


def command(directory):
    return [
        sys.executable,
        "-u",
        "scripts/train_decision_ppo.py",
        "--continuous",
        "--auto-resume",
        "--output-dir",
        str(directory),
        "--snapshot-every",
        "32",
        "--rollout-size",
        "16",
        "--epochs",
        "1",
        "--log-seconds",
        "0.01",
    ]


def stop(directory):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_decision_ppo.py",
            "--stop",
            "--output-dir",
            str(directory),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_continuous_stop_resume_signals_and_kill(tmp_path):
    directory = tmp_path / "run"
    prior_step = 0
    prior_games = 0
    prior_seconds = 0
    # Independent sessions exercise the task stop, terminal Ctrl+C, hard kill,
    # and recovery from that hard kill followed by SIGTERM.
    for index, method in enumerate(("request", "sigint", "kill", "sigterm")):
        log = tmp_path / f"session-{index}.log"
        with log.open("w") as stream:
            process = subprocess.Popen(
                command(directory), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
            try:
                wait_for(
                    process,
                    lambda: status(directory).get("steps", 0) >= prior_step + 48,
                    log,
                )
                if method == "request":
                    assert "Stop requested" in stop(directory)
                else:
                    process.send_signal(
                        {
                            "sigint": signal.SIGINT,
                            "sigterm": signal.SIGTERM,
                            "kill": signal.SIGKILL,
                        }[method]
                    )
                process.wait(timeout=30)
                assert process.returncode == (
                    -signal.SIGKILL if method == "kill" else 0
                ), log.read_text()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        durable = torch.load(directory / "checkpoint.pt", weights_only=True)
        assert durable["steps"] >= prior_step + 48
        assert durable["metrics"]["completed_games_total"] >= prior_games
        assert durable["metrics"]["configuration"]["steps"] is None
        assert durable["metrics"]["continuous"]
        measured = durable["metrics"]
        assert measured["training_duration_complete"]
        assert measured["training_seconds_session"] > 0
        assert measured["training_seconds_total"] == pytest.approx(
            prior_seconds + measured["training_seconds_session"]
        )
        prior_seconds = measured["training_seconds_total"]
        if method != "kill":
            assert durable["metrics"]["status"] == "stopped"
            assert status(directory)["status"] == "stopped"
            assert "CLEANLY STOPPED" in log.read_text()
        assert "Laufbudget" not in log.read_text() and "Restzeit" not in log.read_text()
        assert "Continuous until stopped" in log.read_text()
        prior_step = durable["steps"]
        prior_games = durable["metrics"]["completed_games_total"]
    rows = [
        json.loads(x) for x in (directory / "episodes.jsonl").read_text().splitlines()
    ]
    assert (
        len(rows)
        == prior_games
        == sum(r["games"] for r in durable["league"]["pfsp"]["results"])
    )
    assert "No training is active" in stop(directory)


def test_vscode_tasks_use_active_trainer_and_same_run():
    data = json.loads((ROOT / ".vscode/tasks.json").read_text())
    tasks = data["tasks"]
    start = next(t for t in tasks if "Training starten" in t["label"])
    end = next(t for t in tasks if "Training sauber stoppen" in t["label"])
    for t in (start, end):
        assert any(a.endswith("/scripts/train_decision_ppo.py") for a in t["args"])
        assert "run_fresh_pair_pfsp.py" not in " ".join(t["args"])
    assert "--continuous" in start["args"] and "--auto-resume" in start["args"]
    assert start["args"][start["args"].index("--eval-every-games") + 1] == "5000"
    assert start["args"][start["args"].index("--workers") + 1] == "4"
    assert "--steps" not in start["args"] and "--stop" in end["args"]
    assert (
        start["args"][start["args"].index("--output-dir") + 1]
        == end["args"][end["args"].index("--output-dir") + 1]
    )
