#!/usr/bin/env python3
"""Measure the current trainer with a bounded, disposable league run."""

import argparse
import json
from pathlib import Path
import tempfile
import time

from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.selfplay.decision_league import LeagueConfig
from sevenwonders_rl.utils.decision_ppo import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.workers < 1:
        parser.error("Steps and workers must be positive")
    with tempfile.TemporaryDirectory(prefix="sevenwonders-benchmark-") as directory:
        env = DecisionEnv(args.engine)
        started = time.monotonic()
        try:
            _, metrics = train(
                env,
                args.steps,
                args.seed,
                directory,
                workers=args.workers,
                league_config=LeagueConfig(),
            )
        finally:
            env.close()
        elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "steps": metrics["actually_applied"],
                "workers": args.workers,
                "completed_games": metrics["completed_games_total"],
                "total_seconds": elapsed,
                "training_seconds": metrics["training_seconds_session"],
                "decisions_per_second_including_startup": args.steps / elapsed,
                "note": "A technical throughput measurement, not a learning-quality comparison.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
