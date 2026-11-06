#!/usr/bin/env python3
"""Evaluate an immutable protocol-3 snapshot independently of ongoing training."""

import argparse
import json
from pathlib import Path

from sevenwonders_rl.evaluation.decision_evaluation import evaluate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument(
        "--evaluation-dir", type=Path, help="New directory for a different fixed panel"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=40,
        help="Seed blocks; 27 games per block per model",
    )
    parser.add_argument("--seed", type=int, default=159001)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel CPU games (default: 4; use 1 for serial evaluation)",
    )
    parser.add_argument(
        "--reference-steps",
        type=int,
        nargs=3,
        help="Three distinct trained snapshots; default: first three",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        summary, path = evaluate_run(
            args.run_dir,
            args.engine,
            args.evaluation_dir,
            args.blocks,
            args.seed,
            args.reference_steps,
            None if args.json else print,
            workers=args.workers,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Evaluation not started: {exc}\n")
    if args.json:
        print(
            json.dumps({**summary, "directory": str(path)}, indent=2, allow_nan=False)
        )
    if not summary["valid"]:
        parser.exit(1, "Evaluation incomplete; results and errors have been saved.\n")


if __name__ == "__main__":
    main()
