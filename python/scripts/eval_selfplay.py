"""Neural-only holdout evaluation, paired against the run's untrained policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from sevenwonders_rl.evaluation.runner import evaluate_model
from sevenwonders_rl.evaluation.selfplay import evaluation_plan, paired_progress, reference
from sevenwonders_rl.utils.checkpoints import load_model, file_hash
from sevenwonders_rl.utils.runtime import atomic_json, exchange_io_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", help="Defaults to the run's latest.pt")
    parser.add_argument("--reference", nargs="+", help="Optional frozen trained opponents from this run")
    parser.add_argument("--blocks", type=int, default=20, help="Independent seeds; 12 games per block with two references")
    parser.add_argument("--seed", type=int, default=9001, help="Separate from model-selection seeds")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--io-root")
    parser.add_argument("--action-mode", choices=["stochastic", "greedy"], default="stochastic")
    parser.add_argument("--opponent-inference", choices=["batched", "serial"], default="batched")
    args = parser.parse_args(argv)
    if args.blocks <= 0 or args.seed < 0:
        parser.error("blocks must be positive and seed nonnegative")
    torch.set_num_threads(1)
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    stored = json.loads((run_dir / "references.json").read_text())["references"]
    references = [reference(path) for path in args.reference] if args.reference else stored
    plan = evaluation_plan(references, args.seed, args.blocks)
    candidate = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "latest.pt"
    initial = Path(stored[0]["checkpoint"])
    if file_hash(initial) != stored[0]["sha256"]:
        raise ValueError("Initial reference changed")
    io_root = exchange_io_root(args.io_root, output / "exchange", "holdout")
    root = Path(__file__).resolve().parents[2]
    summaries = {}
    try:
        for name, path in (("initial", initial), ("candidate", candidate)):
            print(f"Evaluating {name}: {len(plan)} games", flush=True)
            summaries[name] = evaluate_model(load_model(path), "cpu", plan, root,
                io_root / name, output / f"{name}.jsonl", action_mode=args.action_mode,
                checkpoint_id=file_hash(path), batch_opponents=args.opponent_inference == "batched")
        summaries["progress"] = paired_progress(output / "candidate.jsonl", output / "initial.jsonl")
        summaries.update(seed=args.seed, seed_blocks=args.blocks, references=references,
                         io_root=str(io_root), opponent_inference=args.opponent_inference,
                         candidate= {**summaries["candidate"], "checkpoint": str(candidate),
                                     "sha256": file_hash(candidate)})
        atomic_json(output / "summary.json", summaries)
        print(json.dumps(summaries, indent=2), flush=True)
    finally:
        shutil.rmtree(io_root)
    if not all(summaries[name]["valid_for_selection"] for name in ("initial", "candidate")):
        raise RuntimeError("Holdout evaluation contains incomplete games; inspect records")
    return summaries


if __name__ == "__main__":
    main()
