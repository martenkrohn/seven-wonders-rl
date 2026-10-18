"""Analyze PPO training evaluation results with detailed action descriptives.

This script reads evaluation JSONL files from training runs and produces
detailed behavioral analysis similar to simulate_specialists_descriptives.py.

Usage:
    # Analyze a specific evaluation checkpoint
    python scripts/ppo_training/analyze_eval.py runs/200k_test/eval_50000.jsonl

    # Analyze all evals in a directory
    python scripts/ppo_training/analyze_eval.py runs/200k_test/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_eval(records: List[Dict[str, Any]]) -> str:
    """Generate summary report from evaluation JSONL records."""
    if not records:
        return "No records found."

    lines = []
    lines.append("=" * 80)
    lines.append(f"EVALUATION SUMMARY ({len(records)} games)")
    lines.append("=" * 80)

    # Aggregate metrics
    wins = [r["win"] for r in records]
    vp_diffs = [r["vp_diff"] for r in records]
    agent_vps = [r["agent_vp"] for r in records]

    lines.append(f"\nOverall Performance:")
    lines.append(f"  Win rate:     {np.mean(wins):.1%}")
    lines.append(f"  VP diff:      {np.mean(vp_diffs):+.2f} ± {np.std(vp_diffs):.2f}")
    lines.append(f"  Agent VP:     {np.mean(agent_vps):.2f} ± {np.std(agent_vps):.2f}")

    # Action counts (if available)
    if "action_counts" in records[0]:
        lines.append(f"\nAction Distribution (per game):")
        action_keys = ["discard", "build_wonder", "prod_build", "other"]
        
        for key in action_keys:
            counts = [r["action_counts"].get(key, 0) for r in records]
            m = np.mean(counts)
            s = np.std(counts)
            lines.append(f"  {key:12s}: {m:5.2f} ± {s:5.2f}")

    # Points breakdown (if available)
    if "points_breakdown" in records[0]:
        lines.append(f"\nVP Breakdown (per game):")
        vp_keys = ["civilian", "commercial", "guild", "military", "scientific", "wonder"]
        
        for key in vp_keys:
            vps = [r["points_breakdown"].get(key, 0) for r in records]
            m = np.mean(vps)
            s = np.std(vps)
            lines.append(f"  {key:12s}: {m:5.2f} ± {s:5.2f}")

    # Per-opponent breakdown (if available)
    if "opponents" in records[0]:
        lines.append(f"\nPer-Opponent Breakdown:")
        
        # Group by opponent combination
        opponent_groups = defaultdict(list)
        for r in records:
            opp = r.get("opponents", {})
            key = (opp.get("left", "unknown"), opp.get("right", "unknown"))
            opponent_groups[key].append(r)

        for (left, right), games in opponent_groups.items():
            wins = [g["win"] for g in games]
            vp_diffs = [g["vp_diff"] for g in games]
            
            lines.append(f"  vs {left} + {right} ({len(games)} games):")
            lines.append(f"    Win rate: {np.mean(wins):.1%}  |  VP diff: {np.mean(vp_diffs):+.2f} ± {np.std(vp_diffs):.2f}")

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PPO evaluation results")
    parser.add_argument("path", type=str, help="Path to JSONL file or directory containing eval_*.jsonl files")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        # Single file
        jsonl_files = [path]
    elif path.is_dir():
        # Directory: find all eval_*.jsonl files
        jsonl_files = sorted(path.glob("eval_*.jsonl"))
        if not jsonl_files:
            print(f"No eval_*.jsonl files found in {path}")
            return
    else:
        print(f"Path not found: {path}")
        return

    for jsonl_path in jsonl_files:
        print(f"\n{'='*80}")
        print(f"FILE: {jsonl_path}")
        print(f"{'='*80}")
        
        records = load_jsonl(jsonl_path)
        report = summarize_eval(records)
        print(report)


if __name__ == "__main__":
    main()
