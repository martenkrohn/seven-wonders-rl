#!/usr/bin/env python3
"""Round-robin evaluation of all specialist policies.

This script runs tournaments comparing all bot policies against each other
to validate that they have distinct strategies and strengths/weaknesses.
"""

from pathlib import Path
import subprocess
import json
import numpy as np
import pandas as pd
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO_ROOT / "python" / ".venv" / "bin" / "python"
EVAL_SCRIPT = REPO_ROOT / "python" / "scripts" / "eval_policy.py"
RUNS_DIR = REPO_ROOT / "python" / "runs"

# All policies to evaluate
POLICIES = [
    "random",
    "heuristic",
    "military",
    "science",
    "economy",
    "wonder",
    "vp_finisher",
]

# Opponent configurations
OPPONENT_CONFIGS = [
    "random",     # vs 2 random opponents
    "heuristic",  # vs 2 heuristic opponents
    "mixed",      # vs 1 heuristic, 1 random
]


def run_evaluation(
    agent_policy: str,
    opponents: str,
    games: int,
    seed: int,
    jsonl_path: Path,
) -> Dict:
    """Run a single evaluation and return summary statistics."""
    
    cmd = [
        str(VENV_PYTHON),
        str(EVAL_SCRIPT),
        "--games", str(games),
        "--seed", str(seed),
        "--agent-policy", agent_policy,
        "--opponents", opponents,
        "--jsonl", str(jsonl_path),
    ]
    
    print(f"Running: {agent_policy} vs {opponents} ({games} games)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {agent_policy} vs {opponents} failed!")
        print(result.stderr)
        return None
    
    # Parse output to extract summary
    output_lines = result.stdout.strip().split("\n")
    summary_line = output_lines[-1] if output_lines else ""
    
    # Also read JSONL to compute additional stats
    games_data = []
    with jsonl_path.open("r") as f:
        for line in f:
            games_data.append(json.loads(line))
    
    if not games_data:
        return None
    
    wins = sum(g["win"] for g in games_data)
    vp_diffs = [g["vp_diff"] for g in games_data]
    agent_vps = [g["agent_total"] for g in games_data]
    
    discard_counts = [g["discard_count_agent"] for g in games_data]
    wonder_counts = [g["build_wonder_count_agent"] for g in games_data]
    prod_counts = [g["prod_build_count_agent"] for g in games_data]
    
    return {
        "agent_policy": agent_policy,
        "opponents": opponents,
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "vp_diff_mean": float(np.mean(vp_diffs)),
        "vp_diff_std": float(np.std(vp_diffs)),
        "agent_vp_mean": float(np.mean(agent_vps)),
        "agent_vp_std": float(np.std(agent_vps)),
        "discard_avg": float(np.mean(discard_counts)),
        "wonder_avg": float(np.mean(wonder_counts)),
        "prod_avg": float(np.mean(prod_counts)),
    }


def main():
    print("=" * 80)
    print("ROUND-ROBIN POLICY EVALUATION")
    print("=" * 80)
    print()
    
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    games_per_matchup = 30
    seed_base = 42000
    
    all_results = []
    
    # Run all matchups
    for agent_policy in POLICIES:
        for opponents in OPPONENT_CONFIGS:
            jsonl_path = RUNS_DIR / f"roundrobin_{agent_policy}_vs_{opponents}.jsonl"
            
            result = run_evaluation(
                agent_policy=agent_policy,
                opponents=opponents,
                games=games_per_matchup,
                seed=seed_base,
                jsonl_path=jsonl_path,
            )
            
            if result:
                all_results.append(result)
    
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Create DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Print results by opponent type
    for opp in OPPONENT_CONFIGS:
        print(f"\n### VS {opp.upper()} ###")
        subset = df[df["opponents"] == opp].sort_values("win_rate", ascending=False)
        
        print(f"\n{'Policy':<15} {'Win%':>6} {'VP Diff':>8} {'Agent VP':>9} {'Discard':>7} {'Wonder':>7} {'Prod':>6}")
        print("-" * 80)
        
        for _, row in subset.iterrows():
            print(
                f"{row['agent_policy']:<15} "
                f"{row['win_rate']*100:5.1f}% "
                f"{row['vp_diff_mean']:7.2f} "
                f"{row['agent_vp_mean']:8.2f} "
                f"{row['discard_avg']:6.2f} "
                f"{row['wonder_avg']:6.2f} "
                f"{row['prod_avg']:5.2f}"
            )
    
    # Overall rankings
    print("\n\n### OVERALL RANKINGS (averaged across opponent types) ###\n")
    
    overall = df.groupby("agent_policy").agg({
        "win_rate": "mean",
        "vp_diff_mean": "mean",
        "agent_vp_mean": "mean",
        "discard_avg": "mean",
        "wonder_avg": "mean",
        "prod_avg": "mean",
    }).sort_values("win_rate", ascending=False)
    
    print(f"{'Policy':<15} {'Avg Win%':>9} {'Avg VP Diff':>12} {'Avg Agent VP':>13} {'Discard':>7} {'Wonder':>7} {'Prod':>6}")
    print("-" * 90)
    
    for policy, row in overall.iterrows():
        print(
            f"{policy:<15} "
            f"{row['win_rate']*100:8.1f}% "
            f"{row['vp_diff_mean']:11.2f} "
            f"{row['agent_vp_mean']:12.2f} "
            f"{row['discard_avg']:6.2f} "
            f"{row['wonder_avg']:6.2f} "
            f"{row['prod_avg']:5.2f}"
        )
    
    # Strategy diversity analysis
    print("\n\n### STRATEGY DIVERSITY (action patterns) ###\n")
    
    diversity_df = df.groupby("agent_policy").agg({
        "discard_avg": ["mean", "std"],
        "wonder_avg": ["mean", "std"],
        "prod_avg": ["mean", "std"],
    })
    
    print(diversity_df)
    
    # Save full results
    results_path = RUNS_DIR / "roundrobin_results.json"
    with results_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nFull results saved to: {results_path}")
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print()
    
    # Key insights
    print("KEY INSIGHTS:")
    print("-------------")
    
    # Find strongest overall
    best_policy = overall.index[0]
    best_winrate = overall.iloc[0]["win_rate"]
    print(f"• Strongest overall: {best_policy} ({best_winrate*100:.1f}% avg win rate)")
    
    # Find most distinct strategies by action patterns
    action_variance = df.groupby("agent_policy")[["discard_avg", "wonder_avg", "prod_avg"]].std().sum(axis=1)
    most_varied = action_variance.idxmax()
    print(f"• Most unique strategy: {most_varied} (highest action pattern variance)")
    
    # Performance against tough opponents
    vs_heuristic = df[df["opponents"] == "heuristic"].sort_values("win_rate", ascending=False)
    best_vs_heuristic = vs_heuristic.iloc[0]["agent_policy"]
    best_vs_heuristic_wr = vs_heuristic.iloc[0]["win_rate"]
    print(f"• Best vs heuristic: {best_vs_heuristic} ({best_vs_heuristic_wr*100:.1f}% win rate)")
    
    print()


if __name__ == "__main__":
    main()
