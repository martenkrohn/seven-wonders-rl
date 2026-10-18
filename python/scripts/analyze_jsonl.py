#!/usr/bin/env python3
"""Analyze JSONL output from eval_policy.py"""

import argparse
import json
from pathlib import Path
import numpy as np


def analyze_jsonl(jsonl_path: Path) -> None:
    games = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            games.append(json.loads(line))
    
    n = len(games)
    print(f"\n{'='*60}")
    print(f"Analysis of {jsonl_path.name}")
    print(f"{'='*60}")
    print(f"Total games: {n}\n")
    
    # Truncation analysis
    if "truncated" in games[0]:
        truncated = np.array([g.get("truncated", False) for g in games])
        terminated = np.array([g.get("terminated", False) for g in games])
        turn_ends = np.array([g.get("turn_end", -1) for g in games])
        
        print("=== Completion Status ===")
        print(f"Terminated (normal): {terminated.sum()}/{n} ({terminated.mean()*100:.1f}%)")
        print(f"Truncated (timeout):  {truncated.sum()}/{n} ({truncated.mean()*100:.1f}%)")
        if truncated.sum() > 0:
            trunc_reasons = [g.get("truncated_reason", "unknown") for g in games if g.get("truncated", False)]
            from collections import Counter
            reason_counts = Counter(trunc_reasons)
            for reason, count in reason_counts.items():
                print(f"  - {reason}: {count}")
        print(f"\nTurn distribution:")
        print(f"  Turn 20 (full): {(turn_ends == 20).sum()}/{n}")
        print(f"  Turn < 20:      {(turn_ends < 20).sum()}/{n}")
        if (turn_ends < 20).sum() > 0:
            early_turns = turn_ends[turn_ends < 20]
            print(f"    Early turn range: {early_turns.min()} - {early_turns.max()}")
            print(f"    Early turn mean:  {early_turns.mean():.1f}")
        print()
    
    # Win rate and VP stats
    wins = np.array([g["win"] for g in games])
    vp_diffs = np.array([g["vp_diff"] for g in games])
    agent_totals = np.array([g["agent_total"] for g in games])
    opp_bests = np.array([g["opp_best"] for g in games])
    
    # Mean-based metrics (if available)
    opp_means = None
    vp_diffs_from_mean = None
    agent_ranks = None
    if "opp_mean" in games[0]:
        opp_means = np.array([g.get("opp_mean", 0) for g in games])
        vp_diffs_from_mean = np.array([g.get("vp_diff_from_mean", 0) for g in games])
        
        # Compute ranks
        agent_ranks = []
        for g in games:
            totals = g.get("totals", [])
            if totals:
                agent_total = g["agent_total"]
                sorted_opps = sorted([t for i, t in enumerate(totals) if i != 0 or len(totals) <= 2])
                # Rank is position when sorted descending
                rank = 1 + sum(1 for t in totals if t > agent_total and t != agent_total)
                agent_ranks.append(rank)
        agent_ranks = np.array(agent_ranks) if agent_ranks else None
    
    print("=== Overall Performance ===")
    print(f"Win rate:        {wins.mean():.3f} ({wins.sum()}/{n})")
    print(f"Agent VP:        {agent_totals.mean():.2f} ± {agent_totals.std():.2f}")
    print(f"Opponent best:   {opp_bests.mean():.2f} ± {opp_bests.std():.2f}")
    if opp_means is not None:
        print(f"Opponent mean:   {opp_means.mean():.2f} ± {opp_means.std():.2f}")
    print(f"VP difference (vs best):   {vp_diffs.mean():.2f} ± {vp_diffs.std():.2f}")
    print(f"  Min/Max:       {vp_diffs.min():.0f} / {vp_diffs.max():.0f}")
    print(f"  Median:        {np.median(vp_diffs):.2f}")
    print(f"  25th/75th:     {np.percentile(vp_diffs, 25):.2f} / {np.percentile(vp_diffs, 75):.2f}")
    if vp_diffs_from_mean is not None:
        print(f"\nVP difference (vs mean):   {vp_diffs_from_mean.mean():.2f} ± {vp_diffs_from_mean.std():.2f}")
        print(f"  Min/Max:       {vp_diffs_from_mean.min():.0f} / {vp_diffs_from_mean.max():.0f}")
    
    if agent_ranks is not None:
        print(f"\nAgent Rank:      {agent_ranks.mean():.2f} ± {agent_ranks.std():.2f}")
        print(f"  1st place:     {(agent_ranks == 1).sum()}/{n}")
        print(f"  2nd place:     {(agent_ranks == 2).sum()}/{n}")
        print(f"  3rd place:     {(agent_ranks == 3).sum()}/{n}")
    
    # Points breakdown
    print("\n=== Points Breakdown (Agent) ===")
    points_keys = ["civilian", "commercial", "guild", "military", "scientific", "wonder"]
    for key in points_keys:
        values = np.array([g["points"].get(key, 0) for g in games])
        print(f"{key:12s}:  {values.mean():5.2f} ± {values.std():4.2f}  (min={values.min():.0f}, max={values.max():.0f})")
    
    # Wonder stages
    wonder_stages = np.array([g.get("wonder_stage", 0) for g in games])
    print(f"\nWonder stages:   {wonder_stages.mean():.2f} ± {wonder_stages.std():.2f}")
    print(f"  0 stages: {(wonder_stages == 0).sum()}/{n}")
    print(f"  1 stage:  {(wonder_stages == 1).sum()}/{n}")
    print(f"  2 stages: {(wonder_stages == 2).sum()}/{n}")
    print(f"  3+ stages: {(wonder_stages >= 3).sum()}/{n}")
    
    # Action counts
    print("\n=== Action Counts (per game, 21 total) ===")
    discards = np.array([g.get("discard_count_agent", 0) for g in games])
    wonders = np.array([g.get("build_wonder_count_agent", 0) for g in games])
    prods = np.array([g.get("prod_build_count_agent", 0) for g in games])
    invalids = np.array([g.get("invalid_requested_count_agent", 0) for g in games])
    
    print(f"Discard:         {discards.mean():.2f} ± {discards.std():.2f}  ({discards.mean()/21*100:.1f}% of actions)")
    print(f"Build wonder:    {wonders.mean():.2f} ± {wonders.std():.2f}  ({wonders.mean()/21*100:.1f}% of actions)")
    print(f"Build prod:      {prods.mean():.2f} ± {prods.std():.2f}  ({prods.mean()/21*100:.1f}% of actions)")
    print(f"Other builds:    {(21 - discards - wonders - prods).mean():.2f} ± {(21 - discards - wonders - prods).std():.2f}  ({(21 - discards - wonders - prods).mean()/21*100:.1f}% of actions)")
    
    if invalids.sum() > 0:
        print(f"\nInvalid fallbacks: {invalids.sum():.0f}/{n*21:.0f} total actions ({invalids.sum()/(n*21)*100:.2f}%)")
        print(f"  Games with invalids: {(invalids > 0).sum()}/{n} ({(invalids > 0).mean()*100:.1f}%)")
        print(f"  Mean per game: {invalids.mean():.2f} ± {invalids.std():.2f}")
    
    # Final production
    if "final_prod_self" in games[0]:
        print("\n=== Final Production (Agent) ===")
        prod_names = ["wood", "ore", "clay", "stone", "loom", "glass", "papyrus"]
        final_prods = np.array([g["final_prod_self"] for g in games])
        for i, name in enumerate(prod_names):
            values = final_prods[:, i]
            print(f"{name:8s}: {values.mean():.2f} ± {values.std():.2f}  (max={values.max():.0f})")
        
        total_raw = final_prods[:, :4].sum(axis=1)
        total_manuf = final_prods[:, 4:].sum(axis=1)
        print(f"\nTotal raw:   {total_raw.mean():.2f} ± {total_raw.std():.2f}")
        print(f"Total manuf: {total_manuf.mean():.2f} ± {total_manuf.std():.2f}")
    
    print(f"\n{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze eval_policy.py JSONL output")
    parser.add_argument("jsonl", type=str, help="Path to JSONL file")
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found")
        return
    
    analyze_jsonl(jsonl_path)


if __name__ == "__main__":
    main()
