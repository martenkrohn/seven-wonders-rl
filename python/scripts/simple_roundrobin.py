#!/usr/bin/env python3
"""Simple round-robin evaluation without pandas dependency."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

import numpy as np
from sevenwonders_rl.envs import SevenWondersEnv
from sevenwonders_rl.policies import (
    BaselinePolicy,
    MilitaryTempoPolicy,
    MadScientistPolicy,
)


def run_matchup(agent_policy, opp1_policy, opp2_policy, games=50, seed=42):
    """Run a matchup and return win rate for agent."""
    
    env = SevenWondersEnv(
        repo_root=str(REPO_ROOT),
        agent_seat=0,
        seed=seed,
    )
    
    policies = [agent_policy, opp1_policy, opp2_policy]
    wins = 0
    total_vp = 0
    
    for game_idx in range(games):
        obs, info = env.reset(seed=seed + game_idx)
        done = False
        
        while not done:
            seat = info.get("current_seat", 0)
            policy = policies[seat]
            
            action = policy.select_action(obs, env._rng)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Check winner
        status = info.get("status", {})
        players = status.get("players", {})
        
        vp_scores = {int(seat): players[seat]["points"]["total"] for seat in players}
        winner = max(vp_scores, key=vp_scores.get)
        
        if winner == 0:
            wins += 1
        
        total_vp += vp_scores[0]
    
    env.close()
    
    return wins / games, total_vp / games


def main():
    policies = {
        "Baseline": BaselinePolicy(),
        "Military": MilitaryTempoPolicy(),
        "MadScientist": MadScientistPolicy(),
    }
    
    print("=" * 70)
    print("ROUND-ROBIN TOURNAMENT (50 games per matchup)")
    print("=" * 70)
    print()
    
    # Results table
    results = {}
    
    for agent_name, agent_policy in policies.items():
        print(f"Testing {agent_name}...")
        
        matchup_wins = []
        matchup_vps = []
        
        for opp_name, opp_policy in policies.items():
            if opp_name == agent_name:
                continue  # Skip self-play
            
            # Agent vs 2 opponents (same policy)
            win_rate, avg_vp = run_matchup(
                agent_policy, opp_policy, opp_policy, 
                games=50, seed=100
            )
            
            matchup_wins.append(win_rate)
            matchup_vps.append(avg_vp)
            
            print(f"  vs {opp_name:12s}: {win_rate*100:5.1f}% (VP: {avg_vp:.1f})")
        
        overall_win_rate = np.mean(matchup_wins)
        overall_avg_vp = np.mean(matchup_vps)
        
        results[agent_name] = {
            "win_rate": overall_win_rate,
            "avg_vp": overall_avg_vp,
        }
        
        print(f"  Overall:       {overall_win_rate*100:5.1f}% (VP: {overall_avg_vp:.1f})")
        print()
    
    # Summary ranking
    print("=" * 70)
    print("OVERALL RANKING")
    print("=" * 70)
    
    ranked = sorted(results.items(), key=lambda x: x[1]["win_rate"], reverse=True)
    
    for rank, (name, data) in enumerate(ranked, 1):
        print(f"{rank}. {name:12s}  {data['win_rate']*100:5.1f}%  (VP: {data['avg_vp']:.1f})")
    
    print()


if __name__ == "__main__":
    main()
