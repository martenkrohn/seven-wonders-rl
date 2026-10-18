#!/usr/bin/env python3
"""
Population Performance Tournament

Evaluates each policy as an agent for N games against a balanced opponent pool.
Reports both outcome metrics (win rate, VP diff) and style metrics (science, military, etc).

Key feature: balanced opponent scheduling ensures each opponent appears roughly equally.
"""

from pathlib import Path
import sys
import json
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

from sevenwonders_rl.envs import SevenWondersEnv
from sevenwonders_rl.policies import (
    BaselinePolicy,
    MilitaryTempoPolicy,
    MadScientistPolicy,
)

# ============================================================================
# Configuration
# ============================================================================

POLICY_POOL = {
    "Baseline": BaselinePolicy,
    "Military": MilitaryTempoPolicy,
    "MadScientist": MadScientistPolicy,
}

N_GAMES = 200  # Games per policy evaluation
SEED_BASE = 42


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class GameRecord:
    """Single game result."""
    game_id: int
    agent_policy: str
    opponent_1: str
    opponent_2: str
    seed: int
    
    # Outcome
    agent_vp: int
    agent_win: int
    best_opponent_vp: int
    vp_diff: int  # agent_vp - best_opponent_vp
    
    # Agent style
    agent_science_points: int
    agent_military_points: int
    agent_wonder_points: int
    agent_coin_end: int
    agent_discard_count: int
    agent_trade_spend: float


@dataclass
class PolicyResults:
    """Aggregated results for one policy."""
    policy_name: str
    n_games: int
    games: list = field(default_factory=list)
    
    # Opponent balance tracking
    opponent_counts: dict = field(default_factory=lambda: defaultdict(int))
    
    def add_game(self, record: GameRecord):
        """Add a game record."""
        self.games.append(record)
        self.opponent_counts[record.opponent_1] += 1
        self.opponent_counts[record.opponent_2] += 1
    
    def win_rate(self) -> float:
        """Win rate (%)."""
        if not self.games:
            return 0.0
        wins = sum(g.agent_win for g in self.games)
        return 100.0 * wins / len(self.games)
    
    def mean_vp(self) -> float:
        """Mean VP scored by agent."""
        if not self.games:
            return 0.0
        return sum(g.agent_vp for g in self.games) / len(self.games)
    
    def mean_vp_diff(self) -> float:
        """Mean VP diff (agent - best opponent)."""
        if not self.games:
            return 0.0
        return sum(g.vp_diff for g in self.games) / len(self.games)
    
    def vp_diff_percentiles(self) -> tuple:
        """(P10, P50, P90) of VP diff."""
        if not self.games:
            return (0.0, 0.0, 0.0)
        diffs = sorted([g.vp_diff for g in self.games])
        n = len(diffs)
        p10 = diffs[max(0, int(n * 0.1) - 1)]
        p50 = diffs[n // 2]
        p90 = diffs[min(n - 1, int(n * 0.9))]
        return (p10, p50, p90)
    
    def vp_diff_std(self) -> float:
        """Standard deviation of VP diff."""
        if not self.games:
            return 0.0
        diffs = [g.vp_diff for g in self.games]
        return float(np.std(diffs))
    
    def mean_science(self) -> float:
        """Mean science points."""
        if not self.games:
            return 0.0
        return sum(g.agent_science_points for g in self.games) / len(self.games)
    
    def mean_military(self) -> float:
        """Mean military points."""
        if not self.games:
            return 0.0
        return sum(g.agent_military_points for g in self.games) / len(self.games)
    
    def mean_wonder(self) -> float:
        """Mean wonder points."""
        if not self.games:
            return 0.0
        return sum(g.agent_wonder_points for g in self.games) / len(self.games)
    
    def mean_coins_end(self) -> float:
        """Mean coins at end of game."""
        if not self.games:
            return 0.0
        return sum(g.agent_coin_end for g in self.games) / len(self.games)
    
    def discard_rate(self) -> float:
        """Proportion of turns that were discards."""
        total_actions = 0
        total_discards = 0
        for g in self.games:
            # Rough estimate: ~20 actions per game
            total_actions += 20
            total_discards += g.agent_discard_count
        if total_actions == 0:
            return 0.0
        return 100.0 * total_discards / total_actions
    
    def mean_trade_spend(self) -> float:
        """Mean estimated trade spend per game."""
        if not self.games:
            return 0.0
        return sum(g.agent_trade_spend for g in self.games) / len(self.games)


# ============================================================================
# Opponent scheduling
# ============================================================================

def build_balanced_opponent_schedule(pool: list, n_games: int, seed: int = 42) -> list:
    """
    Build balanced opponent pairs for n_games.
    
    Each policy in pool appears roughly equally in the opponent slots.
    
    Args:
        pool: list of policy names
        n_games: number of games to schedule
        seed: random seed for shuffling
    
    Returns:
        list of (opponent_1, opponent_2) tuples, length n_games
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    
    # Total opponent slots
    total_slots = 2 * n_games
    
    # Build slot list: repeat each policy floor(total_slots / len(pool)) times
    base_count = total_slots // len(pool)
    remainder = total_slots % len(pool)
    
    slot_list = []
    for i, policy in enumerate(pool):
        count = base_count + (1 if i < remainder else 0)
        slot_list.extend([policy] * count)
    
    # Shuffle
    rng.shuffle(slot_list)
    
    # Pair them up
    pairs = [(slot_list[2*i], slot_list[2*i+1]) for i in range(n_games)]
    
    return pairs


# ============================================================================
# Game execution
# ============================================================================

def run_game(
    agent_policy_obj,
    opponent_1_policy_obj,
    opponent_2_policy_obj,
    seed: int,
    repo_root: str,
) -> dict:
    """
    Run a single 3-player game and return result dict.
    
    Agent is in seat 0, opponents in seats 1 and 2.
    """
    env = SevenWondersEnv(
        repo_root=repo_root,
        agent_seat=0,
        seed=seed,
    )
    
    obs, info = env.reset(seed=seed)
    done = False
    turns = 0
    
    # Track actions by agent
    discard_count = 0
    trade_spend_total = 0.0
    
    policies = [agent_policy_obj, opponent_1_policy_obj, opponent_2_policy_obj]
    
    try:
        while not done and turns < 25:
            # Get current player (0, 1, or 2)
            player = info.get("current_player", 0)
            
            mask = obs["action_mask"]
            legal = np.flatnonzero(mask == 1)
            
            if len(legal) == 0:
                break
            
            # Select action based on policy
            if player == 0:  # Agent
                action = agent_policy_obj.select_action(obs, env._rng)
                
                # Track discard
                if action // 7 == 3:
                    discard_count += 1
                
                # Track trade spend (estimate)
                feats = obs.get("action_feats")
                if feats is not None:
                    from sevenwonders_rl.features.action_feature_schema import IDX
                    if action < feats.shape[0]:
                        action_type = action // 7
                        if action_type in (0, 2):
                            trade_spend_total += float(feats[action, IDX.get("est_trade_cost_coins", 0)])
                        elif action_type == 1:
                            trade_spend_total += float(feats[action, IDX.get("wonder_est_trade_cost_coins", 0)])
            else:
                # Opponents play
                action = policies[player].select_action(obs, env._rng)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            turns += 1
        
        env.close()
        
        # Extract results
        status = info.get("status", {})
        players = status.get("players", {})
        
        agent_data = players.get("0", {})
        opponent_1_data = players.get("1", {})
        opponent_2_data = players.get("2", {})
        
        agent_vp = agent_data.get("points", {}).get("total", 0)
        opp_1_vp = opponent_1_data.get("points", {}).get("total", 0)
        opp_2_vp = opponent_2_data.get("points", {}).get("total", 0)
        
        best_opponent_vp = max(opp_1_vp, opp_2_vp)
        vp_diff = agent_vp - best_opponent_vp
        agent_win = 1 if agent_vp > opp_1_vp and agent_vp > opp_2_vp else 0
        
        # Style metrics
        agent_science = agent_data.get("points", {}).get("science", 0)
        agent_military = agent_data.get("points", {}).get("military", 0)
        agent_wonder = agent_data.get("points", {}).get("wonder", 0)
        agent_coins = agent_data.get("coins", 0)
        
        return {
            "agent_vp": agent_vp,
            "best_opponent_vp": best_opponent_vp,
            "vp_diff": vp_diff,
            "agent_win": agent_win,
            "agent_science": agent_science,
            "agent_military": agent_military,
            "agent_wonder": agent_wonder,
            "agent_coins": agent_coins,
            "discard_count": discard_count,
            "trade_spend": trade_spend_total,
        }
    
    except Exception as e:
        print(f"    Error in game: {e}")
        env.close()
        return None


# ============================================================================
# Main tournament
# ============================================================================

def run_population_tournament():
    """Run full population performance tournament."""
    
    print("=" * 90)
    print("POPULATION PERFORMANCE TOURNAMENT")
    print("=" * 90)
    print(f"Games per policy: {N_GAMES}")
    print(f"Opponent pool: {list(POLICY_POOL.keys())}")
    print(f"Seed base: {SEED_BASE}")
    print()
    
    results_by_policy = {}
    
    # Evaluate each policy
    for policy_name, policy_class in POLICY_POOL.items():
        print(f"Evaluating {policy_name}...")
        
        agent_policy = policy_class()
        policy_results = PolicyResults(policy_name=policy_name, n_games=N_GAMES)
        
        # Build balanced opponent schedule
        pool_names = list(POLICY_POOL.keys())
        opponent_pairs = build_balanced_opponent_schedule(
            pool_names, N_GAMES, seed=SEED_BASE
        )
        
        # Run games
        for game_id, (opp_1_name, opp_2_name) in enumerate(opponent_pairs):
            if (game_id + 1) % 50 == 0:
                print(f"  Game {game_id + 1}/{N_GAMES}...")
            
            opp_1_policy = POLICY_POOL[opp_1_name]()
            opp_2_policy = POLICY_POOL[opp_2_name]()
            
            seed = SEED_BASE + game_id
            result = run_game(
                agent_policy,
                opp_1_policy,
                opp_2_policy,
                seed=seed,
                repo_root=str(REPO_ROOT),
            )
            
            if result is None:
                continue
            
            # Create record
            record = GameRecord(
                game_id=game_id,
                agent_policy=policy_name,
                opponent_1=opp_1_name,
                opponent_2=opp_2_name,
                seed=seed,
                agent_vp=result["agent_vp"],
                agent_win=result["agent_win"],
                best_opponent_vp=result["best_opponent_vp"],
                vp_diff=result["vp_diff"],
                agent_science_points=result["agent_science"],
                agent_military_points=result["agent_military"],
                agent_wonder_points=result["agent_wonder"],
                agent_coin_end=result["agent_coins"],
                agent_discard_count=result["discard_count"],
                agent_trade_spend=result["trade_spend"],
            )
            
            policy_results.add_game(record)
        
        results_by_policy[policy_name] = policy_results
        print(f"  ✓ Completed {len(policy_results.games)} games\n")
    
    # Print results
    print_tournament_results(results_by_policy)


def print_tournament_results(results: dict):
    """Print formatted tournament results."""
    
    print("\n" + "=" * 90)
    print("OUTCOME METRICS (sorted by mean VP diff)")
    print("=" * 90)
    
    # Sort by mean VP diff (descending)
    sorted_results = sorted(
        results.values(),
        key=lambda r: r.mean_vp_diff(),
        reverse=True
    )
    
    print(f"\n{'Policy':<15} {'Win%':>7} {'Mean VP':>10} {'VP Diff':>10} {'Std Dev':>10} {'P10/P50/P90':>20}")
    print("-" * 90)
    
    for r in sorted_results:
        p10, p50, p90 = r.vp_diff_percentiles()
        percentiles_str = f"{p10:+.1f}/{p50:+.1f}/{p90:+.1f}"
        
        print(
            f"{r.policy_name:<15} {r.win_rate():>6.1f}% {r.mean_vp():>10.1f} "
            f"{r.mean_vp_diff():>+9.1f} {r.vp_diff_std():>10.2f} {percentiles_str:>20}"
        )
    
    # Style metrics
    print("\n" + "=" * 90)
    print("STYLE METRICS")
    print("=" * 90)
    
    print(f"\n{'Policy':<15} {'Science':>10} {'Military':>10} {'Wonder':>10} {'Coins':>10} {'Discard%':>10} {'Trade Avg':>10}")
    print("-" * 90)
    
    for r in sorted_results:
        print(
            f"{r.policy_name:<15} {r.mean_science():>10.1f} {r.mean_military():>10.1f} "
            f"{r.mean_wonder():>10.1f} {r.mean_coins_end():>10.1f} {r.discard_rate():>9.1f}% {r.mean_trade_spend():>10.1f}"
        )
    
    # Opponent balance
    print("\n" + "=" * 90)
    print("OPPONENT POOL BALANCE (how many times each appeared as opponent)")
    print("=" * 90)
    
    for r in sorted_results:
        print(f"\n{r.policy_name}:")
        total_opponent_slots = sum(r.opponent_counts.values())
        for opp_name in sorted(r.opponent_counts.keys()):
            count = r.opponent_counts[opp_name]
            expected = total_opponent_slots / len(r.opponent_counts)
            pct = 100.0 * count / total_opponent_slots if total_opponent_slots > 0 else 0.0
            balance_str = "✓" if abs(count - expected) < 10 else "⚠"
            print(f"  {balance_str} {opp_name:<15} {count:>3} times ({pct:>5.1f}%)")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    run_population_tournament()
