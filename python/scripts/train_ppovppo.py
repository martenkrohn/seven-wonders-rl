"""Train PPO with self-play against frozen checkpoint snapshots.

This script implements a self-play league where:
1. One learner trains continuously
2. Snapshots are saved periodically to form a "pool" of opponents
3. Each episode, two opponents are sampled from the pool
4. The pool grows over time, creating gradually harder opponents
5. A lag parameter prevents training against too-recent snapshots
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

# Prevent CPU oversubscription with parallel envs
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from sevenwonders_rl.envs import SevenWondersEnv, SeatPolicy
from sevenwonders_rl.envs.sevenwonders_env import N_ACTIONS
from sevenwonders_rl.features.action_feature_schema import FEATURE_NAMES, K as K_ACTION_FEATS
from sevenwonders_rl.policies.ppo_checkpoint_policy import PPOCheckpointPolicy
from sevenwonders_rl.selfplay.opponent_pool import OpponentPool
from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.ppo_utils import flatten_global, masked_logits, select_action_from_obs

# Fixed evaluation matchups (for deterministic cross-checkpoint comparison)
EVAL_FIXED_MATCHUPS = [
    ("baseline", "baseline"),
    ("mad_scientist", "mad_scientist"),
    ("military", "military"),
    ("random", "random"),
]

# Fixed seed list for evaluation (deterministic across all checkpoints)
EVAL_FIXED_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def _fmt_hours(seconds: float) -> str:
    if seconds <= 0:
        return "0.00h"
    return f"{seconds / 3600.0:.2f}h"


def _training_speed(steps_done: int, elapsed_sec: float) -> float:
    if elapsed_sec <= 0:
        return 0.0
    return steps_done / elapsed_sec


def _eta_seconds(total_steps: int, steps_done: int, steps_per_sec: float) -> float:
    remaining = max(0, total_steps - steps_done)
    if steps_per_sec <= 1e-12:
        return 0.0
    return remaining / steps_per_sec


def _print_eval_block(
    timestep: int,
    total_timesteps: int,
    start_time: float,
    metrics: Dict[str, Any],
    new_best: bool,
    best_vp_diff: float,
) -> None:
    elapsed_sec = time.time() - start_time
    steps_per_sec = _training_speed(timestep, elapsed_sec)
    eta_sec = _eta_seconds(total_timesteps, timestep, steps_per_sec)
    percent = 100.0 * timestep / total_timesteps if total_timesteps > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"  ✓ Evaluation Results ({metrics['total_games']} games):")
    print(
        f"    Win rate: {metrics['win_rate']:.1%}  |  VP diff: {metrics['vp_diff_mean']:+.2f}±{metrics['vp_diff_std']:.2f}"
    )
    print(
        f"    Agent VP: {metrics['agent_vp_mean']:.1f}  |  Progress: {percent:.1f}% ({timestep:,}/{total_timesteps:,})"
    )
    print(f"    Training time: {_fmt_hours(elapsed_sec)}")
    print(f"    JSONL: {metrics['jsonl_path']}")
    print(
        f"    Elapsed: {_fmt_hours(elapsed_sec)}  |  ETA: {_fmt_hours(eta_sec)}  |  Speed: {steps_per_sec:.0f} steps/sec"
    )
    if new_best:
        print(f"    New best model! Best VP diff so far: {best_vp_diff:+.2f}")
    print("=" * 80)


def _evaluate_fixed_opponents(
    model: ActorCritic,
    device: torch.device,
    agent_seat: int,
    timestep: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run fixed-matchup evaluation for deterministic cross-checkpoint comparison.

    Returns aggregated metrics + writes per-game JSONL.
    """
    from sevenwonders_rl.features.action_feature_schema import IDX
    from sevenwonders_rl.policies.registry import make_policy

    repo_root = Path(__file__).resolve().parents[2]

    all_games = []

    for left_policy_name, right_policy_name in EVAL_FIXED_MATCHUPS:
        for seed in EVAL_FIXED_SEEDS:
            # Build deterministic opponent policies
            left_policy = make_policy(left_policy_name, seed=seed + 10000)
            right_policy = make_policy(right_policy_name, seed=seed + 20000)

            # Assign to seats (agent at agent_seat, others get left/right)
            other_seats = [s for s in (0, 1, 2) if s != agent_seat]
            opponent_policies = [
                SeatPolicy("random"),
                SeatPolicy("random"),
                SeatPolicy("random"),
            ]
            opponent_policies[other_seats[0]] = SeatPolicy(
                left_policy_name, policy=left_policy
            )
            opponent_policies[other_seats[1]] = SeatPolicy(
                right_policy_name, policy=right_policy
            )

            env = SevenWondersEnv(
                repo_root=str(repo_root),
                agent_seat=agent_seat,
                opponent_policies=opponent_policies,
                reward_scale=1.0,
            )

            obs, info = env.reset(seed=seed)
            done = False

            # Track action types
            action_counts = {
                "discard": 0,
                "build_wonder": 0,
                "prod_build": 0,
                "other": 0,
            }

            while not done:
                action = select_action_from_obs(
                    obs, model, device, deterministic=True
                )

                # Count action type
                action_feats = obs.get("action_feats")
                if action_feats is not None and action < len(action_feats):
                    row = action_feats[action]
                    if row[IDX["action_type_3"]] > 0.5:
                        action_counts["discard"] += 1
                    elif row[IDX["action_type_1"]] > 0.5:
                        action_counts["build_wonder"] += 1
                    elif (
                        sum(
                            row[IDX[f"delta_prod_{r}"]]
                            for r in [
                                "wood",
                                "ore",
                                "clay",
                                "stone",
                                "loom",
                                "glass",
                                "papyrus",
                            ]
                        )
                        > 0
                        or row[IDX["delta_prod_choice_raw"]] > 0
                        or row[IDX["delta_prod_choice_manuf"]] > 0
                    ):
                        action_counts["prod_build"] += 1
                    else:
                        action_counts["other"] += 1

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            status = info.get("status", {})
            players = status.get("players", {})
            totals = [int(players[str(sid)]["points"]["total"]) for sid in (0, 1, 2)]
            agent_total = totals[agent_seat]
            opp_best = max(t for i, t in enumerate(totals) if i != agent_seat)
            vp_diff = agent_total - opp_best
            win = 1 if vp_diff > 0 else 0

            points = players[str(agent_seat)]["points"]

            game_record = {
                "checkpoint_step": timestep,
                "seed": seed,
                "opponents": {"left": left_policy_name, "right": right_policy_name},
                "win": win,
                "agent_vp": agent_total,
                "vp_diff": vp_diff,
                "totals": totals,
                "points_breakdown": {
                    "civilian": int(points.get("civilian", 0)),
                    "commercial": int(points.get("commercial", 0)),
                    "guild": int(points.get("guild", 0)),
                    "military": int(points.get("military", 0)),
                    "scientific": int(points.get("scientific", 0)),
                    "wonder": int(points.get("wonder", 0)),
                    "total": agent_total,
                },
                "action_counts": action_counts,
            }
            all_games.append(game_record)

            env.close()

    # Write JSONL
    jsonl_path = out_dir / f"eval_{timestep}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for game in all_games:
            f.write(json.dumps(game) + "\n")

    # Aggregate metrics
    wins = np.array([g["win"] for g in all_games], dtype=np.float32)
    vp_diffs = np.array([g["vp_diff"] for g in all_games], dtype=np.float32)
    agent_totals = np.array([g["agent_vp"] for g in all_games], dtype=np.float32)

    # Per-matchup breakdown
    matchup_stats = {}
    for left, right in EVAL_FIXED_MATCHUPS:
        matchup_key = f"{left}_vs_{right}"
        matchup_games = [
            g
            for g in all_games
            if g["opponents"]["left"] == left
            and g["opponents"]["right"] == right
        ]
        if matchup_games:
            matchup_wins = [g["win"] for g in matchup_games]
            matchup_vp_diffs = [g["vp_diff"] for g in matchup_games]
            matchup_stats[matchup_key] = {
                "win_rate": float(np.mean(matchup_wins)),
                "vp_diff_mean": float(np.mean(matchup_vp_diffs)),
                "games": len(matchup_games),
            }

    return {
        "win_rate": float(wins.mean()),
        "vp_diff_mean": float(vp_diffs.mean()),
        "vp_diff_std": float(vp_diffs.std(ddof=0)),
        "agent_vp_mean": float(agent_totals.mean()),
        "total_games": len(all_games),
        "matchup_breakdown": matchup_stats,
        "jsonl_path": str(jsonl_path),
    }


def _evaluate_primary_selfplay(
    model: ActorCritic,
    device: torch.device,
    agent_seat: int,
    timestep: int,
    out_dir: Path,
    pool: OpponentPool,
    rng: np.random.Generator,
    n_games: int,
    policy_cache: "OrderedDict[str, PPOCheckpointPolicy]",
    cache_max: int,
) -> Optional[Dict[str, Any]]:
    """
    Primary evaluation: PPO vs snapshot pool.

    This is used once snapshots exist (Option A).
    It samples opponent snapshots from the pool and runs n_games episodes.
    """
    # Need at least 2 eligible snapshots to populate both opponent seats
    if pool.pool_size(timestep) < 2:
        return None

    def _get_cached_policy(path: str) -> PPOCheckpointPolicy:
        if path in policy_cache:
            pol = policy_cache.pop(path)
            policy_cache[path] = pol
            return pol
        pol = PPOCheckpointPolicy(path, device="cpu", deterministic=False)
        policy_cache[path] = pol
        while len(policy_cache) > cache_max:
            policy_cache.popitem(last=False)
        return pol

    repo_root = Path(__file__).resolve().parents[2]
    all_games: List[Dict[str, Any]] = []

    for _ in range(n_games):
        p1 = pool.sample_path(rng, timestep)
        p2 = pool.sample_path(rng, timestep)
        if p1 is None or p2 is None:
            continue

        pol1 = _get_cached_policy(p1)
        pol2 = _get_cached_policy(p2)
        pol1.set_seed(int(rng.integers(0, 2**31 - 1)))
        pol2.set_seed(int(rng.integers(0, 2**31 - 1)))

        other_seats = [s for s in (0, 1, 2) if s != agent_seat]
        opponent_policies = [
            SeatPolicy("random"),
            SeatPolicy("random"),
            SeatPolicy("random"),
        ]
        opponent_policies[other_seats[0]] = SeatPolicy("ppo_snapshot", policy=pol1)
        opponent_policies[other_seats[1]] = SeatPolicy("ppo_snapshot", policy=pol2)

        env = SevenWondersEnv(
            repo_root=str(repo_root),
            agent_seat=agent_seat,
            opponent_policies=opponent_policies,
            reward_scale=1.0,
        )

        seed = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            action = select_action_from_obs(obs, model, device, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        status = info.get("status", {})
        players = status.get("players", {})
        totals = [int(players[str(sid)]["points"]["total"]) for sid in (0, 1, 2)]
        agent_total = totals[agent_seat]
        opp_best = max(t for i, t in enumerate(totals) if i != agent_seat)
        vp_diff = agent_total - opp_best
        win = 1 if vp_diff > 0 else 0

        all_games.append(
            {
                "checkpoint_step": timestep,
                "seed": seed,
                "opponents": {"left": p1, "right": p2},
                "win": win,
                "agent_vp": agent_total,
                "vp_diff": vp_diff,
                "totals": totals,
            }
        )
        env.close()

    if len(all_games) == 0:
        return None

    jsonl_path = out_dir / f"eval_primary_{timestep}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for g in all_games:
            f.write(json.dumps(g) + "\n")

    wins = np.array([g["win"] for g in all_games], dtype=np.float32)
    vp_diffs = np.array([g["vp_diff"] for g in all_games], dtype=np.float32)
    agent_vps = np.array([g["agent_vp"] for g in all_games], dtype=np.float32)

    return {
        "win_rate": float(wins.mean()),
        "vp_diff_mean": float(vp_diffs.mean()),
        "vp_diff_std": float(vp_diffs.std(ddof=0)),
        "agent_vp_mean": float(agent_vps.mean()),
        "total_games": int(len(all_games)),
        "jsonl_path": str(jsonl_path),
        "pool_total": int(pool.total_size()),
        "pool_eligible": int(pool.pool_size(timestep)),
    }


def _evaluate_vs_pool(
    model: ActorCritic,
    device: torch.device,
    agent_seat: int,
    timestep: int,
    out_dir: Path,
    pool: OpponentPool,
    rng: np.random.Generator,
    n_games: int,
    policy_cache: "OrderedDict[str, PPOCheckpointPolicy]",
    cache_max: int,
) -> Optional[Dict[str, Any]]:
    """Evaluate agent versus snapshot pool opponents."""
    if pool.pool_size(timestep) < 2:
        return None

    def _get_cached_policy(path: str) -> PPOCheckpointPolicy:
        if path in policy_cache:
            policy = policy_cache.pop(path)
            policy_cache[path] = policy
            return policy
        policy = PPOCheckpointPolicy(path, device="cpu", deterministic=False)
        policy_cache[path] = policy
        while len(policy_cache) > cache_max:
            policy_cache.popitem(last=False)
        return policy

    repo_root = Path(__file__).resolve().parents[2]
    all_games = []

    for i in range(n_games):
        p1 = pool.sample_path(rng, timestep)
        p2 = pool.sample_path(rng, timestep)
        if p1 is None or p2 is None:
            continue

        pol1 = _get_cached_policy(p1)
        pol2 = _get_cached_policy(p2)
        pol1.set_seed(int(rng.integers(0, 2**31 - 1)))
        pol2.set_seed(int(rng.integers(0, 2**31 - 1)))

        other_seats = [s for s in (0, 1, 2) if s != agent_seat]
        opponent_policies = [SeatPolicy("random"), SeatPolicy("random"), SeatPolicy("random")]
        opponent_policies[other_seats[0]] = SeatPolicy("ppo_snapshot", policy=pol1)
        opponent_policies[other_seats[1]] = SeatPolicy("ppo_snapshot", policy=pol2)

        env = SevenWondersEnv(
            repo_root=str(repo_root),
            agent_seat=agent_seat,
            opponent_policies=opponent_policies,
            reward_scale=1.0,
        )

        seed = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            action = select_action_from_obs(obs, model, device, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        status = info.get("status", {})
        players = status.get("players", {})
        totals = [int(players[str(sid)]["points"]["total"]) for sid in (0, 1, 2)]
        agent_total = totals[agent_seat]
        opp_best = max(t for i, t in enumerate(totals) if i != agent_seat)
        vp_diff = agent_total - opp_best
        win = 1 if vp_diff > 0 else 0

        all_games.append(
            {
                "checkpoint_step": timestep,
                "seed": seed,
                "opponents": {"left": p1, "right": p2},
                "win": win,
                "agent_vp": agent_total,
                "vp_diff": vp_diff,
                "totals": totals,
            }
        )

        env.close()

    if not all_games:
        return None

    jsonl_path = out_dir / f"eval_pool_{timestep}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for game in all_games:
            f.write(json.dumps(game) + "\n")

    wins = np.array([g["win"] for g in all_games], dtype=np.float32)
    vp_diffs = np.array([g["vp_diff"] for g in all_games], dtype=np.float32)
    agent_totals = np.array([g["agent_vp"] for g in all_games], dtype=np.float32)

    return {
        "win_rate": float(wins.mean()),
        "vp_diff_mean": float(vp_diffs.mean()),
        "vp_diff_std": float(vp_diffs.std(ddof=0)),
        "agent_vp_mean": float(agent_totals.mean()),
        "total_games": len(all_games),
        "jsonl_path": str(jsonl_path),
    }


def _feature_schema_hash() -> str:
    import hashlib

    data = "\n".join(FEATURE_NAMES).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _git_commit() -> str:
    for key in ("GIT_COMMIT", "GITHUB_SHA"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        )
        return bool(out.strip())
    except Exception:
        return False


def sample_episode_opponents(
    pool: OpponentPool,
    env_rng: np.random.Generator,
    current_step: int,
    agent_seat: int,
    bootstrap_mode: str,
    pool_min_size: int = 3,
    policy_cache: Optional["OrderedDict[str, PPOCheckpointPolicy]"] = None,
    cache_max: int = 16,
) -> List[SeatPolicy]:
    """
    Sample two opponent policies for an episode.

    Uses a bootstrap strategy:
    - If pool is too small, use bots (baseline/mad_scientist/military/random)
    - Otherwise, sample from pool with 70% chance

    Args:
        pool: OpponentPool instance
        env_rng: numpy RNG for sampling
        current_step: current training step
        agent_seat: which seat is the learner
        bootstrap_mode: "bots" or "random" for early pool
        pool_min_size: minimum pool size before using snapshots

    Returns:
        List of SeatPolicy objects for seats 0, 1, 2
    """
    from sevenwonders_rl.policies.registry import make_policy
    
    other_seats = [s for s in (0, 1, 2) if s != agent_seat]

    def _get_cached_policy(path: str) -> PPOCheckpointPolicy:
        if policy_cache is None:
            return PPOCheckpointPolicy(path, device="cpu", deterministic=False)
        if path in policy_cache:
            policy = policy_cache.pop(path)
            policy_cache[path] = policy
            return policy
        policy = PPOCheckpointPolicy(path, device="cpu", deterministic=False)
        policy_cache[path] = policy
        while len(policy_cache) > cache_max:
            policy_cache.popitem(last=False)
        return policy

    # Default: all random
    opp = [SeatPolicy("random"), SeatPolicy("random"), SeatPolicy("random")]

    # Check if pool has enough snapshots
    pool_size = pool.pool_size(current_step)

    if pool_size < pool_min_size:
        # Bootstrap phase: use fixed bots
        if bootstrap_mode == "bots":
            bot_names = ["baseline", "mad_scientist"]
            for idx, seat in enumerate(other_seats):
                bot_policy = make_policy(bot_names[idx], seed=int(env_rng.integers(0, 2**31 - 1)))
                opp[seat] = SeatPolicy(bot_names[idx], policy=bot_policy)
        # else: stay random
        return opp

    # Pool available: mix between snapshots and bots
    # With 70% probability, use snapshots; 30% use bots for diversity
    if env_rng.random() < 0.7:
        # Sample from pool
        p1 = pool.sample_path(env_rng, current_step)
        p2 = pool.sample_path(env_rng, current_step)

        if p1 is not None and p2 is not None:
            try:
                pol1 = _get_cached_policy(p1)
                pol2 = _get_cached_policy(p2)

                # Per-episode seeding for reproducibility
                pol1.set_seed(int(env_rng.integers(0, 2**31 - 1)))
                pol2.set_seed(int(env_rng.integers(0, 2**31 - 1)))

                opp[other_seats[0]] = SeatPolicy("ppo_snapshot", policy=pol1)
                opp[other_seats[1]] = SeatPolicy("ppo_snapshot", policy=pol2)
            except (FileNotFoundError, ValueError):
                # Fallback to bots if snapshot load fails
                if bootstrap_mode == "bots":
                    bot_names = ["baseline", "mad_scientist"]
                    for idx, seat in enumerate(other_seats):
                        bot_policy = make_policy(bot_names[idx], seed=int(env_rng.integers(0, 2**31 - 1)))
                        opp[seat] = SeatPolicy(bot_names[idx], policy=bot_policy)
    else:
        # Use bots for diversity
        if bootstrap_mode == "bots":
            bot_names = ["baseline", "military"]
            for idx, seat in enumerate(other_seats):
                bot_policy = make_policy(bot_names[idx], seed=int(env_rng.integers(0, 2**31 - 1)))
                opp[seat] = SeatPolicy(bot_names[idx], policy=bot_policy)

    return opp


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO with self-play")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-seat", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument(
        "--n-envs", type=int, default=2, help="Number of parallel envs"
    )
    parser.add_argument(
        "--rollout-horizon", type=int, default=256, help="Steps per env before update"
    )
    parser.add_argument(
        "--io-root", type=str, default=None, help="IO root dir (default: repo_root/io_train)"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minibatch-epochs", type=int, default=4)
    parser.add_argument(
        "--reward-scale", type=float, default=10.0, help="Scale terminal reward"
    )
    parser.add_argument("--eval-every", type=int, default=256_000)
    parser.add_argument(
        "--eval-vs-pool-games",
        type=int,
        default=0,
        help="Optional extra eval games vs snapshot pool (0 disables)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="Number of games for the primary evaluation (pool eval after snapshots exist).",
    )
    parser.add_argument(
        "--disable-fixed-eval-after-snapshots",
        action="store_true",
        help="Stop fixed bot evaluation once any snapshots exist in league_dir.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50_000,
        help="Progress log every N timesteps",
    )
    parser.add_argument("--out-dir", type=str, default="python/runs/ppo_selfplay")
    # Self-play specific args
    parser.add_argument("--league-dir", type=str, default=None, help="Dir for checkpoints (default: out_dir)")
    parser.add_argument("--pool-max", type=int, default=80, help="Max snapshots in pool")
    parser.add_argument(
        "--pool-lag-steps",
        type=int,
        default=256_000,
        help="Don't sample snapshots newer than (current_step - lag)",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=128_000,
        help="Save snapshots every N steps",
    )
    parser.add_argument(
        "--bootstrap",
        choices=["random", "bots"],
        default="bots",
        help="Bootstrap strategy when pool is empty",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve league_dir (where snapshots are saved)
    league_dir = Path(args.league_dir) if args.league_dir else out_dir
    league_dir.mkdir(parents=True, exist_ok=True)

    # Resolve io_root
    if args.io_root:
        io_root = Path(args.io_root)
    else:
        io_root = repo_root / "io_train"
    io_root.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    from sevenwonders_rl.utils.timing import TimingBuckets

    timers = TimingBuckets()

    print("=" * 80)
    print("PPO SELF-PLAY TRAINING SETUP")
    print("=" * 80)
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Eval every: {args.eval_every:,} steps")
    print(f"Snapshot every: {args.snapshot_every:,} steps")
    print(f"Pool max size: {args.pool_max}")
    print(f"Pool lag: {args.pool_lag_steps:,} steps")
    print(f"Bootstrap mode: {args.bootstrap}")
    print(f"Output dir: {out_dir}")
    print(f"League dir: {league_dir}")
    print(f"Device: cpu")
    print()
    print("Creating environments...")

    # Create opponent pool
    pool = OpponentPool(
        ckpt_dir=str(league_dir),
        max_size=args.pool_max,
        min_step_lag=args.pool_lag_steps,
        latest_k=8,
    )

    # Per-env RNGs for opponent sampling
    env_rngs = [
        np.random.default_rng(args.seed + 10_000 + i) for i in range(args.n_envs)
    ]

    pool_eval_rng = np.random.default_rng(args.seed + 99_999)
    policy_cache: "OrderedDict[str, PPOCheckpointPolicy]" = OrderedDict()

    # Create vectorized envs with unique io directories
    envs = []
    for rank in range(args.n_envs):
        env_io_dir = io_root / f"env_{rank:03d}" / "io"
        env_io_dir.mkdir(parents=True, exist_ok=True)

        # Start with random opponents (will be sampled before each reset)
        env = SevenWondersEnv(
            repo_root=str(repo_root),
            agent_seat=args.agent_seat,
            io_dir=str(env_io_dir),
            opponent_policies=[
                SeatPolicy("random"),
                SeatPolicy("random"),
                SeatPolicy("random"),
            ],
            seed=args.seed,
            env_id=rank,
            reward_scale=args.reward_scale,
            timers=timers,
        )
        envs.append(env)

    # Initialize all envs with sampled opponents
    obs_list = []
    for i, (env, env_rng) in enumerate(zip(envs, env_rngs)):
        opps = sample_episode_opponents(
            pool=pool,
            env_rng=env_rng,
            current_step=0,
            agent_seat=args.agent_seat,
            bootstrap_mode=args.bootstrap,
            policy_cache=policy_cache,
        )
        env.opponent_policies = opps
        obs, info = env.reset(seed=args.seed + i)
        obs_list.append(obs)

    print(f"✓ Created {args.n_envs} environments")
    print()
    print("Initializing model...")

    # Get dimensions from first env
    global_dim = flatten_global(obs_list[0]).shape[0]
    action_feat_dim = obs_list[0]["action_feats"].shape[1]
    obs_keys = sorted(list(obs_list[0].keys()))

    device = torch.device("cpu")
    model = ActorCritic(global_dim, action_feat_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "timestep,win_rate,vp_diff_mean,vp_diff_std,agent_vp_mean\n",
            encoding="utf-8",
        )

    timestep = 0
    episode_seeds = [args.seed + 1000 + i for i in range(args.n_envs)]
    best_vp_diff = float("-inf")

    print(
        f"✓ Model initialized (global_dim={global_dim}, action_feat_dim={action_feat_dim})"
    )
    print()
    print("=" * 80)
    print("STARTING SELF-PLAY TRAINING")
    print("=" * 80)
    print(f"Rollout horizon: {args.rollout_horizon} steps/env")
    print(f"Learning rate: {args.learning_rate}")
    print(f"First eval at: {args.eval_every:,} steps")
    print()

    start_time = time.time()
    update_idx = 0
    next_eval_at = args.eval_every
    next_snapshot_at = args.snapshot_every
    next_log_at = args.log_every

    prev_log_step = 0
    prev_log_time = start_time
    ema_steps_per_sec: Optional[float] = None

    episodes_total = 0
    episodes_selfplay = 0
    episodes_bootstrap = 0

    while timestep < args.total_timesteps:
        update_idx += 1
        # Rollout storage
        global_buf = []
        action_feats_buf = []
        mask_buf = []
        actions_buf = []
        logprob_buf = []
        value_buf = []
        reward_buf = []
        done_buf = []

        for step_idx in range(args.rollout_horizon):
            # Collect batch from all envs
            batch_global = []
            batch_action_feats = []
            batch_mask = []

            for obs in obs_list:
                batch_global.append(flatten_global(obs))
                batch_action_feats.append(obs["action_feats"])
                batch_mask.append(obs["action_mask"])

            batch_global = torch.tensor(
                np.array(batch_global), dtype=torch.float32, device=device
            )
            batch_action_feats = torch.tensor(
                np.array(batch_action_feats), dtype=torch.float32, device=device
            )
            batch_mask = torch.tensor(
                np.array(batch_mask), dtype=torch.float32, device=device
            )

            with timers.time("train.model_forward"):
                with torch.no_grad():
                    logits, values = model.forward_batch(
                        batch_global, batch_action_feats
                    )
                    logits = masked_logits(logits, batch_mask)

            with timers.time("train.sample_action"):
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            # Two-phase stepping
            prev_turns = []
            for env_idx, (env, action) in enumerate(zip(envs, actions)):
                prev_turn = env.send_action(
                    int(action.item()), status=obs_list[env_idx].get("_status")
                )
                prev_turns.append(prev_turn)

            # Phase 2: Wait for all envs
            new_obs_list = []
            for env_idx, (env, prev_turn, action) in enumerate(
                zip(envs, prev_turns, actions)
            ):
                obs, reward, terminated, truncated, info = env.wait_next(prev_turn)
                done = terminated or truncated

                # Store transitions
                global_buf.append(batch_global[env_idx].cpu().numpy())
                action_feats_buf.append(batch_action_feats[env_idx].cpu().numpy())
                mask_buf.append(batch_mask[env_idx].cpu().numpy())
                actions_buf.append(int(action.item()))
                logprob_buf.append(float(logprobs[env_idx].item()))
                value_buf.append(float(values[env_idx].item()))
                reward_buf.append(float(reward))
                done_buf.append(float(done))

                timestep += 1

                if done:
                    episode_seeds[env_idx] += args.n_envs

                    # Sample fresh self-play opponents for next episode
                    opps = sample_episode_opponents(
                        pool=pool,
                        env_rng=env_rngs[env_idx],
                        current_step=timestep,
                        agent_seat=args.agent_seat,
                        bootstrap_mode=args.bootstrap,
                        policy_cache=policy_cache,
                    )
                    env.opponent_policies = opps

                    episodes_total += 1
                    if any(p.name == "ppo_snapshot" for p in opps):
                        episodes_selfplay += 1
                    else:
                        episodes_bootstrap += 1

                    with timers.time("env.reset_total"):
                        obs, info = env.reset(seed=episode_seeds[env_idx])

                new_obs_list.append(obs)

            obs_list = new_obs_list

            if timestep >= next_log_at:
                elapsed_sec = time.time() - start_time
                interval_steps = timestep - prev_log_step
                interval_time = time.time() - prev_log_time
                interval_sps = (
                    interval_steps / interval_time if interval_time > 1e-9 else 0.0
                )
                if ema_steps_per_sec is None:
                    ema_steps_per_sec = interval_sps
                else:
                    ema_steps_per_sec = 0.2 * interval_sps + 0.8 * ema_steps_per_sec

                steps_per_sec = ema_steps_per_sec if ema_steps_per_sec else _training_speed(timestep, elapsed_sec)
                eta_sec = _eta_seconds(args.total_timesteps, timestep, steps_per_sec)
                percent = (
                    100.0 * timestep / args.total_timesteps
                    if args.total_timesteps > 0
                    else 0.0
                )
                frac_selfplay = (
                    episodes_selfplay / episodes_total
                    if episodes_total > 0
                    else 0.0
                )
                print(
                    f"[progress] {percent:5.1f}% | step {timestep:,}/{args.total_timesteps:,} | "
                    f"speed {steps_per_sec:.0f} steps/s | ETA {_fmt_hours(eta_sec)} | "
                    f"selfplay {frac_selfplay:.1%} | pool eligible {pool.pool_size(timestep)} / total {pool.total_size()}"
                )
                prev_log_step = timestep
                prev_log_time = time.time()
                while next_log_at <= timestep:
                    next_log_at += args.log_every

            if timestep >= args.total_timesteps:
                break

        # Convert buffers
        global_buf = torch.tensor(
            np.array(global_buf), dtype=torch.float32, device=device
        )
        action_feats_buf = torch.tensor(
            np.array(action_feats_buf), dtype=torch.float32, device=device
        )
        mask_buf = torch.tensor(np.array(mask_buf), dtype=torch.float32, device=device)
        actions_buf = torch.tensor(
            np.array(actions_buf), dtype=torch.int64, device=device
        )
        logprob_buf = torch.tensor(
            np.array(logprob_buf), dtype=torch.float32, device=device
        )
        value_buf = torch.tensor(
            np.array(value_buf), dtype=torch.float32, device=device
        )
        reward_buf = torch.tensor(
            np.array(reward_buf), dtype=torch.float32, device=device
        )
        done_buf = torch.tensor(
            np.array(done_buf), dtype=torch.float32, device=device
        )

        # Compute last values for GAE bootstrap
        last_values_list = []
        for obs in obs_list:
            global_t = torch.from_numpy(flatten_global(obs)).to(device)
            action_feats_t = torch.from_numpy(obs["action_feats"]).to(device)
            with torch.no_grad():
                _, last_value = model.forward_batch(
                    global_t.unsqueeze(0), action_feats_t.unsqueeze(0)
                )
                last_values_list.append(float(last_value.item()))

        # GAE(lambda)
        with timers.time("train.gae_returns"):
            advantages = torch.zeros_like(reward_buf)
            last_adv = 0.0
            last_value_idx = 0

            for t in reversed(range(len(reward_buf))):
                mask_next = 1.0 - done_buf[t]
                if t == len(reward_buf) - 1:
                    next_value = last_values_list[last_value_idx % args.n_envs]
                    last_value_idx += 1
                else:
                    next_value = value_buf[t + 1]

                delta = (
                    reward_buf[t]
                    + args.gamma * next_value * mask_next
                    - value_buf[t]
                )

                if done_buf[t] > 0.5:
                    last_adv = 0.0

                last_adv = (
                    delta + args.gamma * args.gae_lambda * mask_next * last_adv
                )
                advantages[t] = last_adv

            returns = advantages + value_buf
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        batch_size = len(reward_buf)
        idxs = np.arange(batch_size)

        torch.set_num_threads(4)

        with timers.time("train.ppo_update_total"):
            for _ in range(args.minibatch_epochs):
                np.random.shuffle(idxs)
                for start in range(0, batch_size, args.batch_size):
                    end = start + args.batch_size
                    mb = idxs[start:end]

                    with timers.time("train.ppo_forward"):
                        logits, values = model.forward_batch(
                            global_buf[mb], action_feats_buf[mb]
                        )
                        logits = masked_logits(logits, mask_buf[mb])
                        dist = torch.distributions.Categorical(logits=logits)

                        new_logprob = dist.log_prob(actions_buf[mb])
                        entropy = dist.entropy().mean()

                        ratio = (new_logprob - logprob_buf[mb]).exp()
                        adv = advantages[mb]

                        pg_loss_1 = ratio * adv
                        pg_loss_2 = (
                            torch.clamp(
                                ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps
                            )
                            * adv
                        )
                        policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()

                        value_loss = ((returns[mb] - values) ** 2).mean()

                        loss = (
                            policy_loss
                            + args.vf_coef * value_loss
                            - args.ent_coef * entropy
                        )

                    with timers.time("train.ppo_backward"):
                        optimizer.zero_grad()
                        loss.backward()

                    with timers.time("train.optimizer_step"):
                        optimizer.step()

        torch.set_num_threads(1)

        # Save snapshots periodically (more frequently than eval)
        if timestep >= next_snapshot_at:
            snapshot_path = league_dir / f"checkpoint_{timestep}.pt"
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "timestep": timestep,
                "config": {
                    **config,
                    "global_dim": global_dim,
                    "action_feat_dim": action_feat_dim,
                    "hidden": 128,
                },
                "feature_schema_version": 1,
                "meta": {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": _git_commit(),
                    "dirty": _git_dirty(),
                    "repo_root": str(repo_root),
                    "feature_schema_hash": _feature_schema_hash(),
                    "k_action_feats": int(K_ACTION_FEATS),
                    "obs_keys": obs_keys,
                    "global_obs_dim": int(global_dim),
                    "action_feats_dim": int(action_feat_dim),
                    "n_actions": int(N_ACTIONS),
                    "algo": "ppo",
                    "lr": float(args.learning_rate),
                    "gamma": float(args.gamma),
                    "gae_lambda": float(args.gae_lambda),
                    "clip_eps": float(args.clip_eps),
                    "ent_coef": float(args.ent_coef),
                    "vf_coef": float(args.vf_coef),
                    "rollout_horizon": int(args.rollout_horizon),
                    "batch_size": int(args.batch_size),
                    "epochs": int(args.minibatch_epochs),
                    "seed": int(args.seed),
                    "n_envs": int(args.n_envs),
                    "reward_scale": float(args.reward_scale),
                    "timesteps": int(timestep),
                    "training_return_mean": None,
                },
            }
            torch.save(checkpoint, snapshot_path)
            while next_snapshot_at <= timestep:
                next_snapshot_at += args.snapshot_every

        # Evaluate periodically
        if timestep >= next_eval_at:
            snapshots_exist = pool.total_size() > 0
            use_selfplay_eval = args.disable_fixed_eval_after_snapshots and snapshots_exist

            primary_metrics: Optional[Dict[str, Any]] = None

            if use_selfplay_eval:
                primary_metrics = _evaluate_primary_selfplay(
                    model=model,
                    device=device,
                    agent_seat=args.agent_seat,
                    timestep=timestep,
                    out_dir=out_dir,
                    pool=pool,
                    rng=pool_eval_rng,
                    n_games=args.eval_games,
                    policy_cache=policy_cache,
                    cache_max=16,
                )
                # If pool not ready yet (e.g. not enough eligible snapshots), skip eval this cycle
                if primary_metrics is None:
                    print(
                        f"[eval] skipped selfplay eval (pool eligible {pool.pool_size(timestep)} / total {pool.total_size()})"
                    )
            else:
                # Old deterministic bot-suite eval (40 games)
                primary_metrics = _evaluate_fixed_opponents(
                    model=model,
                    device=device,
                    agent_seat=args.agent_seat,
                    timestep=timestep,
                    out_dir=out_dir,
                )

            # If we got metrics, log + save checkpoints using them
            if primary_metrics is not None:
                with (out_dir / "train_log.csv").open("a", encoding="utf-8") as f:
                    f.write(
                        f"{timestep},{primary_metrics['win_rate']:.4f},{primary_metrics['vp_diff_mean']:.4f},{primary_metrics['vp_diff_std']:.4f},{primary_metrics['agent_vp_mean']:.4f}\n"
                    )

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "timestep": timestep,
                    "config": {
                        **config,
                        "global_dim": global_dim,
                        "action_feat_dim": action_feat_dim,
                        "hidden": 128,
                    },
                    "feature_schema_version": 1,
                    "meta": {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "git_commit": _git_commit(),
                        "dirty": _git_dirty(),
                        "repo_root": str(repo_root),
                        "feature_schema_hash": _feature_schema_hash(),
                        "k_action_feats": int(K_ACTION_FEATS),
                        "obs_keys": obs_keys,
                        "global_obs_dim": int(global_dim),
                        "action_feats_dim": int(action_feat_dim),
                        "n_actions": int(N_ACTIONS),
                        "algo": "ppo",
                        "lr": float(args.learning_rate),
                        "gamma": float(args.gamma),
                        "gae_lambda": float(args.gae_lambda),
                        "clip_eps": float(args.clip_eps),
                        "ent_coef": float(args.ent_coef),
                        "vf_coef": float(args.vf_coef),
                        "rollout_horizon": int(args.rollout_horizon),
                        "batch_size": int(args.batch_size),
                        "epochs": int(args.minibatch_epochs),
                        "seed": int(args.seed),
                        "n_envs": int(args.n_envs),
                        "reward_scale": float(args.reward_scale),
                        "timesteps": int(timestep),
                        "eval_metrics": primary_metrics,
                        "training_return_mean": None,
                    },
                }

                torch.save(checkpoint, out_dir / "latest.pt")

                new_best = False
                if primary_metrics["vp_diff_mean"] > best_vp_diff:
                    best_vp_diff = primary_metrics["vp_diff_mean"]
                    torch.save(checkpoint, out_dir / "best.pt")
                    new_best = True

                _print_eval_block(
                    timestep=timestep,
                    total_timesteps=args.total_timesteps,
                    start_time=start_time,
                    metrics=primary_metrics,
                    new_best=new_best,
                    best_vp_diff=best_vp_diff,
                )

            while next_eval_at <= timestep:
                next_eval_at += args.eval_every

    # Cleanup
    for env in envs:
        env.close()

    total_time = time.time() - start_time
    total_env_steps = timestep
    steps_per_sec = timestep / total_time if total_time > 0 else 0

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(
        f"  Total time: {_fmt_hours(total_time)}  |  Total steps: {timestep:,}  |  Speed: {steps_per_sec:.1f} steps/sec"
    )
    print(f"  Best VP diff: {best_vp_diff:+.2f}")
    print(f"  Out: {out_dir}  |  best.pt / latest.pt saved")
    print(f"  Pool dir: {league_dir}")
    eligible = pool.pool_size(timestep)
    total = pool.total_size()
    print(f"Final pool size: eligible {eligible} / total {total} snapshots")
    print("=" * 80)


if __name__ == "__main__":
    raise SystemExit("Historical trainer disabled: use scripts/ppo_training/train_unified.py --help (corrected GAE/protocol).")
