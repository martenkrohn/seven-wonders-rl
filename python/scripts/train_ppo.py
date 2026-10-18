from __future__ import annotations

import argparse
import math
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.optim as optim

# Prevent CPU oversubscription with parallel envs
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from sevenwonders_rl.envs import SevenWondersEnv, SeatPolicy
from sevenwonders_rl.envs.sevenwonders_env import N_ACTIONS
from sevenwonders_rl.features.action_feature_schema import FEATURE_NAMES, K as K_ACTION_FEATS
from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.ppo_utils import flatten_global, masked_logits, select_action_from_obs

# Opponent mixture for PPO training (episode-level sampling)
OPP_WEIGHTS = {
    "baseline": 0.40,
    "mad_scientist": 0.30,
    "military": 0.20,
    "random": 0.10,
}

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
    print(f"    Win rate: {metrics['win_rate']:.1%}  |  VP diff: {metrics['vp_diff_mean']:+.2f}±{metrics['vp_diff_std']:.2f}")
    print(f"    Agent VP: {metrics['agent_vp_mean']:.1f}  |  Progress: {percent:.1f}% ({timestep:,}/{total_timesteps:,})")
    print(f"    Training time: {_fmt_hours(elapsed_sec)}")
    print(f"    JSONL: {metrics['jsonl_path']}")
    print(f"    Elapsed: {_fmt_hours(elapsed_sec)}  |  ETA: {_fmt_hours(eta_sec)}  |  Speed: {steps_per_sec:.0f} steps/sec")
    if new_best:
        print(f"    New best model! Best VP diff so far: {best_vp_diff:+.2f}")
    print("=" * 80)


def build_opponents(opponents: str, agent_seat: int, seed: int) -> List[SeatPolicy]:
    """Build opponent policies for seats 0, 1, 2 based on strategy."""
    rng = np.random.default_rng(seed)
    opponent_policies = [SeatPolicy("random"), SeatPolicy("random"), SeatPolicy("random")]
    
    if opponents == "heuristic":
        for s in (0, 1, 2):
            if s != agent_seat:
                opponent_policies[s] = SeatPolicy("heuristic")
    elif opponents == "mixed":
        seats = [s for s in (0, 1, 2) if s != agent_seat]
        rng.shuffle(seats)
        opponent_policies[seats[0]] = SeatPolicy("heuristic")
    # else: all random (default)
    
    return opponent_policies


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
    from sevenwonders_rl.policies.registry import make_policy
    from sevenwonders_rl.features.action_feature_schema import IDX
    
    repo_root = Path(__file__).resolve().parents[2]
    
    all_games = []
    
    for left_policy_name, right_policy_name in EVAL_FIXED_MATCHUPS:
        for seed in EVAL_FIXED_SEEDS:
            # Build deterministic opponent policies
            left_policy = make_policy(left_policy_name, seed=seed + 10000)
            right_policy = make_policy(right_policy_name, seed=seed + 20000)
            
            # Assign to seats (agent at agent_seat, others get left/right)
            other_seats = [s for s in (0, 1, 2) if s != agent_seat]
            opponent_policies = [SeatPolicy("random"), SeatPolicy("random"), SeatPolicy("random")]
            opponent_policies[other_seats[0]] = SeatPolicy(left_policy_name, policy=left_policy)
            opponent_policies[other_seats[1]] = SeatPolicy(right_policy_name, policy=right_policy)
            
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
                action = select_action_from_obs(obs, model, device, deterministic=True)
                
                # Count action type
                action_feats = obs.get("action_feats")
                if action_feats is not None and action < len(action_feats):
                    row = action_feats[action]
                    if row[IDX["action_type_3"]] > 0.5:
                        action_counts["discard"] += 1
                    elif row[IDX["action_type_1"]] > 0.5:
                        action_counts["build_wonder"] += 1
                    elif sum(row[IDX[f"delta_prod_{r}"]] for r in ["wood", "ore", "clay", "stone", "loom", "glass", "papyrus"]) > 0 or row[IDX["delta_prod_choice_raw"]] > 0 or row[IDX["delta_prod_choice_manuf"]] > 0:
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
        matchup_games = [g for g in all_games if g["opponents"]["left"] == left and g["opponents"]["right"] == right]
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


def _feature_schema_hash() -> str:
    import hashlib

    data = "\n".join(FEATURE_NAMES).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _git_commit() -> str:
    for key in ("GIT_COMMIT", "GITHUB_SHA"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return bool(out.strip())
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on SevenWondersEnv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-seat", type=int, default=0)
    parser.add_argument("--opponents", choices=["random", "heuristic", "mixed"], default="heuristic")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=2, help="Number of parallel envs (optimal: 2-3)")
    parser.add_argument("--rollout-horizon", type=int, default=256, help="Steps per env before update")
    parser.add_argument("--io-root", type=str, default=None, help="IO root dir (default: repo_root/io_train)")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minibatch-epochs", type=int, default=4)
    parser.add_argument("--reward-scale", type=float, default=10.0, help="Scale terminal reward for value fn")
    parser.add_argument("--eval-every", type=int, default=256_000, help="Eval every N timesteps")
    parser.add_argument("--eval-games", type=int, default=20, help="Games per eval")
    parser.add_argument("--out-dir", type=str, default="python/runs/ppo")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve io_root
    if args.io_root:
        io_root = Path(args.io_root)
    else:
        io_root = repo_root / "io_train"
    io_root.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    
    # Create timing buckets for profiling
    from sevenwonders_rl.utils.timing import TimingBuckets
    timers = TimingBuckets()
    
    print("=" * 80)
    print("PPO TRAINING SETUP")
    print("=" * 80)
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Eval every: {args.eval_every:,} steps")
    print(f"Output dir: {out_dir}")
    print(f"Device: cpu")
    print(f"Opponent mixture: {OPP_WEIGHTS}")
    print()
    print("Creating environments...")
    
    # Create vectorized envs with unique io directories
    envs = []
    for rank in range(args.n_envs):
        env_io_dir = io_root / f"env_{rank:03d}" / "io"
        env_io_dir.mkdir(parents=True, exist_ok=True)
        
        # Use opponent_weights for episode-level sampling (ignores --opponents flag)
        env = SevenWondersEnv(
            repo_root=str(repo_root),
            agent_seat=args.agent_seat,
            io_dir=str(env_io_dir),
            opponent_weights=OPP_WEIGHTS,
            seed=args.seed,
            env_id=rank,
            reward_scale=args.reward_scale,
            timers=timers,
        )
        envs.append(env)
    
    # Initialize all envs
    obs_list = []
    for i, env in enumerate(envs):
        obs, info = env.reset(seed=args.seed + i)
        obs_list.append(obs)
    
    print(f"✓ Created {args.n_envs} environments")
    print()
    print("Initializing model...")
    
    # Get dimensions from first env
    global_dim = flatten_global(obs_list[0]).shape[0]
    action_feat_dim = obs_list[0]["action_feats"].shape[1]
    obs_keys = sorted(list(obs_list[0].keys()))

    device = torch.device("cpu")  # Force CPU (CUDA 6.1 incompatible with PyTorch)
    model = ActorCritic(global_dim, action_feat_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("timestep,win_rate,vp_diff_mean,vp_diff_std,agent_vp_mean\n", encoding="utf-8")

    timestep = 0
    episode_seeds = [args.seed + 1000 + i for i in range(args.n_envs)]
    best_vp_diff = float('-inf')

    print(f"✓ Model initialized (global_dim={global_dim}, action_feat_dim={action_feat_dim})")
    print()
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Rollout horizon: {args.rollout_horizon} steps/env")
    print(f"Learning rate: {args.learning_rate}")
    print(f"First eval at: {args.eval_every:,} steps")
    print()
    print("Training in progress (silent until first eval)...")
    print()

    # Progress tracking
    start_time = time.time()
    update_idx = 0
    next_eval_at = args.eval_every

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
            
            batch_global = torch.tensor(np.array(batch_global), dtype=torch.float32, device=device)
            batch_action_feats = torch.tensor(np.array(batch_action_feats), dtype=torch.float32, device=device)
            batch_mask = torch.tensor(np.array(batch_mask), dtype=torch.float32, device=device)

            with timers.time("train.model_forward"):
                with torch.no_grad():
                    logits, values = model.forward_batch(batch_global, batch_action_feats)
                    logits = masked_logits(logits, batch_mask)
            
            with timers.time("train.sample_action"):
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            # Two-phase stepping: send all actions, then wait for all
            # Phase 1: Send actions to all envs (non-blocking)
            prev_turns = []
            for env_idx, (env, action) in enumerate(zip(envs, actions)):
                prev_turn = env.send_action(int(action.item()), status=obs_list[env_idx].get("_status"))
                prev_turns.append(prev_turn)
            
            # Phase 2: Wait for all envs to finish (engines run in parallel during this phase)
            new_obs_list = []
            for env_idx, (env, prev_turn, action) in enumerate(zip(envs, prev_turns, actions)):
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
                    with timers.time("env.reset_total"):
                        obs, info = env.reset(seed=episode_seeds[env_idx])
                
                new_obs_list.append(obs)

            obs_list = new_obs_list

            if timestep >= args.total_timesteps:
                break

        # Convert buffers
        global_buf = torch.tensor(np.array(global_buf), dtype=torch.float32, device=device)
        action_feats_buf = torch.tensor(np.array(action_feats_buf), dtype=torch.float32, device=device)
        mask_buf = torch.tensor(np.array(mask_buf), dtype=torch.float32, device=device)
        actions_buf = torch.tensor(np.array(actions_buf), dtype=torch.int64, device=device)
        logprob_buf = torch.tensor(np.array(logprob_buf), dtype=torch.float32, device=device)
        value_buf = torch.tensor(np.array(value_buf), dtype=torch.float32, device=device)
        reward_buf = torch.tensor(np.array(reward_buf), dtype=torch.float32, device=device)
        done_buf = torch.tensor(np.array(done_buf), dtype=torch.float32, device=device)

        # Compute last values for GAE bootstrap (batch over all envs)
        last_values_list = []
        for obs in obs_list:
            global_t = torch.from_numpy(flatten_global(obs)).to(device)
            action_feats_t = torch.from_numpy(obs["action_feats"]).to(device)
            with torch.no_grad():
                _, last_value = model.forward_batch(global_t.unsqueeze(0), action_feats_t.unsqueeze(0))
                last_values_list.append(float(last_value.item()))

        # GAE(lambda) - need to handle multiple parallel trajectories
        # For simplicity, compute advantages for entire buffer treating each timestep independently
        with timers.time("train.gae_returns"):
            advantages = torch.zeros_like(reward_buf)
            last_adv = 0.0
            last_value_idx = 0
            
            for t in reversed(range(len(reward_buf))):
                mask_next = 1.0 - done_buf[t]
                if t == len(reward_buf) - 1:
                    # Use appropriate last value for this env
                    next_value = last_values_list[last_value_idx % args.n_envs]
                    last_value_idx += 1
                else:
                    next_value = value_buf[t + 1]
                
                delta = reward_buf[t] + args.gamma * next_value * mask_next - value_buf[t]
                
                # Reset GAE trace on episode boundaries
                if done_buf[t] > 0.5:
                    last_adv = 0.0
                
                last_adv = delta + args.gamma * args.gae_lambda * mask_next * last_adv
                advantages[t] = last_adv

            returns = advantages + value_buf

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = len(reward_buf)
        idxs = np.arange(batch_size)

        # Optimize torch threads for PPO update (engines idle, so we can use all cores)
        torch.set_num_threads(4)  # Use physical core count for updates
        
        with timers.time("train.ppo_update_total"):
            for _ in range(args.minibatch_epochs):
                np.random.shuffle(idxs)
                for start in range(0, batch_size, args.batch_size):
                    end = start + args.batch_size
                    mb = idxs[start:end]

                    with timers.time("train.ppo_forward"):
                        logits, values = model.forward_batch(global_buf[mb], action_feats_buf[mb])
                        logits = masked_logits(logits, mask_buf[mb])
                        dist = torch.distributions.Categorical(logits=logits)

                        new_logprob = dist.log_prob(actions_buf[mb])
                        entropy = dist.entropy().mean()

                        ratio = (new_logprob - logprob_buf[mb]).exp()
                        adv = advantages[mb]

                        pg_loss_1 = ratio * adv
                        pg_loss_2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                        policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()

                        value_loss = ((returns[mb] - values) ** 2).mean()

                        loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                    with timers.time("train.ppo_backward"):
                        optimizer.zero_grad()
                        loss.backward()
                    
                    with timers.time("train.optimizer_step"):
                        optimizer.step()
        
        # Return torch threads to 1 for next rollout (avoid contention with parallel envs)
        torch.set_num_threads(1)

        if timestep >= next_eval_at:
            metrics = _evaluate_fixed_opponents(
                model=model,
                device=device,
                agent_seat=args.agent_seat,
                timestep=timestep,
                out_dir=out_dir,
            )

            # Append to CSV log
            with (out_dir / "train_log.csv").open("a", encoding="utf-8") as f:
                f.write(f"{timestep},{metrics['win_rate']:.4f},{metrics['vp_diff_mean']:.4f},{metrics['vp_diff_std']:.4f},{metrics['agent_vp_mean']:.4f}\n")
            
            # Save latest checkpoint
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
                    "opponent_setup": args.opponents,
                    "reward_definition": "final_vp_diff_vs_best_opponent_scaled",
                    "timesteps": int(timestep),
                    "eval_metrics": metrics,
                    "training_return_mean": None,
                },
            }

            latest_path = out_dir / "latest.pt"
            torch.save(checkpoint, latest_path)
            
            # Save best checkpoint based on vp_diff_mean
            new_best = False
            if metrics['vp_diff_mean'] > best_vp_diff:
                best_vp_diff = metrics['vp_diff_mean']
                best_path = out_dir / "best.pt"
                torch.save(checkpoint, best_path)
                new_best = True
            
            # Also save periodic numbered checkpoints
            ckpt_path = out_dir / f"checkpoint_{timestep}.pt"
            torch.save(checkpoint, ckpt_path)

            # Print a single clean eval block
            _print_eval_block(
                timestep=timestep,
                total_timesteps=args.total_timesteps,
                start_time=start_time,
                metrics=metrics,
                new_best=new_best,
                best_vp_diff=best_vp_diff,
            )

            # Schedule next eval (robust even if timestep jumps past multiple intervals)
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
    print(f"  Total time: {_fmt_hours(total_time)}  |  Total steps: {timestep:,}  |  Speed: {steps_per_sec:.1f} steps/sec")
    print(f"  Best VP diff: {best_vp_diff:+.2f}")
    print(f"  Out: {out_dir}  |  best.pt / latest.pt saved")
    print("=" * 80)


if __name__ == "__main__":
    main()
