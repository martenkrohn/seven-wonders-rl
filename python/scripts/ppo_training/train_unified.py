"""CPU PPO with per-environment GAE, seeded engine games and strict evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from sevenwonders_rl.envs import SevenWondersEnv, SeatPolicy
from sevenwonders_rl.envs.sevenwonders_env import EngineError
from sevenwonders_rl.evaluation.runner import evaluate_model
from sevenwonders_rl.evaluation.selfplay import reference, evaluation_plan, paired_progress, rolling_references, plan_identity
from sevenwonders_rl.policies.registry import make_policy
from sevenwonders_rl.policies.ppo_checkpoint_policy import PPOCheckpointPolicy
from sevenwonders_rl.policies.opponent_batch import opponent_actions_batch
from sevenwonders_rl.selfplay.opponent_pool import OpponentPool
from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.ppo_utils import flatten_global, masked_logits
from sevenwonders_rl.utils.gae import compute_gae, normalize_advantages
from sevenwonders_rl.utils.seeding import derive_seed
from sevenwonders_rl.utils.checkpoints import manifest, save_checkpoint, file_hash
from sevenwonders_rl.utils.runtime import atomic_json, run_io_root, exchange_io_root
from sevenwonders_rl.utils.timing import TimingBuckets
from sevenwonders_rl.utils.training_console import TrainingConsole
from sevenwonders_rl.evaluation.telemetry import status_snapshot

EVAL_FIXED_MATCHUPS = [("baseline", "baseline"), ("mad_scientist", "mad_scientist"),
                       ("military", "military"), ("random", "random")]

def sample_fixed_opponents(
    weights: Dict[str, float],
    env_rng: np.random.Generator,
    agent_seat: int,
) -> List[SeatPolicy]:
    """Sample two bot opponents from fixed weights (FIXED mode)."""
    bot_names = list(weights.keys())
    probs = np.array([weights[name] for name in bot_names])
    probs = probs / probs.sum()

    other_seats = [s for s in (0, 1, 2) if s != agent_seat]
    opps = [SeatPolicy("random"), SeatPolicy("random"), SeatPolicy("random")]

    for seat in other_seats:
        bot_name = env_rng.choice(bot_names, p=probs)
        bot_policy = make_policy(bot_name, seed=int(env_rng.integers(0, 2**31 - 1)))
        opps[seat] = SeatPolicy(bot_name, policy=bot_policy)

    return opps


def sample_selfplay_opponents(
    pool: OpponentPool,
    env_rng: np.random.Generator,
    current_step: int,
    agent_seat: int,
    policy_cache: Optional[OrderedDict] = None,
) -> List[SeatPolicy]:
    """Neural opponents only. Empty pools and load failures are fatal."""
    other_seats = [s for s in (0, 1, 2) if s != agent_seat]
    opps = [SeatPolicy("random"), SeatPolicy("random"), SeatPolicy("random")]

    def _get_cached_policy(path: str) -> PPOCheckpointPolicy:
        if policy_cache is not None and path in policy_cache:
            policy = policy_cache.pop(path)
            policy_cache[path] = policy
            return policy
        try:
            policy = PPOCheckpointPolicy(path, device="cpu", deterministic=False)
        except Exception as exc:
            raise RuntimeError(f"Cannot load neural opponent {path}: {exc}") from exc
        if policy_cache is not None:
            policy_cache[path] = policy
            while len(policy_cache) > 16:
                policy_cache.popitem(last=False)
        return policy

    for seat in other_seats:
        path = pool.sample_path(env_rng, current_step)
        if path is None:
            raise ValueError("Self-play requires its initial neural checkpoint_0.pt")
        opps[seat] = SeatPolicy("ppo_snapshot", policy=_get_cached_policy(path))
    return opps


def tensor_observations(observations):
    return (torch.from_numpy(np.stack([flatten_global(o) for o in observations])),
            torch.from_numpy(np.stack([o["action_feats"] for o in observations])),
            torch.from_numpy(np.stack([o["action_mask"] for o in observations])))


def collect_rollout(model, envs, observations, horizon, action_generator, reset_env, timers):
    """Preserve [time, env] through collection, including pre-reset next values."""
    buffers = {k: [] for k in ("global", "features", "masks", "actions", "logprobs", "values",
                               "rewards", "terminated", "truncated", "next_values")}
    for _ in range(horizon):
        with timers.time("collection.prepare"):
            global_obs, features, masks = tensor_observations(observations)
        with timers.time("collection.inference"), torch.no_grad():
            logits, values = model.forward_batch(global_obs, features)
            logits = masked_logits(logits, masks)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = torch.multinomial(distribution.probs, 1, generator=action_generator).squeeze(-1)
            logprobs = distribution.log_prob(actions)
        with timers.time("collection.send"):
            opponent_actions = opponent_actions_batch(envs, timers=timers)
            previous_turns = [env.send_action(int(action), opponent_actions=prepared)
                              for env, action, prepared in zip(envs, actions, opponent_actions)]
        following, rewards, terminals, truncations, infos = [], [], [], [], []
        with timers.time("collection.wait"):
            for env, previous_turn in zip(envs, previous_turns):
                obs, reward, terminated, truncated, info = env.wait_next(previous_turn)
                if not info.get("transition_valid", True):
                    raise EngineError(info.get("truncated_reason", "invalid_transition"),
                                      f"Invalid transition at {env.io_dir}: {info.get('error', '')}")
                following.append(obs)
                rewards.append(reward)
                terminals.append(terminated)
                truncations.append(truncated)
                infos.append(info)
        with timers.time("collection.bootstrap"), torch.no_grad():
            next_global, next_features, _ = tensor_observations(following)
            _, next_values = model.forward_batch(next_global, next_features)
            next_values = next_values.masked_fill(torch.tensor(terminals), 0.0)
        rows = (global_obs, features, masks, actions, logprobs, values,
                torch.tensor(rewards, dtype=torch.float32), torch.tensor(terminals),
                torch.tensor(truncations), next_values)
        for key, row in zip(buffers, rows):
            buffers[key].append(row.detach())
        for i, (terminated, truncated) in enumerate(zip(terminals, truncations)):
            if terminated or truncated:
                with timers.time("collection.reset"):
                    following[i] = reset_env(i, infos[i], terminated)
        observations = following
    return {key: torch.stack(rows) for key, rows in buffers.items()}, observations


def ppo_update(model, optimizer, rollout, args, minibatch_rng):
    raw_adv, returns = compute_gae(rollout["rewards"], rollout["values"],
        rollout["terminated"], rollout["truncated"], rollout["next_values"], args.gamma, args.gae_lambda)
    advantages = normalize_advantages(raw_adv).flatten()
    returns = returns.flatten()
    batch = {key: value.flatten(0, 1) for key, value in rollout.items()}
    count = advantages.numel()
    metrics = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "gradient_norm")}
    for _ in range(args.minibatch_epochs):
        indices = minibatch_rng.permutation(count)
        for start in range(0, count, args.batch_size):
            mb = indices[start:start + args.batch_size]
            logits, values = model.forward_batch(batch["global"][mb], batch["features"][mb])
            dist = torch.distributions.Categorical(logits=masked_logits(logits, batch["masks"][mb]))
            log_ratio = dist.log_prob(batch["actions"][mb]) - batch["logprobs"][mb]
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[mb]
            clipped = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * advantages[mb]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (returns[mb] - values).square().mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite PPO parameters")
            for key, value in zip(metrics, (policy_loss, value_loss, entropy,
                    ((ratio - 1) - log_ratio).mean(), ((ratio - 1).abs() > args.clip_eps).float().mean(), grad_norm)):
                metrics[key].append(float(value.detach()))
    result = {key: float(np.mean(values)) for key, values in metrics.items()}
    for key, values in (("reward", rollout["rewards"]), ("value", rollout["values"]),
                        ("return", returns), ("advantage", raw_adv)):
        result.update({f"{key}_mean": float(values.mean()), f"{key}_std": float(values.std(correction=0))})
    result.update(transitions=count, terminated=int(rollout["terminated"].sum()),
                  truncated=int(rollout["truncated"].sum()))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixed", "selfplay"], default="selfplay")
    defaults = {"seed": 0, "agent-seat": 0, "total-timesteps": 2_000_000, "n-envs": 2,
        "rollout-horizon": 256, "batch-size": 256, "minibatch-epochs": 4,
        "eval-every": 32_768, "log-every": 4096, "save-every": 4096,
        "eval-games-per-matchup": 10, "eval-seed": 42, "pool-max": 80,
        "pool-lag-steps": 0, "snapshot-every": 4096, "engine-poll-us": 1000}
    for name, default in defaults.items():
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in {"learning-rate": 3e-4, "gamma": .999, "gae-lambda": .95,
        "clip-eps": .2, "ent-coef": .01, "vf-coef": .5, "reward-scale": 10.,
        "opponent-baseline": .5, "opponent-scientist": .3, "opponent-military": .2,
        "poll-seconds": .001}.items():
        parser.add_argument("--" + name, type=float, default=default)
    for name in ("io-root", "out-dir", "league-dir"):
        parser.add_argument("--" + name)
    parser.add_argument("--eval-action-mode", choices=["greedy", "stochastic"], default="stochastic")
    parser.add_argument("--opponent-inference", choices=["batched", "serial"], default="batched",
                        help="Group frozen neural opponents across environments; serial is the comparison path")
    args = parser.parse_args(argv)
    positive = ("total_timesteps", "n_envs", "rollout_horizon", "batch_size", "minibatch_epochs",
                "eval_every", "log_every", "save_every", "eval_games_per_matchup", "pool_max", "snapshot_every")
    if any(getattr(args, key) <= 0 for key in positive):
        parser.error("Counts and intervals must be positive")
    if args.seed < 0 or args.eval_seed < 0 or args.pool_lag_steps < 0 or args.agent_seat not in (0, 1, 2):
        parser.error("Invalid seed, pool lag or agent seat")
    if args.total_timesteps % args.n_envs:
        parser.error("total-timesteps must be divisible by n-envs")
    if not all(math.isfinite(v) for v in vars(args).values() if isinstance(v, float)):
        parser.error("Numeric parameters must be finite")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("gamma and gae-lambda must be in [0,1]")
    if args.learning_rate <= 0 or args.reward_scale <= 0 or not 0 < args.clip_eps < 1 or min(args.ent_coef, args.vf_coef) < 0:
        parser.error("Invalid optimization parameters")
    weights = (args.opponent_baseline, args.opponent_scientist, args.opponent_military)
    if min(weights) < 0 or sum(weights) <= 0:
        parser.error("Opponent weights must be nonnegative with a positive sum")
    if not 0 <= args.engine_poll_us <= 1_000_000 or args.poll_seconds < 0:
        parser.error("Invalid polling interval")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    repo_root = Path(__file__).resolve().parents[3]
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = run_io_root(Path("runs/ppo_unified"), args.mode)
    league_dir = Path(args.league_dir).expanduser().resolve() if args.league_dir else out_dir
    league_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "selfplay" and any(league_dir.glob("checkpoint_*.pt")):
        raise ValueError("Self-play requires a fresh league directory for this run")
    io_root = exchange_io_root(args.io_root, repo_root / "io_train", "training",
                               prefer_ram=args.mode == "selfplay")
    config = {"evaluation_strategy": "rolling_trained_snapshots" if args.mode == "selfplay" else "fixed_bots", **vars(args), "out_dir": str(out_dir), "io_root": str(io_root), "league_dir": str(league_dir)}
    atomic_json(out_dir / "config.json", config)
    run_manifest = manifest(repo_root, config)
    atomic_json(out_dir / "manifest.json", run_manifest)
    console = TrainingConsole(args, out_dir, io_root)
    timers = TimingBuckets()
    pool = OpponentPool(str(league_dir), max_size=args.pool_max,
                        min_step_lag=args.pool_lag_steps, latest_k=8) if args.mode == "selfplay" else None
    env_rngs = [np.random.default_rng(derive_seed(args.seed, "opponents", i)) for i in range(args.n_envs)]
    minibatch_rng = np.random.default_rng(derive_seed(args.seed, "minibatch"))
    action_generator = torch.Generator().manual_seed(derive_seed(args.seed, "learner"))
    weights = {"baseline": args.opponent_baseline, "mad_scientist": args.opponent_scientist, "military": args.opponent_military}
    policy_cache = OrderedDict()
    envs, episodes = [], [0] * args.n_envs
    active_games, checkpoint_hashes = {}, {}
    previous_evaluation_path = None
    previous_plan_id = None
    latest_evaluated_step = None
    timestep, updates, best_vp_diff = 0, 0, float("-inf")
    started = time.perf_counter()
    status = "failed"
    next_eval, next_save, next_snapshot = args.eval_every, args.save_every, args.snapshot_every
    def reset_env(i, final_info=None, terminated=False):
        if final_info is not None:
            with (out_dir / "episodes.jsonl").open("a") as f:
                f.write(json.dumps({**active_games[i],
                    "status": "completed" if terminated else "truncated",
                    "totals": final_info.get("final_totals"), "reason": final_info.get("truncated_reason"),
                    "final_state": status_snapshot(final_info['status'], 3) if terminated else None}) + "\n")
            episodes[i] += 1
        seat = args.agent_seat if args.mode == "fixed" else (args.agent_seat + i + episodes[i] * args.n_envs) % 3
        envs[i].agent_seat = seat
        if args.mode == "fixed":
            policies = sample_fixed_opponents(weights, env_rngs[i], seat)
        else:
            policies = sample_selfplay_opponents(pool, env_rngs[i], timestep, seat, policy_cache)
        envs[i].opponent_policies = policies
        seed = derive_seed(args.seed, "engine", i, episodes[i])
        opponents = {}
        for opponent_seat, policy in enumerate(policies):
            if opponent_seat == seat:
                continue
            descriptor = {"name": policy.name}
            if policy.name == "ppo_snapshot":
                path = str(Path(policy.policy.ckpt_path).resolve())
                if path not in checkpoint_hashes:
                    checkpoint_hashes[path] = file_hash(path)
                descriptor.update(checkpoint=path, sha256=checkpoint_hashes[path])
            opponents[str(opponent_seat)] = descriptor
        active_games[i] = {"env": i, "episode": episodes[i], "engine_seed": seed,
                           "agent_seat": seat, "selection_step": timestep, "opponents": opponents}
        with (out_dir / "opponents.jsonl").open("a") as f:
            f.write(json.dumps(active_games[i]) + "\n")
        obs, _ = envs[i].reset(seed=seed)
        return obs
    try:
        for i in range(args.n_envs):
            envs.append(SevenWondersEnv(repo_root=str(repo_root), agent_seat=args.agent_seat,
                io_dir=str(io_root / "train" / f"env_{i:03d}" / "io"),
                reward_scale=args.reward_scale, invalid_action_fallback="raise", timers=timers,
                engine_poll_us=args.engine_poll_us, poll_seconds=args.poll_seconds,
                batch_opponents=args.opponent_inference == "batched"))
        # Build the initial policy before starting any game: neural self-play is
        # available from the first action, including with a single snapshot.
        zero_obs = {key: np.zeros(space.shape, dtype=space.dtype)
                    for key, space in envs[0].observation_space.spaces.items()}
        global_dim = len(flatten_global(zero_obs))
        feature_dim = zero_obs["action_feats"].shape[1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed(args.seed, "model"))
            model = ActorCritic(global_dim, feature_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        config.update(global_dim=global_dim, action_feat_dim=feature_dim, hidden=128,
                      model_architecture_version=model.architecture_version)
        atomic_json(out_dir / "config.json", config)
        atomic_json(out_dir / "manifest.json", run_manifest)
        def save(path, metrics=None):
            with timers.time("checkpoint"):
                save_checkpoint(path, model, optimizer, timestep, config, run_manifest, metrics)
        if args.mode == "selfplay":
            save(league_dir / "checkpoint_0.pt")
            reference_dir = out_dir / "references"
            reference_dir.mkdir()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(derive_seed(args.seed, "model", 1))
                independent = ActorCritic(global_dim, feature_dim)
            independent_path = reference_dir / "independent_initial.pt"
            save_checkpoint(independent_path, independent, None, 0, config, run_manifest)
            references = [reference(league_dir / "checkpoint_0.pt"), reference(independent_path)]
            atomic_json(out_dir / "references.json", {"references": references,
                "purpose": "optional post-training initial references; in-training evaluation uses rolling trained snapshots"})
            print("Checkpoint-Evaluationen: trainierte Vorgänger; keine Anfangsevaluation. "
                  "Erste Messung, sobald ältere trainierte Snapshots vorhanden sind.", flush=True)
        with timers.time("initialization.reset"):
            observations = [reset_env(i) for i in range(args.n_envs)]
        while timestep < args.total_timesteps:
            horizon = min(args.rollout_horizon, (args.total_timesteps - timestep) // args.n_envs)
            with timers.time("collection"):
                rollout, observations = collect_rollout(model, envs, observations, horizon,
                                                        action_generator, reset_env, timers)
            with timers.time("ppo_update"):
                metrics = ppo_update(model, optimizer, rollout, args, minibatch_rng)
            digest = hashlib.sha256()
            for key in ("global", "features", "masks", "actions", "rewards", "terminated", "truncated"):
                digest.update(rollout[key].numpy().tobytes())
            metrics["trajectory_sha256"] = digest.hexdigest()
            timestep += metrics["transitions"]
            updates += 1
            metrics.update(timestep=timestep, update=updates)
            with (out_dir / "updates.jsonl").open("a") as f:
                f.write(json.dumps(metrics, allow_nan=False) + "\n")
            if updates == 1 or timestep >= next_save:
                save(out_dir / "latest.pt")
                while next_save <= timestep:
                    next_save += args.save_every
            if args.mode == "selfplay" and timestep >= next_snapshot:
                save(league_dir / f"checkpoint_{timestep}.pt")
                while next_snapshot <= timestep:
                    next_snapshot += args.snapshot_every
            if timestep >= next_eval or (args.mode == "selfplay" and timestep == args.total_timesteps):
                references = rolling_references(league_dir, timestep) if args.mode == 'selfplay' else None
                specs = evaluation_plan(references, args.eval_seed, args.eval_games_per_matchup) if references else []
                if args.mode == 'fixed':
                    specs = [{"left": left, "right": right, "agent_seat": args.agent_seat,
                          "engine_seed": derive_seed(args.eval_seed, "evaluation", i)}
                         for left, right in EVAL_FIXED_MATCHUPS for i in range(args.eval_games_per_matchup)]
                current_plan_id = plan_identity(specs, args.eval_action_mode)
                if not specs:
                    print(f"Evaluation bei Schritt {timestep} verschoben: noch kein älterer trainierter Snapshot.", flush=True)
                else:
                    console.evaluation_start(timestep, len(specs), references=references)
                    with timers.time("evaluation"):
                        evaluation = evaluate_model(model, "cpu", specs, repo_root,
                            io_root / "evaluation" / str(timestep), out_dir / f"eval_{timestep}.jsonl",
                            args.eval_action_mode, args.engine_poll_us, args.poll_seconds, f"step:{timestep}",
                            batch_opponents=args.opponent_inference == "batched")
                    evaluation.update(plan_id=current_plan_id, references=references, step=timestep,
                                      evaluation_strategy=config['evaluation_strategy'])
                    if previous_plan_id == current_plan_id and previous_evaluation_path is not None:
                        evaluation.update(paired_progress(out_dir / f"eval_{timestep}.jsonl", previous_evaluation_path))
                        evaluation['comparison_label'] = 'Gegenüber vorheriger Evaluation'
                    previous_plan_id = current_plan_id
                    previous_evaluation_path = out_dir / f"eval_{timestep}.jsonl"
                    with (out_dir / "train_log.csv").open("a") as f:
                        if f.tell() == 0:
                            f.write("timestep,win_rate,vp_diff_mean,vp_diff_std,agent_vp_mean,completed,truncated,failed,win_share,plan_id,reference_steps\n")
                        f.write(",".join(str(v) if v is not None else "" for v in
                            (timestep, evaluation["win_rate"], evaluation["vp_diff_mean"], evaluation["vp_diff_std"],
                             evaluation["agent_vp_mean"], evaluation["completed"], evaluation["truncated"], evaluation["failed"],
                             evaluation["win_share"], current_plan_id,
                             ';'.join(str(r['step']) for r in references or []))) + "\n")
                    new_best = False
                    if args.mode == 'selfplay':
                        if evaluation['valid_for_selection']:
                            save(out_dir / 'evaluated.pt', evaluation)
                            latest_evaluated_step = timestep
                            evaluation['saved_evaluated_checkpoint'] = True
                    else:
                        score = evaluation['vp_diff_mean']
                        new_best = evaluation['valid_for_selection'] and score > best_vp_diff
                        if new_best:
                            best_vp_diff = score
                            save(out_dir / 'best.pt', evaluation)
                    atomic_json(out_dir / f"eval_{timestep}_summary.json", evaluation)
                    console.evaluation_end(evaluation, new_best=new_best)
                while next_eval <= timestep:
                    next_eval += args.eval_every
            if timestep % args.log_every < args.n_envs * horizon or timestep == args.total_timesteps:
                console.progress(timestep, updates)
        save(out_dir / "latest.pt")
        status = "completed"
    except BaseException as exc:
        for i, env in enumerate(envs):
            error_log = env.io_dir.parent / "engine_stderr.log"
            if error_log.exists():
                shutil.copyfile(error_log, out_dir / f"engine_{i}_stderr.log")
        atomic_json(out_dir / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                    "reason": getattr(exc, "reason", None), "last_completed_update": updates,
                    "timestep": timestep, "pending_rollout_discarded": True,
                    "last_statuses": [env._last_status for env in envs]})
        raise
    finally:
        for env in envs:
            env.close()
        shutil.rmtree(io_root)
        elapsed = time.perf_counter() - started
        atomic_json(out_dir / "timing.json", timers.as_dict(elapsed))
        atomic_json(out_dir / "result.json", {"status": status, "timestep": timestep, "updates": updates,
                    "elapsed_seconds": elapsed, "steps_per_second": timestep / elapsed,
                    "best_vp_diff": best_vp_diff if math.isfinite(best_vp_diff) else None,
                    "selection_metric": "latest_valid_evaluation" if args.mode == "selfplay" else "vp_diff_mean",
                    "latest_evaluated_step": latest_evaluated_step,
                    "best_selection_score": best_vp_diff if math.isfinite(best_vp_diff) else None})
    console.finish(out_dir)
    return out_dir


if __name__ == "__main__":
    main()
