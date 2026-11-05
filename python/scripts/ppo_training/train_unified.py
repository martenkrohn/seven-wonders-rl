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
from sevenwonders_rl.selfplay.opponent_pool import OpponentPool, FrozenOpponentPool, MixedOpponentPool, read_population
from sevenwonders_rl.selfplay.pair_pfsp import PairPFSPPool
from sevenwonders_rl.utils.ppo_model import (
    ActorCritic, MultiHeadActorCritic, extend_public_inputs, extend_observation_memory)
from sevenwonders_rl.utils.ppo_utils import flatten_global, action_features, masked_logits
from sevenwonders_rl.utils.gae import compute_gae, normalize_advantages
from sevenwonders_rl.utils.seeding import derive_seed
from sevenwonders_rl.utils.checkpoints import manifest, save_checkpoint, file_hash, load_model
from sevenwonders_rl.rewards import REWARD_PROFILES, RewardSpec
from sevenwonders_rl.utils.runtime import atomic_json, run_io_root, exchange_io_root
from sevenwonders_rl.utils.timing import TimingBuckets
from sevenwonders_rl.utils.training_console import TrainingConsole
from sevenwonders_rl.evaluation.telemetry import status_snapshot
from sevenwonders_rl.evaluation.science_diagnostics import ScienceDiagnostics
from sevenwonders_rl.evaluation.milestones import MilestoneEvaluations, milestone_steps, fixed_panel
from sevenwonders_rl.utils.training_restart import training_directory_lock, prepare_existing_run, matching_configuration
from sevenwonders_rl.utils.phase_resume import runtime_snapshot, restore_runtime, trim_uncommitted_logs

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
        bot_name = str(env_rng.choice(bot_names, p=probs))
        bot_policy = make_policy(bot_name, seed=int(env_rng.integers(0, 2**31 - 1)))
        opps[seat] = SeatPolicy(bot_name, policy=bot_policy)

    return opps


def sample_selfplay_opponents(
    pool: OpponentPool,
    env_rng: np.random.Generator,
    current_step: int,
    agent_seat: int,
    policy_cache: Optional[OrderedDict] = None,
    cache_size: Optional[int] = None,
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
            limit = cache_size if cache_size is not None else max(pool.max_size, 16)
            while len(policy_cache) > limit:
                policy_cache.popitem(last=False)
        return policy

    paired = None
    if isinstance(pool, PairPFSPPool):
        left, right, metadata = pool.sample_pair(env_rng, current_step)
        paired = {(agent_seat - 1) % 3: (left, metadata), (agent_seat + 1) % 3: (right, metadata)}
    for seat in other_seats:
        path, metadata = (paired[seat] if paired is not None else
                          pool.sample_entry(env_rng, current_step) if isinstance(pool, MixedOpponentPool)
                          else (pool.sample_path(env_rng, current_step), None))
        if path is None:
            raise ValueError("Self-play requires its initial neural checkpoint_0.pt")
        opps[seat] = SeatPolicy("ppo_snapshot", policy=_get_cached_policy(path), sampling_metadata=metadata)
    return opps


def tensor_observations(observations, model=None):
    return (torch.from_numpy(np.stack([flatten_global(o, model) for o in observations])),
            torch.from_numpy(np.stack([action_features(o, model) for o in observations])),
            torch.from_numpy(np.stack([o["action_mask"] for o in observations])))


def initialize_model(args, global_dim, feature_dim):
    """Weight initialization only; never import optimizer/RNG/episode state."""
    memory = getattr(args, 'observation_memory', False)
    from sevenwonders_rl.features.observation_history import MEMORY_DIM
    base_global_dim = global_dim - MEMORY_DIM if memory else global_dim
    if args.init_checkpoint:
        path = Path(args.init_checkpoint)
        source_hash = file_hash(path)
        model = load_model(path)
        source = torch.load(path, map_location='cpu', weights_only=True)
        if file_hash(path) != source_hash:
            raise ValueError('Initial checkpoint changed during loading')
        initialization = dict(kind='checkpoint', checkpoint=str(path.resolve()), sha256=source_hash,
                              source_timestep=int(source['timestep']),
                              source_reward_profile=source['config'].get('reward_profile', 'standard'),
                              optimizer_restored=False)
    else:
        seed = derive_seed(args.seed, 'model')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = ActorCritic(base_global_dim, feature_dim,
                                public_information=getattr(args, 'public_information', False))
        initialization = dict(kind='random', seed=seed, optimizer_restored=False)
    if getattr(model, 'memory_enabled', False) and model.num_heads != args.num_heads:
        raise ValueError('Memory checkpoint requires the same number of heads')
    if args.num_heads > 1:
        if model.num_heads == 1:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(derive_seed(args.seed, 'multihead'))
                expanded = MultiHeadActorCritic(model.global_mlp[0].in_features, model.action_mlp[0].in_features,
                    hidden=model.global_mlp[0].out_features, num_heads=args.num_heads,
                    prior_strength=args.prior_strength,
                    public_information=getattr(model, 'public_information', False))
            if args.init_checkpoint:
                expanded.copy_single_head(model)
            model = expanded
        elif model.num_heads != args.num_heads or model.prior_strength != args.prior_strength:
            raise ValueError('Initial multi-head checkpoint must match num-heads and prior-strength')
    elif model.num_heads != 1:
        raise ValueError('Multi-head checkpoint requires matching --num-heads and --prior-strength')
    if getattr(args, 'public_information', False) and not getattr(model, 'public_information', False):
        model = extend_public_inputs(model)
        initialization['public_input_migration'] = 'append_zero_columns_preserve_prior_function'
    if memory and not getattr(model, 'memory_enabled', False):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed(args.seed, 'model', 2))
            model = extend_observation_memory(model)
        initialization['memory_input_migration'] = 'shared_zero_residual_preserve_prior_function'
    if (getattr(model, 'global_dim', model.global_mlp[0].in_features) != global_dim or
            model.action_mlp[0].in_features != feature_dim or
            getattr(model, 'memory_enabled', False) != memory or
            getattr(model, 'public_information', False) != getattr(args, 'public_information', False)):
        raise ValueError('Initial checkpoint observation dimensions/contract do not match this environment')
    return model.train(), initialization


def collect_rollout(model, envs, observations, horizon, action_generator, reset_env, timers,
                    on_transition=None, head_ids=None):
    """Preserve [time, env] through collection, including pre-reset next values."""
    buffers = {k: [] for k in ("global", "features", "masks", "actions", "logprobs", "values",
                               "rewards", "terminated", "truncated", "next_values")}
    multihead = getattr(model, 'num_heads', 1) > 1
    if multihead:
        if head_ids is None:
            raise ValueError('Multi-head collection requires per-environment head IDs')
        buffers['head_ids'] = []
    for _ in range(horizon):
        # Snapshot before reset_env mutates the live per-environment assignment.
        selected_heads = torch.tensor(head_ids, dtype=torch.long) if multihead else None
        with timers.time("collection.prepare"):
            global_obs, features, masks = tensor_observations(observations, model)
        with timers.time("collection.inference"), torch.no_grad():
            logits, values = (model.forward_batch(global_obs, features, selected_heads) if multihead
                              else model.forward_batch(global_obs, features))
            logits = masked_logits(logits, masks)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = torch.multinomial(distribution.probs, 1, generator=action_generator).squeeze(-1)
            logprobs = distribution.log_prob(actions)
        with timers.time("collection.send"):
            before_statuses = [env._last_status for env in envs] if on_transition else None
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
        if on_transition:
            for i, env in enumerate(envs):
                on_transition(i, before_statuses[i], infos[i]['status'], observations[i],
                              int(actions[i]), distribution.probs[i].detach().numpy())
        with timers.time("collection.bootstrap"), torch.no_grad():
            if all(terminals):
                next_values = torch.zeros(len(envs), dtype=values.dtype)
            else:
                next_global = torch.from_numpy(np.stack([flatten_global(o, model) for o in following]))
                next_values = (model.forward_values(next_global, selected_heads) if multihead
                               else model.forward_values(next_global))
                next_values = next_values.masked_fill(torch.tensor(terminals), 0.0)
        rows = (global_obs, features, masks, actions, logprobs, values,
                torch.tensor(rewards, dtype=torch.float32), torch.tensor(terminals),
                torch.tensor(truncations), next_values)
        for key, row in zip(buffers, rows):
            buffers[key].append(row.detach())
        if multihead:
            buffers['head_ids'].append(selected_heads)
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
            inputs = (batch['global'][mb], batch['features'][mb])
            logits, values = (model.forward_batch(*inputs, batch['head_ids'][mb])
                             if model.num_heads > 1 else model.forward_batch(*inputs))
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
    if model.num_heads > 1:
        result['head_transitions'] = torch.bincount(batch['head_ids'], minlength=model.num_heads).tolist()
        # Compare all heads on the same observed states, rather than confusing
        # different hands/opponents with genuine policy differences.
        with torch.no_grad():
            n = min(64, count)
            probabilities = []
            for h in range(model.num_heads):
                logits, _ = model.forward_batch(batch['global'][:n], batch['features'][:n],
                                               torch.full((n,), h, dtype=torch.long))
                probabilities.append(masked_logits(logits, batch['masks'][:n]).softmax(-1))
            probs = torch.stack(probabilities)
            average = probs.mean(0).clamp_min(1e-12)
            result['head_js_divergence'] = float((probs * (probs.clamp_min(1e-12).log() - average.log())).sum(-1).mean())
            result['head_greedy_disagreement'] = float((probs.argmax(-1) != probs.argmax(-1)[0]).float().mean())
    for key, values in (("reward", rollout["rewards"]), ("value", rollout["values"]),
                        ("return", returns), ("advantage", raw_adv)):
        result.update({f"{key}_mean": float(values.mean()), f"{key}_std": float(values.std(correction=0))})
    result.update(transitions=count, terminated=int(rollout["terminated"].sum()),
                  truncated=int(rollout["truncated"].sum()))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--public-information', action='store_true',
                        help='Append public played-card identities and wonder identities, stage costs/effects')
    parser.add_argument('--observation-memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Share complete own hand, action information and observed history with actor/critic (default: enabled); enables public information')
    parser.add_argument('--prior-strength', type=float, default=0.0,
                        help='Frozen action-logit prior strength; multi-head only, zero disables priors')
    parser.add_argument('--opponent-population', help='JSON with fixed weighted opponent groups')
    parser.add_argument('--pair-pfsp', action='store_true',
                        help='Grow an own-snapshot league and prioritize ordered opponent pairs by discounted win share')
    parser.add_argument('--pfsp-power', type=float, default=2.)
    parser.add_argument('--pfsp-control-fraction', type=float, default=.10,
                        help='Uniform ordered-pair control sampling, also used for training')
    parser.add_argument('--pfsp-half-life', type=float, default=131_072.,
                        help='Online result half-life in learner transitions')
    parser.add_argument('--pfsp-prior-games', type=float, default=3.,
                        help='Pseudo-count for the neutral three-player win-share estimate of 1/3')
    parser.add_argument('--mixed-league', action='store_true',
                        help='Add 50%% recent and 15%% archive opponents to fixed groups (weights sum to 0.35)')
    parser.add_argument('--mixed-recent-weight', type=float, default=.50)
    parser.add_argument('--mixed-archive-weight', type=float, default=.15)
    parser.add_argument('--historical-checkpoints', nargs='+', help='Ordered previous-phase checkpoints to seed archive')
    parser.add_argument('--evaluation-population', help='Separate fixed opponent groups used only in milestone evaluation')
    parser.add_argument('--milestone-evaluations', action='store_true',
                        help='Detailed fixed-panel evaluations at 0/25/50/75/100%% with one evolving dashboard')
    parser.add_argument('--continuous', action='store_true',
                        help='No end budget; pair-PFSP with fixed-panel evaluations every --eval-every steps')
    parser.add_argument('--eval-panel', help='JSON ordered reference pairs named group:index')
    parser.add_argument('--dashboard-reference', help='Pinned dashboard reference JSON; evaluated once on the same milestone panel')
    parser.add_argument("--mode", choices=["fixed", "selfplay"], default="selfplay")
    parser.add_argument('--init-checkpoint', help='Initialize compatible weights; fresh optimizer, counters and league')
    parser.add_argument('--frozen-opponents', nargs='+',
                        help='Fixed neural checkpoints instead of the growing self-play pool; copied into the run')
    parser.add_argument('--reward-profile', choices=REWARD_PROFILES, default='standard')
    parser.add_argument('--specialist-weight', type=float, default=1.0,
                        help='Additional final category-point weight for science/military (standard ignores it)')
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
    parser.add_argument("--science-diagnostics", action="store_true",
                        help="Passively record science opportunities, policy probabilities and game outcomes")
    parser.add_argument("--opponent-cache-size", type=int, default=None,
                        help="Loaded snapshots; default covers the eligible pool and active seats")
    parser.add_argument("--eval-workers", type=int, default=1,
                        help="Independent evaluation worker processes (one Torch thread each)")
    parser.add_argument("--throughput-preset", action="store_true",
                        help="Use 8 environments and horizon 64 (512 samples/update); requires default env settings")
    parser.add_argument('--restart-incomplete', action='store_true',
                        help='Archive an incomplete matching run and start fresh; does not resume weights')
    parser.add_argument('--resume', action='store_true',
                        help='Restore latest weights/optimizer/RNGs; abandon and restart unfinished games explicitly')
    parser.add_argument('--skip-completed', action='store_true',
                        help='Skip a matching run that already completed its requested budget')
    args = parser.parse_args(argv)
    if args.observation_memory:
        args.public_information = True
    if args.continuous and (not args.pair_pfsp or not args.milestone_evaluations or not args.out_dir):
        parser.error('--continuous requires --pair-pfsp, --milestone-evaluations and --out-dir')
    args.dashboard_reference_spec = None
    if args.dashboard_reference:
        if not args.milestone_evaluations:
            parser.error('--dashboard-reference requires --milestone-evaluations')
        from sevenwonders_rl.evaluation.dashboard_reference import read_dashboard_config
        args.dashboard_reference = str(Path(args.dashboard_reference).resolve())
        dashboard = read_dashboard_config(args.dashboard_reference, Path(__file__).resolve().parents[2])
        # Dashboard routing is presentation state, not part of the training identity.
        args.dashboard_reference_spec = dict(version=dashboard['version'], reference=dashboard['reference'])
    if args.pair_pfsp:
        if (args.mode != 'selfplay' or args.frozen_opponents or args.opponent_population
                or args.mixed_league or args.historical_checkpoints or args.league_dir):
            parser.error('--pair-pfsp requires an internal own-snapshot selfplay league without fixed training opponents')
        if (min(args.pfsp_power, args.pfsp_half_life, args.pfsp_prior_games) <= 0
                or not 0 < args.pfsp_control_fraction <= 1):
            parser.error('Pair-PFSP needs positive power, half-life, prior-games and control fraction in (0,1]')
    if args.num_heads < 1 or args.prior_strength < 0 or not math.isfinite(args.prior_strength):
        parser.error('num-heads must be positive; prior-strength finite and nonnegative')
    if args.num_heads == 1 and args.prior_strength:
        parser.error('Priors require --num-heads greater than one')
    args.population_spec = None
    if args.opponent_population:
        if args.frozen_opponents:
            parser.error('--opponent-population and --frozen-opponents are mutually exclusive')
        try:
            args.population_spec = read_population(args.opponent_population)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            parser.error(f'Invalid opponent population: {exc}')
        args.opponent_population = str(Path(args.opponent_population).expanduser().resolve())
        args.frozen_opponents = [p for group in args.population_spec for p in group['checkpoints']]
    if args.frozen_opponents:
        if args.mode != 'selfplay':
            parser.error('--frozen-opponents requires neural --mode selfplay')
        args.frozen_opponents = list(dict.fromkeys(str(Path(p).expanduser().resolve()) for p in args.frozen_opponents))
        for path in args.frozen_opponents:
            if not Path(path).is_file():
                parser.error(f'Frozen opponent does not exist: {path}')
            if args.out_dir and Path(path).is_relative_to(Path(args.out_dir).expanduser().resolve()):
                parser.error('Frozen opponent sources must be outside the new output directory')
    if args.mixed_league:
        if not args.population_spec or not any(g['name'] == 'anchors' for g in args.population_spec):
            parser.error('--mixed-league requires --opponent-population with anchors group')
        if min(args.mixed_recent_weight, args.mixed_archive_weight) <= 0 or not math.isclose(
                sum(g['weight'] for g in args.population_spec) + args.mixed_recent_weight + args.mixed_archive_weight,
                1.0, abs_tol=1e-8):
            parser.error('Mixed league positive dynamic weights and fixed group weights must sum to one')
    args.evaluation_population_spec = None
    if args.evaluation_population:
        if not args.milestone_evaluations:
            parser.error('--evaluation-population requires --milestone-evaluations')
        args.evaluation_population = str(Path(args.evaluation_population).expanduser().resolve())
        args.evaluation_population_spec = read_population(args.evaluation_population)
    if args.historical_checkpoints:
        if not args.mixed_league:
            parser.error('--historical-checkpoints requires --mixed-league')
        args.historical_checkpoints = [str(Path(p).expanduser().resolve()) for p in args.historical_checkpoints]
        if len(set(args.historical_checkpoints)) != len(args.historical_checkpoints) or any(
                not Path(p).is_file() for p in args.historical_checkpoints):
            parser.error('Historical checkpoints must be unique existing files')
    if args.milestone_evaluations:
        if not (args.population_spec or args.evaluation_population_spec) or not args.eval_panel:
            parser.error('--milestone-evaluations requires a training or evaluation population and --eval-panel')
        args.eval_panel = str(Path(args.eval_panel).expanduser().resolve())
        if not Path(args.eval_panel).is_file():
            parser.error('Evaluation panel does not exist')
    if not math.isfinite(args.specialist_weight) or args.specialist_weight < 0:
        parser.error('--specialist-weight must be finite and nonnegative')
    if args.init_checkpoint:
        source = Path(args.init_checkpoint).expanduser().resolve()
        if not source.is_file():
            parser.error(f'Initial checkpoint does not exist: {source}')
        if args.out_dir and source.is_relative_to(Path(args.out_dir).expanduser().resolve()):
            parser.error('Initial checkpoint must be outside the new output directory')
        args.init_checkpoint = str(source)
    if args.throughput_preset:
        if (args.n_envs, args.rollout_horizon) != (2, 256):
            parser.error("--throughput-preset cannot be combined with explicit environment/horizon settings")
        args.n_envs, args.rollout_horizon = 8, 64
    if args.eval_workers <= 0 or (args.opponent_cache_size is not None and args.opponent_cache_size <= 0):
        parser.error("Worker and cache counts must be positive")
    if (args.restart_incomplete or args.skip_completed) and not args.out_dir:
        parser.error('--restart-incomplete/--skip-completed require --out-dir')
    if args.resume and (not args.out_dir or not args.milestone_evaluations or args.science_diagnostics
                        or args.restart_incomplete or args.skip_completed or args.league_dir):
        parser.error('--resume requires --out-dir, milestone evaluations, internal league, no restart/skip/science-diagnostics')
    positive = ("total_timesteps", "n_envs", "rollout_horizon", "batch_size", "minibatch_epochs",
                "eval_every", "log_every", "save_every", "eval_games_per_matchup", "pool_max", "snapshot_every")
    if any(getattr(args, key) <= 0 for key in positive):
        parser.error("Counts and intervals must be positive")
    if args.seed < 0 or args.eval_seed < 0 or args.pool_lag_steps < 0 or args.agent_seat not in (0, 1, 2):
        parser.error("Invalid seed, pool lag or agent seat")
    if not args.continuous and args.total_timesteps % args.n_envs:
        parser.error("total-timesteps must be divisible by n-envs")
    if args.continuous and args.eval_every % args.n_envs:
        parser.error('Continuous eval-every must be divisible by n-envs')
    if args.milestone_evaluations and not args.continuous:
        try:
            milestone_steps(args.total_timesteps, args.n_envs)
        except ValueError as exc:
            parser.error(str(exc))
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
    if args.continuous:
        args.total_timesteps = None
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.out_dir:
        output = Path(args.out_dir).expanduser().resolve()
        with training_directory_lock(output):
            if args.resume:
                matching_configuration(output, args)
                if not (output / 'latest.pt').is_file():
                    raise ValueError('Resume requires latest.pt with phase runtime state')
                old_manifest = json.loads((output / 'manifest.json').read_text())
                if old_manifest['engine_sha256'] != file_hash(Path(__file__).resolve().parents[3] / 'build/7Wonders'):
                    raise ValueError('Resume engine changed; preserve this run and start a new phase')
                load_model(output / 'latest.pt')
                saved_config = json.loads((output / 'config.json').read_text())
                specs = fixed_panel(saved_config.get('evaluation_opponent_references',
                                    saved_config.get('frozen_opponent_references', [])), args.eval_panel,
                                    args.eval_seed, args.eval_games_per_matchup)
                plan_path = output / 'evaluation_plan.json'
                saved_plan = json.loads(plan_path.read_text()) if plan_path.is_file() else None
                checkpoint_step = int(torch.load(output / 'latest.pt', weights_only=True)['timestep'])
                if saved_plan is None and checkpoint_step != 0:
                    raise ValueError('Resume requires the saved evaluation plan')
                if saved_plan is not None and plan_identity(specs, args.eval_action_mode) != saved_plan['plan_id']:
                    raise ValueError('Resume evaluation panel changed')
                result_path = output / 'result.json'
                if not args.continuous and result_path.is_file() and json.loads(result_path.read_text()).get('status') == 'completed':
                    print(f'Bereits abgeschlossen: {output}', flush=True)
                    return output
                return train(args)
            if prepare_existing_run(output, args):
                return output
            return train(args)
    return train(args)


def train(args):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    repo_root = Path(__file__).resolve().parents[3]
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
        if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
            raise ValueError(f"Output directory is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = run_io_root(Path("runs/ppo_unified"), args.mode)
    league_dir = Path(args.league_dir).expanduser().resolve() if args.league_dir else out_dir
    league_dir.mkdir(parents=True, exist_ok=True)
    stop_request = out_dir / 'stop_requested.json'
    if args.continuous:
        stop_request.unlink(missing_ok=True)
    if args.mode == "selfplay" and any(league_dir.glob("checkpoint_*.pt")) and not args.resume:
        raise ValueError("Self-play requires a fresh league directory for this run")
    io_root = exchange_io_root(args.io_root, repo_root / "io_train", "training",
                               prefer_ram=args.mode == "selfplay")
    reward_spec = RewardSpec(args.reward_profile, args.specialist_weight, args.reward_scale)
    config = {"evaluation_strategy": ('fixed_neural_opponents' if args.frozen_opponents else
              "rolling_trained_snapshots" if args.mode == "selfplay" else "fixed_bots"), **vars(args), "out_dir": str(out_dir), "io_root": str(io_root), "league_dir": str(league_dir),
              'reward_spec': reward_spec.describe()}
    atomic_json(out_dir / "config.json", config)
    run_manifest = manifest(repo_root, config)
    atomic_json(out_dir / "manifest.json", run_manifest)
    console = TrainingConsole(args, out_dir, io_root)
    timers = TimingBuckets()
    pool = OpponentPool(str(league_dir), max_size=args.pool_max,
                        min_step_lag=args.pool_lag_steps, latest_k=8, auto_refresh=False) if args.mode == "selfplay" else None
    if args.pair_pfsp:
        pool = PairPFSPPool(pool, power=args.pfsp_power, control_fraction=args.pfsp_control_fraction,
                           half_life=args.pfsp_half_life, prior_games=args.pfsp_prior_games)
        config['training_opponent_strategy'] = 'ordered_pair_pfsp'
        if args.milestone_evaluations:
            config['evaluation_strategy'] = 'fixed_external_control_panel'
    env_rngs = [np.random.default_rng(derive_seed(args.seed, "opponents", i)) for i in range(args.n_envs)]
    minibatch_rng = np.random.default_rng(derive_seed(args.seed, "minibatch"))
    action_generator = torch.Generator().manual_seed(derive_seed(args.seed, "learner"))
    weights = {"baseline": args.opponent_baseline, "mad_scientist": args.opponent_scientist, "military": args.opponent_military}
    policy_cache = OrderedDict()
    envs, episodes = [], [0] * args.n_envs
    head_ids = [0] * args.n_envs
    head_rngs = [np.random.default_rng(derive_seed(args.seed, 'head', i)) for i in range(args.n_envs)]
    active_games, checkpoint_hashes = {}, {}
    science_diagnostics = ScienceDiagnostics(out_dir / 'science_episodes.jsonl') if args.science_diagnostics else None
    observed_transitions = 0
    previous_evaluation_path = None
    previous_plan_id = None
    latest_evaluated_step = None
    timestep, updates, best_vp_diff = 0, 0, float("-inf")
    resumed_step = 0
    started = time.perf_counter()
    status = "failed"
    next_eval, next_save, next_snapshot = args.eval_every, args.save_every, args.snapshot_every
    milestones = None
    def reset_env(i, final_info=None, terminated=False):
        if final_info is not None:
            if isinstance(pool, PairPFSPPool) and terminated:
                game = active_games[i]
                pair = next(iter(game['opponents'].values()))
                pool.record_result(pair['pair_left_step'], pair['pair_right_step'],
                                   final_info['final_totals'], game['agent_seat'], timestep)
            if science_diagnostics is not None and terminated:
                science_diagnostics.finish(i, final_info['status'], observed_transitions)
            with (out_dir / "episodes.jsonl").open("a") as f:
                f.write(json.dumps({**active_games[i],
                    "status": "completed" if terminated else "truncated",
                    "completed_update": updates + 1,
                    "totals": final_info.get("final_totals"), "reason": final_info.get("truncated_reason"),
                    'reward_components': final_info.get('reward_components'),
                    "final_state": status_snapshot(final_info['status'], 3) if terminated else None}) + "\n")
            episodes[i] += 1
        head_ids[i] = int(head_rngs[i].integers(args.num_heads)) if args.num_heads > 1 else 0
        seat = args.agent_seat if args.mode == "fixed" else (args.agent_seat + i + episodes[i] * args.n_envs) % 3
        envs[i].agent_seat = seat
        if args.mode == "fixed":
            policies = sample_fixed_opponents(weights, env_rngs[i], seat)
        else:
            policies = sample_selfplay_opponents(pool, env_rngs[i], timestep, seat, policy_cache,
                cache_size=args.opponent_cache_size or max(args.pool_max, 16) + 2 * args.n_envs)
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
                if policy.sampling_metadata is not None:
                    descriptor.update(policy.sampling_metadata)
                elif path in frozen_metadata:
                    descriptor.update(frozen_metadata[path])
            opponents[str(opponent_seat)] = descriptor
        active_games[i] = {"env": i, "episode": episodes[i], "engine_seed": seed,
                           "agent_seat": seat, "selection_step": timestep, "opponents": opponents,
                           "head_id": head_ids[i]}
        obs, _ = envs[i].reset(seed=seed)
        for opponent_seat, descriptor in opponents.items():
            if int(opponent_seat) in envs[i]._opponent_head_ids:
                descriptor['head_id'] = envs[i]._opponent_head_ids[int(opponent_seat)]
        with (out_dir / 'opponents.jsonl').open('a') as f:
            f.write(json.dumps(active_games[i]) + '\n')
        if science_diagnostics is not None:
            science_diagnostics.start(i, active_games[i], observed_transitions)
        return obs
    def observe_transition(i, before, after, obs, action, probabilities):
        nonlocal observed_transitions
        observed_transitions += 1
        science_diagnostics.observe(i, before, after, obs, action, probabilities, timestep)
    frozen_metadata = {}
    try:
        frozen_references = []
        if args.frozen_opponents:
            copies = out_dir / 'frozen_opponents'
            copies.mkdir(exist_ok=args.resume)
            for index, source in enumerate(args.frozen_opponents):
                source_hash = file_hash(source)
                path = copies / f'opponent_{index:03d}.pt'
                if not args.resume:
                    shutil.copyfile(source, path)
                if file_hash(path) != source_hash or file_hash(source) != source_hash:
                    raise ValueError(f'Frozen opponent changed while copying: {source}')
                load_model(path)  # fail before any simulation on incompatible weights
                payload = torch.load(path, map_location='cpu', weights_only=True)
                group = next((g for g in args.population_spec or [] if source in g['checkpoints']), None)
                metadata = {'group': group['name'] if group else 'frozen',
                            'sampling_weight': group['weight'] / len(group['checkpoints']) if group else 1.0}
                frozen_metadata[str(path.resolve())] = metadata
                frozen_references.append({**reference(path), 'source_checkpoint': source,
                                          'step': int(payload['timestep']), **metadata})
            pool = FrozenOpponentPool(tuple(ref['checkpoint'] for ref in frozen_references),
                tuple(ref['sampling_weight'] for ref in frozen_references) if args.population_spec else ())
            if args.mixed_league:
                groups = [dict(name=g['name'], weight=g['weight'], paths=[r['checkpoint']
                          for r in frozen_references if r['group'] == g['name']]) for g in args.population_spec]
                historical = []
                if args.historical_checkpoints:
                    archive_dir = out_dir / 'historical_opponents'
                    archive_dir.mkdir(exist_ok=args.resume)
                    for i, source in enumerate(args.historical_checkpoints):
                        digest = file_hash(source)
                        target = archive_dir / f'historical_{i:03d}.pt'
                        if not args.resume:
                            shutil.copyfile(source, target)
                        if file_hash(target) != digest or file_hash(source) != digest:
                            raise ValueError('Historical checkpoint changed')
                        load_model(target)
                        historical.append(dict(checkpoint=str(target.resolve()), source_checkpoint=source, sha256=digest))
                config['historical_opponent_references'] = historical
                pool = MixedOpponentPool(OpponentPool(str(league_dir), max_size=args.pool_max,
                    min_step_lag=args.pool_lag_steps, auto_refresh=False), groups,
                    recent_weight=args.mixed_recent_weight, archive_weight=args.mixed_archive_weight,
                    historical_paths=tuple(r['checkpoint'] for r in historical))
                config['training_opponent_strategy'] = dict(recent=args.mixed_recent_weight,
                    archive=args.mixed_archive_weight, fixed=sum(g['weight'] for g in groups))
            config['frozen_opponent_references'] = frozen_references
            run_manifest['frozen_opponent_references'] = frozen_references
        evaluation_references = frozen_references
        if args.evaluation_population_spec:
            evaluation_references = []
            directory = out_dir / 'evaluation_opponents'
            directory.mkdir(exist_ok=args.resume)
            for group in args.evaluation_population_spec:
                for source in group['checkpoints']:
                    path = directory / f"opponent_{len(evaluation_references):03d}.pt"
                    digest = file_hash(source)
                    if not args.resume:
                        shutil.copyfile(source, path)
                    if file_hash(path) != digest or file_hash(source) != digest:
                        raise ValueError('Evaluation opponent changed')
                    load_model(path)
                    evaluation_references.append(dict(**reference(path), source_checkpoint=source,
                        group=group['name'], step=int(torch.load(path, weights_only=True)['timestep'])))
            config['evaluation_opponent_references'] = evaluation_references
            run_manifest['evaluation_opponent_references'] = evaluation_references
        for i in range(args.n_envs):
            envs.append(SevenWondersEnv(repo_root=str(repo_root), agent_seat=args.agent_seat,
                io_dir=str(io_root / "train" / f"env_{i:03d}" / "io"),
                reward_scale=args.reward_scale, invalid_action_fallback="raise", timers=timers,
                engine_poll_us=args.engine_poll_us, poll_seconds=args.poll_seconds,
                batch_opponents=args.opponent_inference == "batched",
                reward_profile=args.reward_profile, specialist_weight=args.specialist_weight,
                public_information=args.public_information, observation_memory=args.observation_memory))
        # Build the initial policy before starting any game: neural self-play is
        # available from the first action, including with a single snapshot.
        zero_obs = {key: np.zeros(space.shape, dtype=space.dtype)
                    for key, space in envs[0].observation_space.spaces.items()}
        global_dim = len(flatten_global(zero_obs, public_information=args.public_information,
                                       observation_memory=args.observation_memory))
        feature_dim = action_features(zero_obs, public_information=args.public_information).shape[1]
        model, initialization = initialize_model(args, global_dim, feature_dim)
        if args.resume:
            resumed = torch.load(out_dir / 'latest.pt', map_location='cpu', weights_only=True)
            # load_model validates schema/architecture in addition to strict weight shapes.
            model = load_model(out_dir / 'latest.pt').train()
            initialization = resumed['config']['initialization']
            timestep = resumed_step = int(resumed['timestep'])
            if not args.continuous and timestep > args.total_timesteps:
                raise ValueError('Resume checkpoint exceeds the requested total budget')
            updates = int(resumed['runtime_state']['updates'])
            restore_runtime(resumed['runtime_state'], episodes, env_rngs, head_rngs, minibatch_rng, action_generator)
            if isinstance(pool, PairPFSPPool):
                pool.load_state_dict(resumed['runtime_state']['pair_pfsp'])
                atomic_json(out_dir / 'league_status.json', pool.report(timestep))
            trim_uncommitted_logs(out_dir, timestep, updates)
            with (out_dir / 'resume_events.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=timestep, restored_optimizer=True,
                    abandoned_games=list(resumed['runtime_state']['active_games'].values()),
                    bit_exact=False, reason='Fresh engine games after restoring the last complete PPO update')) + '\n')
            next_save = (timestep // args.save_every + 1) * args.save_every
            next_snapshot = (timestep // args.snapshot_every + 1) * args.snapshot_every
            console.start_step = timestep
        if args.init_checkpoint:
            print(f"Phasenstartgewichte: {args.init_checkpoint} "
                  f"(Quellschritt {initialization['source_timestep']}); "
                  + ("Optimizer und Schrittzähler werden wiederhergestellt." if args.resume else
                     "neuer Optimizer und Schrittzähler."), flush=True)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
        if args.resume:
            optimizer.load_state_dict(resumed['optimizer_state'])
            print(f'Fortgesetzt ab Schritt {timestep}; {args.n_envs} laufende Spiele beginnen mit neuen Seeds.', flush=True)
        hidden = model.global_mlp[0].out_features
        config.update(global_dim=global_dim, action_feat_dim=feature_dim, hidden=hidden,
                      model_architecture_version=model.architecture_version, initialization=initialization)
        run_manifest['initialization'] = initialization
        run_manifest['model_architecture_version'] = model.architecture_version
        atomic_json(out_dir / "config.json", config)
        atomic_json(out_dir / "manifest.json", run_manifest)
        def save(path, metrics=None):
            with timers.time("checkpoint"):
                is_snapshot = path.name.startswith("checkpoint_")
                runtime = (None if is_snapshot else runtime_snapshot(updates, episodes, active_games,
                    env_rngs, head_rngs, minibatch_rng, action_generator))
                if runtime is not None and isinstance(pool, PairPFSPPool):
                    runtime['pair_pfsp'] = pool.state_dict()
                save_checkpoint(path, model, None if is_snapshot else optimizer, timestep, config, run_manifest,
                                metrics, runtime_state=runtime)
                if is_snapshot and isinstance(pool, (OpponentPool, MixedOpponentPool, PairPFSPPool)):
                    pool.register(path)
                if isinstance(pool, PairPFSPPool):
                    atomic_json(out_dir / 'league_status.json', pool.report(timestep))
        if args.mode == "selfplay" and not args.resume:
            save(league_dir / "checkpoint_0.pt")
            references = [reference(league_dir / "checkpoint_0.pt")]
            if not args.init_checkpoint:
                reference_dir = out_dir / "references"
                reference_dir.mkdir()
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(derive_seed(args.seed, "model", 1))
                    from sevenwonders_rl.features.observation_history import MEMORY_DIM
                    base_dim = global_dim - MEMORY_DIM if args.observation_memory else global_dim
                    independent = (MultiHeadActorCritic(base_dim, feature_dim,
                        num_heads=args.num_heads, prior_strength=args.prior_strength,
                        public_information=args.public_information)
                        if args.num_heads > 1 else ActorCritic(base_dim, feature_dim,
                            public_information=args.public_information))
                    if args.observation_memory:
                        independent = extend_observation_memory(independent)
                independent_path = reference_dir / "independent_initial.pt"
                save_checkpoint(independent_path, independent, None, 0, config, run_manifest)
                references.append(reference(independent_path))
            atomic_json(out_dir / "references.json", {"references": references,
                'initialization': initialization,
                "purpose": "optional post-training initial references; in-training evaluation uses rolling trained snapshots"})
            print(('Training: eigene eingefrorene Zwischenstände, ergebnisabhängige Gegnerpaare (PFSP).' if args.pair_pfsp else
                   'Training: gemischte Snapshot-Liga und feste Referenzgruppen.' if args.mixed_league else
                   'Training und Evaluation: fest eingefrorene neuronale Gegner; '
                   'eigene Snapshots werden nicht als Gegner verwendet.' if args.frozen_opponents else
                   "Checkpoint-Evaluationen: trainierte Vorgänger; keine Anfangsevaluation. "
                   "Erste Messung, sobald ältere trainierte Snapshots vorhanden sind."), flush=True)
        if args.milestone_evaluations:
            # Even an interruption during the initial evaluation is recoverable.
            save(out_dir / 'latest.pt')
            milestones = MilestoneEvaluations(out_dir, args, evaluation_references, repo_root, io_root, save,
                                              resume_step=timestep if args.resume else None)
            if milestones.baseline is None:
                console.evaluation_start(0, len(milestones.specs), references=frozen_references)
                with timers.time('evaluation'):
                    initial_evaluation = milestones.evaluate(model, 0)
                console.evaluation_end(initial_evaluation)
            else:
                latest_evaluated_step = max((r['step'] for r in milestones.curve if r['step'] > 0), default=None)
                # An interruption may happen after saving the update but during its evaluation,
                # including at 100%. Complete that report before collecting more samples.
                if milestones.is_due(timestep) and not any(r['step'] == timestep for r in milestones.curve):
                    console.evaluation_start(timestep, len(milestones.specs), references=frozen_references)
                    with timers.time('evaluation'):
                        evaluation = milestones.evaluate(model, timestep)
                    save(out_dir / 'evaluated.pt', evaluation)
                    latest_evaluated_step = timestep
                    console.evaluation_end(evaluation)
            print(f'Dashboard während des Laufs: {out_dir / "dashboard.html"}', flush=True)
        with timers.time("initialization.reset"):
            observations = ([reset_env(i) for i in range(args.n_envs)]
                            if (args.continuous or timestep < args.total_timesteps) and not
                            (args.continuous and stop_request.exists()) else [])
        while (args.continuous or timestep < args.total_timesteps) and not (args.continuous and stop_request.exists()):
            boundary = (milestones.next_step(timestep)
                        if milestones is not None else args.total_timesteps)
            horizon = min(args.rollout_horizon, (boundary - timestep) // args.n_envs)
            with timers.time("collection"):
                rollout, observations = collect_rollout(model, envs, observations, horizon,
                    action_generator, reset_env, timers,
                    on_transition=observe_transition if science_diagnostics is not None else None,
                    head_ids=head_ids if args.num_heads > 1 else None)
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
            if (milestones is not None and milestones.is_due(timestep)
                    and not (args.continuous and stop_request.exists())):
                # The model at a scheduled evaluation must be durable before its report.
                save(out_dir / 'latest.pt')
                console.evaluation_start(timestep, len(milestones.specs), references=frozen_references)
                with timers.time('evaluation'):
                    evaluation = milestones.evaluate(model, timestep)
                save(out_dir / 'evaluated.pt', evaluation)
                latest_evaluated_step = timestep
                save(out_dir / 'latest.pt')
                console.evaluation_end(evaluation)
            if milestones is None and (timestep >= next_eval or (args.mode == "selfplay" and timestep == args.total_timesteps)):
                references = (frozen_references if args.frozen_opponents else
                              rolling_references(league_dir, timestep) if args.mode == 'selfplay' else None)
                specs = evaluation_plan(references, args.eval_seed, args.eval_games_per_matchup,
                                        all_pairs=bool(args.population_spec)) if references else []
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
                            batch_opponents=args.opponent_inference == "batched", workers=args.eval_workers)
                    evaluation.update(plan_id=current_plan_id, references=references, step=timestep,
                                      evaluation_strategy=config['evaluation_strategy'],
                                      evaluation_reward_profile='standard', training_reward_profile=args.reward_profile)
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
                if milestones is not None:
                    milestones.update_status(timestep, 'completed' if milestones.finished(timestep) else 'training')
        save(out_dir / "latest.pt")
        status = 'stopped' if args.continuous else 'completed'
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
        if milestones is not None:
            milestones.update_status(timestep, status)
            milestones.publish_dashboard()
        for env in envs:
            env.close()
        shutil.rmtree(io_root)
        elapsed = time.perf_counter() - started
        atomic_json(out_dir / "timing.json", timers.as_dict(elapsed))
        atomic_json(out_dir / "result.json", {"status": status, "timestep": timestep, "updates": updates,
                    "elapsed_seconds": elapsed, "steps_per_second": (timestep - resumed_step) / elapsed,
                    "resumed_from_step": resumed_step,
                    "best_vp_diff": best_vp_diff if math.isfinite(best_vp_diff) else None,
                    "selection_metric": "latest_valid_evaluation" if args.mode == "selfplay" else "vp_diff_mean",
                    "latest_evaluated_step": latest_evaluated_step,
                    "best_selection_score": best_vp_diff if math.isfinite(best_vp_diff) else None})
    console.finish(out_dir, status=status)
    return out_dir


if __name__ == "__main__":
    main()
