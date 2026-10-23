"""Batch frozen opponent inference without sharing randomness or game state."""
from contextlib import nullcontext

from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.policies.ppo_checkpoint_policy import PPOCheckpointPolicy


def opponent_actions_batch(envs, statuses=None, timers=None):
    """Return per-environment seat overrides, grouping the same cached policy.

    Every observation comes from its own pre-turn status. Unsupported policies
    and legacy opponent samplers continue through the environment's serial path.
    """
    actions = [{} for _ in envs]
    groups = {}
    def timing(name):
        return timers.time(name) if timers is not None else nullcontext()
    with timing("policies.opponents"):
        with timing("policies.opponents.prepare"):
            for index, env in enumerate(envs):
                if not getattr(env, "batch_opponents", False) or getattr(env, "_opponent_sampler", None) is not None:
                    continue
                status = statuses[index] if statuses is not None else None
                if status is None:
                    status = env._last_status
                if status is None:
                    continue
                for seat, wrapper in enumerate(env.opponent_policies):
                    if seat == env.agent_seat or not isinstance(wrapper.policy, PPOCheckpointPolicy):
                        continue
                    policy = wrapper.policy
                    group = groups.setdefault(id(policy), (policy, []))[1]
                    group.append((index, seat, build_obs(status, seat), env._seat_rngs[seat]))
        for policy, requests in groups.values():
            with timing("policies.opponents.inference"):
                selected = policy.select_actions([r[2] for r in requests], [r[3] for r in requests])
            if len(selected) != len(requests):
                raise ValueError("Batched opponent returned an incomplete action list")
            if timers is not None:
                timers.add_ns("policies.opponents.requests", 0, calls=len(requests))
            for (index, seat, obs, _), action in zip(requests, selected):
                if not 0 <= action < len(obs["action_mask"]) or not obs["action_mask"][action]:
                    raise ValueError(f"Batched neural opponent returned illegal action {action} for seat {seat}")
                actions[index][seat] = action
    return actions
