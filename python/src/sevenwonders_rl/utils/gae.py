"""GAE on [time, environment] arrays, before any minibatch flattening."""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_gae(rewards, values, terminated, truncated, next_values,
                gamma: float = 0.999, gae_lambda: float = 0.95):
    """Return raw advantages and returns; next_values are pre-reset values.

    Valid external truncations bootstrap but cut the trace. Transport failures
    are not transitions and must be rejected by the collector before calling.
    """
    shape = rewards.shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("GAE requires nonempty [time, environment] tensors")
    if any(x.shape != shape for x in (values, terminated, truncated, next_values)):
        raise ValueError("All GAE tensors must have the same shape")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    if not all(torch.isfinite(x).all() for x in (rewards, values, next_values)):
        raise ValueError("Non-finite GAE input")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(shape[0])):
        bootstrap = (~terminated[t].bool()).to(rewards.dtype)
        trace = (~(terminated[t].bool() | truncated[t].bool())).to(rewards.dtype)
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        carry = delta + gamma * gae_lambda * trace * carry
        advantages[t] = carry
    return advantages, advantages + values


def normalize_advantages(advantages):
    return (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)
