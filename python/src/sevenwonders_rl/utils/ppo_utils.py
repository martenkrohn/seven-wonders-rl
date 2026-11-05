from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch


def flatten_global(obs: Dict[str, Any], model=None, *, public_information=False,
                   observation_memory=False) -> np.ndarray:
    # table_prod includes each wonder's starting resource as part of the current
    # production totals. No wonder-stage cost/effect table is required for this.
    parts = [
        obs["global_table"].astype(np.float32).reshape(-1),
        obs["table_points"].astype(np.float32).reshape(-1),
        obs["table_prod"].astype(np.float32).reshape(-1),
        obs["table_amount"].astype(np.float32).reshape(-1),
        obs["wonder_stage"].astype(np.float32).reshape(-1),
        obs["game_meta"].astype(np.float32).reshape(-1),
        obs['science_jokers'].astype(np.float32).reshape(-1),
        obs['table_discounts'].astype(np.float32).reshape(-1),
    ]
    if public_information or getattr(model, 'public_information', False):
        parts.append(obs['public_state'].astype(np.float32).reshape(-1))
    if observation_memory or getattr(model, 'memory_enabled', False):
        from sevenwonders_rl.features.observation_history import MEMORY_KEY, MEMORY_DIM
        if MEMORY_KEY not in obs or obs[MEMORY_KEY].shape != (MEMORY_DIM,):
            raise ValueError('Observation-memory model requires an environment with complete history')
        parts.append(obs[MEMORY_KEY].astype(np.float32))
    return np.concatenate(parts, axis=0)


def action_features(obs, model=None, *, public_information=False):
    if public_information or getattr(model, 'public_information', False):
        return np.concatenate((obs['action_feats'], obs['wonder_action_feats']), axis=-1)
    return obs['action_feats']


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.all((mask > 0).any(dim=-1)):
        raise ValueError("Cannot select an action with an empty legal-action mask")
    neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    return torch.where(mask > 0, logits, neg_inf)


def select_action_from_obs(
    obs: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    deterministic: bool = False,
    rng: np.random.Generator | None = None,
    head_id: int | None = None,
) -> int:
    global_flat = torch.from_numpy(flatten_global(obs, model)).to(device)
    action_feats = torch.from_numpy(action_features(obs, model)).to(device)
    mask = torch.from_numpy(obs["action_mask"]).to(device)

    with torch.no_grad():
        inputs = (global_flat.unsqueeze(0), action_feats.unsqueeze(0))
        logits, _ = (model.forward_batch(*inputs, [head_id]) if getattr(model, 'num_heads', 1) > 1
                     else model.forward_batch(*inputs))
        logits = masked_logits(logits.squeeze(0), mask)
        if deterministic:
            return int(torch.argmax(logits).item())
        if rng is None:
            raise ValueError("Stochastic action selection requires a local RNG")
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        return int(rng.choice(len(probs), p=probs / probs.sum()))
