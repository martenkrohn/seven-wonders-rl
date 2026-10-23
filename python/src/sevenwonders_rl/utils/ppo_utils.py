from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch


def flatten_global(obs: Dict[str, Any]) -> np.ndarray:
    # table_prod includes each wonder's starting resource as part of the current
    # production totals. No wonder-stage cost/effect table is required for this.
    parts = [
        obs["global_table"].astype(np.float32).reshape(-1),
        obs["table_points"].astype(np.float32).reshape(-1),
        obs["table_prod"].astype(np.float32).reshape(-1),
        obs["table_amount"].astype(np.float32).reshape(-1),
        obs["wonder_stage"].astype(np.float32).reshape(-1),
        obs["game_meta"].astype(np.float32).reshape(-1),
    ]
    return np.concatenate(parts, axis=0)


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
) -> int:
    global_flat = torch.from_numpy(flatten_global(obs)).to(device)
    action_feats = torch.from_numpy(obs["action_feats"]).to(device)
    mask = torch.from_numpy(obs["action_mask"]).to(device)

    with torch.no_grad():
        logits, _ = model.forward_batch(global_flat.unsqueeze(0), action_feats.unsqueeze(0))
        logits = masked_logits(logits.squeeze(0), mask)
        if deterministic:
            return int(torch.argmax(logits).item())
        if rng is None:
            raise ValueError("Stochastic action selection requires a local RNG")
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        return int(rng.choice(len(probs), p=probs / probs.sum()))
