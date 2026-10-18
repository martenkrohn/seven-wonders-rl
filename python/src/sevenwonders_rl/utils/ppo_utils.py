from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch


def flatten_global(obs: Dict[str, Any]) -> np.ndarray:
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
    neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    return torch.where(mask > 0, logits, neg_inf)


def select_action_from_obs(
    obs: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    deterministic: bool = False,
) -> int:
    global_flat = torch.from_numpy(flatten_global(obs)).to(device)
    action_feats = torch.from_numpy(obs["action_feats"]).to(device)
    mask = torch.from_numpy(obs["action_mask"]).to(device)

    with torch.no_grad():
        logits, _ = model.forward_batch(global_flat.unsqueeze(0), action_feats.unsqueeze(0))
        logits = masked_logits(logits.squeeze(0), mask)
        if deterministic:
            return int(torch.argmax(logits).item())
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())
