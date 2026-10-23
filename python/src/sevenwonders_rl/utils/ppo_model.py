from __future__ import annotations

import torch
import torch.nn as nn


MODEL_ARCHITECTURE_VERSION = 2


class ActorCritic(nn.Module):
    def __init__(self, global_dim: int, action_feat_dim: int, hidden: int = 128,
                 *, architecture_version: int = MODEL_ARCHITECTURE_VERSION):
        super().__init__()
        if architecture_version not in (1, MODEL_ARCHITECTURE_VERSION):
            raise ValueError(f"Unsupported model architecture version: {architecture_version}")
        self.architecture_version = architecture_version
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # A linear head here makes the global contribution identical for every
        # action, so it cancels in softmax. Joint nonlinear processing lets the
        # policy learn interactions between the state and each candidate action.
        # Version 1 is retained only for explicitly requested historical inference.
        self.policy_head = (nn.Linear(hidden * 2, 1) if architecture_version == 1 else
                            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                          nn.Linear(hidden, 1)))
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward_batch(self, global_flat: torch.Tensor, action_feats: torch.Tensor):
        # global_flat: (B, G), action_feats: (B, A, F)
        global_ctx = self.global_mlp(global_flat)  # (B, H)
        action_emb = self.action_mlp(action_feats)  # (B, A, H)

        global_expand = global_ctx.unsqueeze(1).expand(-1, action_emb.size(1), -1)
        logits = self.policy_head(torch.cat([action_emb, global_expand], dim=-1)).squeeze(-1)  # (B, A)
        values = self.value_head(global_ctx).squeeze(-1)  # (B,)
        return logits, values
