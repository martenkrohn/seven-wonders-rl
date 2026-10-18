from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, global_dim: int, action_feat_dim: int, hidden: int = 128):
        super().__init__()
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
        self.policy_head = nn.Linear(hidden * 2, 1)
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
