"""PPO checkpoint policy wrapper for opponent snapshots in self-play."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.ppo_utils import flatten_global, masked_logits


@dataclass
class PPOCheckpointPolicy:
    """
    Loads a PPO checkpoint and provides action selection via select_action(obs, rng).

    Compatible with SeatPolicy(name, policy=obj) injection pattern.
    """

    ckpt_path: str
    device: str = "cpu"
    deterministic: bool = False

    def __post_init__(self) -> None:
        """Load checkpoint and initialize model."""
        p = Path(self.ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        ckpt = torch.load(p, map_location=self.device)

        # Extract config from checkpoint metadata
        cfg = ckpt.get("config", {})
        global_dim = int(cfg.get("global_dim", 0))
        action_feat_dim = int(cfg.get("action_feat_dim", 0))

        if global_dim <= 0 or action_feat_dim <= 0:
            raise ValueError(
                f"Invalid config in checkpoint {p}: "
                f"global_dim={global_dim}, action_feat_dim={action_feat_dim}"
            )

        # Initialize model
        self.model = ActorCritic(global_dim, action_feat_dim).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self._rng = np.random.default_rng(0)

    def set_seed(self, seed: int) -> None:
        """Set seed for stochastic action selection."""
        self._rng = np.random.default_rng(seed)

    def reset(self, obs: Dict[str, Any]) -> None:
        """Optional reset hook (for compatibility with other policies)."""
        pass

    def select_action(
        self, obs: Dict[str, Any], rng: Optional[np.random.Generator] = None
    ) -> int:
        """
        Select an action from observation.

        Args:
            obs: observation dict with keys: global_table, wonder_stage, hand_ids,
                 action_feats, action_mask
            rng: optional RNG (used if provided, else internal RNG)

        Returns:
            Action index (0-27)
        """
        if rng is None:
            rng = self._rng

        global_vec = flatten_global(obs)
        action_feats = obs["action_feats"]
        mask = obs["action_mask"]

        global_t = torch.tensor(
            global_vec[None, :], dtype=torch.float32, device=self.device
        )
        feats_t = torch.tensor(
            action_feats[None, :, :], dtype=torch.float32, device=self.device
        )
        mask_t = torch.tensor(mask[None, :], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, _ = self.model.forward_batch(global_t, feats_t)
            logits = masked_logits(logits, mask_t)

            if self.deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())

        return action
