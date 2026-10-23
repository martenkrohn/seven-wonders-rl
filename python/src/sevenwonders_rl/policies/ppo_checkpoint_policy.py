"""PPO checkpoint policy wrapper for opponent snapshots in self-play."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.ppo_utils import flatten_global, masked_logits
from sevenwonders_rl.utils.checkpoints import load_model


@dataclass
class PPOCheckpointPolicy:
    """
    Loads a PPO checkpoint and provides action selection via select_action(obs, rng).

    Compatible with SeatPolicy(name, policy=obj) injection pattern.
    """

    ckpt_path: str
    device: str = "cpu"
    deterministic: bool = False
    allow_legacy: bool = False

    def __post_init__(self) -> None:
        """Load checkpoint and initialize model."""
        p = Path(self.ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        self.model = load_model(p, self.device, allow_legacy=self.allow_legacy)

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
        return self.select_actions([obs], [rng])[0]

    def select_actions(self, observations, rngs):
        """One forward pass for this frozen policy; one local RNG per seat/game."""
        if len(observations) != len(rngs):
            raise ValueError("Each opponent observation requires its own RNG")
        if not observations:
            return []
        global_t = torch.from_numpy(np.stack([flatten_global(obs) for obs in observations])).to(self.device)
        feats_t = torch.from_numpy(np.stack([obs["action_feats"] for obs in observations])).to(self.device, dtype=torch.float32)
        mask_t = torch.from_numpy(np.stack([obs["action_mask"] for obs in observations])).to(self.device)

        with torch.no_grad():
            logits, _ = self.model.forward_batch(global_t, feats_t)
            logits = masked_logits(logits, mask_t)

            if self.deterministic:
                return logits.argmax(-1).cpu().tolist()
            else:
                probs = logits.softmax(-1).cpu().numpy().astype(np.float64)
                return [int(rng.choice(len(row), p=row / row.sum())) for row, rng in zip(probs, rngs)]
