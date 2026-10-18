"""Random policy that selects uniformly from legal actions."""

from __future__ import annotations
import numpy as np
from typing import Any


class RandomPolicy:
    """Pure random baseline: uniform sampling over legal actions.
    
    Args:
        seed: Random seed for reproducibility (optional)
    """
    
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, obs: dict[str, Any], rng: np.random.Generator | None = None) -> int:
        """Select random legal action from mask.
        
        Args:
            obs: Observation dict containing 'action_mask'
            rng: Optional RNG (uses self.rng if not provided)
        
        Returns:
            Action index (0-27)
        """
        if rng is None:
            rng = self.rng
        
        mask = obs.get("action_mask", None)
        if mask is None:
            raise ValueError("RandomPolicy requires 'action_mask' in observation")
        
        legal = np.flatnonzero(mask == 1)
        if legal.size == 0:
            return 0  # Fallback (shouldn't happen)
        
        return int(rng.choice(legal))
    
    def reset(self, observation: dict[str, Any] | None = None) -> None:
        """Reset policy state (no-op for stateless random policy)."""
        pass
    
    def __repr__(self) -> str:
        return "RandomPolicy()"
