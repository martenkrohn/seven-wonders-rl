"""Deterministic opponent sampling for PPO training.

Samples two opponents per episode from a fixed distribution with:
- Deterministic RNG based on (seed, env_id, episode_idx)
- Independent sampling (duplicates allowed)
- Normalized probability weights
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class OpponentSampler:
    """Deterministic opponent sampler for episode-level diversity.
    
    Args:
        weights: Dictionary mapping policy names to sampling probabilities
        seed: Base random seed for reproducibility
        env_id: Environment worker ID (for multiprocessing)
    
    Example:
        >>> sampler = OpponentSampler(
        ...     weights={"baseline": 0.4, "mad_scientist": 0.3, "military": 0.2, "random": 0.1},
        ...     seed=42,
        ...     env_id=0
        ... )
        >>> left, right = sampler.sample_two(episode_idx=0)
        >>> # Returns deterministic pair for this episode
    """
    
    weights: dict[str, float]
    seed: int
    env_id: int = 0

    def __post_init__(self) -> None:
        """Normalize weights and prepare for efficient sampling."""
        if not self.weights:
            raise ValueError("OpponentSampler requires non-empty weights dict")
        
        names = list(self.weights.keys())
        probs = np.array([self.weights[n] for n in names], dtype=np.float64)
        
        if np.any(probs < 0):
            raise ValueError("All weights must be non-negative")
        if np.sum(probs) == 0:
            raise ValueError("At least one weight must be positive")
        
        # Normalize to probabilities
        probs = probs / probs.sum()
        
        self._names = names
        self._probs = probs

    def sample_two(self, episode_idx: int) -> tuple[str, str]:
        """Sample two opponents for an episode (independent, duplicates allowed).
        
        Args:
            episode_idx: Episode number (incremented per reset)
        
        Returns:
            Tuple of (left_opponent_name, right_opponent_name)
        """
        # Deterministic RNG per (seed, env_id, episode)
        # Use prime multipliers to avoid collisions
        s = (self.seed + 10007 * self.env_id + 1000003 * episode_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(s)
        
        # Sample independently (duplicates allowed)
        a = rng.choice(self._names, p=self._probs)
        b = rng.choice(self._names, p=self._probs)
        
        return str(a), str(b)
    
    def __repr__(self) -> str:
        weight_str = ", ".join(f"{k}={v:.2f}" for k, v in self.weights.items())
        return f"OpponentSampler({weight_str}, seed={self.seed}, env_id={self.env_id})"
