"""Opponent pool for self-play league training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_CKPT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


@dataclass
class OpponentPool:
    """
    Manages a pool of frozen PPO checkpoint snapshots for self-play training.

    Features:
    - Discovers checkpoints from a directory
    - Filters by step lag (don't sample too-recent snapshots)
    - Enforces max pool size (FIFO by step)
    - Samples with mixture distribution: 50% from latest_k, 50% uniform
    """

    ckpt_dir: str
    max_size: int = 50
    min_step_lag: int = 0
    latest_k: int = 5

    def _list_checkpoints(self) -> List[Tuple[int, Path]]:
        """
        List all checkpoint_*.pt files in ckpt_dir, sorted by step.

        Returns:
            List of (step, path) tuples, sorted by step ascending
        """
        p = Path(self.ckpt_dir)
        if not p.exists():
            return []

        items = []
        for f in p.glob("checkpoint_*.pt"):
            m = _CKPT_RE.search(f.name)
            if not m:
                continue
            step = int(m.group(1))
            items.append((step, f))

        items.sort(key=lambda x: x[0])
        return items

    def eligible(self, current_step: int) -> List[Tuple[int, Path]]:
        """
        Get list of eligible checkpoints for sampling given current training step.

        Filters by:
        - min_step_lag: exclude snapshots newer than (current_step - lag)
        - max_size: keep only the most recent max_size checkpoints

        Args:
            current_step: current timestep in training

        Returns:
            Filtered list of (step, path) tuples
        """
        items = self._list_checkpoints()

        # Apply lag filter
        if self.min_step_lag > 0:
            items = [
                (s, f) for (s, f) in items if s == 0 or s <= (current_step - self.min_step_lag)
            ]

        # Apply max_size (keep newest)
        if self.max_size > 0 and len(items) > self.max_size:
            items = items[-self.max_size :]

        return items

    def sample_path(
        self, rng: np.random.Generator, current_step: int
    ) -> Optional[str]:
        """
        Sample a checkpoint path from the eligible pool.

        Uses mixture distribution:
        - 50% from latest_k snapshots
        - 50% uniform from full eligible pool

        Args:
            rng: numpy RNG for sampling
            current_step: current training step

        Returns:
            Path to checkpoint file, or None if pool is empty
        """
        items = self.eligible(current_step)
        if not items:
            return None

        # Split pool: latest_k vs rest
        latest = items[-min(self.latest_k, len(items)) :]

        # Sample: 50% from latest, 50% uniform
        if rng.random() < 0.5:
            idx = int(rng.integers(0, len(latest)))
            _, path = latest[idx]
        else:
            idx = int(rng.integers(0, len(items)))
            _, path = items[idx]

        return str(path)

    def pool_size(self, current_step: int) -> int:
        """Return current number of eligible checkpoints."""
        return len(self.eligible(current_step))

    def total_size(self) -> int:
        """Return total number of checkpoints on disk (ignores lag)."""
        return len(self._list_checkpoints())
