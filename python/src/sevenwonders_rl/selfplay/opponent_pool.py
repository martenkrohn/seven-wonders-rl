"""Opponent pool for self-play league training."""

from __future__ import annotations

import re
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_CKPT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


def read_population(path):
    """Load fixed groups; paths are relative to the invoking working directory.

    Select a group by weight, then a member uniformly, independently per seat.
    Returning the normalized spec lets restart checks detect content changes.
    """
    groups = json.loads(Path(path).read_text())['groups']
    if not groups:
        raise ValueError('Opponent population is empty')
    result, seen, names = [], set(), set()
    for group in groups:
        name, weight = group['name'], float(group['weight'])
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Population group names must be unique nonempty strings')
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Population group weights must be finite and positive')
        paths = [str(Path(p).expanduser().resolve()) for p in group['checkpoints']]
        if not paths or len(set(paths)) != len(paths) or seen.intersection(paths):
            raise ValueError('Population checkpoints must be nonempty and unique across groups')
        if any(not Path(p).is_file() for p in paths):
            raise ValueError(f'Missing population checkpoint in group {name}')
        names.add(name)
        seen.update(paths)
        result.append(dict(name=name, weight=weight, checkpoints=paths))
    return result


@dataclass(frozen=True)
class FrozenOpponentPool:
    """Explicit fixed neural opponents; learner snapshots never enter this pool."""
    paths: Tuple[str, ...]
    weights: Tuple[float, ...] = ()

    def __post_init__(self):
        if self.weights and (len(self.weights) != len(self.paths) or
                any(not math.isfinite(w) or w <= 0 for w in self.weights) or
                not math.isfinite(sum(self.weights))):
            raise ValueError('Frozen opponent weights must match paths and be finite and positive')

    @property
    def max_size(self):
        return len(self.paths)

    def sample_path(self, rng, current_step):
        if not self.paths:
            raise ValueError('Frozen opponent pool is empty')
        if self.weights:
            probs = np.asarray(self.weights) / sum(self.weights)
            return self.paths[int(rng.choice(len(self.paths), p=probs))]
        return self.paths[int(rng.integers(len(self.paths)))]

    def pool_size(self, current_step):
        return len(self.paths)

    def total_size(self):
        return len(self.paths)


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
    auto_refresh: bool = True
    _checkpoints: Optional[List[Tuple[int, Path]]] = field(default=None, init=False, repr=False)

    def refresh(self):
        """Discover external changes; training registers its own atomic saves."""
        self._checkpoints = None
        self._checkpoints = self._list_checkpoints()

    def register(self, path):
        path = Path(path)
        match = _CKPT_RE.fullmatch(path.name)
        if not match or path.parent.resolve() != Path(self.ckpt_dir).resolve() or not path.is_file():
            raise ValueError(f"Not a published pool snapshot: {path}")
        if self._checkpoints is None:
            self.refresh()
        items = {step: saved for step, saved in self._checkpoints}
        items[int(match.group(1))] = path
        self._checkpoints = sorted(items.items())

    def _list_checkpoints(self) -> List[Tuple[int, Path]]:
        """
        List all checkpoint_*.pt files in ckpt_dir, sorted by step.

        Returns:
            List of (step, path) tuples, sorted by step ascending
        """
        if not self.auto_refresh and self._checkpoints is not None:
            return list(self._checkpoints)
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
        if not self.auto_refresh:
            self._checkpoints = items
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


@dataclass
class MixedOpponentPool:
    """Fixed group weights, lagged recent snapshots and a spread-out archive.

    Archive members are selected by time, not by measured strategic diversity.
    Missing dynamic groups fall back to the mature anchor group, never random play.
    """
    league: OpponentPool
    groups: list
    recent_weight: float = .50
    archive_weight: float = .15
    recent_count: int = 6
    archive_count: int = 6
    historical_paths: tuple = ()

    def __post_init__(self):
        if not any(g['name'] == 'anchors' and g['paths'] for g in self.groups):
            raise ValueError('Mixed league requires a nonempty anchors group')
        if any(not math.isfinite(w) or w <= 0 for w in (self.recent_weight, self.archive_weight)):
            raise ValueError('Dynamic group weights must be positive and finite')
        self.max_size = self.league.max_size + len(self.historical_paths) + sum(len(g['paths']) for g in self.groups)

    def register(self, path):
        self.league.register(path)

    def dynamic_members(self, current_step):
        items = [(s, p) for s, p in self.league.eligible(current_step) if s > 0]
        recent = items[-self.recent_count:]
        older = [(None, Path(p)) for p in self.historical_paths] + items[:-self.recent_count]
        if len(older) > self.archive_count:
            indices = np.linspace(0, len(older) - 1, self.archive_count, dtype=int)
            older = [older[i] for i in indices]
        return {'recent': [str(p.resolve()) for _, p in recent],
                'archive': [str(p.resolve()) for _, p in older]}

    def sample_entry(self, rng, current_step):
        dynamic = self.dynamic_members(current_step)
        groups = [dict(name='recent', weight=self.recent_weight, paths=dynamic['recent']),
                  dict(name='archive', weight=self.archive_weight, paths=dynamic['archive'])] + self.groups
        weights = np.asarray([g['weight'] for g in groups], dtype=float)
        group = groups[int(rng.choice(len(groups), p=weights / weights.sum()))]
        paths = group['paths']
        fallback = not paths
        if fallback:
            paths = next(g['paths'] for g in self.groups if g['name'] == 'anchors')
        path = paths[int(rng.integers(len(paths)))]
        return path, dict(group=group['name'], fallback_to_anchors=fallback,
                          group_weight=group['weight'] / float(weights.sum()),
                          eligible_members=len(paths))

    def sample_path(self, rng, current_step):
        return self.sample_entry(rng, current_step)[0]

    def pool_size(self, current_step):
        dynamic = self.dynamic_members(current_step)
        return sum(map(len, dynamic.values())) + sum(len(g['paths']) for g in self.groups)

    def total_size(self):
        return self.league.total_size() + len(self.historical_paths) + sum(len(g['paths']) for g in self.groups)
