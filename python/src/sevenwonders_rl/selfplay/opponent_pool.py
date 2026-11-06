"""Published frozen-snapshot inventory for the active training league."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

_CKPT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


@dataclass
class OpponentPool:
    """Published checkpoints, registered explicitly by the durable league."""

    ckpt_dir: str
    max_size: int = 50
    min_step_lag: int = 0
    latest_k: int = 5
    auto_refresh: bool = True
    _checkpoints: Optional[List[Tuple[int, Path]]] = field(
        default=None, init=False, repr=False
    )

    def refresh(self):
        """Discover external changes; training registers its own atomic saves."""
        self._checkpoints = None
        self._checkpoints = self._list_checkpoints()

    def register(self, path):
        path = Path(path)
        match = _CKPT_RE.fullmatch(path.name)
        if (
            not match
            or path.parent.resolve() != Path(self.ckpt_dir).resolve()
            or not path.is_file()
        ):
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

    def total_size(self) -> int:
        """Return total number of checkpoints on disk (ignores lag)."""
        return len(self._list_checkpoints())
