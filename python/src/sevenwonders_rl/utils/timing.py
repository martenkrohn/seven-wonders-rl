"""Lightweight timing buckets for performance profiling."""
from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import contextmanager
from time import perf_counter_ns
from typing import Dict, Iterator, Optional, List, Tuple
from collections import deque
import numpy as np

@dataclass
class Bucket:
    ns: int = 0
    calls: int = 0
    samples: deque = field(default_factory=lambda: deque(maxlen=4096))

@dataclass
class TimingBuckets:
    buckets: Dict[str, Bucket] = field(default_factory=dict)
    started_ns: int = field(default_factory=perf_counter_ns)

    def add_ns(self, name: str, dt_ns: int, calls: int = 1) -> None:
        b = self.buckets.get(name)
        if b is None:
            b = Bucket()
            self.buckets[name] = b
        b.ns += dt_ns
        b.calls += calls
        b.samples.append(dt_ns / calls if calls else 0)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        t0 = perf_counter_ns()
        try:
            yield
        finally:
            self.add_ns(name, perf_counter_ns() - t0)

    def reset(self) -> None:
        self.buckets.clear()
        self.started_ns = perf_counter_ns()

    def as_dict(self, elapsed_seconds=None):
        elapsed = elapsed_seconds if elapsed_seconds is not None else (perf_counter_ns() - self.started_ns) / 1e9
        return {"wall_seconds": elapsed,
                "semantics": "inclusive scopes overlap; percentages use wall time, never summed; quantiles use last 4096 calls",
                "buckets": {name: {"seconds": b.ns / 1e9, "calls": b.calls,
                    "wall_percent": 100 * b.ns / 1e9 / elapsed if elapsed > 0 else 0,
                    "mean_ms": b.ns / 1e6 / b.calls if b.calls else 0,
                    "recent_p50_ms": float(np.median(b.samples)) / 1e6 if b.samples else 0,
                    "recent_p95_ms": float(np.percentile(b.samples, 95)) / 1e6 if b.samples else 0}
                    for name, b in self.buckets.items()}}

    def snapshot(self) -> Dict[str, Tuple[int, int]]:
        # name -> (ns, calls)
        return {k: (v.ns, v.calls) for k, v in self.buckets.items()}

    def report(self, title: str = "", top_k: Optional[int] = None) -> str:
        items: List[Tuple[str, int, int]] = [
            (k, v.ns, v.calls) for k, v in self.buckets.items()
        ]
        total_ns = max(1, perf_counter_ns() - self.started_ns)
        items.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            items = items[:top_k]

        lines = []
        if title:
            lines.append(title)
        lines.append("Inclusive scopes; percentages of wall time may overlap.")
        lines.append(f"{'bucket':32s} {'total_ms':>10s} {'%':>7s} {'calls':>10s} {'avg_ms':>10s}")
        lines.append("-" * 75)

        for name, ns, calls in items:
            total_ms = ns / 1e6
            pct = 100.0 * ns / total_ns
            avg_ms = (ns / calls) / 1e6 if calls else 0.0
            lines.append(f"{name:32s} {total_ms:10.2f} {pct:7.2f} {calls:10d} {avg_ms:10.4f}")

        lines.append("-" * 75)
        lines.append(f"{'WALL':32s} {(total_ns/1e6):10.2f} {100.0:7.2f}")
        return "\n".join(lines)
