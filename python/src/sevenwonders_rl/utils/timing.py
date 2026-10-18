"""Lightweight timing buckets for performance profiling."""
from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import contextmanager
from time import perf_counter_ns
from typing import Dict, Iterator, Optional, List, Tuple

@dataclass
class Bucket:
    ns: int = 0
    calls: int = 0

@dataclass
class TimingBuckets:
    buckets: Dict[str, Bucket] = field(default_factory=dict)

    def add_ns(self, name: str, dt_ns: int, calls: int = 1) -> None:
        b = self.buckets.get(name)
        if b is None:
            b = Bucket()
            self.buckets[name] = b
        b.ns += dt_ns
        b.calls += calls

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        t0 = perf_counter_ns()
        try:
            yield
        finally:
            self.add_ns(name, perf_counter_ns() - t0)

    def reset(self) -> None:
        self.buckets.clear()

    def snapshot(self) -> Dict[str, Tuple[int, int]]:
        # name -> (ns, calls)
        return {k: (v.ns, v.calls) for k, v in self.buckets.items()}

    def report(self, title: str = "", top_k: Optional[int] = None) -> str:
        items: List[Tuple[str, int, int]] = [
            (k, v.ns, v.calls) for k, v in self.buckets.items()
        ]
        total_ns = sum(ns for _, ns, _ in items) or 1
        items.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            items = items[:top_k]

        lines = []
        if title:
            lines.append(title)
        lines.append(f"{'bucket':32s} {'total_ms':>10s} {'%':>7s} {'calls':>10s} {'avg_ms':>10s}")
        lines.append("-" * 75)

        for name, ns, calls in items:
            total_ms = ns / 1e6
            pct = 100.0 * ns / total_ns
            avg_ms = (ns / calls) / 1e6 if calls else 0.0
            lines.append(f"{name:32s} {total_ms:10.2f} {pct:7.2f} {calls:10d} {avg_ms:10.4f}")

        lines.append("-" * 75)
        lines.append(f"{'TOTAL':32s} {(total_ns/1e6):10.2f} {100.0:7.2f}")
        return "\n".join(lines)
