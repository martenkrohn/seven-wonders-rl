"""Isolated exchange directories and persistent, strict JSON output."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

RAM_IO_ROOT = Path("/dev/shm/sevenwonders")


def exchange_io_root(requested, fallback, prefix="run", *, prefer_ram=True) -> Path:
    """Auto-select RAM for transient exchange only; explicit paths never fall back."""
    if requested is not None:
        return run_io_root(requested, prefix)
    if prefer_ram and RAM_IO_ROOT.parent.is_dir():
        try:
            return run_io_root(RAM_IO_ROOT, prefix)
        except OSError:
            # Platforms without writable shared memory retain the disk transport.
            pass
    return run_io_root(fallback, prefix)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def run_io_root(root, prefix="run") -> Path:
    base = Path(root).expanduser().resolve()
    path = base / f"{prefix}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def seat_order(agent_seat: int):
    return agent_seat, (agent_seat - 1) % 3, (agent_seat + 1) % 3
