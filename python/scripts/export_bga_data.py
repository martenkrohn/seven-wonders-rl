#!/usr/bin/env python3
"""Copy versioned neutral engine exports; --check never writes or parses C++."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine-root", type=Path, default=ROOT / "engine")
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "python/src/sevenwonders_rl/data"
    )
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    source_hash = None
    for kind in ("cards", "wonders"):
        data = (a.engine_root / f"data/generated/{kind}-v1.json").read_bytes()
        document = json.loads(data)
        if document["format_version"] != 1:
            raise SystemExit("Unsupported BGA export format")
        digest = document["source"]["sha256"]
        if source_hash is not None and digest != source_hash:
            raise SystemExit("Card/wonder source hash differs")
        source_hash = digest
        target = a.output_dir / f"bga_{kind}-v1.json"
        if a.check:
            if not target.is_file() or target.read_bytes() != data:
                raise SystemExit(f"Stale BGA resource: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print("BGA exports checked" if a.check else "BGA exports copied")


if __name__ == "__main__":
    main()
