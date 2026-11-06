#!/usr/bin/env python3
"""Verify that the engine checkout matches the RL repository's explicit lock."""

import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "engine.lock.json").read_text())
    engine = root / "engine"
    if not (engine / "engine-contract.json").is_file():
        raise SystemExit("Engine missing: run git submodule update --init --recursive")
    actual = subprocess.check_output(
        ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != lock["commit"]:
        raise SystemExit(
            f"Engine commit differs from lock: {actual} != {lock['commit']}"
        )
    entry = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--stage", "engine"], text=True
    ).split()
    if len(entry) < 2 or entry[0] != "160000" or entry[1] != actual:
        raise SystemExit(
            "The staged engine submodule commit does not match engine.lock.json"
        )
    contract = json.loads((engine / "engine-contract.json").read_text())
    for key in (
        "release",
        "ruleset",
        "protocol_version",
        "players",
        "second_edition_implemented",
    ):
        if contract[key] != lock[key]:
            raise SystemExit(f"Engine contract differs from lock: {key}")
    dirty = subprocess.check_output(
        ["git", "-C", str(engine), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit(
            "Engine checkout has uncommitted changes; commit and update the pin before checking"
        )
    print(
        f"Engine pinned to {actual}: {lock['release']}, {lock['ruleset']}, protocol {lock['protocol_version']}"
    )


if __name__ == "__main__":
    main()
