"""Shared parsing and read-only checks for the legacy C++ data exporters."""

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def parse_enum(file_path, namespace_name):
    """Read enumerators inside the braces; enum syntax never consumes an ID."""
    content = re.sub(r"//[^\n]*|/\*.*?\*/", "", file_path.read_text(), flags=re.S)
    pattern = rf"namespace\s+{re.escape(namespace_name)}\s*\{{\s*enum(?:\s+\w+)?\s*\{{([^}}]*)\}}"
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Enum in namespace {namespace_name} not found in {file_path}")
    forward, reverse, current = {}, {}, 0
    for declaration in match[1].split(","):
        declaration = declaration.strip()
        if not declaration:
            continue
        parts = declaration.split("=", 1)
        name = parts[0].strip()
        if not re.fullmatch(r"\w+", name):
            raise ValueError(f"Unsupported enum entry: {declaration}")
        if len(parts) == 2:
            current = int(parts[1].strip(), 0)
        if name in forward or current in reverse:
            raise ValueError(f"Duplicate enum identity: {declaration}")
        forward[name], reverse[current] = current, name
        current += 1
    return forward, reverse


def export_arguments(argv, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--engine-root", type=Path, default=ROOT / "engine",
                        help="Legacy engine source checkout (default: repository submodule)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "python/src/sevenwonders_rl/data",
                        help="Export destination; use a temporary directory to review changes")
    parser.add_argument("--check", action="store_true", help="Compare JSON without writing; exit 1 on differences")
    args = parser.parse_args(argv)
    contract = json.loads((args.engine_root / "engine-contract.json").read_text())
    if (contract.get("ruleset") != "legacy-dmag-with-downstream-fixes"
            or contract.get("second_edition_implemented") is not False):
        parser.error("These exporters require the legacy engine contract; BGA targets need their own generator")
    return args


def check_export(output_dir, outputs):
    """Compare semantic JSON, reporting every changed output without creating files."""
    identical = True
    for name, expected in outputs.items():
        path = output_dir / name
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            actual = None
        if actual != expected:
            print(f"Export differs: {path}")
            identical = False
    return 0 if identical else 1
