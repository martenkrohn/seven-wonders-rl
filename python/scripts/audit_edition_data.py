#!/usr/bin/env python3
"""Read-only comparison of current game data with the pinned upstream snapshot.

Prints evidence as JSON. Does not regenerate or overwrite package data.
External edition rules are assessed separately in docs/reference/EDITION_AUDIT.md.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

import extract_card_db_from_cpp as cards
import extract_wonder_db_from_cpp as wonders

ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "engine"
BASELINE = "a0bc333a7a7f21f83156de8a2d522e7baa913f0a"
DATA = ROOT / "python/src/sevenwonders_rl/data"


def upstream(path):
    return subprocess.check_output(
        ["git", "-C", str(ENGINE_ROOT), "show", f"{BASELINE}:{path}"], text=True)


def regeneration(module):
    outputs = {}
    original = module.write_json_atomic
    module.write_json_atomic = lambda data, path: outputs.update(
        {path.name: json.loads(json.dumps(data))})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            module.main([])
    finally:
        module.write_json_atomic = original
    result = {}
    for name, generated in outputs.items():
        current = json.loads((DATA / name).read_text())
        result[name] = {
            "identical": current == generated,
            "missing_ids": sorted(set(current) - set(generated)),
            "new_ids": sorted(set(generated) - set(current)),
            "changed_fields": {
                key: [field for field in sorted(set(current[key]) | set(generated[key]))
                      if current[key].get(field) != generated[key].get(field)]
                for key in sorted(set(current) & set(generated))
                if isinstance(current[key], dict) and current[key] != generated[key]
            },
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Exit 1 if package data differs from the legacy export')
    args = parser.parse_args(argv)
    current_main = (ENGINE_ROOT / "main.cpp").read_text()
    pattern = r"(?:cards|guild_cards)\.push_back\(Card\(.*?\)\);"
    old = re.findall(pattern, upstream("main.cpp"))
    new = re.findall(pattern, current_main)
    card_ids, _ = cards.parse_enum(ENGINE_ROOT / "include/card.h", "CARD_ID")
    types, _ = cards.parse_enum(ENGINE_ROOT / "include/card.h", "CARD_TYPE")
    parsed = cards.parse_cards_from_main(ENGINE_ROOT / "main.cpp", card_ids, types)
    card_db = json.loads((DATA / "card_db.json").read_text())
    resources = ["wood", "ore", "clay", "stone", "loom", "glass", "papyrus", "coins"]
    mismatches = []
    for card in parsed:
        data = card_db[str(card["id"])]
        if data["cost"] != dict(zip(resources, card["cost_vec"])):
            mismatches.append({"id": card["id"], "field": "cost"})
        if card["free_with_raw"] != -1 and data["chain_from"] != card["free_with_raw"]:
            mismatches.append({"id": card["id"], "field": "chain_from"})
    files = ["main.cpp", "src/card.cpp", "src/player.cpp", "src/wonder.cpp",
             "include/card.h", "include/wonder.h", "include/resources.h",
             "python/scripts/legacy_export_common.py",
             "python/scripts/extract_card_db_from_cpp.py",
             "python/scripts/extract_wonder_db_from_cpp.py",
             "python/src/sevenwonders_rl/data/card_db.json",
             "python/src/sevenwonders_rl/data/wonder_db.json",
             "python/src/sevenwonders_rl/data/mechanics.json"]
    result = {
        "baseline": BASELINE,
        "working_tree_sha256": {
            ("engine/" + path if not path.startswith("python/") else path):
                hashlib.sha256(((ROOT if path.startswith("python/") else ENGINE_ROOT) / path).read_bytes()).hexdigest()
                for path in files},
        "card_declarations": {"upstream": len(old), "current": len(new), "identical": old == new},
        "card_cpp_unchanged": upstream("src/card.cpp") == (ENGINE_ROOT / "src/card.cpp").read_text(),
        "card_id_header_unchanged": upstream("include/card.h") == (ENGINE_ROOT / "include/card.h").read_text(),
        "unique_card_ids": len(card_db),
        "card_era_rows": len(parsed),
        "engine_contract": json.loads((ENGINE_ROOT / 'engine-contract.json').read_text()),
        "card_cost_or_single_chain_mismatches": mismatches,
        "regeneration": {**regeneration(cards), **regeneration(wonders)},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.check and (mismatches or any(not row['identical'] for row in result['regeneration'].values())):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
