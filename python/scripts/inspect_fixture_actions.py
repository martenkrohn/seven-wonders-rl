"""
From python/, with the virtual environment active:
python scripts/inspect_fixture_actions.py --fixture tests/fixtures/pool/g000_e1_t00.json --seat 1
"""


#!/usr/bin/env python3
"""Inspect action masks and features from a fixture for debugging."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

# Adjust these imports to match your project layout.
from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.features.action_feature_schema import IDX  # dict: feature_name -> column index
from sevenwonders_rl.data.wonders import wonder_by_id

ACTION_TYPES = ("build_structure", "build_wonder", "build_hand_free", "discard")
N_SLOTS = 7

# Cost/trade features to always display (even if zero)
COST_KEYS = [
    "card_cost_raw_sum", "card_cost_total_sum",
    "missing_raw_sum",
    "coins_after_card_cost", "coins_after_est_total_cost",
    "est_trade_cost_left", "est_trade_cost_right",
    "est_trade_cost_total",
]

# Card colors/eras
CARD_COLORS = {
    'brown': 'Brown Raw',
    'gray': 'Gray Manuf',
    'blue': 'Blue Civilian',
    'yellow': 'Yellow Commercial',
    'red': 'Red Military',
    'green': 'Green Scientific',
    'purple': 'Purple Guild',
}


def decode_action(action_idx: int) -> tuple[str, int]:
    """Decode flat action index to (action_type, slot)."""
    a_type = action_idx // N_SLOTS
    slot = action_idx % N_SLOTS
    return ACTION_TYPES[a_type], slot


def load_status(path: Path) -> dict:
    """Load game_status.json fixture."""
    return json.loads(path.read_text())


def format_resources(res_dict: dict) -> str:
    """Format resources dict nicely, showing non-zero values."""
    non_zero = {k: v for k, v in res_dict.items() if v != 0}
    if not non_zero:
        return "(all zero)"
    return ", ".join(f"{k}:{v}" for k, v in sorted(non_zero.items()))


def print_game_overview(status: dict, seat: int) -> None:
    """Print high-level game state information."""
    game = status["game"]
    print(f"\n{'='*70}")
    print(f"GAME STATE: Era {game['era']}, Turn {game['turn']}")
    print(f"{'='*70}")
    
    for s in range(3):
        p = status["players"][str(s)]
        wonder_name = p.get("wonder_name", "?")
        points = p.get("points", {})
        total_vp = points.get("total", 0)
        marker = " <-- ACTIVE SEAT" if s == seat else ""
        print(f"  Seat {s}: {wonder_name:20s} VP={total_vp:2d}{marker}")


def print_player_overview(p: dict, seat_key: str) -> None:
    """Print detailed player information."""
    print(f"\n{'='*70}")
    print(f"PLAYER {seat_key} OVERVIEW")
    print(f"{'='*70}")
    
    # Wonder info
    wonder_id = p.get("wonder_id")
    wonder_name = p.get("wonder_name", "?")
    wonder_stage = p.get("wonder_stage", 0)
    wonder_obj = wonder_by_id(wonder_id) if wonder_id else None
    starting_res = wonder_obj.get("starting_resource") if wonder_obj else "?"
    
    print(f"\nWonder: {wonder_name} (ID={wonder_id})")
    print(f"  Starting Resource: {starting_res}")
    print(f"  Stages Built: {wonder_stage}")
    
    # Points breakdown
    points = p.get("points", {})
    print(f"\nPoints:")
    for point_type in ["civilian", "commercial", "military", "scientific", "guild", "wonder"]:
        val = points.get(point_type, 0)
        print(f"  {point_type:12s}: {val:2d}")
    print(f"  {'TOTAL':12s}: {points.get('total', 0):2d}")
    
    # Resources
    resources = p.get("resources", {})
    print(f"\nCurrent Resources:")
    raw_res = ["clay", "ore", "stone", "wood"]
    manuf_res = ["loom", "glass", "papyrus"]
    coins = resources.get("coins", 0)
    shields = resources.get("shields", 0)
    
    print(f"  Raw:  {', '.join(f'{r}:{resources.get(r, 0)}' for r in raw_res)}")
    print(f"  Manuf: {', '.join(f'{r}:{resources.get(r, 0)}' for r in manuf_res)}")
    print(f"  Special: coins={coins}, shields={shields}")
    
    # Card counts by color
    cards_played = p.get("cards_played", [])
    color_count = defaultdict(int)
    for card_name in cards_played:
        card = None  # Would need name->id lookup
        # For now, just count
        color_count[card_name] = color_count[card_name] + 1
    
    print(f"\nCards Built: {len(cards_played)}")
    if cards_played:
        print(f"  {', '.join(cards_played)}")
    
    # Flags
    can_wonder = p.get("can_build_wonder", False)
    can_free = p.get("can_build_hand_free", False)
    print(f"\nAction Flags:")
    print(f"  can_build_wonder: {can_wonder}")
    print(f"  can_build_hand_free: {can_free}")


def print_action_analysis(status: dict, seat: int, mask: np.ndarray, feats: np.ndarray, args) -> None:
    """Print detailed action analysis."""
    seat_key = str(seat)
    p = status["players"][seat_key]
    
    hand = p.get("cards_hand", [])
    playable = set(p.get("cards_playable", []))
    can_wonder = bool(p.get("can_build_wonder", False))
    can_free = bool(p.get("can_build_hand_free", False))
    
    inv_idx = {v: k for k, v in IDX.items()}
    
    def card_at(slot: int) -> str:
        return hand[slot] if 0 <= slot < len(hand) else "<PAD>"

    def show_cost_keys(row):
        """Display all cost/trade features (even if zero) for debugging."""
        for k in COST_KEYS:
            if k in IDX:
                print(f"      {k:30s} {row[IDX[k]]:+.3f}")

    print(f"\n{'='*70}")
    print(f"HAND & ACTION ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nHand (7 slots):")
    for i in range(N_SLOTS):
        card = card_at(i)
        is_playable = card in playable
        marker = " [playable]" if is_playable and card != "<PAD>" else ""
        print(f"  {i}: {card:20s}{marker}")
    
    print(f"\nEngine Legal Actions (cards_playable):")
    for c in sorted(playable):
        print(f"  - {c}")
    
    print(f"\n{'='*70}")
    print(f"ACTION DETAILS (28 actions total)")
    print(f"{'='*70}")
    
    legal_count = 0
    for a in range(4 * N_SLOTS):
        legal = int(mask[a])
        if legal == 0 and not args.show_illegal:
            continue
        
        legal_count += legal

        action_type, slot = decode_action(a)
        card = card_at(slot)

        # Quick expectation checks (based on your stated contract)
        expected = None
        if card == "<PAD>":
            expected = 0
        elif action_type == "discard":
            expected = 1
        elif action_type == "build_structure":
            expected = 1 if card in playable else 0
        elif action_type == "build_wonder":
            expected = 1 if can_wonder else 0
        elif action_type == "build_hand_free":
            expected = 1 if can_free else 0

        warn = ""
        if expected is not None and expected != legal:
            warn = f"  ⚠️  [MASK MISMATCH: expected={expected}, got={legal}]"

        row = feats[a]
        nz = [(inv_idx[i], float(row[i])) for i in range(row.shape[0]) if abs(row[i]) > args.eps]
        nz.sort(key=lambda kv: abs(kv[1]), reverse=True)
        nz = nz[: args.topk]

        status_str = "✓ LEGAL" if legal else "✗ illegal"
        print(f"\n  a={a:02d} {status_str:10s} {action_type:15s} slot={slot} card={card}{warn}")
        
        # Always show cost/trade features
        print("    Cost/Trade features:")
        show_cost_keys(row)
        
        # Show top-k other non-zero features
        if not nz:
            print("    Other features: (none)")
        else:
            print("    Top non-zero features:")
            for k, v in nz:
                if k not in COST_KEYS:  # Don't duplicate cost features
                    print(f"      {k:30s} {v:+.3f}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {legal_count} legal actions out of 28")
    print(f"{'='*70}")

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect action masks and features from a game_status.json fixture"
    )
    ap.add_argument("--fixture", type=Path, required=True, help="Path to a fixture json (game_status-like).")
    ap.add_argument("--seat", type=int, default=0, choices=[0, 1, 2], help="Player seat to inspect (0, 1, or 2)")
    ap.add_argument("--topk", type=int, default=20, help="How many feature entries to show per action.")
    ap.add_argument("--eps", type=float, default=1e-6, help="Non-zero threshold for printing features.")
    ap.add_argument("--show-illegal", action="store_true", help="Also print illegal actions (mask==0).")
    args = ap.parse_args()

    status = load_status(args.fixture)
    seat_key = str(args.seat)

    obs = build_obs(status, args.seat)
    mask = np.asarray(obs["action_mask"], dtype=np.int32)
    feats = np.asarray(obs["action_feats"], dtype=np.float32)

    # Print comprehensive fixture information
    print_game_overview(status, args.seat)
    print_player_overview(status["players"][seat_key], seat_key)
    print_action_analysis(status, args.seat, mask, feats, args)


if __name__ == "__main__":
    main()
