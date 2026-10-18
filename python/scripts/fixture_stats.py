#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


N_PLAYERS = 3


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p, 2)
    return ent


def approx_state_signature(status: Dict[str, Any]) -> Tuple:
    """
    A cheap, stable-ish signature to estimate state uniqueness without
    needing the engine. This does NOT guarantee true uniqueness.
    """
    g = status["game"]
    sig = [int(g["era"]), int(g["turn"]), bool(g["clockwise"])]

    # Per player: wonder id/stage + compact resource/points totals + board size
    for sid in ("0", "1", "2"):
        p = status["players"][sid]
        r = p["resources"]
        pts = p["points"]

        raw_total = int(r["wood"] + r["ore"] + r["clay"] + r["stone"])
        manu_total = int(r["loom"] + r["glass"] + r["papyrus"])

        sig.extend([
            int(p.get("wonder_id", -1)),
            int(p.get("wonder_stage", -1)),
            int(r["coins"]),
            int(r["shields"]),
            raw_total,
            manu_total,
            int(pts.get("total", 0)),
            len(p.get("cards_played", [])),
            len(p.get("cards_hand", [])),
        ])
    return tuple(sig)


@dataclass
class Stats:
    n_files: int = 0
    by_era: Counter = None
    by_era_turn: Dict[int, Counter] = None
    hand_sizes: Counter = None
    playable_sizes: Counter = None
    wonder_names: Counter = None
    wonder_stages: Counter = None
    can_build_wonder: Counter = None
    can_build_hand_free: Counter = None
    points_total: Counter = None  # bucketed
    points_total_min: int = 10**9
    points_total_max: int = -10**9
    coins_min: int = 10**9
    coins_max: int = -10**9
    negative_total_points_count: int = 0
    card_in_hand: Counter = None
    card_played: Counter = None
    unique_state_sigs: set = None
    games: Counter = None  # from filename prefix g### if present

    def __post_init__(self):
        self.by_era = Counter()
        self.by_era_turn = {1: Counter(), 2: Counter(), 3: Counter()}
        self.hand_sizes = Counter()
        self.playable_sizes = Counter()
        self.wonder_names = Counter()
        self.wonder_stages = Counter()
        self.can_build_wonder = Counter()
        self.can_build_hand_free = Counter()
        self.points_total = Counter()
        self.card_in_hand = Counter()
        self.card_played = Counter()
        self.unique_state_sigs = set()
        self.games = Counter()


def bucket_points(x: int) -> str:
    # coarse buckets so the distribution is readable
    if x <= -5:
        return "<=-5"
    if -4 <= x <= -1:
        return "-4..-1"
    if 0 <= x <= 4:
        return "0..4"
    if 5 <= x <= 9:
        return "5..9"
    if 10 <= x <= 19:
        return "10..19"
    if 20 <= x <= 39:
        return "20..39"
    return "40+"


def iter_pool_files(pool_dir: Path) -> Iterable[Path]:
    yield from sorted(pool_dir.glob("*.json"))


def parse_game_idx_from_name(name: str) -> str | None:
    # expects something like g113_e2_t07.json
    if name.startswith("g") and "_e" in name:
        return name.split("_")[0]  # g113
    return None


def update_stats(st: Stats, path: Path, status: Dict[str, Any]) -> None:
    st.n_files += 1

    g = status["game"]
    era = int(g["era"])
    turn = int(g["turn"])
    st.by_era[era] += 1
    if era in st.by_era_turn:
        st.by_era_turn[era][turn] += 1

    gidx = parse_game_idx_from_name(path.name)
    if gidx:
        st.games[gidx] += 1

    st.unique_state_sigs.add(approx_state_signature(status))

    for sid in ("0", "1", "2"):
        p = status["players"][sid]

        st.hand_sizes[len(p.get("cards_hand", []))] += 1
        st.playable_sizes[len(p.get("cards_playable", []))] += 1

        st.wonder_names[str(p.get("wonder_name", "UNKNOWN"))] += 1
        st.wonder_stages[int(p.get("wonder_stage", -1))] += 1

        st.can_build_wonder[bool(p.get("can_build_wonder", False))] += 1
        st.can_build_hand_free[bool(p.get("can_build_hand_free", False))] += 1

        pts_total = int(p.get("points", {}).get("total", 0))
        st.points_total[bucket_points(pts_total)] += 1
        st.points_total_min = min(st.points_total_min, pts_total)
        st.points_total_max = max(st.points_total_max, pts_total)
        if pts_total < 0:
            st.negative_total_points_count += 1

        r = p.get("resources", {})
        coins = int(r.get("coins", 0))
        st.coins_min = min(st.coins_min, coins)
        st.coins_max = max(st.coins_max, coins)

        for c in p.get("cards_hand", []):
            st.card_in_hand[c] += 1
        for c in p.get("cards_played", []):
            st.card_played[c] += 1


def print_report(st: Stats, top_k_cards: int = 15, all_cards: bool = False) -> None:
    print("\n=== Fixture Pool Diversity Report ===\n")

    print(f"Snapshots: {st.n_files}")
    print(f"Approx unique state signatures: {len(st.unique_state_sigs)} ({len(st.unique_state_sigs)/max(1,st.n_files):.2%} unique)\n")

    print("Era counts:", dict(st.by_era))
    for era in (1, 2, 3):
        if st.by_era_turn[era]:
            turns = st.by_era_turn[era]
            print(f"  Era {era}: turns seen = {len(turns)}; min={min(turns)} max={max(turns)}")
    print()

    print("Hand size distribution:", dict(sorted(st.hand_sizes.items())))
    print("Playable count distribution:", dict(sorted(st.playable_sizes.items())))
    print()

    print("Wonder stage distribution:", dict(sorted(st.wonder_stages.items())))
    print(f"Wonder name entropy: {shannon_entropy(st.wonder_names):.3f} bits over {len(st.wonder_names)} names")
    print("Top wonders:", st.wonder_names.most_common(10))
    print()

    print("can_build_wonder:", dict(st.can_build_wonder))
    print("can_build_hand_free:", dict(st.can_build_hand_free))
    print()

    print(f"points.total range: [{st.points_total_min}, {st.points_total_max}]")
    print("points.total buckets:", dict(st.points_total))
    print(f"Snapshots with negative points.total: {st.negative_total_points_count}")
    print(f"coins range: [{st.coins_min}, {st.coins_max}]")
    print()

    print(f"All {len(st.card_in_hand)} cards in hand (sorted by count):")
    for card, count in st.card_in_hand.most_common():
        print(f"  {card}: {count}")
    print()

    print(f"All {len(st.card_played)} cards played (sorted by count):")
    for card, count in st.card_played.most_common():
        print(f"  {card}: {count}")
    print()

    if st.games:
        print(f"Games detected from filenames: {len(st.games)}")
        print("Snapshots per game (min/median/max-ish): "
              f"min={min(st.games.values())} max={max(st.games.values())}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", type=Path, default=Path("python/tests/fixtures/pool"))
    ap.add_argument("--top-k-cards", type=int, default=15)
    ap.add_argument("--all-cards", action="store_true", help="Display all cards instead of just top-k")
    args = ap.parse_args()

    pool_dir = args.pool_dir
    if not pool_dir.exists():
        raise FileNotFoundError(f"Pool dir not found: {pool_dir}")

    st = Stats()
    for p in iter_pool_files(pool_dir):
        try:
            status = load_json(p)
        except Exception as e:
            print(f"[warn] failed to load {p.name}: {e}")
            continue
        update_stats(st, p, status)

    print_report(st, all_cards=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
