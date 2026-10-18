#!/usr/bin/env python3
"""
Capture frozen io/game_status.json snapshots into python/tests/fixtures/.

This script:
  - launches the C++ engine in an isolated working directory
  - auto-plays a goal-seeking policy for all 3 seats
  - watches game_status.json every turn and saves snapshots when conditions match
  - repeats across games until all target fixture files are captured

Why this exists:
  - The engine is currently nondeterministic (time-seeded shuffles), so we can't
    "replay" golden states reliably in tests.
  - Frozen fixtures let us write deterministic engine-free component tests.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


N_PLAYERS = 3


CHOICE_PRODUCTION_CARDS = {
    # Brown choice production
    "Tree Farm",
    "Excavation",
    "Clay Pit",
    "Timber Yard",
    "Forest Cave",
    "Mine",
    # Yellow choice production (often non-tradable per estimator assumptions)
    "Forum",
    "Caravansery",
}


def repo_root_from_this_file() -> Path:
    # .../python/scripts/capture_fixtures.py -> parents[0]=scripts, [1]=python, [2]=repo_root
    return Path(__file__).resolve().parents[2]


def default_engine_path(repo_root: Path) -> Path:
    return repo_root / "build" / "7Wonders"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def write_pretty_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_action_file(path: Path, subcommand: str, argument: str, extra: str = "") -> None:
    payload = {"command": {"subcommand": subcommand, "argument": argument, "extra": extra}}
    path.write_text(json.dumps(payload) + "\n")


def write_ready(path: Path, n_players: int = N_PLAYERS) -> None:
    path.write_text(("ready\n" * n_players))


def wait_for_file(path: Path, timeout_s: float = 10.0, poll_s: float = 0.05) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for {path}")


def wait_for_turn_advance(
    status_path: Path,
    prev_turn: int,
    timeout_s: float = 10.0,
    poll_s: float = 0.02,
) -> Dict[str, Any]:
    t0 = time.time()
    last_seen: Optional[Dict[str, Any]] = None
    while time.time() - t0 < timeout_s:
        if status_path.exists() and status_path.stat().st_size > 0:
            try:
                st = read_json(status_path)
                last_seen = st
                turn = int(st["game"]["turn"])
                if turn != prev_turn:
                    return st
            except Exception:
                # file may be mid-write; try again
                pass
        time.sleep(poll_s)
    raise TimeoutError(
        f"Timed out waiting for turn advance from {prev_turn}. "
        f"Last status seen: {None if last_seen is None else last_seen.get('game', {})}"
    )


def dump_pool_status(pool_dir: Path, game_idx: int, turn: int, status: Dict[str, Any]) -> None:
    era = int(status["game"]["era"])
    path = pool_dir / f"g{game_idx:03d}_e{era}_t{turn:02d}.json"
    if not path.exists():
        write_pretty_json(path, status)


@dataclass(frozen=True)
class Targets:
    status_age1_turn0: str = "status_age1_turn0.json"
    status_age2_turn0: str = "status_age2_turn0.json"
    status_age3_turn0: str = "status_age3_turn0.json"
    status_can_build_wonder: str = "status_can_build_wonder.json"
    status_can_build_hand_free: str = "status_can_build_hand_free.json"
    status_choice_production: str = "status_choice_production.json"
    status_near_endgame: str = "status_near_endgame.json"


def has_choice_production_played(status: Dict[str, Any], seat: int) -> bool:
    p = status["players"][str(seat)]
    played = p.get("cards_played", [])
    return any(name in CHOICE_PRODUCTION_CARDS for name in played)


def is_age_start(status: Dict[str, Any], seat: int, era: int) -> bool:
    if int(status["game"]["era"]) != era:
        return False
    hand = status["players"][str(seat)]["cards_hand"]
    return len(hand) == 7


def is_can_build_wonder(status: Dict[str, Any], seat: int) -> bool:
    return bool(status["players"][str(seat)].get("can_build_wonder", False))


def is_can_build_hand_free(status: Dict[str, Any], seat: int) -> bool:
    return bool(status["players"][str(seat)].get("can_build_hand_free", False))


def is_near_endgame(status: Dict[str, Any], seat: int) -> bool:
    # Robust “late” indicator that doesn't rely on exact turn numbering:
    # Age III and the hand is down to 2 or fewer cards.
    if int(status["game"]["era"]) != 3:
        return False
    hand = status["players"][str(seat)]["cards_hand"]
    return 0 < len(hand) <= 2


def is_game_over(status: Dict[str, Any]) -> bool:
    # Conservative: stop if engine turn hits terminal, or hands are empty.
    g = status.get("game", {})
    if int(g.get("era", 0)) == 3 and int(g.get("turn", -1)) >= 20:
        return True
    for sid in range(N_PLAYERS):
        hand = status["players"][str(sid)]["cards_hand"]
        if len(hand) == 0:
            return True
    return False


def capture_if_missing(
    *,
    fixtures_dir: Path,
    filename: str,
    status: Dict[str, Any],
    overwrite: bool,
    captured: Dict[str, bool],
) -> None:
    out_path = fixtures_dir / filename
    if captured.get(filename, False):
        return
    if out_path.exists() and not overwrite:
        captured[filename] = True
        return
    write_pretty_json(out_path, status)
    captured[filename] = True
    print(f"[captured] {filename}")


def write_index(fixtures_dir: Path, filename: str, seat: int) -> None:
    idx_path = fixtures_dir / "index.json"
    idx: Dict[str, Dict[str, int]] = {}
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
    idx[filename] = {"seat": seat}
    idx_path.write_text(json.dumps(idx, indent=2, sort_keys=True) + "\n")


def all_captured(captured: Dict[str, bool], targets: Targets) -> bool:
    need = [
        targets.status_age1_turn0,
        targets.status_age2_turn0,
        targets.status_age3_turn0,
        targets.status_can_build_wonder,
        targets.status_can_build_hand_free,
        targets.status_choice_production,
        targets.status_near_endgame,
    ]
    return all(captured.get(n, False) for n in need)


def pick_playable_card(status: Dict[str, Any], seat: int, prefer_names: Optional[Set[str]] = None) -> Optional[str]:
    p = status["players"][str(seat)]
    hand = p["cards_hand"]
    playable = set(p["cards_playable"])
    prefer_names = prefer_names or set()

    for c in hand:
        if c in playable and c in prefer_names:
            return c
    for c in hand:
        if c in playable:
            return c
    return None


def pick_sacrifice_card(status: Dict[str, Any], seat: int) -> str:
    """Pick a card to sacrifice for wonder building, preferring non-playable cards."""
    p = status["players"][str(seat)]
    hand = p["cards_hand"]
    playable = set(p["cards_playable"])
    for c in hand:
        if c not in playable:
            return c
    return hand[0]


def choose_action_for_seat(
    status: Dict[str, Any],
    seat: int,
    missing: Set[str],
) -> Tuple[str, str]:
    p = status["players"][str(seat)]
    hand = p["cards_hand"]
    playable = set(p["cards_playable"])
    can_wonder = bool(p.get("can_build_wonder", False))
    wonder_name = str(p.get("wonder_name", "")).lower()

    if not hand:
        raise RuntimeError("No cards in hand")

    # FORCE: whenever a wonder stage is legal, always build it
    if can_wonder:
        return ("build_wonder", pick_sacrifice_card(status, seat))

    # 1) Choice-production hunting
    if "status_choice_production.json" in missing:
        for c in hand:
            if c in CHOICE_PRODUCTION_CARDS and c in playable:
                return ("build_structure", c)

    # 2) Capture can_build_wonder sooner: build stages when possible
    if "status_can_build_wonder.json" in missing and can_wonder:
        return ("build_wonder", hand[0])

    # 3) Olympia hunting for can_build_hand_free
    if "status_can_build_hand_free.json" in missing:
        if "olympia" in wonder_name and can_wonder:
            return ("build_wonder", hand[0])

    # 4) Build any playable structure to grow economy
    c = pick_playable_card(status, seat)
    if c is not None:
        return ("build_structure", c)

    # 5) Fallback: discard
    return ("discard", hand[0])


def main() -> int:
    repo_root = repo_root_from_this_file()
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, default=default_engine_path(repo_root))
    parser.add_argument("--max-games", type=int, default=30)
    parser.add_argument("--max-steps-per-game", type=int, default=50)
    parser.add_argument("--capture-pool", action="store_true")
    parser.add_argument("--pool-dir", type=Path, default=None)
    parser.add_argument("--pool-every-turn", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=repo_root / ".tmp_fixture_capture",
        help="Scratch directory used as engine CWD (engine writes ./io inside it)",
    )
    args = parser.parse_args()

    engine = args.engine.resolve()
    if not engine.exists():
        raise FileNotFoundError(f"Engine executable not found: {engine}")

    fixtures_dir = repo_root / "python" / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    pool_dir = args.pool_dir or (fixtures_dir / "pool")
    if args.capture_pool:
        pool_dir.mkdir(parents=True, exist_ok=True)

    targets = Targets()
    captured: Dict[str, bool] = {}

    # Respect existing files unless overwrite is requested.
    for name in [
        targets.status_age1_turn0,
        targets.status_age2_turn0,
        targets.status_age3_turn0,
        targets.status_can_build_wonder,
        targets.status_can_build_hand_free,
        targets.status_choice_production,
        targets.status_near_endgame,
    ]:
        if (fixtures_dir / name).exists() and not args.overwrite:
            captured[name] = True

    args.workdir.mkdir(parents=True, exist_ok=True)

    for game_idx in range(args.max_games):
        if all_captured(captured, targets) and not args.capture_pool:
            break

        run_dir = args.workdir / f"game_{game_idx:03d}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        (run_dir / "io").mkdir(parents=True, exist_ok=True)

        status_path = run_dir / "io" / "game_status.json"
        ready_path = run_dir / "io" / "ready.txt"
        player_paths = [run_dir / "io" / f"player_{i+1}.json" for i in range(N_PLAYERS)]

        print(f"[run] starting engine (game {game_idx}) in {run_dir}")
        proc = subprocess.Popen(
            [str(engine)],
            cwd=str(run_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            wait_for_file(status_path, timeout_s=10.0)
            status = read_json(status_path)
            prev_turn = int(status["game"]["turn"])

            for step_idx in range(args.max_steps_per_game):
                if args.capture_pool:
                    dump_pool_status(pool_dir, game_idx, int(status["game"]["turn"]), status)

                # --- Capture targets if conditions match ---
                for seat in range(N_PLAYERS):
                    if is_age_start(status, seat, era=1):
                        capture_if_missing(
                            fixtures_dir=fixtures_dir,
                            filename=targets.status_age1_turn0,
                            status=status,
                            overwrite=args.overwrite,
                            captured=captured,
                        )
                    if is_age_start(status, seat, era=2):
                        capture_if_missing(
                            fixtures_dir=fixtures_dir,
                            filename=targets.status_age2_turn0,
                            status=status,
                            overwrite=args.overwrite,
                            captured=captured,
                        )
                    if is_age_start(status, seat, era=3):
                        capture_if_missing(
                            fixtures_dir=fixtures_dir,
                            filename=targets.status_age3_turn0,
                            status=status,
                            overwrite=args.overwrite,
                            captured=captured,
                        )

                if not captured.get(targets.status_can_build_wonder, False):
                    for seat in range(N_PLAYERS):
                        if is_can_build_wonder(status, seat):
                            write_pretty_json(fixtures_dir / targets.status_can_build_wonder, status)
                            write_index(fixtures_dir, targets.status_can_build_wonder, seat)
                            captured[targets.status_can_build_wonder] = True
                            print(f"[captured] {targets.status_can_build_wonder} (seat {seat})")
                            break

                if not captured.get(targets.status_can_build_hand_free, False):
                    for seat in range(N_PLAYERS):
                        if is_can_build_hand_free(status, seat):
                            write_pretty_json(fixtures_dir / targets.status_can_build_hand_free, status)
                            write_index(fixtures_dir, targets.status_can_build_hand_free, seat)
                            captured[targets.status_can_build_hand_free] = True
                            print(f"[captured] {targets.status_can_build_hand_free} (seat {seat})")
                            break

                if not captured.get(targets.status_choice_production, False):
                    for seat in range(N_PLAYERS):
                        if has_choice_production_played(status, seat):
                            write_pretty_json(fixtures_dir / targets.status_choice_production, status)
                            write_index(fixtures_dir, targets.status_choice_production, seat)
                            captured[targets.status_choice_production] = True
                            print(f"[captured] {targets.status_choice_production} (seat {seat})")
                            break

                if not captured.get(targets.status_near_endgame, False):
                    for seat in range(N_PLAYERS):
                        if is_near_endgame(status, seat):
                            capture_if_missing(
                                fixtures_dir=fixtures_dir,
                                filename=targets.status_near_endgame,
                                status=status,
                                overwrite=args.overwrite,
                                captured=captured,
                            )
                            break

                if all_captured(captured, targets) and not args.capture_pool:
                    break

                if is_game_over(status):
                    break

                # --- Advance turn by submitting actions for all players ---
                missing = set()
                for name in [
                    targets.status_age1_turn0,
                    targets.status_age2_turn0,
                    targets.status_age3_turn0,
                    targets.status_can_build_wonder,
                    targets.status_can_build_hand_free,
                    targets.status_choice_production,
                    targets.status_near_endgame,
                ]:
                    if not (fixtures_dir / name).exists():
                        missing.add(name)

                for sid in range(N_PLAYERS):
                    subcmd, arg = choose_action_for_seat(status, sid, missing)
                    write_action_file(player_paths[sid], subcmd, arg, extra="")

                write_ready(ready_path, N_PLAYERS)

                status = wait_for_turn_advance(status_path, prev_turn, timeout_s=10.0)
                prev_turn = int(status["game"]["turn"])

                if args.capture_pool:
                    dump_pool_status(pool_dir, game_idx, int(status["game"]["turn"]), status)

        except TimeoutError as e:
            if "turn advance from 20" in str(e):
                pass
            else:
                print(f"[warn] run ended early: {e}")
        except Exception as e:
            print(f"[warn] run ended early: {e}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)

    missing = []
    for name in [
        targets.status_age1_turn0,
        targets.status_age2_turn0,
        targets.status_age3_turn0,
        targets.status_can_build_wonder,
        targets.status_can_build_hand_free,
        targets.status_choice_production,
        targets.status_near_endgame,
    ]:
        if not (fixtures_dir / name).exists():
            missing.append(name)

    if missing:
        print("[done] some fixtures are still missing:")
        for m in missing:
            print(f"  - {m}")
        print("Tip: increase --max-games, or run again with --overwrite.")
        return 1

    print("[done] all fixtures captured in python/tests/fixtures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
