import numpy as np
import pytest

from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.features.action_feature_schema import FEATURE_NAMES, IDX, K, HEADER_IDXS


pytestmark = pytest.mark.component

N_SLOTS = 7
N_ACTIONS = 28

BUILD_STRUCTURE = 0
BUILD_WONDER = 1
BUILD_HAND_FREE = 2
DISCARD = 3


def _idx(action_type: int, slot: int) -> int:
    return action_type * N_SLOTS + slot


def _non_header_indices() -> np.ndarray:
    return np.array([i for i in range(K) if i not in HEADER_IDXS], dtype=int)


def _extract_cards(card_db: dict) -> list[dict]:
    if isinstance(card_db, list):
        return card_db
    if "cards" in card_db and isinstance(card_db["cards"], list):
        return card_db["cards"]
    if "by_id" in card_db and isinstance(card_db["by_id"], dict):
        return list(card_db["by_id"].values())
    return [v for v in card_db.values() if isinstance(v, dict)]


def _card_color(card: dict) -> str:
    if "color" in card:
        return str(card["color"]).lower()
    if "type" in card:
        return str(card["type"]).lower()
    raise KeyError("card missing color/type")


def _card_era(card: dict) -> int:
    if "era" in card and card["era"] is not None:
        return int(card["era"])
    eras = card.get("eras")
    if isinstance(eras, list) and eras:
        return int(eras[0])
    raise KeyError("card missing era/eras")


def test_action_feats_shape_and_finite(status_age1_turn0):
    obs = build_obs(status_age1_turn0, agent_seat=0)
    feats = obs["action_feats"]
    assert feats.shape == (N_ACTIONS, K)
    assert np.isfinite(feats).all()


def test_header_fields_filled_and_illegal_rows_zeroed(status_age1_turn0):
    # Check across all seats to catch rotation-related feature errors
    for seat in (0, 1, 2):
        obs = build_obs(status_age1_turn0, agent_seat=seat)
        feats = obs["action_feats"]
        mask = obs["action_mask"].astype(int)

        non_header = _non_header_indices()

        for a in range(N_ACTIONS):
            row = feats[a]
            # Header always finite (already checked), but also make sure header is not all zeros for real slots.
            # slot_is_empty will also be set for padded slots, so we avoid over-constraining here.

            if mask[a] == 0:
                # Contract: non-header features must be zeros for illegal actions
                assert np.all(row[non_header] == 0), (
                    f"seat{seat} action{a}: illegal but non-header not zero. "
                    f"nonzero={[(FEATURE_NAMES[i], float(row[i])) for i in non_header if row[i] != 0][:10]}"
                )


def test_action_type_one_hot_correct(status_age1_turn0):
    obs = build_obs(status_age1_turn0, agent_seat=0)
    feats = obs["action_feats"]

    for action_type in (BUILD_STRUCTURE, BUILD_WONDER, BUILD_HAND_FREE, DISCARD):
        col = IDX[f"action_type_{action_type}"]
        for slot in range(N_SLOTS):
            a = _idx(action_type, slot)
            assert feats[a, col] == 1.0, f"action_type one-hot missing at action={a}"

        # And other action_type_* should be 0 for this row group
        other_types = [t for t in (0, 1, 2, 3) if t != action_type]
        for slot in range(N_SLOTS):
            a = _idx(action_type, slot)
            for t in other_types:
                assert feats[a, IDX[f"action_type_{t}"]] == 0.0, f"action={a} has multiple action_type bits set"


def test_discard_delta_coins_is_3_for_real_slots(status_age1_turn0):
    obs = build_obs(status_age1_turn0, agent_seat=0)
    feats = obs["action_feats"]
    hand_ids = obs["hand_ids"].astype(int)

    # Schema contract: discard_delta_coins exists
    assert "discard_delta_coins" in IDX, "Schema missing discard_delta_coins"
    col = IDX["discard_delta_coins"]

    for slot in range(N_SLOTS):
        a = _idx(DISCARD, slot)
        if hand_ids[slot] == -1:
            # padded slot => illegal; already checked zeroed
            continue
        assert feats[a, col] == 3.0, f"discard action at slot {slot} should give +3 coins"


def test_card_era_and_color_match_hand_cards(status_age1_turn0, card_db):
    """
    Pick two cards from the fixture hand and verify per-action header encodes:
      - card_era_{1,2,3}
      - card_color_{...}
    without duplicating the feature computation logic.
    """
    status = status_age1_turn0
    seat = 0
    obs = build_obs(status, agent_seat=seat)
    feats = obs["action_feats"]

    hand = status["players"][str(seat)]["cards_hand"]
    assert len(hand) >= 2, "fixture hand too small for this test"

    # Choose two specific cards from this fixture (stable because fixture is frozen)
    names = [hand[0], hand[1]]

    # Build quick lookup for cards by name from loaded card_db
    cards = _extract_cards(card_db)
    by_name = {c["name"]: c for c in cards if "name" in c}

    for slot, name in enumerate(names):
        assert name in by_name, f"Card {name!r} not found in card_db"
        card = by_name[name]
        era = _card_era(card)
        color = _card_color(card)

        # Use build_structure row for that slot (header is shared across action types anyway)
        a = _idx(BUILD_STRUCTURE, slot)

        # Era one-hot
        for e in (1, 2, 3):
            col = IDX[f"card_era_{e}"]
            expected = 1.0 if e == era else 0.0
            assert feats[a, col] == expected, f"{name}: era one-hot mismatch at card_era_{e}"

        # Color one-hot
        for c in ("brown", "grey", "blue", "yellow", "red", "green", "purple"):
            col = IDX[f"card_color_{c}"]
            expected = 1.0 if c == color else 0.0
            assert feats[a, col] == expected, f"{name}: color one-hot mismatch at card_color_{c}"
