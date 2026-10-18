import numpy as np
import pytest

from sevenwonders_rl.features.obs_builder import compute_action_mask

pytestmark = pytest.mark.component

N_SLOTS = 7

# action_index = action_type * 7 + slot
BUILD_STRUCTURE = 0
BUILD_WONDER = 1
BUILD_HAND_FREE = 2
DISCARD = 3


def _idx(action_type: int, slot: int) -> int:
    return action_type * N_SLOTS + slot


def _assert_slot_all_false(mask: np.ndarray, slot: int, ctx: str) -> None:
    for a in (BUILD_STRUCTURE, BUILD_WONDER, BUILD_HAND_FREE, DISCARD):
        assert mask[_idx(a, slot)] == 0, f"{ctx}: expected slot {slot} to be fully illegal, but {a=} was legal"


def _assert_slot_rule(mask: np.ndarray, slot: int, *, can_wonder: bool, can_free: bool, playable: bool, ctx: str) -> None:
    assert mask[_idx(DISCARD, slot)] == 1, f"{ctx}: discard should always be legal for real hand slots"

    assert mask[_idx(BUILD_STRUCTURE, slot)] == (1 if playable else 0), (
        f"{ctx}: build_structure mismatch. playable={playable}"
    )

    assert mask[_idx(BUILD_WONDER, slot)] == (1 if can_wonder else 0), (
        f"{ctx}: build_wonder mismatch. can_build_wonder={can_wonder}"
    )

    assert mask[_idx(BUILD_HAND_FREE, slot)] == (1 if can_free else 0), (
        f"{ctx}: build_hand_free mismatch. can_build_hand_free={can_free}"
    )


def test_action_mask_matches_contract_on_named_fixtures(
    status_age1_turn0,
    status_can_build_wonder,
    status_can_build_hand_free,
):
    # Run the same assertions on a few fixtures that exercise flags
    for status, label in [
        (status_age1_turn0, "age1_turn0"),
        (status_can_build_wonder, "can_build_wonder"),
        (status_can_build_hand_free, "can_build_hand_free"),
    ]:
        for seat in (0, 1, 2):
            p = status["players"][str(seat)]
            hand = p["cards_hand"]
            playable_set = set(p["cards_playable"])
            can_wonder = bool(p["can_build_wonder"])
            can_free = bool(p["can_build_hand_free"])

            mask = compute_action_mask(status, seat_id=seat)
            assert mask.shape == (28,), f"{label}.seat{seat}: mask shape {mask.shape}"
            assert set(np.unique(mask)).issubset({0, 1}), f"{label}.seat{seat}: mask must be 0/1 only"

            # Real slots
            for slot in range(min(len(hand), N_SLOTS)):
                card = hand[slot]
                playable = card in playable_set
                _assert_slot_rule(
                    mask,
                    slot,
                    can_wonder=can_wonder,
                    can_free=can_free,
                    playable=playable,
                    ctx=f"{label}.seat{seat}.slot{slot}.{card}",
                )

            # Padded slots
            for slot in range(len(hand), N_SLOTS):
                _assert_slot_all_false(mask, slot, ctx=f"{label}.seat{seat}.slot{slot}.PAD")


def test_action_mask_padded_slots_always_illegal_over_pool(status_pool):
    # Broad invariant over real game states; sample every 10th for speed.
    for path, status in status_pool[::10]:
        for seat in (0, 1, 2):
            hand = status["players"][str(seat)]["cards_hand"]
            mask = compute_action_mask(status, seat_id=seat)
            for slot in range(len(hand), N_SLOTS):
                _assert_slot_all_false(mask, slot, ctx=f"{path.name}.seat{seat}.PAD{slot}")
