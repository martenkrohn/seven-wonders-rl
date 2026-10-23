import numpy as np
import pytest
import copy

from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.features.action_feature_schema import K as K_ACTION_FEATS
from sevenwonders_rl.utils.ppo_utils import flatten_global


pytestmark = pytest.mark.component


def _expected_global_row_from_status_player(p: dict) -> np.ndarray:
    r = p["resources"]
    coins = int(r["coins"])
    shields = int(r["shields"])
    gear = int(r["gear"])
    compass = int(r["compass"])
    tablet = int(r["tablet"])
    raw_total = int(r["wood"] + r["ore"] + r["clay"] + r["stone"])
    manu_total = int(r["loom"] + r["glass"] + r["papyrus"])
    return np.asarray([coins, shields, gear, compass, tablet, raw_total, manu_total], dtype=np.float32)


def _assert_obs_shapes(obs: dict, ctx: str) -> None:
    required_keys = {
        "global_table",
        "wonder_stage",
        "hand_ids",
        "action_mask",
        "action_feats",
        "table_points",
        "table_prod",
        "table_amount",
        "game_meta",
    }
    missing = sorted(required_keys - set(obs.keys()))
    assert not missing, f"{ctx}: missing obs keys: {missing}. present={sorted(obs.keys())}"

    assert obs["global_table"].shape == (3, 7), f"{ctx}: global_table shape {obs['global_table'].shape}"
    assert obs["wonder_stage"].shape == (3,), f"{ctx}: wonder_stage shape {obs['wonder_stage'].shape}"
    assert obs["hand_ids"].shape == (7,), f"{ctx}: hand_ids shape {obs['hand_ids'].shape}"
    assert obs["action_mask"].shape == (28,), f"{ctx}: action_mask shape {obs['action_mask'].shape}"
    assert obs["action_feats"].shape == (28, K_ACTION_FEATS), (
        f"{ctx}: action_feats shape {obs['action_feats'].shape}, expected (28,{K_ACTION_FEATS})"
    )
    assert obs["table_points"].shape == (3, 7), f"{ctx}: table_points shape {obs['table_points'].shape}"
    assert obs["table_prod"].shape == (3, 7), f"{ctx}: table_prod shape {obs['table_prod'].shape}"
    assert obs["table_amount"].shape == (3, 7), f"{ctx}: table_amount shape {obs['table_amount'].shape}"
    assert obs["game_meta"].shape == (3,), f"{ctx}: game_meta shape {obs['game_meta'].shape}"

    # Basic numeric sanity
    for k in ("global_table", "action_feats", "table_points", "table_prod", "table_amount", "game_meta"):
        arr = obs[k]
        assert np.isfinite(arr).all(), f"{ctx}: {k} contains NaN/inf"


def _rotation_for_seat(seat: int) -> tuple[int, int, int]:
    # Contract: rotate into [self, left, right] based on fixed seat order
    # For 3 players with seats [0,1,2], left = (seat+1)%3, right = (seat+2)%3
    return (seat, (seat - 1) % 3, (seat + 1) % 3)


def test_build_obs_shapes_all_eras(status_age1_turn0, status_age2_turn0, status_age3_turn0):
    for era_status, label in [
        (status_age1_turn0, "age1"),
        (status_age2_turn0, "age2"),
        (status_age3_turn0, "age3"),
    ]:
        for seat in (0, 1, 2):
            obs = build_obs(era_status, seat)
            _assert_obs_shapes(obs, ctx=f"{label}.seat{seat}")


def test_global_table_matches_resources_and_rotation_on_age1_turn0(status_age1_turn0):
    status = status_age1_turn0

    for seat in (0, 1, 2):
        obs = build_obs(status, seat)
        gt = obs["global_table"]

        s_self, s_left, s_right = _rotation_for_seat(seat)

        exp_self = _expected_global_row_from_status_player(status["players"][str(s_self)])
        exp_left = _expected_global_row_from_status_player(status["players"][str(s_left)])
        exp_right = _expected_global_row_from_status_player(status["players"][str(s_right)])

        np.testing.assert_array_equal(gt[0], exp_self, err_msg=f"seat{seat}: self row mismatch")
        np.testing.assert_array_equal(gt[1], exp_left, err_msg=f"seat{seat}: left row mismatch")
        np.testing.assert_array_equal(gt[2], exp_right, err_msg=f"seat{seat}: right row mismatch")


def test_game_meta_matches_status(status_age1_turn0):
    status = status_age1_turn0
    obs = build_obs(status, agent_seat=0)

    era = int(status["game"]["era"])
    turn = int(status["game"]["turn"])
    clockwise = bool(status["game"]["clockwise"])

    # Contract: game_meta = [era, turn, clockwise]
    # Clockwise stored as 1/0 in array form is acceptable.
    gm = obs["game_meta"]
    assert int(gm[0]) == era
    assert int(gm[1]) == turn
    assert int(gm[2]) == (1 if clockwise else 0)


def test_hand_ids_padded_and_matches_status(status_pool):
    # Use pool to hit smaller hand sizes (6..2) deterministically.
    # Keep it light: every 10th snapshot.
    for path, status in status_pool[::10]:
        for seat in (0, 1, 2):
            obs = build_obs(status, seat)
            hand_ids = list(status["players"][str(seat)]["cards_hand_id"])
            # build_obs contract: 7 slots padded with -1
            expected = hand_ids[:7] + [-1] * max(0, 7 - len(hand_ids))
            got = obs["hand_ids"].astype(int).tolist()
            assert got == expected, f"{path.name}: seat{seat} hand_ids mismatch. got={got} exp={expected}"


@pytest.mark.parametrize("resource", ["wood", "ore", "clay", "stone", "loom", "glass", "papyrus"])
def test_starting_resource_reaches_model_input_for_every_relative_seat(status_age1_turn0, resource):
    # Keep all public resource totals unchanged except this one starting resource.
    # Check the flattened network input too, not just an unused observation key.
    status = copy.deepcopy(status_age1_turn0)
    for agent_seat in (0, 1, 2):
        for resource_seat in (0, 1, 2):
            before = build_obs(status, agent_seat)
            status["players"][str(resource_seat)]["resources"][resource] += 1
            after = build_obs(status, agent_seat)
            row = _rotation_for_seat(agent_seat).index(resource_seat)
            column = ["wood", "ore", "clay", "stone", "loom", "glass", "papyrus"].index(resource)
            assert after["table_prod"][row, column] == before["table_prod"][row, column] + 1
            # Resource totals occupy the third 3x7 block in the global vector.
            index = 42 + row * 7 + column
            assert flatten_global(after)[index] == flatten_global(before)[index] + 1
            status["players"][str(resource_seat)]["resources"][resource] -= 1
