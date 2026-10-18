import pytest

pytestmark = pytest.mark.component

REQUIRED_GAME_KEYS = ("era", "turn", "clockwise")

REQUIRED_PLAYER_KEYS = (
    "amount",
    "can_build_hand_free",
    "can_build_wonder",
    "cards_hand",
    "cards_hand_id",
    "cards_playable",
    "points",
    "resources",
    "wonder_stage",
)

REQUIRED_RES_KEYS = (
    "wood", "ore", "clay", "stone", "loom", "glass", "papyrus",
    "gear", "compass", "tablet",
    "coins", "shields",
)

REQUIRED_POINTS_KEYS = (
    "civilian", "commercial", "guild", "military", "scientific", "wonder", "total"
)

REQUIRED_AMOUNT_KEYS = (
    "civilian", "commercial", "guild", "military", "scientific",
    "raw_material", "manufactured_goods",
)


def _assert_has_keys(d: dict, keys: tuple, ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    assert not missing, f"{ctx}: missing keys {missing}. present={sorted(d.keys())}"


def _assert_int_fields(d: dict, keys: tuple, ctx: str) -> None:
    bad = [(k, type(d.get(k)).__name__, d.get(k)) for k in keys if not isinstance(d.get(k), int)]
    assert not bad, f"{ctx}: expected int fields, got {bad}"


def _assert_bool_field(d: dict, key: str, ctx: str) -> None:
    assert isinstance(d.get(key), bool), (
        f"{ctx}: expected bool for {key}, got {type(d.get(key)).__name__}={d.get(key)}"
    )


def test_pool_status_schema_and_types(status_pool):
    for path, status in status_pool:
        ctx = f"{path.name}"

        assert isinstance(status, dict), f"{ctx}: status is not a dict"
        assert "game" in status and isinstance(status["game"], dict), f"{ctx}: missing/invalid status['game']"
        assert "players" in status and isinstance(status["players"], dict), f"{ctx}: missing/invalid status['players']"

        game = status["game"]
        _assert_has_keys(game, REQUIRED_GAME_KEYS, f"{ctx}.game")
        _assert_int_fields(game, ("era", "turn"), f"{ctx}.game")
        _assert_bool_field(game, "clockwise", f"{ctx}.game")

        era = game["era"]
        assert era in (1, 2, 3), f"{ctx}: game.era out of range: {era}"
        turn = game["turn"]
        assert 0 <= turn <= 30, f"{ctx}: game.turn looks wrong: {turn}"

        # Contract from engine writer: clockwise == (era==1 or era==3)
        assert game["clockwise"] == (era in (1, 3)), f"{ctx}: game.clockwise inconsistent with era={era}"

        players = status["players"]
        for sid in ("0", "1", "2"):
            assert sid in players, f"{ctx}: missing players['{sid}']"
            p = players[sid]
            assert isinstance(p, dict), f"{ctx}: players['{sid}'] is not a dict"
            _assert_has_keys(p, REQUIRED_PLAYER_KEYS, f"{ctx}.players[{sid}]")

            hand = p["cards_hand"]
            hand_ids = p["cards_hand_id"]
            playable = p["cards_playable"]

            assert isinstance(hand, list) and all(isinstance(x, str) for x in hand), (
                f"{ctx}.players[{sid}]: cards_hand invalid"
            )
            assert isinstance(hand_ids, list) and all(isinstance(x, int) for x in hand_ids), (
                f"{ctx}.players[{sid}]: cards_hand_id invalid"
            )
            assert len(hand) == len(hand_ids), f"{ctx}.players[{sid}]: len(cards_hand) != len(cards_hand_id)"
            assert 0 <= len(hand) <= 7, f"{ctx}.players[{sid}]: unexpected hand size {len(hand)}"

            assert isinstance(playable, list) and all(isinstance(x, str) for x in playable), (
                f"{ctx}.players[{sid}]: cards_playable invalid"
            )
            for c in playable:
                assert c in hand, f"{ctx}.players[{sid}]: playable card not in hand: {c}"

            _assert_bool_field(p, "can_build_wonder", f"{ctx}.players[{sid}]")
            _assert_bool_field(p, "can_build_hand_free", f"{ctx}.players[{sid}]")

            assert isinstance(p["wonder_stage"], int), f"{ctx}.players[{sid}]: wonder_stage not int"

            res = p["resources"]
            pts = p["points"]
            amt = p["amount"]
            assert isinstance(res, dict), f"{ctx}.players[{sid}]: resources not dict"
            assert isinstance(pts, dict), f"{ctx}.players[{sid}]: points not dict"
            assert isinstance(amt, dict), f"{ctx}.players[{sid}]: amount not dict"

            _assert_has_keys(res, REQUIRED_RES_KEYS, f"{ctx}.players[{sid}].resources")
            _assert_has_keys(pts, REQUIRED_POINTS_KEYS, f"{ctx}.players[{sid}].points")
            _assert_has_keys(amt, REQUIRED_AMOUNT_KEYS, f"{ctx}.players[{sid}].amount")

            _assert_int_fields(res, REQUIRED_RES_KEYS, f"{ctx}.players[{sid}].resources")
            _assert_int_fields(pts, REQUIRED_POINTS_KEYS, f"{ctx}.players[{sid}].points")
            _assert_int_fields(amt, REQUIRED_AMOUNT_KEYS, f"{ctx}.players[{sid}].amount")

            assert res["coins"] >= 0, f"{ctx}.players[{sid}]: coins negative"
            assert res["shields"] >= 0, f"{ctx}.players[{sid}]: shields negative"

            # Contract: total must equal sum of breakdown (allows negatives from military defeats)
            breakdown_keys = [k for k in REQUIRED_POINTS_KEYS if k != "total"]
            expected_total = sum(int(pts[k]) for k in breakdown_keys)
            assert pts["total"] == expected_total, (
                f"{ctx}.players[{sid}]: points.total mismatch. "
                f"total={pts['total']} expected_sum={expected_total} "
                f"breakdown={{{', '.join(f'{k}: {pts[k]}' for k in breakdown_keys)}}}"
            )
