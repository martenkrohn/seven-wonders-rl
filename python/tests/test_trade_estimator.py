import pytest

from sevenwonders_rl.features.trade_estimator import estimate_trade_cost
from sevenwonders_rl.data.mechanics import get_trade_costs

pytestmark = pytest.mark.unit


def _count_choice_tokens(choice_supply: list[set[str]], res: str) -> int:
    return sum(1 for s in choice_supply if res in s)


def _assert_invariants(est):
    assert est.total_cost == est.cost_left + est.cost_right, (
        f"total_cost mismatch: total={est.total_cost} left={est.cost_left} right={est.cost_right}"
    )
    if est.feasible:
        assert est.missing_raw == 0 and est.missing_manuf == 0, (
            f"feasible but missing: raw={est.missing_raw} manuf={est.missing_manuf}"
        )


def _assert_plan_not_exceed_supply(
    est,
    left_supply: dict[str, int],
    left_choice_supply: list[set[str]],
    right_supply: dict[str, int],
    right_choice_supply: list[set[str]],
):
    for res, bought in est.plan_left.items():
        max_avail = int(left_supply.get(res, 0)) + _count_choice_tokens(left_choice_supply, res)
        assert bought <= max_avail, f"plan_left overbuys {res}: bought={bought} max_avail={max_avail}"
    for res, bought in est.plan_right.items():
        max_avail = int(right_supply.get(res, 0)) + _count_choice_tokens(right_choice_supply, res)
        assert bought <= max_avail, f"plan_right overbuys {res}: bought={bought} max_avail={max_avail}"


def test_no_missing_cost_zero():
    est = estimate_trade_cost(
        required={"wood": 1, "stone": 1},
        own_production={"wood": 1, "stone": 1},
        own_choice_resources=[],
        left_supply={},
        left_choice_supply=[],
        right_supply={},
        right_choice_supply=[],
        player_flags=set(),
    )
    _assert_invariants(est)
    assert est.feasible is True
    assert est.total_cost == 0
    assert est.missing_raw == 0
    assert est.missing_manuf == 0


def test_own_choice_used_once_unique_best_need():
    # Need wood=2, stone=1, and we have one own choice token {wood, stone}.
    # Unique best need is wood (2 > 1), so the choice should reduce wood by 1.
    # No neighbors: remaining should be wood=1, stone=1 => missing_raw=2.
    est = estimate_trade_cost(
        required={"wood": 2, "stone": 1},
        own_production={},
        own_choice_resources=[{"wood", "stone"}],
        left_supply={},
        left_choice_supply=[],
        right_supply={},
        right_choice_supply=[],
        player_flags=set(),
    )
    _assert_invariants(est)
    assert est.feasible is False
    assert est.missing_manuf == 0
    assert est.missing_raw == 2  # wood(1) + stone(1)


def test_left_raw_discount_applies_only_left():
    trade = get_trade_costs()
    assert trade["raw"] == 2 and trade.get("raw_discount", 1) == 1

    est = estimate_trade_cost(
        required={"wood": 1},
        own_production={},
        own_choice_resources=[],
        left_supply={"wood": 1},
        left_choice_supply=[],
        right_supply={"wood": 1},
        right_choice_supply=[],
        player_flags={"raw_cheap_east"},
    )
    _assert_invariants(est)
    _assert_plan_not_exceed_supply(est, {"wood": 1}, [], {"wood": 1}, [])
    assert est.feasible is True
    assert est.cost_left == 1
    assert est.cost_right == 0
    assert est.total_cost == 1
    assert est.plan_left.get("wood", 0) == 1


def test_right_raw_discount_applies_only_right():
    est = estimate_trade_cost(
        required={"wood": 1},
        own_production={},
        own_choice_resources=[],
        left_supply={"wood": 1},
        left_choice_supply=[],
        right_supply={"wood": 1},
        right_choice_supply=[],
        player_flags={"raw_cheap_west"},
    )
    _assert_invariants(est)
    _assert_plan_not_exceed_supply(est, {"wood": 1}, [], {"wood": 1}, [])
    assert est.feasible is True
    assert est.cost_left == 0
    assert est.cost_right == 1
    assert est.total_cost == 1
    assert est.plan_right.get("wood", 0) == 1


def test_manuf_discount_applies_both_directions():
    trade = get_trade_costs()
    assert trade["manuf"] == 2 and trade.get("manuf_discount", 1) == 1

    est = estimate_trade_cost(
        required={"glass": 1},
        own_production={},
        own_choice_resources=[],
        left_supply={"glass": 1},
        left_choice_supply=[],
        right_supply={"glass": 1},
        right_choice_supply=[],
        player_flags={"manuf_cheap"},
    )
    _assert_invariants(est)
    _assert_plan_not_exceed_supply(est, {"glass": 1}, [], {"glass": 1}, [])
    assert est.feasible is True
    # tie => left
    assert est.total_cost == 1
    assert est.cost_left == 1
    assert est.cost_right == 0


def test_cost_splits_across_neighbors_when_needed():
    est = estimate_trade_cost(
        required={"wood": 2},
        own_production={},
        own_choice_resources=[],
        left_supply={"wood": 1},
        left_choice_supply=[],
        right_supply={"wood": 1},
        right_choice_supply=[],
        player_flags=set(),
    )
    _assert_invariants(est)
    _assert_plan_not_exceed_supply(est, {"wood": 1}, [], {"wood": 1}, [])
    assert est.feasible is True
    assert est.total_cost == 4  # 2 * 2 coins
    assert est.cost_left == 2
    assert est.cost_right == 2
    assert est.plan_left.get("wood", 0) == 1
    assert est.plan_right.get("wood", 0) == 1


def test_infeasible_when_no_supply():
    est = estimate_trade_cost(
        required={"papyrus": 1},
        own_production={},
        own_choice_resources=[],
        left_supply={},
        left_choice_supply=[],
        right_supply={},
        right_choice_supply=[],
        player_flags=set(),
    )
    _assert_invariants(est)
    assert est.feasible is False
    assert est.missing_raw == 0
    assert est.missing_manuf == 1
    assert est.total_cost == 0


def test_neighbor_choice_token_must_be_exclusive_across_resources():
    # One left neighbor choice token can provide wood OR ore, but not both.
    est = estimate_trade_cost(
        required={"wood": 1, "ore": 1},
        own_production={},
        own_choice_resources=[],
        left_supply={},
        left_choice_supply=[{"wood", "ore"}],  # single token
        right_supply={},
        right_choice_supply=[],
        player_flags=set(),
    )
    _assert_invariants(est)
    assert est.feasible is False
    assert est.missing_raw == 1
