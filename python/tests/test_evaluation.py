import pytest

from sevenwonders_rl.evaluation.runner import run_game, aggregate
from sevenwonders_rl.envs.sevenwonders_env import EngineError

pytestmark = pytest.mark.unit


class FakeEnv:
    agent_seat = 0

    def __init__(self, outcome):
        self.outcome = outcome
        self.closed = False

    def reset(self, seed):
        if self.outcome == "startup":
            raise EngineError("engine_exit", "start failed")
        return {}, {}

    def step(self, action):
        if self.outcome == "bug":
            raise TypeError("programming error")
        if self.outcome in ("timeout", "engine_exit"):
            return {}, 0., False, True, {"transition_valid": False, "truncated_reason": self.outcome}
        if self.outcome == "external_limit":
            return {}, 0., False, True, {"transition_valid": True}
        return {}, 2., True, False, {"transition_valid": True, "status": {
            "game": {"phase": "finished"}, "players": {
                str(s): {"points": {"total": t}} for s, t in enumerate((12, 10, 9))}}}

    def close(self):
        self.closed = True


def test_mixed_attempts_have_honest_denominators_and_no_partial_scores():
    records = []
    for kind in ("completed", "timeout", "engine_exit", "startup"):
        env = FakeEnv(kind)
        result, _ = run_game(env, lambda obs: 0, {"engine_seed": 0})
        assert env.closed
        if result["status"] != "completed":
            assert result["win"] is None and result["totals"] is None
        records.append(result)
    metrics = aggregate(records)
    assert (metrics["attempted"], metrics["completed"], metrics["truncated"], metrics["failed"]) == (4, 1, 1, 2)
    assert metrics["win_rate"] == 1 and metrics["vp_diff_mean"] == 2
    assert not metrics["valid_for_selection"]
    assert aggregate(records[1:])["win_rate"] is None


def test_programming_error_propagates_and_resources_close():
    env = FakeEnv("bug")
    with pytest.raises(TypeError, match="programming"):
        run_game(env, lambda obs: 0, {"engine_seed": 0})
    assert env.closed


def test_external_limit_is_truncation_without_a_final_score():
    result, _ = run_game(FakeEnv("external_limit"), lambda obs: 0, {"engine_seed": 0})
    assert result["status"] == "truncated" and result["reason"] == "external_limit"
    assert result["vp_diff"] is None
