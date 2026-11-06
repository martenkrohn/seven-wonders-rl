"""Ordered-pair sampling probabilities, decay and checkpoint state."""

import json
import numpy as np
import pytest
from sevenwonders_rl.selfplay.opponent_pool import OpponentPool
from sevenwonders_rl.selfplay.pair_pfsp import PairPFSPPool

pytestmark = pytest.mark.component


def pool_at(tmp_path, steps=(0, 100), **kwargs):
    for step in steps:
        (tmp_path / f"checkpoint_{step}.pt").touch()
    return PairPFSPPool(OpponentPool(str(tmp_path), auto_refresh=False), **kwargs)


def test_neutral_start_new_pairs_and_future_snapshot_filter(tmp_path):
    pool = pool_at(tmp_path)
    pool.league.max_size = 1
    pairs, rates, counts, _, probabilities = pool.distribution(0)
    assert [(a[0], b[0]) for a, b in pairs] == [(0, 0)]
    np.testing.assert_allclose(rates, [1 / 3])
    np.testing.assert_array_equal(counts, [0])
    np.testing.assert_array_equal(probabilities, [1])
    pool.league.max_size = 50
    for _ in range(30):
        pool.record_result(0, 0, [90, 20, 30], 0, 0)
    pairs, rates, counts, _, probabilities = pool.distribution(100)
    assert [(a[0], b[0]) for a, b in pairs] == [(0, 0), (0, 100), (100, 0), (100, 100)]
    assert rates[0] > 0.9 and all(r == pytest.approx(1 / 3) for r in rates[1:])
    assert counts[0] > 0 and not counts[1:].any()
    assert probabilities[0] < min(probabilities[1:])
    assert min(probabilities) >= pool.control_fraction / 4
    assert sum(probabilities) == pytest.approx(1)


def test_ordered_pairs_ties_decay_and_immutable_checkpoint_state(tmp_path):
    pool = pool_at(tmp_path, half_life=100)
    pool.record_result(0, 100, [20, 20, 10], 0, 100)
    pool.record_result(100, 0, [10, 20, 30], 0, 100)
    _, rates, counts, _, _ = pool.distribution(100)
    assert counts[1] == counts[2] == 1
    assert rates[1] == pytest.approx(1.5 / 4)
    assert rates[2] == pytest.approx(1.0 / 4)
    _, later_rates, later_counts, _, _ = pool.distribution(200)
    assert later_counts[1] == 0.5
    assert abs(later_rates[1] - 1 / 3) < abs(rates[1] - 1 / 3)
    saved = pool.state_dict()
    restored = pool_at(tmp_path, half_life=100)
    restored.load_state_dict(saved)
    a, b = np.random.default_rng(97), np.random.default_rng(97)
    assert [pool.sample_pair(a, 200) for _ in range(30)] == [
        restored.sample_pair(b, 200) for _ in range(30)
    ]
    pool.record_result(0, 100, [10, 10, 10], 0, 200)
    assert saved == restored.state_dict() and saved != pool.state_dict()
    assert pool._results[(0, 100)]["score"] == pytest.approx(0.25 + 1 / 3)
    broken = json.loads(json.dumps(saved))
    broken["results"][0]["score"] = -1
    with pytest.raises(ValueError, match="Invalid"):
        restored.load_state_dict(broken)
    with pytest.raises(ValueError, match="Incompatible"):
        pool_at(tmp_path, power=3).load_state_dict(saved)
