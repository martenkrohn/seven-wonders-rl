"""Significance uses independent game-seed blocks, including exact small samples."""
import itertools
import json

import numpy as np
import pytest

from sevenwonders_rl.evaluation.selfplay import block_significance, paired_progress


@pytest.mark.unit
def test_exact_sign_flip_matches_enumeration_and_is_two_sided():
    values = np.array([[1., 0.], [2., 0.], [-.5, 0.], [3., 0.]])
    observed = abs(values[:, 0].mean())
    distribution = [abs(np.mean(values[:, 0] * s)) for s in itertools.product([-1, 1], repeat=4)]
    expected = np.mean(np.asarray(distribution) >= observed)
    result = block_significance(values)
    assert result['exact'] and result['permutations'] == 16
    assert result['p_values'] == [expected, 1.]
    assert block_significance(-values)['p_values'] == result['p_values']
    assert block_significance(np.ones((12, 1)))['p_values'][0] == 2 / 4096
    assert block_significance(np.ones((1, 1))) is None


@pytest.mark.unit
def test_randomized_sign_flip_is_reproducible_and_never_returns_zero():
    state = np.random.get_state()
    result = block_significance(np.ones((18, 2)))
    assert result == block_significance(np.ones((18, 2)))
    assert not result['exact'] and result['permutations'] == 65535
    assert all(0 < p < .001 for p in result['p_values'])
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.unit
def test_seat_rotations_do_not_inflate_significance_and_duplicate_games_are_rejected(tmp_path):
    def record(seed, seat, difference):
        return dict(engine_seed=seed, agent_seat=seat, opponents={'left':'a','right':'b'},
                    status='completed', vp_diff=difference, win_share=difference / 10)
    baseline = [record(seed, seat, 0) for seed in range(3) for seat in range(3)]
    current = [record(seed, seat, 2) for seed in range(3) for seat in range(3)]
    a,b = tmp_path/'a.jsonl',tmp_path/'b.jsonl'
    def write(path, rows):
        path.write_text('\n'.join(map(json.dumps, rows)))
    write(a,current); write(b,baseline)
    result = paired_progress(a,b)
    assert result['seed_blocks'] == 3
    assert result['vp_diff_gain_p_value'] == .25  # 3 blocks, not 9 independent games
    assert result['significance_test']['method'] == 'paired_seed_block_sign_flip'
    write(a,current+current);write(b,baseline+baseline)
    with pytest.raises(ValueError, match='plans differ'):
        paired_progress(a,b)
