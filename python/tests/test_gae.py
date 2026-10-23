import pytest
import torch

from sevenwonders_rl.utils.gae import compute_gae, normalize_advantages

pytestmark = pytest.mark.unit


def gae(rewards, values=None, terminated=None, truncated=None, next_values=None, **kwargs):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    zeros = torch.zeros_like(rewards)
    convert = lambda x: zeros if x is None else torch.tensor(x)
    return compute_gae(rewards, convert(values), convert(terminated), convert(truncated),
                       convert(next_values), **kwargs)


def test_independent_environments_and_permutation():
    advantages, returns = gae([[0., 0.], [0., 1.]], gamma=1, gae_lambda=1)
    torch.testing.assert_close(advantages, torch.tensor([[0., 1.], [0., 1.]]))
    changed, _ = gae([[0., 3.], [0., 9.]], gamma=1, gae_lambda=1)
    torch.testing.assert_close(changed[:, 0], advantages[:, 0])
    permuted, _ = gae([[0., 0.], [1., 0.]], gamma=1, gae_lambda=1)
    torch.testing.assert_close(permuted, advantages.flip(1))
    torch.testing.assert_close(advantages, returns)


def test_distinct_bootstraps_and_nonzero_values():
    advantages, returns = gae([[0., 0.]], values=[[2., 3.]], next_values=[[10., 20.]], gamma=.5)
    torch.testing.assert_close(advantages, torch.tensor([[3., 7.]]))
    torch.testing.assert_close(returns, torch.tensor([[5., 10.]]))


def test_staggered_endings_cannot_leak_across_reset():
    advantages, _ = gae([[1., 0.], [100., 2.]], terminated=[[1, 0], [0, 1]],
                       gamma=1, gae_lambda=1)
    torch.testing.assert_close(advantages, torch.tensor([[1., 2.], [100., 2.]]))


def test_truncation_bootstraps_from_final_observation_but_cuts_trace():
    advantages, _ = gae([[1.], [100.]], truncated=[[1], [0]],
                        next_values=[[10.], [0.]], gamma=1, gae_lambda=1)
    torch.testing.assert_close(advantages, torch.tensor([[11.], [100.]]))
    terminal, _ = gae([[1.]], terminated=[[1]], next_values=[[999.]], gamma=1)
    assert terminal.item() == 1


def test_one_sample_and_invalid_inputs():
    advantages, _ = gae([[5.]])
    assert normalize_advantages(advantages).item() == 0
    with pytest.raises(ValueError, match="same shape"):
        compute_gae(torch.zeros(2, 2), *[torch.zeros(1, 2)] * 4)
    with pytest.raises(ValueError, match="Non-finite"):
        gae([[float("nan")]])
