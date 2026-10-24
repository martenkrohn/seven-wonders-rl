"""Batching must preserve seat-local randomness and immutable snapshot weights."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sevenwonders_rl.envs import SeatPolicy
from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.policies.opponent_batch import opponent_actions_batch
from sevenwonders_rl.policies.ppo_checkpoint_policy import PPOCheckpointPolicy
from sevenwonders_rl.utils.timing import TimingBuckets
from test_selfplay import checkpoint

pytestmark = pytest.mark.component


@pytest.mark.parametrize("deterministic", [False, True])
def test_shared_snapshot_batches_four_seats_without_mixing_rngs_or_states(tmp_path, status_age1_turn0, monkeypatch, deterministic):
    checkpoint(tmp_path / "shared.pt")
    policy = PPOCheckpointPolicy(str(tmp_path / "shared.pt"), deterministic=deterministic)
    snapshots = copy.deepcopy(policy.model.state_dict())
    statuses = [copy.deepcopy(status_age1_turn0) for _ in range(2)]
    statuses[1]["players"]["0"]["resources"]["coins"] += 10
    envs = [SimpleNamespace(batch_opponents=True, _opponent_sampler=None, _last_status=status,
            agent_seat=seat, opponent_policies=[SeatPolicy("ppo_snapshot", policy=policy) for _ in range(3)],
            _seat_rngs=[np.random.default_rng(100 * index + s) for s in range(3)])
            for index, (status, seat) in enumerate(zip(statuses, (0, 2)))]
    serial_rngs = [[np.random.default_rng(100 * index + s) for s in range(3)] for index in range(2)]
    calls = []
    predictions = []
    forward = policy.model.forward_batch
    def observed(global_obs, features):
        calls.append(len(global_obs))
        result = forward(global_obs, features)
        predictions.append(tuple(value.detach().clone() for value in result))
        return result
    monkeypatch.setattr(policy.model, "forward_batch", observed)
    before = torch.get_rng_state().clone()
    expected = [{seat: policy.select_action(build_obs(status, seat), serial_rngs[index][seat])
                 for seat in range(3) if seat != envs[index].agent_seat}
                for index, status in enumerate(statuses)]
    assert calls == [1, 1, 1, 1]
    serial_predictions = tuple(torch.cat([prediction[i] for prediction in predictions]) for i in (0, 1))
    calls.clear()
    predictions.clear()
    timers = TimingBuckets()
    actual = opponent_actions_batch(envs, timers=timers)
    # Batch-sized matrix multiplication can round identical/near-tied greedy
    # scores differently. Compare every score/value, and only permit a changed
    # deterministic choice when both scores are equal at float32 precision.
    torch.testing.assert_close(predictions[0], serial_predictions, rtol=1e-5, atol=1e-6)
    if deterministic:
        row = 0
        for index, seats in enumerate(expected):
            assert actual[index].keys() == seats.keys()
            for seat, action in seats.items():
                logits = serial_predictions[0][row]
                torch.testing.assert_close(logits[actual[index][seat]], logits[action], rtol=0, atol=1e-6)
                row += 1
    else:
        assert actual == expected
    assert calls == [4]
    assert timers.buckets["policies.opponents.requests"].calls == 4
    assert timers.buckets["policies.opponents.inference"].calls == 1
    for index, env in enumerate(envs):
        for seat in range(3):
            assert env._seat_rngs[seat].bit_generator.state == serial_rngs[index][seat].bit_generator.state
    torch.testing.assert_close(before, torch.get_rng_state())
    for name, value in snapshots.items():
        torch.testing.assert_close(value, policy.model.state_dict()[name])
    assert all(p.grad is None for p in policy.model.parameters())


def test_distinct_snapshots_and_serial_or_sampler_paths_are_separate(tmp_path, status_age1_turn0, monkeypatch):
    checkpoint(tmp_path / "first.pt")
    checkpoint(tmp_path / "second.pt")
    first, second = (PPOCheckpointPolicy(str(tmp_path / name)) for name in ("first.pt", "second.pt"))
    calls = []
    for name, policy in (("first", first), ("second", second)):
        original = policy.select_actions
        def observed(observations, rngs, original=original, name=name):
            calls.append((name, len(observations)))
            return original(observations, rngs)
        monkeypatch.setattr(policy, "select_actions", observed)
    env = SimpleNamespace(batch_opponents=True, _opponent_sampler=None, _last_status=status_age1_turn0,
        agent_seat=1, opponent_policies=[SeatPolicy("ppo_snapshot", policy=first), SeatPolicy("random"),
                                       SeatPolicy("ppo_snapshot", policy=second)],
        _seat_rngs=[np.random.default_rng(s) for s in range(3)])
    assert set(opponent_actions_batch([env])[0]) == {0, 2}
    assert calls == [("first", 1), ("second", 1)]
    env.batch_opponents = False
    assert opponent_actions_batch([env]) == [{}]
    env.batch_opponents, env._opponent_sampler = True, object()
    assert opponent_actions_batch([env]) == [{}]
    assert len(calls) == 2


def test_illegal_batched_action_is_rejected_without_random_fallback(tmp_path, status_age1_turn0, monkeypatch):
    checkpoint(tmp_path / "shared.pt")
    policy = PPOCheckpointPolicy(str(tmp_path / "shared.pt"))
    monkeypatch.setattr(policy, "select_actions", lambda observations, rngs: [999] * len(observations))
    env = SimpleNamespace(batch_opponents=True, _opponent_sampler=None, _last_status=status_age1_turn0,
        agent_seat=0, opponent_policies=[SeatPolicy("ppo_snapshot", policy=policy) for _ in range(3)],
        _seat_rngs=[np.random.default_rng(s) for s in range(3)])
    with pytest.raises(ValueError, match="illegal action"):
        opponent_actions_batch([env])
