from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sevenwonders_rl.envs.sevenwonders_env import EngineError
from sevenwonders_rl.utils.timing import TimingBuckets
from test_training_contracts import trainer_module

pytestmark = pytest.mark.component


def observation(value):
    return {"global_table": np.full((3, 7), value, np.float32),
            "table_points": np.zeros((3, 7), np.float32),
            "table_prod": np.zeros((3, 7), np.float32),
            "table_amount": np.zeros((3, 7), np.float32),
            "wonder_stage": np.zeros(3, np.float32), "game_meta": np.zeros(3, np.float32),
            "action_feats": np.zeros((28, 92), np.float32), "action_mask": np.ones(28, np.int8)}


class ValueModel:
    def forward_batch(self, global_obs, features):
        return torch.zeros(len(global_obs), 28), global_obs[:, 0]


class FakeEnvironment:
    io_dir = "fake"

    def __init__(self, terminated=False, invalid=False):
        self.terminated, self.invalid = terminated, invalid

    def send_action(self, action, *, opponent_actions=None):
        return 0

    def wait_next(self, previous_turn):
        return observation(10), 1., self.terminated, not self.terminated, {"transition_valid": not self.invalid}


def test_collector_uses_pre_reset_observation_and_preserves_environment_axis():
    trainer = trainer_module()
    resets = []
    def reset(i, info, terminal):
        resets.append(i)
        return observation(1000 + i)
    rollout, following = trainer.collect_rollout(ValueModel(),
        [FakeEnvironment(), FakeEnvironment(terminated=True)], [observation(2), observation(3)],
        1, torch.Generator().manual_seed(0), reset, TimingBuckets())
    assert rollout["rewards"].shape == (1, 2)
    torch.testing.assert_close(rollout["next_values"], torch.tensor([[10., 0.]]))
    torch.testing.assert_close(rollout["values"], torch.tensor([[2., 3.]]))
    assert resets == [0, 1] and following[0]["global_table"][0,0] == 1000


def test_collector_rejects_transport_failure_instead_of_learning_from_it():
    trainer = trainer_module()
    with pytest.raises(EngineError, match="Invalid transition"):
        trainer.collect_rollout(ValueModel(), [FakeEnvironment(invalid=True)], [observation(0)],
            1, torch.Generator().manual_seed(0), lambda *args: pytest.fail("Must not reset invalid transition"), TimingBuckets())
