import copy
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.features.action_feature_schema import IDX
from sevenwonders_rl.utils.checkpoints import load_model, save_checkpoint, schema_hash, FEATURE_SEMANTICS_VERSION
from sevenwonders_rl.utils.ppo_model import ActorCritic, MODEL_ARCHITECTURE_VERSION
from sevenwonders_rl.utils.ppo_utils import select_action_from_obs
from sevenwonders_rl.utils.seeding import derive_seed

pytestmark = pytest.mark.component


def test_game_features_use_engine_fields_and_requested_seat(status_age3_turn0):
    status = copy.deepcopy(status_age3_turn0)
    status["game"]["turn"] = 17
    for seat in (0, 1, 2):
        player = status["players"][str(seat)]
        player["cards_hand"] = player["cards_hand"][:7-seat]
        player["cards_hand_id"] = player["cards_hand_id"][:7-seat]
        obs = build_obs(status, seat)
        row = obs["action_feats"][np.flatnonzero(obs["action_mask"])[0]]
        assert [row[IDX[f"age_{a}"]] for a in (1, 2, 3)] == [0, 0, 1]
        assert row[IDX["round_number_norm"]] == .5
        assert row[IDX["cards_remaining_norm"]] == pytest.approx((7-seat)/7)
        left, right = (seat - 1) % 3, (seat + 1) % 3
        shields = status["players"][str(seat)]["resources"]["shields"]
        assert row[IDX["military_balance_left"]] == shields - status["players"][str(left)]["resources"]["shields"]
        assert row[IDX["military_balance_right"]] == shields - status["players"][str(right)]["resources"]["shields"]


def test_seed_roles_and_local_policy_rng_preserve_global_torch_state(status_age1_turn0):
    assert derive_seed(0, "engine", 1, 2) == derive_seed(0, "engine", 1, 2)
    assert derive_seed(0, "engine", 1, 2) != derive_seed(0, "policy", 1, 2)
    model = ActorCritic(90, 92)
    obs = build_obs(status_age1_turn0, 0)
    state = torch.get_rng_state().clone()
    first = [select_action_from_obs(obs, model, "cpu", rng=np.random.default_rng(s)) for s in range(8)]
    torch.testing.assert_close(state, torch.get_rng_state())
    torch.rand(10)
    second = [select_action_from_obs(obs, model, "cpu", rng=np.random.default_rng(s)) for s in range(8)]
    assert first == second


def test_checkpoint_roundtrip_semantics_hidden_size_and_rng(tmp_path):
    model = ActorCritic(90, 92, hidden=16)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "latest.pt"
    meta = {"feature_schema_hash": schema_hash(), "feature_semantics_version": FEATURE_SEMANTICS_VERSION}
    save_checkpoint(path, model, optimizer, 4, {"global_dim": 90, "action_feat_dim": 92, "hidden": 16}, meta)
    before = torch.get_rng_state().clone()
    loaded = load_model(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["config"]["model_architecture_version"] == MODEL_ARCHITECTURE_VERSION
    assert checkpoint["meta"]["model_architecture_version"] == MODEL_ARCHITECTURE_VERSION
    torch.testing.assert_close(before, torch.get_rng_state())
    for a, b in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(a, b)
    inputs, actions = torch.randn(2, 90), torch.randn(2, 28, 92)
    for expected, actual in zip(model.forward_batch(inputs, actions), loaded.forward_batch(inputs, actions)):
        torch.testing.assert_close(expected, actual)
    from sevenwonders_rl.policies.ppo_checkpoint_policy import PPOCheckpointPolicy
    policy = PPOCheckpointPolicy(str(path))
    policy.set_seed(123)
    assert policy.model.global_mlp[0].out_features == 16
    save_checkpoint(path, model, optimizer, 4, {"global_dim": 90, "action_feat_dim": 92, "hidden": 16}, {})
    with pytest.raises(ValueError, match="semantics"):
        load_model(path)
    load_model(path, allow_legacy=True)


def test_old_linear_head_requires_legacy_override_and_preserves_predictions(tmp_path):
    old = ActorCritic(90, 92, hidden=8, architecture_version=1)
    path = tmp_path / "old.pt"
    # Before architecture versioning, even feature-v2 checkpoints had no model version.
    payload = {"state_dict": old.state_dict(),
               "config": {"global_dim": 90, "action_feat_dim": 92, "hidden": 8},
               "meta": {"feature_schema_hash": schema_hash(),
                        "feature_semantics_version": FEATURE_SEMANTICS_VERSION}}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_model(path)
    before = torch.get_rng_state().clone()
    loaded = load_model(path, allow_legacy=True)
    torch.testing.assert_close(before, torch.get_rng_state())
    assert loaded.architecture_version == 1
    inputs, actions = torch.randn(2, 90), torch.randn(2, 28, 92)
    for expected, actual in zip(old.forward_batch(inputs, actions), loaded.forward_batch(inputs, actions)):
        torch.testing.assert_close(expected, actual)
    payload["config"]["model_architecture_version"] = MODEL_ARCHITECTURE_VERSION
    torch.save(payload, path)
    with pytest.raises(ValueError, match="metadata"):
        load_model(path, allow_legacy=True)


def trainer_module():
    path = Path(__file__).parents[1] / "scripts/ppo_training/train_unified.py"
    spec = importlib.util.spec_from_file_location("unified", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_training_configuration_rejected_before_output():
    trainer = trainer_module()
    for args in (["--n-envs", "0"], ["--total-timesteps", "3", "--n-envs", "2"],
                 ["--gamma", "nan"], ["--opponent-baseline", "-1"]):
        with pytest.raises(SystemExit):
            trainer.parse_args(args)
    from sevenwonders_rl.envs.sevenwonders_env import SevenWondersEnv
    with pytest.raises(ValueError, match="end in /io"):
        SevenWondersEnv(io_dir="incorrect-exchange-path")


def test_final_publication_between_read_and_process_exit_is_not_lost(tmp_path, status_age3_turn0):
    from sevenwonders_rl.envs.sevenwonders_env import SevenWondersEnv
    root = Path(__file__).parents[2]
    env = SevenWondersEnv(repo_root=str(root), io_dir=str(tmp_path / "io"))
    env.engine_seed = 0
    before = copy.deepcopy(status_age3_turn0)
    before.update(protocol_version=2)
    before["game"].update(turn=20, phase="playing", engine_seed=0)
    final = copy.deepcopy(before)
    final["game"].update(turn=21, era=4, phase="finished")
    statuses = iter((before, final))
    env._read_status = lambda: next(statuses)
    env.proc = SimpleNamespace(poll=lambda: 0, returncode=0)
    assert env._wait_for_turn_change(20) == final
    env.proc = None
