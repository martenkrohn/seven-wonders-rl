import copy
from collections import OrderedDict
import json

import numpy as np
import pytest
import torch

from sevenwonders_rl.evaluation.selfplay import evaluation_plan, paired_progress, reference
from sevenwonders_rl.selfplay.opponent_pool import OpponentPool
from sevenwonders_rl.utils.checkpoints import save_checkpoint, FEATURE_SEMANTICS_VERSION
from sevenwonders_rl.utils.ppo_model import ActorCritic
from test_training_contracts import trainer_module

pytestmark = pytest.mark.component


def checkpoint(path):
    model = ActorCritic(90, 92, hidden=8)
    save_checkpoint(path, model, None, 0, {"global_dim": 90, "action_feat_dim": 92, "hidden": 8},
                    {"feature_semantics_version": FEATURE_SEMANTICS_VERSION})
    return model


def test_neural_only_start_with_one_snapshot_and_fatal_load_errors(tmp_path, monkeypatch):
    trainer = trainer_module()
    monkeypatch.setattr(trainer, "make_policy", lambda *a, **kw: pytest.fail("No bots in self-play"))
    path = tmp_path / "checkpoint_0.pt"
    model = checkpoint(path)
    original = copy.deepcopy(model.state_dict())
    pool = OpponentPool(str(tmp_path), min_step_lag=99999)
    cache = OrderedDict()
    policies = trainer.sample_selfplay_opponents(pool, np.random.default_rng(0), 0, 1, cache)
    for seat in (0, 2):
        assert policies[seat].name == "ppo_snapshot"
        assert policies[seat].policy.model is not model
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    for key, value in original.items():
        torch.testing.assert_close(policies[0].policy.model.state_dict()[key], value)
    path.unlink()
    with pytest.raises(ValueError, match="initial neural"):
        trainer.sample_selfplay_opponents(pool, np.random.default_rng(0), 0, 1, cache)
    path.write_bytes(b"broken checkpoint")
    with pytest.raises(RuntimeError, match="Cannot load neural opponent.*checkpoint_0"):
        trainer.sample_selfplay_opponents(pool, np.random.default_rng(0), 0, 1)


def test_plan_balances_seats_and_opponent_orders_and_paired_uncertainty(tmp_path):
    refs = [{"checkpoint": "a", "sha256": "A"}, {"checkpoint": "b", "sha256": "B"}]
    plan = evaluation_plan(refs, 42, 3)
    assert len(plan) == 36
    assert {p["engine_seed"] for p in plan}.isdisjoint(p["engine_seed"] for p in evaluation_plan(refs, 9001, 3))
    for seed in {p["engine_seed"] for p in plan}:
        block = [p for p in plan if p["engine_seed"] == seed]
        assert [p["agent_seat"] for p in block].count(0) == 4
        assert len({(p["left"]["checkpoint"], p["right"]["checkpoint"]) for p in block}) == 4
    baseline = [{**p, "status": "completed", "opponents": {"left": p["left"], "right": p["right"]},
                 "vp_diff": 0, "win_share": 0.} for p in plan]
    candidate = [{**row, "vp_diff": 2, "win_share": .5} for row in baseline]
    initial_path, candidate_path = tmp_path / "initial.jsonl", tmp_path / "candidate.jsonl"
    def write(path, records):
        path.write_text("\n".join(json.dumps(row) for row in records))
    write(initial_path, baseline)
    write(candidate_path, candidate)
    progress = paired_progress(candidate_path, initial_path)
    assert progress["seed_blocks"] == 3
    assert progress["vp_diff_gain_ci95"] == [2., 2.]
    assert progress["win_share_gain"] == .5
    candidate[0]["status"] = "failed"
    write(candidate_path, candidate)
    assert not paired_progress(candidate_path, initial_path)["paired_valid"]
    write(candidate_path, candidate[:-1])
    with pytest.raises(ValueError, match="plans differ"):
        paired_progress(candidate_path, initial_path)
