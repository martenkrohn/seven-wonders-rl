import importlib.util
import json
from pathlib import Path

import pytest
import torch

from sevenwonders_rl.evaluation.runner import evaluate_model
from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.seeding import derive_seed


def detail_module():
    path = Path(__file__).parents[1] / "scripts/ppo_training/eval_ppo_descriptives.py"
    spec = importlib.util.spec_from_file_location("detail_eval", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module  # dataclass annotations resolve through module name
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_evaluators_agree_and_do_not_consume_training_rng(require_engine, tmp_path):
    root = Path(__file__).parents[2]
    model = ActorCritic(90, 92, hidden=16)
    model.train()
    before = torch.get_rng_state().clone()
    specs = [{"engine_seed": derive_seed(123, "evaluation", i), "left": "baseline", "right": "baseline"}
             for i in range(2)]
    output = tmp_path / "fixed.jsonl"
    metrics = evaluate_model(model, "cpu", specs, root, tmp_path / "fixed-io", output)
    assert metrics["completed"] == 2 and metrics["valid_for_selection"]
    assert model.training
    detail = detail_module()
    _, attempts = detail.run_evaluation(model, "cpu", 2, {"baseline": 1.}, 123,
                                       out_dir=tmp_path / "detail", io_root=tmp_path / "detail-io")
    fixed = [json.loads(line) for line in output.read_text().splitlines()]
    for a, b in zip(fixed, attempts):
        for key in ("totals", "win", "vp_tie", "vp_diff", "action_counts"):
            assert a[key] == b[key]
    torch.testing.assert_close(before, torch.get_rng_state())
    assert not list((tmp_path / "detail-io").glob("evaluation-*"))


@pytest.mark.component
def test_discard_cannot_gain_science_or_shields_from_missing_feature_names(status_age1_turn0):
    detail = detail_module()
    class DiscardModel:
        def forward_batch(self, global_obs, features):
            logits = torch.zeros(1, 28)
            logits[0, 21] = 100
            return logits, torch.zeros(1)
    obs = build_obs(status_age1_turn0, 0)
    logger = detail.PPOPolicyLogger("test", DiscardModel(), "cpu")
    assert logger.select_action(obs) == 21
    event = logger.events[-1]
    assert event.is_discard and event.shields_gain == 0 and event.science_gain == 0


@pytest.mark.integration
def test_neural_evaluation_preserves_rng_and_checks_reference_identity(require_engine, tmp_path, monkeypatch):
    import sevenwonders_rl.evaluation.runner as runner
    from sevenwonders_rl.evaluation.selfplay import reference, evaluation_plan
    from test_selfplay import checkpoint
    model = checkpoint(tmp_path / "reference.pt")
    plan = evaluation_plan([reference(tmp_path / "reference.pt")], 17, 1)[:2]
    monkeypatch.setattr(runner, "make_policy", lambda *a, **kw: pytest.fail("Neural evaluation must not build bots"))
    before = torch.get_rng_state().clone()
    metrics = evaluate_model(model, "cpu", plan, Path(__file__).parents[2],
                             tmp_path / "io", tmp_path / "games.jsonl", action_mode="stochastic")
    assert metrics["completed"] == 2
    torch.testing.assert_close(before, torch.get_rng_state())
    plan[0]["left"] = {**plan[0]["left"], "sha256": "invalid"}
    with pytest.raises(ValueError, match="reference changed"):
        evaluate_model(model, "cpu", plan, Path(__file__).parents[2], tmp_path / "bad-io", tmp_path / "bad.jsonl")
    assert model.training
