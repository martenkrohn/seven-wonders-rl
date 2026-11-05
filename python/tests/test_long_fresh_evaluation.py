"""Balanced direct plans and separation from old training/evaluation seeds."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def module():
    directory = Path(__file__).resolve().parents[1] / 'scripts/ppo_training'
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location('long_fresh', directory / 'eval_long_training_fresh.py')
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.remove(str(directory))


def test_direct_panel_always_contains_the_other_two_models_and_balances_seats():
    m = module()
    models = {name: dict(checkpoint=name, sha256=name) for name in ('initial', 'quarter', 'candidate')}
    plans = m.direct_plans(models, 130002, 32)
    for name, plan in plans.items():
        assert len(plan) == 192
        assert len({s['engine_seed'] for s in plan}) == 32
        assert {s['agent_seat'] for s in plan} == {0, 1, 2}
        for s in plan:
            assert {s['left']['group'], s['right']['group']} == set(models) - {name}
    assert len({s['engine_seed'] for plan in plans.values() for s in plan}) == 32


def test_fresh_seed_audit_rejects_previously_used_seeds(tmp_path):
    m = module()
    (tmp_path / 'opponents.jsonl').write_text(json.dumps(dict(engine_seed=42)) + '\n')
    assert m.fresh_seed_audit(tmp_path, {43})['overlap'] == 0
    with pytest.raises(ValueError, match='overlap'):
        m.fresh_seed_audit(tmp_path, {42})


@pytest.mark.integration
def test_direct_plan_runs_with_existing_evaluator_and_report(tmp_path, require_engine):
    from test_selfplay import checkpoint
    from sevenwonders_rl.utils.checkpoints import file_hash, load_model
    from sevenwonders_rl.evaluation.runner import evaluate_model
    from sevenwonders_rl.evaluation.selfplay import read_records
    from sevenwonders_rl.evaluation.report import write_report
    m = module()
    models = {}
    for name in ('initial', 'quarter', 'candidate'):
        path = tmp_path / f'{name}.pt'
        checkpoint(path)
        models[name] = dict(checkpoint=str(path), sha256=file_hash(path), label=name)
    plans = m.direct_plans(models, 130501, 1)
    records = {}
    for name in models:
        raw = tmp_path / f'{name}.jsonl'
        result = evaluate_model(load_model(models[name]['checkpoint']), 'cpu', plans[name],
            Path(__file__).resolve().parents[2], tmp_path / 'io' / name, raw,
            action_mode='stochastic', detailed=True)
        assert result['completed'] == 6 and result['valid_for_selection']
        records[name] = read_records(raw)
    write_report(tmp_path, records, dict(seed=130501, model_descriptors=models, references=[]), {})
    assert (tmp_path / 'dashboard.html').is_file()
