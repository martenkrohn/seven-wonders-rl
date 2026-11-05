"""A single pinned historical model and fair, cached future comparisons."""
import json
from pathlib import Path

import pytest

from sevenwonders_rl.utils.checkpoints import file_hash
from sevenwonders_rl.evaluation.dashboard_reference import read_dashboard_config, reference_entry
from test_selfplay import checkpoint
from test_training_contracts import trainer_module


def pin_reference(tmp_path, source, active):
    historical = tmp_path / 'historical.json'
    historical.write_text(json.dumps({'models': {'candidate': {'win_share': .6}, 'initial': {'win_share': .1}},
                                     'paired_progress': {'initial': {'paired_valid': True}}}))
    config = tmp_path / 'dashboard.json'
    config.write_text(json.dumps(dict(version=1, active_run=str(active),
        reference=dict(model='candidate', label='Fixed reference', report=str(historical),
                       report_sha256=file_hash(historical), checkpoint=str(source), checkpoint_sha256=file_hash(source)))))
    return config, historical


@pytest.mark.component
def test_pinned_model_integrity_and_old_comparisons_are_removed(tmp_path):
    source = tmp_path / 'model.pt'
    checkpoint(source)
    config, historical = pin_reference(tmp_path, source, tmp_path / 'run')
    selected = read_dashboard_config(config, tmp_path)
    entry = reference_entry(selected)
    assert list(entry['report']['models']) == ['candidate']
    assert entry['report']['paired_progress'] == {}
    assert entry['labels']['candidate'] == 'Fixed reference'
    historical.write_text('changed')
    with pytest.raises(ValueError, match='report changed'):
        read_dashboard_config(config, tmp_path)


@pytest.mark.integration
def test_reference_comparison_uses_same_panel_and_cache_survives_resume(tmp_path, require_engine, monkeypatch):
    trainer = trainer_module()
    source = tmp_path / 'model.pt'
    checkpoint(source)
    output = tmp_path / 'training'
    config, historical = pin_reference(tmp_path, source, output)
    original_historical = historical.read_bytes()
    population, panel = tmp_path / 'population.json', tmp_path / 'panel.json'
    population.write_text(json.dumps({'groups': [dict(name='external', weight=1., checkpoints=[str(source)])]}))
    panel.write_text(json.dumps({'pairs': [['external:0', 'external:0']]}))
    args = ['--out-dir', str(output), '--pair-pfsp', '--milestone-evaluations',
        '--dashboard-reference', str(config), '--evaluation-population', str(population), '--eval-panel', str(panel),
        '--total-timesteps', '128', '--n-envs', '2', '--rollout-horizon', '16',
        '--batch-size', '32', '--minibatch-epochs', '1', '--snapshot-every', '32', '--save-every', '32',
        '--eval-games-per-matchup', '1', '--seed', '951']
    import sevenwonders_rl.evaluation.runner as runner
    evaluate = runner.evaluate_model
    calls = []

    def reference_evaluation(*args, **kwargs):
        calls.append(1)
        assert len(calls) == 1, 'Reference must be evaluated once, including across resume'
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, 'evaluate_model', reference_evaluation)
    milestone_evaluation = trainer.MilestoneEvaluations.evaluate

    def interrupt(self, model, step):
        if step == 64:
            raise RuntimeError('controlled interruption')
        return milestone_evaluation(self, model, step)

    monkeypatch.setattr(trainer.MilestoneEvaluations, 'evaluate', interrupt)
    with pytest.raises(RuntimeError, match='controlled interruption'):
        trainer.main(args)
    cached = output / 'reference_evaluation' / 'candidate.jsonl'
    cache_time = cached.stat().st_mtime_ns
    monkeypatch.setattr(trainer.MilestoneEvaluations, 'evaluate', milestone_evaluation)
    trainer.main([*args, '--resume'])
    assert len(calls) == 1 and cached.stat().st_mtime_ns == cache_time
    assert historical.read_bytes() == original_historical
    raw_reference = [json.loads(s) for s in cached.read_text().splitlines()]
    for folder in (output / 'evaluations').glob('step_*'):
        report = json.loads((folder / 'analysis.json').read_text())
        assert set(report['models']) == {'candidate', 'reference'}
        assert list(report['paired_progress']) == ['reference']
        assert report['paired_progress']['reference']['paired_valid']
        candidate = [json.loads(s) for s in (folder / 'candidate.jsonl').read_text().splitlines()]
        assert [(r['engine_seed'], r['agent_seat'], r['opponents']) for r in candidate] == [
               (r['engine_seed'], r['agent_seat'], r['opponents']) for r in raw_reference]
    from test_dashboard import DashboardParser
    parser = DashboardParser(); parser.feed((output / 'dashboard.html').read_text())
    entries = json.loads(parser.payload)
    assert len(entries) == 6  # five new evaluations and precisely one historical reference
    assert all(set(e['report']['models']) == {'candidate', 'reference'} for e in entries[:-1])
    assert set(entries[-1]['report']['models']) == {'candidate'}
