"""Ordered-pair matchmaking, learning feedback boundaries and durable resume."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from sevenwonders_rl.selfplay.opponent_pool import OpponentPool
from sevenwonders_rl.selfplay.pair_pfsp import PairPFSPPool
from sevenwonders_rl.utils.checkpoints import load_model
from test_training_contracts import trainer_module
from test_selfplay import checkpoint


def pool_at(tmp_path, steps=(0, 100), **kwargs):
    for step in steps:
        (tmp_path / f'checkpoint_{step}.pt').touch()
    return PairPFSPPool(OpponentPool(str(tmp_path), auto_refresh=False), **kwargs)


@pytest.mark.component
def test_neutral_start_new_pairs_and_future_snapshot_filter(tmp_path):
    pool = pool_at(tmp_path)
    pool.league.max_size = 1
    pairs, rates, counts, _, probabilities = pool.distribution(0)
    assert [(a[0], b[0]) for a, b in pairs] == [(0, 0)]
    np.testing.assert_allclose(rates, [1/3])
    np.testing.assert_array_equal(counts, [0])
    np.testing.assert_array_equal(probabilities, [1])
    pool.league.max_size = 50
    for _ in range(30):
        pool.record_result(0, 0, [90, 20, 30], 0, 0)
    pairs, rates, counts, _, probabilities = pool.distribution(100)
    assert [(a[0], b[0]) for a, b in pairs] == [(0, 0), (0, 100), (100, 0), (100, 100)]
    assert rates[0] > .9 and all(r == pytest.approx(1/3) for r in rates[1:])
    assert counts[0] > 0 and not counts[1:].any()
    assert probabilities[0] < min(probabilities[1:])
    assert min(probabilities) >= pool.control_fraction / 4
    assert sum(probabilities) == pytest.approx(1)


@pytest.mark.component
def test_ordered_pairs_ties_decay_and_immutable_checkpoint_state(tmp_path):
    pool = pool_at(tmp_path, half_life=100)
    pool.record_result(0, 100, [20, 20, 10], 0, 100)
    pool.record_result(100, 0, [10, 20, 30], 0, 100)
    _, rates, counts, _, _ = pool.distribution(100)
    assert counts[1] == counts[2] == 1
    assert rates[1] == pytest.approx(1.5 / 4)
    assert rates[2] == pytest.approx(1. / 4)
    _, later_rates, later_counts, _, _ = pool.distribution(200)
    assert later_counts[1] == .5
    assert abs(later_rates[1] - 1/3) < abs(rates[1] - 1/3)
    saved = pool.state_dict()
    restored = pool_at(tmp_path, half_life=100)
    restored.load_state_dict(saved)
    a, b = np.random.default_rng(97), np.random.default_rng(97)
    assert [pool.sample_pair(a, 200) for _ in range(30)] == [restored.sample_pair(b, 200) for _ in range(30)]
    pool.record_result(0, 100, [10, 10, 10], 0, 200)
    assert saved == restored.state_dict() and saved != pool.state_dict()
    assert pool._results[(0, 100)]['score'] == pytest.approx(.25 + 1/3)
    broken = json.loads(json.dumps(saved))
    broken['results'][0]['score'] = -1
    with pytest.raises(ValueError, match='Invalid'):
        restored.load_state_dict(broken)
    with pytest.raises(ValueError, match='Incompatible'):
        pool_at(tmp_path, power=3).load_state_dict(saved)


@pytest.mark.component
def test_feedback_changes_sampling_and_uniform_controls_remain(tmp_path):
    pool = pool_at(tmp_path)
    for _ in range(100):
        pool.record_result(0, 100, [30, 10, 10], 0, 100)
        pool.record_result(100, 0, [10, 30, 20], 0, 100)
    draws = [pool.sample_pair(np.random.default_rng(seed), 100) for seed in range(2000)]
    easy = sum(m['pair_left_step'] == 0 and m['pair_right_step'] == 100 for _, _, m in draws)
    hard = sum(m['pair_left_step'] == 100 and m['pair_right_step'] == 0 for _, _, m in draws)
    assert hard > 10 * easy > 0
    assert 130 < sum(m['pair_selection'] == 'uniform_control' for _, _, m in draws) < 270
    assert all(m['pair_probability'] >= .1 / 4 for _, _, m in draws)


@pytest.mark.component
@pytest.mark.parametrize('seat', [0, 1, 2])
def test_pair_assignment_uses_relative_left_and_right(tmp_path, monkeypatch, seat):
    trainer = trainer_module()
    pool = pool_at(tmp_path)
    left, right = str(tmp_path / 'checkpoint_0.pt'), str(tmp_path / 'checkpoint_100.pt')
    monkeypatch.setattr(pool, 'sample_pair', lambda rng, step: (left, right, {'group': 'pair_pfsp'}))
    monkeypatch.setattr(trainer, 'PPOCheckpointPolicy', lambda path, **kwargs: path)
    policies = trainer.sample_selfplay_opponents(pool, np.random.default_rng(1), 100, seat)
    assert policies[(seat - 1) % 3].policy == left
    assert policies[(seat + 1) % 3].policy == right


@pytest.mark.component
def test_pair_pfsp_rejects_incompatible_training_sources():
    trainer = trainer_module()
    for flags in (['--mode', 'fixed'], ['--mixed-league'], ['--league-dir', '/tmp/other'],
                  ['--pfsp-control-fraction', '0'], ['--pfsp-half-life', '-1'], ['--pfsp-power', 'nan']):
        with pytest.raises(SystemExit):
            trainer.parse_args(['--pair-pfsp', *flags])


@pytest.mark.component
def test_fresh_recipe_validates_external_hashes_without_starting_training(tmp_path):
    from sevenwonders_rl.utils.checkpoints import file_hash
    source = tmp_path / 'evaluation.pt'
    checkpoint(source)
    population, panel = tmp_path / 'evaluation.json', tmp_path / 'panel.json'
    population.write_text(json.dumps({'groups': [dict(name='external', weight=1., checkpoints=[str(source)])]}))
    panel.write_text(json.dumps({'pairs': [['external:0', 'external:0']]}))
    recipe = tmp_path / 'recipe.json'
    settings = dict(mode='selfplay', pair_pfsp=True, observation_memory=True,
                    milestone_evaluations=True, evaluation_population=str(population), eval_panel=str(panel))
    content = dict(version=1, trainer=settings, evaluation_source_hashes={str(source): file_hash(source)})
    recipe.write_text(json.dumps(content))
    output = tmp_path / 'should_not_exist'
    command = [sys.executable, 'scripts/ppo_training/run_fresh_pair_pfsp.py',
               '--config', str(recipe), '--out-dir', str(output), '--dry-run']
    root = Path(__file__).parents[1]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)['trainer_arguments']
    assert '--pair-pfsp' in resolved and '--init-checkpoint' not in resolved
    assert not output.exists()
    content['evaluation_source_hashes'][str(source)] = 'changed'
    recipe.write_text(json.dumps(content))
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'checkpoint changed' in result.stderr
    assert not output.exists()


@pytest.mark.integration
def test_fresh_training_feedback_external_evaluation_dashboard_and_resume(tmp_path, require_engine, monkeypatch):
    trainer = trainer_module()
    source = tmp_path / 'evaluation_only.pt'
    checkpoint(source)
    population, panel = tmp_path / 'evaluation.json', tmp_path / 'panel.json'
    population.write_text(json.dumps({'groups': [dict(name='control', weight=1., checkpoints=[str(source)])]}))
    panel.write_text(json.dumps({'pairs': [['control:0', 'control:0']]}))
    output = tmp_path / 'training'
    args = ['--out-dir', str(output), '--pair-pfsp', '--total-timesteps', '256',
        '--n-envs', '2', '--rollout-horizon', '16', '--batch-size', '32', '--minibatch-epochs', '1',
        '--num-heads', '4', '--prior-strength', '32', '--snapshot-every', '64', '--save-every', '32',
        '--evaluation-population', str(population), '--eval-panel', str(panel),
        '--milestone-evaluations', '--eval-games-per-matchup', '1', '--seed', '751']
    original = trainer.MilestoneEvaluations.evaluate

    def interrupted(self, model, step):
        if step == 128:
            raise RuntimeError('controlled milestone interruption')
        return original(self, model, step)

    monkeypatch.setattr(trainer.MilestoneEvaluations, 'evaluate', interrupted)
    with pytest.raises(RuntimeError, match='controlled milestone'):
        trainer.main(args)
    durable = torch.load(output / 'latest.pt', weights_only=True)
    assert durable['timestep'] == 128 and durable['runtime_state']['pair_pfsp']['results']
    # A non-authoritative dashboard report must never override durable statistics.
    (output / 'league_status.json').write_text(json.dumps({'method': 'ordered_pair_pfsp', 'timestep': 9999}))
    monkeypatch.setattr(trainer.MilestoneEvaluations, 'evaluate', original)
    trainer.main([*args, '--resume'])
    config = json.loads((output / 'config.json').read_text())
    assert config['initialization']['kind'] == 'random' and config['init_checkpoint'] is None
    assert not config['frozen_opponents'] and not config['population_spec']
    assert load_model(output / 'latest.pt').memory_enabled
    selections = [json.loads(line) for line in (output / 'opponents.jsonl').read_text().splitlines()]
    for game in selections:
        seat = game['agent_seat']
        opponents = game['opponents']
        left, right = opponents[str((seat - 1) % 3)], opponents[str((seat + 1) % 3)]
        assert all(Path(o['checkpoint']).parent == output for o in opponents.values())
        assert Path(left['checkpoint']).name == f"checkpoint_{left['pair_left_step']}.pt"
        assert Path(right['checkpoint']).name == f"checkpoint_{right['pair_right_step']}.pt"
        assert left['pair_left_step'] <= game['selection_step']
        assert right['pair_right_step'] <= game['selection_step']
    state = torch.load(output / 'latest.pt', weights_only=True)['runtime_state']['pair_pfsp']
    completed = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    assert sum(r['games'] for r in state['results']) == len(completed) > 0
    assert any((r['left'], r['right']) != (0, 0) for r in state['results'])
    report = json.loads((output / 'league_status.json').read_text())
    assert sum(r['probability'] for r in report['pairs']) == pytest.approx(1)
    curve = json.loads((output / 'learning_curve.json').read_text())['evaluations']
    assert [r['step'] for r in curve] == [0, 64, 128, 192, 256]
    assert 'Trainingsliga' in (output / 'dashboard.html').read_text()
