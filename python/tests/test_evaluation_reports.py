import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sevenwonders_rl.evaluation.telemetry import player_snapshot, distribution, category_statistics, CATEGORIES
from sevenwonders_rl.evaluation.report import summarize_model, write_report, block_intervals
from sevenwonders_rl.evaluation.runner import evaluate_model
from sevenwonders_rl.evaluation.selfplay import reference, evaluation_plan
from sevenwonders_rl.utils.training_console import category_lines


@pytest.mark.unit
@pytest.mark.parametrize('coins', [0, 2, 3, 11])
def test_coins_separate_without_changing_score_and_input(coins):
    player = {'points': {'civilian': 4, 'commercial': 6 + coins // 3, 'military': -2,
                         'scientific': 10, 'guild': 3, 'wonder': 7, 'total': 28 + coins // 3},
              'resources': {'coins': coins}, 'cards_played': ['Lumber Yard']}
    original = copy.deepcopy(player)
    state = player_snapshot(player)
    assert state['points']['yellow'] == 6
    assert state['points']['treasury'] == coins // 3
    assert sum(state['points'][key] for key in CATEGORIES) == state['points']['total']
    assert player == original


@pytest.mark.unit
def test_science_distribution_and_missing_coins_are_honest():
    stats = distribution([0] * 8 + [10, 40])
    assert stats['mean'] == 5 and stats['median'] == 0
    assert stats['zero_fraction'] == .8 and stats['positive_mean'] == 25
    assert stats['p90'] == pytest.approx(13)
    record = {'status': 'completed', 'agent_seat': 0,
              'points_breakdown': {'civilian': 4, 'commercial': 8, 'scientific': 0, 'total': 12}}
    categories = category_statistics([record])
    assert categories['yellow'] is None and categories['treasury'] is None
    assert categories['commercial_combined']['mean'] == 8
    text = '\n'.join(category_lines(categories))
    assert 'nicht erfasst' in text and 'Handel + Münzen' in text
    assert distribution([]) is None


@pytest.mark.unit
def test_correlated_rotations_do_not_narrow_seed_block_intervals():
    rows = [{'status': 'completed', 'engine_seed': s, 'win_share': float(s), 'vp_diff': 2*s-1}
            for s in (0, 1, 2)]
    assert block_intervals(rows) == block_intervals([r for r in rows for _ in range(12)])


@pytest.mark.integration
def test_detail_preserves_games_rng_and_final_age_points(require_engine, tmp_path):
    from test_selfplay import checkpoint
    model = checkpoint(tmp_path / 'model.pt')
    plan = evaluation_plan([reference(tmp_path / 'model.pt')], 923, 1)
    root = Path(__file__).parents[2]
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    results = {}
    for name, detail in [('compact', False), ('detailed', True)]:
        evaluate_model(model, 'cpu', plan, root, tmp_path / name, tmp_path / f'{name}.jsonl',
                       action_mode='stochastic', detailed=detail)
        results[name] = [json.loads(line) for line in (tmp_path / f'{name}.jsonl').read_text().splitlines()]
    for compact, detailed in zip(results['compact'], results['detailed']):
        for key in ('totals', 'win_share', 'vp_diff', 'points_breakdown', 'action_counts'):
            assert compact[key] == detailed[key]
        assert [s['completed_age'] for s in detailed['age_states']] == [1, 2, 3]
        assert [s['game']['era'] for s in detailed['age_states']] == [2, 3, 4]
        assert detailed['age_states'][-1] == detailed['final_state']
        assert len(detailed['actions']) >= 18
        assert all(e['execution'] == 'applied' for e in detailed['actions'])
        for player in detailed['final_state']['players'].values():
            assert sum(player['points'][k] for k in CATEGORIES) == player['points']['total']
    torch.testing.assert_close(torch_before, torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_before[0] == after[0]
    np.testing.assert_array_equal(numpy_before[1], after[1])
    assert numpy_before[2:] == after[2:]
    analysis = summarize_model(results['detailed'])
    assert analysis['coverage']['complete_age_sequences'] == 3
    assert analysis['coverage']['unknown_built_cards'] == 0
    assert all(c['selection_fraction'] is None or c['selection_fraction'] <= 1 for c in analysis['cards'])
    destination = tmp_path / 'report'
    destination.mkdir()
    text = write_report(destination, {'initial': results['compact'], 'candidate': results['detailed']}, {}, {})
    assert 'P10–P90' in text and 'GEBAUTE KARTEN' in text and 'EINZELNE KARTEN' in text
    for name in ('report.md', 'report.txt', 'dashboard.html', 'analysis.json', 'categories.csv', 'cards.csv', 'games.jsonl'):
        assert (destination / name).stat().st_size > 0
