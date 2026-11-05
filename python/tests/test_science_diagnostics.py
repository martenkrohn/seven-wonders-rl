import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sevenwonders_rl.evaluation.science_diagnostics import ScienceDiagnostics, summarize_games
from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.data.cards import by_name_strict

pytestmark = pytest.mark.component


def transition(status_age1_turn0):
    before = copy.deepcopy(status_age1_turn0)
    player = before['players']['0']
    player['cards_hand'] = ['Workshop', 'Apothecary']
    player['cards_hand_id'] = [by_name_strict(n)['id'] for n in player['cards_hand']]
    player['cards_playable'] = player['cards_hand']
    obs = build_obs(before, 0)
    obs['action_mask'][:] = 0
    obs['action_mask'][[0, 1, 14, 21]] = 1
    after = copy.deepcopy(before)
    after['last_actions'] = {'0': {'status': 'applied', 'executed': 'build_structure'}}
    return before, after, obs


def test_paid_and_free_probability_and_acknowledgements(tmp_path, status_age1_turn0):
    before, after, obs = transition(status_age1_turn0)
    tracker = ScienceDiagnostics(tmp_path / 'science.jsonl')
    tracker.start(0, {'agent_seat': 0}, 0)
    probabilities = np.zeros(28)
    probabilities[[0, 1, 14, 21]] = [.1, .2, .3, .4]
    tracker.observe(0, before, after, obs, 0, probabilities, 512)
    event = tracker.games[0]['decisions'][0]
    assert event['green_built'] and event['green_available']
    assert event['green_probability'] == pytest.approx(.6)
    assert event['policy_step'] == 512
    after['last_actions']['0']['status'] = 'skipped'
    tracker.observe(0, before, after, obs, 0, probabilities, 512)
    assert len(tracker.games[0]['decisions']) == 1
    after['last_actions']['0'] = {'status': 'fallback_discard', 'executed': 'discard'}
    tracker.observe(0, before, after, obs, 0, probabilities, 512)
    assert not tracker.games[0]['decisions'][-1]['green_built']


def test_ages_final_result_ties_and_empty_denominators(tmp_path, status_age1_turn0):
    before, after, obs = transition(status_age1_turn0)
    tracker = ScienceDiagnostics(tmp_path / 'science.jsonl')
    tracker.start(0, {'agent_seat': 0, 'engine_seed': 9}, 0)
    after['game']['era'] = 2
    after['players']['0']['cards_played'] = ['Workshop']
    tracker.observe(0, before, after, obs, 0, np.ones(28) / 28, 0)
    for s, total in enumerate([40, 40, 30]):
        after['players'][str(s)]['points']['total'] = total
    after['players']['0']['points']['scientific'] = 16
    tracker.finish(0, after, 42)
    game = json.loads(tracker.path.read_text())
    assert game['ages'][0]['symbols']['gear'] == 1
    assert game['ages'][0]['green_built_total'] == 1
    assert game['vp_margin'] == 0 and game['win_share'] == .5
    summary = summarize_games([game])
    assert summary['early_green_science_ge10_nonloss'] == 1
    assert summary['early_green_science_ge20_nonloss'] == 0
    assert summary['ages']['2']['green_choice_rate'] is None
    assert summary['no_early_green']['science_mean'] is None
    assert summarize_games([])['win_share'] is None
    # Exercise the real offline CLI, including cohort windows and JSON output.
    (tmp_path / 'science_episodes.jsonl').write_text(json.dumps({**game, 'env': 0, 'episode': 0}) + '\n')
    (tmp_path / 'config.json').write_text(json.dumps({
        'science_diagnostics': True, 'seed': 31, 'ent_coef': .01, 'gae_lambda': .95}))
    output = tmp_path / 'summary.json'
    root = Path(__file__).parents[1]
    subprocess.run([sys.executable, 'scripts/ppo_training/analyze_science.py', str(tmp_path),
                    '--window', '64', '--out', str(output)], cwd=root,
                   check=True, capture_output=True, text=True)
    run = json.loads(output.read_text())['runs'][0]
    assert run['completed_games'] == 1 and run['training_result'] is None
    assert run['first_early_green_nonloss_discoveries']['10']['engine_seed'] == 9
    assert run['windows'][0]['start'] == 0
    assert output.with_suffix('.html').exists()


def test_command_generator_is_print_only_and_factorial():
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, 'scripts/ppo_training/science_experiment_commands.py',
                             '--seeds', '31'], cwd=root, capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    training = [line for line in lines if 'train_unified.py' in line]
    assert len(training) == 4 and all('--science-diagnostics' in line for line in training)
    assert all('--restart-incomplete --skip-completed' in line for line in training)
    assert '--ent-coef .03 --gae-lambda 1.0' in training[-1]
