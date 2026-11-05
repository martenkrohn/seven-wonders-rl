import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from sevenwonders_rl.data.cards import by_name_strict
from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.policies.action_constraints import constrained_mask, green_build_mask
from sevenwonders_rl.utils.checkpoints import file_hash
from sevenwonders_rl.utils.ppo_utils import select_action_from_obs


@pytest.mark.component
def test_paid_free_and_illegal_green_never_select_wonder_or_discard(status_age1_turn0):
    obs = build_obs(status_age1_turn0, 0)
    obs['hand_ids'][:] = -1
    obs['hand_ids'][:3] = [by_name_strict(n)['id'] for n in ['Workshop', 'Apothecary', 'Baths']]
    obs['action_mask'][:] = 0
    obs['action_mask'][[0, 2, 7, 15, 21]] = 1
    before = copy.deepcopy(obs)
    assert np.flatnonzero(green_build_mask(obs)).tolist() == [0, 15]
    assert np.flatnonzero(constrained_mask(obs, 'green_priority')).tolist() == [0, 15]
    for key in obs:
        np.testing.assert_array_equal(obs[key], before[key])
    assert constrained_mask(obs, 'none') is obs['action_mask']
    with pytest.raises(ValueError, match='Unknown'):
        constrained_mask(obs, 'invalid')


@pytest.mark.component
def test_no_green_preserves_free_policy_actions_and_rng(status_age1_turn0):
    obs = build_obs(status_age1_turn0, 0)
    obs['hand_ids'][:] = by_name_strict('Baths')['id']
    class Model:
        def forward_batch(self, global_obs, features):
            return torch.arange(28, dtype=torch.float32).reshape(1, 28) / 10, torch.zeros(1)
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    model = Model()
    assert constrained_mask(obs, 'green_priority') is obs['action_mask']
    for _ in range(20):
        assert select_action_from_obs(obs, model, 'cpu', rng=a) == select_action_from_obs(
            {**obs, 'action_mask': constrained_mask(obs, 'green_priority')}, model, 'cpu', rng=b)
    assert a.bit_generator.state == b.bit_generator.state


@pytest.mark.integration
def test_short_paired_green_evaluation_cli_and_offline_report(require_engine, tmp_path):
    from test_selfplay import checkpoint
    first, second = tmp_path / 'first.pt', tmp_path / 'second.pt'
    checkpoint(first)
    checkpoint(second)
    original = file_hash(first)
    root = Path(__file__).parents[1]
    output = tmp_path / 'evaluation'
    process = subprocess.run([sys.executable, 'scripts/eval_green_priority.py',
        '--checkpoint', str(first), '--reference', str(first), str(second), '--blocks', '1',
        '--out-dir', str(output), '--io-root', str(tmp_path / 'exchange')], cwd=root,
        check=True, capture_output=True, text=True, timeout=60)
    saved = json.loads((output / 'summary.json').read_text())
    free = [json.loads(line) for line in (output / 'comparison_1.jsonl').read_text().splitlines()]
    forced = [json.loads(line) for line in (output / 'candidate.jsonl').read_text().splitlines()]
    assert len(free) == len(forced) == 12 and saved['progress']['paired_valid']
    assert saved['comparison_1']['completed'] == saved['candidate']['completed'] == 12
    assert saved['candidate']['constraint_statistics']['restricted_decisions'] > 0
    assert saved['candidate']['constraint_statistics']['violations'] == 0
    for a, b in zip(free, forced):
        for key in ['engine_seed', 'agent_seat', 'opponents', 'policy_seeds', 'checkpoint_id']:
            assert a[key] == b[key]
        for event in b['actions']:
            if 'green' in event['legal_build_colors']:
                assert event['color'] == 'green' and event['action'] in ('build_structure', 'build_hand_free')
    assert file_hash(first) == original
    report = (output / 'report.txt').read_text()
    assert 'Freie Policy' in report and 'Grün-Priorität' in report
    assert 'Untrainiertes Ausgangsmodell' not in report
    assert 'Untrainiertes und trainiertes' not in process.stdout
    analysis = json.loads((output / 'analysis.json').read_text())
    assert analysis['models']['candidate']['constraint_statistics']['violations'] == 0
    assert analysis['model_descriptors']['candidate']['sha256'] == analysis['model_descriptors']['comparison_1']['sha256']
    assert (output / 'dashboard.html').is_file()
    first.unlink()
    subprocess.run([sys.executable, 'scripts/eval_green_priority.py', '--report-only', '--out-dir', str(output)],
        cwd=root, check=True, capture_output=True, text=True, timeout=30)
