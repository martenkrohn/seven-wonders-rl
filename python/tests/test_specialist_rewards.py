"""Specialist objective and weight-initialization contracts."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from sevenwonders_rl.rewards import RewardSpec
from sevenwonders_rl.utils.checkpoints import file_hash


def finished():
    return {'game': {'phase': 'finished'}, 'players': {
        str(seat): {'points': {'total': total, 'scientific': science, 'military': military}}
        for seat, total, science, military in [(0, 40, 20, -3), (1, 45, 4, 10), (2, 38, 1, 0)]}}


@pytest.mark.unit
def test_terminal_reward_preserves_margin_and_signed_category():
    state = finished()
    assert RewardSpec(scale=10).terminal_components(state, 0)['training_reward'] == -.5
    sci = RewardSpec('science', 1, 10).terminal_components(state, 0)
    assert sci['base_reward'] == -.5 and sci['specialist_bonus'] == 2
    assert sci['training_reward'] == 1.5
    mil = RewardSpec('military', 1, 10).terminal_components(state, 0)
    assert mil['training_reward'] == -.8 and mil['raw_specialist_bonus'] == -3
    assert RewardSpec('science', 0, 10).terminal_components(state, 0)['training_reward'] == -.5
    assert RewardSpec('military', 2, 10).terminal_components(state, 1)['training_reward'] == 2.5
    state['game']['phase'] = 'playing'
    with pytest.raises(ValueError, match='completed'):
        RewardSpec('science').terminal_components(state, 0)


@pytest.mark.unit
@pytest.mark.parametrize('kwargs', [dict(profile='bad'), dict(specialist_weight=-1),
    dict(specialist_weight=float('nan')), dict(scale=0), dict(scale=float('inf'))])
def test_reward_spec_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RewardSpec(**kwargs)


@pytest.mark.component
def test_initialization_preserves_weights_rng_hidden_size_and_fresh_optimizer(tmp_path):
    from test_selfplay import checkpoint
    from test_training_contracts import trainer_module
    source = tmp_path / 'source.pt'
    expected = checkpoint(source)
    original_hash = file_hash(source)
    trainer = trainer_module()
    args = trainer.parse_args(['--no-observation-memory', '--init-checkpoint', str(source), '--reward-profile', 'science'])
    before = torch.get_rng_state().clone()
    actual, provenance = trainer.initialize_model(args, 102, 99)
    assert actual.training and actual.global_mlp[0].out_features == 8
    assert provenance['sha256'] == original_hash and not provenance['optimizer_restored']
    assert not torch.optim.Adam(actual.parameters()).state
    assert torch.equal(before, torch.get_rng_state())
    for key, tensor in expected.state_dict().items():
        assert torch.equal(tensor, actual.state_dict()[key])
    assert file_hash(source) == original_hash
    with pytest.raises(ValueError, match='dimensions'):
        trainer.initialize_model(args, 101, 99)


@pytest.mark.integration
@pytest.mark.parametrize('profile', ['science', 'military'])
def test_specialist_training_and_bonus_free_holdout(require_engine, tmp_path, profile):
    from test_selfplay import checkpoint
    source = tmp_path / 'source.pt'
    expected = checkpoint(source)
    original_hash = file_hash(source)
    root = Path(__file__).parents[1]
    output = tmp_path / profile
    subprocess.run([sys.executable, 'scripts/ppo_training/train_unified.py',
        '--no-observation-memory',
        '--init-checkpoint', str(source), '--reward-profile', profile,
        '--total-timesteps', '128', '--n-envs', '2', '--rollout-horizon', '32',
        '--snapshot-every', '64', '--save-every', '64', '--eval-every', '128',
        '--eval-games-per-matchup', '1', '--out-dir', str(output)],
        cwd=root, check=True, capture_output=True, text=True, timeout=60)
    config = json.loads((output / 'config.json').read_text())
    assert config['initialization']['sha256'] == original_hash
    assert config['reward_spec']['profile'] == profile
    initial = torch.load(output / 'checkpoint_0.pt', weights_only=True)
    for key, value in expected.state_dict().items():
        assert torch.equal(value, initial['state_dict'][key])
    assert initial['optimizer_state'] is None
    episodes = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    assert episodes
    for episode in episodes:
        parts = episode['reward_components']
        assert parts['reward_profile'] == profile
        own = episode['final_state']['players'][str(episode['agent_seat'])]['raw_points']
        category = 'scientific' if profile == 'science' else 'military'
        assert parts['raw_specialist_bonus'] == own[category]
        assert parts['training_reward'] == (parts['raw_vp_diff'] + own[category]) / 10
    evaluation = json.loads((output / 'eval_128_summary.json').read_text())
    assert evaluation['evaluation_reward_profile'] == 'standard'
    assert evaluation['training_reward_profile'] == profile
    assert evaluation['valid_for_selection']
    assert file_hash(source) == original_hash
    # The standalone evaluator must call its starting model pretrained in all reports.
    holdout = tmp_path / 'holdout'
    subprocess.run([sys.executable, 'scripts/eval_selfplay.py', '--run-dir', str(output),
        '--reference', str(output / 'checkpoint_64.pt'), '--blocks', '1', '--workers', '1',
        '--out-dir', str(holdout)], cwd=root, check=True, capture_output=True, text=True, timeout=60)
    report = (holdout / 'report.txt').read_text()
    assert 'Vortrainiertes Ausgangsmodell' in report
    assert 'Untrainiertes Ausgangsmodell' not in report


@pytest.mark.component
@pytest.mark.parametrize('profile', ['standard', 'science', 'military'])
def test_environment_only_rewards_real_completion_and_never_timeout(tmp_path, monkeypatch, profile):
    from sevenwonders_rl.envs.sevenwonders_env import SevenWondersEnv, EngineError
    env = SevenWondersEnv(repo_root=str(Path(__file__).parents[2]), io_dir=str(tmp_path / 'io'),
                          reward_profile=profile, reward_scale=10)
    monkeypatch.setattr(env, '_build_obs', lambda status: {})
    monkeypatch.setattr(env, '_build_info', lambda status, obs: {'transition_valid': True})
    state = finished()
    state['game']['phase'] = 'playing'
    env._last_status = state
    monkeypatch.setattr(env, '_wait_for_turn_change', lambda **kwargs: state)
    _, reward, terminated, truncated, info = env.wait_next(0)
    assert reward == 0 and not terminated and not truncated and 'reward_components' not in info
    state['game']['phase'] = 'finished'
    _, reward, terminated, truncated, info = env.wait_next(20)
    assert terminated and not truncated
    assert reward == RewardSpec(profile, 1, 10).terminal_components(state, 0)['training_reward']
    state['game']['phase'] = 'playing'
    def fail(**kwargs):
        raise EngineError('timeout', 'test timeout')
    monkeypatch.setattr(env, '_wait_for_turn_change', fail)
    _, reward, terminated, truncated, info = env.wait_next(5)
    assert reward == 0 and truncated and not terminated
    assert not info['transition_valid'] and 'reward_components' not in info


@pytest.mark.component
def test_restart_rejects_changed_initial_weights(tmp_path):
    from test_selfplay import checkpoint
    from test_training_contracts import trainer_module
    from sevenwonders_rl.utils.training_restart import matching_configuration
    source = tmp_path / 'source.pt'
    checkpoint(source)
    trainer = trainer_module()
    args = trainer.parse_args(['--init-checkpoint', str(source)])
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps({**vars(args), 'initialization': {'sha256': file_hash(source)}}))
    matching_configuration(run, args)
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='contents changed'):
        matching_configuration(run, args)


@pytest.mark.component
def test_command_generator_emits_separate_specialists_without_launching(tmp_path, capsys):
    import importlib.util
    script = Path(__file__).parents[1] / 'scripts/ppo_training/specialist_experiment_commands.py'
    spec = importlib.util.spec_from_file_location('specialist_commands', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / 'source.pt'
    checkpoint.touch()
    module.main(['--init-checkpoint', str(checkpoint), '--reference', str(checkpoint), '--prefix', 'runs/example'])
    commands = capsys.readouterr().out
    assert commands.count('--reward-profile science') == 1
    assert commands.count('--reward-profile military') == 1
    assert '--reward-profile standard' not in commands
    assert '--throughput-preset' in commands and '--init-checkpoint' in commands
    assert not (tmp_path / 'runs').exists()


@pytest.mark.integration
@pytest.mark.parametrize('profile', ['science', 'military'])
def test_frozen_standard_opponents_never_admit_learner_snapshots(require_engine, tmp_path, profile):
    from test_selfplay import checkpoint
    source = tmp_path / 'standard.pt'
    checkpoint(source)
    source_hash = file_hash(source)
    output = tmp_path / profile
    root = Path(__file__).parents[1]
    subprocess.run([sys.executable, 'scripts/ppo_training/train_unified.py',
        '--init-checkpoint', str(source), '--frozen-opponents', str(source),
        '--reward-profile', profile, '--total-timesteps', '128', '--n-envs', '2',
        '--rollout-horizon', '16', '--snapshot-every', '32', '--eval-every', '64',
        '--eval-games-per-matchup', '1', '--out-dir', str(output)], cwd=root,
        check=True, capture_output=True, text=True, timeout=60)
    config = json.loads((output / 'config.json').read_text())
    assert config['evaluation_strategy'] == 'fixed_neural_opponents'
    frozen, = config['frozen_opponent_references']
    assert frozen['sha256'] == source_hash
    assert file_hash(frozen['checkpoint']) == source_hash
    assert file_hash(source) == source_hash
    assert len(list(output.glob('checkpoint_*.pt'))) == 5
    selections = [json.loads(line) for line in (output / 'opponents.jsonl').read_text().splitlines()]
    assert len({row['agent_seat'] for row in selections}) == 3
    for row in selections:
        assert len(row['opponents']) == 2
        for opponent in row['opponents'].values():
            assert opponent['checkpoint'] == frozen['checkpoint']
            assert opponent['sha256'] == source_hash
    first = json.loads((output / 'eval_64_summary.json').read_text())
    last = json.loads((output / 'eval_128_summary.json').read_text())
    assert first['valid_for_selection'] and last['valid_for_selection']
    assert first['plan_id'] == last['plan_id'] and last['paired_valid']
    assert last['evaluation_reward_profile'] == 'standard'
    assert [r['checkpoint'] for r in last['references']] == [frozen['checkpoint']]
