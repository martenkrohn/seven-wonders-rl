"""Continuous PPO survives clean stops, hard kills and interrupted evaluations."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from sevenwonders_rl.utils.checkpoints import file_hash
from sevenwonders_rl.utils.phase_resume import trim_uncommitted_logs
from test_selfplay import checkpoint
from test_training_contracts import trainer_module


def setup_recipe(tmp_path):
    source = tmp_path / 'external.pt'
    checkpoint(source)
    population, panel = tmp_path / 'population.json', tmp_path / 'panel.json'
    population.write_text(json.dumps({'groups': [dict(name='external', weight=1., checkpoints=[str(source)])]}))
    panel.write_text(json.dumps({'pairs': [['external:0', 'external:0']]}))
    recipe = tmp_path / 'recipe.json'
    settings = dict(mode='selfplay', pair_pfsp=True, observation_memory=True,
        milestone_evaluations=True, evaluation_population=str(population), eval_panel=str(panel),
        n_envs=2, rollout_horizon=16, batch_size=32, minibatch_epochs=1, num_heads=1,
        prior_strength=0., total_timesteps=64, snapshot_every=64, log_every=64,
        seed=851, eval_seed=852, eval_games_per_matchup=1, eval_workers=1, io_root=str(tmp_path / 'io'))
    recipe.write_text(json.dumps(dict(version=1, trainer=settings,
                                     evaluation_source_hashes={str(source): file_hash(source)})))
    output = tmp_path / 'training'
    command = [sys.executable, 'scripts/ppo_training/run_fresh_pair_pfsp.py',
               '--config', str(recipe), '--continuous', '--auto-resume', '--out-dir', str(output),
               '--eval-every', '64', '--save-every', '32']
    return command, output


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def await_condition(process, predicate, log, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            pytest.fail(f'Trainer exited ({process.returncode}): {log.read_text()}')
        time.sleep(.025)
    pytest.fail(f'Trainer timed out: {log.read_text()}')


def stop_cleanly(output):
    result = subprocess.run([sys.executable, 'scripts/ppo_training/run_fresh_pair_pfsp.py',
        '--continuous', '--stop', '--out-dir', str(output)], cwd=Path(__file__).parents[1],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_continuous_task_restarts_same_learner_beyond_original_budget(tmp_path, require_engine):
    command, output = setup_recipe(tmp_path)
    root = Path(__file__).parents[1]
    previous_step = 0
    initial_hash = None
    for session in range(2):
        log = tmp_path / f'session_{session}.log'
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                await_condition(process, lambda: any(r['step'] >= previous_step + 128 for r in
                    read_json(output / 'learning_curve.json').get('evaluations', [])), log)
                stop_cleanly(output)
                process.wait(timeout=30)
                assert process.returncode == 0, log.read_text()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        result = read_json(output / 'result.json')
        assert read_json(output / 'config.json')['total_timesteps'] is None
        saved = torch.load(output / 'latest.pt', weights_only=True)
        assert result['status'] == 'stopped'
        assert result['resumed_from_step'] == previous_step
        assert saved['timestep'] == result['timestep'] > previous_step
        assert saved['optimizer_state']['state']
        assert saved['runtime_state']['updates'] == result['updates']
        state = read_json(output / 'training_status.json')
        assert state['total_timesteps'] is None and state['percent'] is None
        assert state['status'] == 'stopped' and state['next_evaluation'] > result['timestep']
        current_hash = file_hash(output / 'checkpoint_0.pt')
        if initial_hash is not None:
            assert current_hash == initial_hash
            assert read_json(output / 'resume_events.jsonl')  # exactly one event in this test
        initial_hash = current_hash
        previous_step = result['timestep']
    assert previous_step > 64  # recipe's former finite budget has no limiting effect
    curve = read_json(output / 'learning_curve.json')['evaluations']
    assert [r['step'] for r in curve] == sorted({r['step'] for r in curve})
    assert all(r['percent'] is None and r['step'] % 64 == 0 for r in curve)
    assert 'Fortlaufendes Training' in (output / 'dashboard.html').read_text()
    episodes = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    assert sum(r['games'] for r in saved['runtime_state']['pair_pfsp']['results']) == len(episodes)


@pytest.mark.integration
@pytest.mark.parametrize('kill_step', [0, 64])
def test_hard_kill_during_evaluation_recovers_and_repairs_partial_log(tmp_path, require_engine, kill_step):
    command, output = setup_recipe(tmp_path)
    root = Path(__file__).parents[1]
    log = tmp_path / 'killed.log'
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            await_condition(process, lambda: (output / 'evaluations' / f'step_{kill_step:07d}' / 'model.pt').exists(), log)
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    durable = torch.load(output / 'latest.pt', weights_only=True)
    assert durable['timestep'] >= kill_step
    if kill_step == 0:
        (output / 'evaluation_plan.json').unlink()  # also recover a pre-plan interruption
    with (output / 'updates.jsonl').open('a') as stream:
        stream.write('{"update":')
    resumed_log = tmp_path / 'resumed.log'
    with resumed_log.open('w') as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            await_condition(process, lambda: read_json(output / 'training_status.json').get('timestep', 0) >= durable['timestep'] + 64,
                            resumed_log)
            stop_cleanly(output)
            process.wait(timeout=30)
            assert process.returncode == 0, resumed_log.read_text()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    assert read_json(output / 'result.json')['resumed_from_step'] == durable['timestep']
    assert '{"update":' in (output / 'updates.jsonl.rolled_back').read_text()
    assert read_json(output / 'evaluations' / f'step_{kill_step:07d}' / 'summary.json')['valid_for_selection']


@pytest.mark.component
def test_continuous_validation_and_dashboard_window(tmp_path):
    trainer = trainer_module()
    for flags in ([], ['--pair-pfsp'], ['--pair-pfsp', '--milestone-evaluations', '--n-envs', '0']):
        with pytest.raises(SystemExit):
            trainer.parse_args(['--continuous', *flags])
    command, output = setup_recipe(tmp_path)
    settings = json.loads(Path(command[command.index('--config') + 1]).read_text())['trainer']
    base = ['--continuous', '--pair-pfsp', '--milestone-evaluations', '--out-dir', str(output),
            '--evaluation-population', settings['evaluation_population'], '--eval-panel', settings['eval_panel'],
            '--n-envs', '2', '--eval-every', '64']
    args = trainer.parse_args(base)
    assert args.total_timesteps is None
    for invalid in (['--n-envs', '0'], ['--eval-every', '3']):
        with pytest.raises(SystemExit):
            trainer.parse_args([*base, *invalid])
    from sevenwonders_rl.evaluation.milestones import MilestoneEvaluations
    reports = object.__new__(MilestoneEvaluations)
    reports.continuous, reports.entries, reports.interval = True, [], 64
    for i in range(50):
        reports.remember_entry(i)
    assert reports.entries == [0, *range(38, 50)]
    assert reports.next_step(64) == 128 and reports.is_due(128) and not reports.is_due(129)


@pytest.mark.component
def test_resume_only_repairs_truncated_final_log_line(tmp_path):
    path = tmp_path / 'updates.jsonl'
    path.write_text('{"update":1}\n{"update":')
    trim_uncommitted_logs(tmp_path, 32, 1)
    assert path.read_text() == '{"update":1}\n'
    path.write_text('bad complete line\n{"update":1}\n')
    with pytest.raises(json.JSONDecodeError):
        trim_uncommitted_logs(tmp_path, 32, 1)
