"""Restart decisions must preserve results and reject simultaneous writers."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sevenwonders_rl.utils.training_restart import training_directory_lock, prepare_existing_run
from test_training_contracts import trainer_module

pytestmark = pytest.mark.component


def setup_run(tmp_path, status='failed'):
    directory = tmp_path / 'run'
    directory.mkdir()
    args = trainer_module().parse_args(['--out-dir', str(directory), '--total-timesteps', '64',
                                        '--restart-incomplete', '--skip-completed'])
    (directory / 'config.json').write_text(json.dumps({**vars(args), 'league_dir': str(directory)}))
    (directory / 'result.json').write_text(json.dumps({'status': status, 'timestep': 64 if status == 'completed' else 32}))
    (directory / 'latest.pt').write_bytes(b'preserved checkpoint')
    return directory, args


def test_failed_run_archived_and_successful_run_skipped(tmp_path):
    directory, args = setup_run(tmp_path)
    with training_directory_lock(directory):
        assert not prepare_existing_run(directory, args)
        assert not directory.exists()
    archived, = (tmp_path / 'archive').iterdir()
    assert (archived / 'latest.pt').read_bytes() == b'preserved checkpoint'
    directory.mkdir()
    (directory / 'config.json').write_text((archived / 'config.json').read_text())
    (directory / 'latest.pt').write_bytes(b'finished')
    (directory / 'result.json').write_text(json.dumps({'status': 'completed', 'timestep': 64}))
    with training_directory_lock(directory):
        assert prepare_existing_run(directory, args)
    assert (directory / 'latest.pt').read_bytes() == b'finished'
    assert len(list((tmp_path / 'archive').iterdir())) == 1


def test_killed_run_without_result_and_changed_config(tmp_path):
    directory, args = setup_run(tmp_path)
    (directory / 'result.json').unlink()
    args.ent_coef = .03
    with pytest.raises(ValueError, match='different settings: ent_coef'):
        prepare_existing_run(directory, args)
    assert directory.exists() and not (tmp_path / 'archive').exists()
    args.ent_coef = .01
    assert not prepare_existing_run(directory, args)
    assert not directory.exists()


def test_lock_prevents_second_writer_and_releases_on_exception(tmp_path):
    directory = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='injected'):
        with training_directory_lock(directory):
            with pytest.raises(RuntimeError, match='already running'):
                with training_directory_lock(directory):
                    pytest.fail('second writer acquired lock')
            raise RuntimeError('injected')
    with training_directory_lock(directory):
        pass


def test_external_league_and_completed_run_without_skip_are_preserved(tmp_path):
    directory, args = setup_run(tmp_path, 'completed')
    args.skip_completed = False
    with pytest.raises(ValueError, match='already completed'):
        prepare_existing_run(directory, args)
    assert directory.exists()
    (directory / 'result.json').unlink()
    args.league_dir = str(tmp_path / 'external')
    with pytest.raises(ValueError, match='league inside'):
        prepare_existing_run(directory, args)
    assert directory.exists()


def test_early_startup_interruption_before_configuration_is_archived(tmp_path):
    directory, args = setup_run(tmp_path)
    (directory / 'result.json').unlink()
    (directory / 'config.json').unlink()
    assert not prepare_existing_run(directory, args)
    archived, = (tmp_path / 'archive').iterdir()
    assert (archived / 'latest.pt').exists()


def test_legacy_running_process_is_not_archived(tmp_path):
    directory, args = setup_run(tmp_path)
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)',
                                'scripts/ppo_training/train_unified.py', '--out-dir', str(directory)])
    try:
        with pytest.raises(RuntimeError, match='still running'):
            with training_directory_lock(directory):
                prepare_existing_run(directory, args)
        assert directory.exists() and not (tmp_path / 'archive').exists()
    finally:
        process.terminate()
        process.wait(timeout=5)
