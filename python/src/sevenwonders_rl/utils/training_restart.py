"""Serialized, non-destructive restart of local Linux training directories."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import uuid


def read_json(path):
    return json.loads(path.read_text())


def matching_configuration(directory, args):
    saved = read_json(directory / 'config.json')
    ignored = {'out_dir', 'io_root', 'league_dir', 'restart_incomplete', 'skip_completed',
               'eval_workers', 'opponent_cache_size', 'throughput_preset', 'resume'}
    defaults = {'init_checkpoint': None, 'reward_profile': 'standard', 'specialist_weight': 1.0,
                'frozen_opponents': None, 'num_heads': 1, 'prior_strength': 0.0,
                'opponent_population': None, 'population_spec': None,
                'public_information': False, 'observation_memory': False, 'mixed_league': False,
                'milestone_evaluations': False, 'eval_panel': None}
    defaults.update(mixed_recent_weight=.50, mixed_archive_weight=.15, historical_checkpoints=None,
                    evaluation_population=None, evaluation_population_spec=None)
    defaults.update(pair_pfsp=False, pfsp_power=2., pfsp_control_fraction=.10,
                    pfsp_half_life=131_072., pfsp_prior_games=3.)
    defaults.update(continuous=False)
    defaults.update(dashboard_reference=None, dashboard_reference_spec=None)
    changed = [key for key, value in vars(args).items()
               if key not in ignored and saved.get(key, defaults.get(key)) != value]
    if changed:
        raise ValueError('Existing run has different settings: ' + ', '.join(changed) +
                         '. Use a new --out-dir for a different experiment.')
    if getattr(args, 'init_checkpoint', None):
        from sevenwonders_rl.utils.checkpoints import file_hash
        if saved.get('initialization', {}).get('sha256') != file_hash(args.init_checkpoint):
            raise ValueError('Initial checkpoint contents changed; preserve this run and use a new output directory.')
    if getattr(args, 'frozen_opponents', None):
        from sevenwonders_rl.utils.checkpoints import file_hash
        for reference in saved.get('frozen_opponent_references', []):
            if file_hash(reference['source_checkpoint']) != reference['sha256']:
                raise ValueError('Frozen opponent source contents changed; use a new output directory.')
    for key in ('historical_opponent_references', 'evaluation_opponent_references'):
        for reference in saved.get(key, []):
            from sevenwonders_rl.utils.checkpoints import file_hash
            if file_hash(reference['source_checkpoint']) != reference['sha256']:
                raise ValueError('Saved historical/evaluation source changed; use a new output directory.')


def assert_no_legacy_process(directory):
    """Also protect trainers started before directory locks were introduced."""
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if not any(Path(arg).name == 'train_unified.py' for arg in argv):
                continue
            output = None
            for i, arg in enumerate(argv):
                if arg == '--out-dir' and i + 1 < len(argv):
                    output = argv[i + 1]
                elif arg.startswith('--out-dir='):
                    output = arg.split('=', 1)[1]
            if output is None:
                continue
            target = Path(output).expanduser()
            if not target.is_absolute():
                target = (proc / 'cwd').resolve() / target
            if target.resolve() == directory:
                raise RuntimeError(f'Training is still running for {directory} (PID {proc.name}).')
        except (OSError, UnicodeError):
            continue


@contextmanager
def training_directory_lock(directory):
    directory = Path(directory).resolve()
    locks = directory.parent / '.training-locks'
    locks.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode after release: deleting it would allow lock races.
    with (locks / (directory.name + '.lock')).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Training is already running for {directory}.') from exc
        assert_no_legacy_process(directory)
        yield


def prepare_existing_run(directory, args):
    """Return True only when an intact, matching completed run can be skipped."""
    directory = Path(directory).resolve()
    if not directory.exists() or not any(directory.iterdir()):
        return False
    if not (args.restart_incomplete or args.skip_completed):
        return False  # retain the trainer's existing refusal of nonempty output
    result_path = directory / 'result.json'
    try:
        result = read_json(result_path) if result_path.exists() else {}
    except (ValueError, OSError):
        result = {}
    if not isinstance(result, dict):
        result = {}
    config_path = directory / 'config.json'
    if config_path.exists():
        matching_configuration(directory, args)
    elif result.get('status') == 'completed':
        raise ValueError(f'Completed run has no config.json: {directory}; preserve it and use a new --out-dir.')
    complete = (result.get('status') == 'completed'
                and result.get('timestep') == args.total_timesteps
                and (directory / 'latest.pt').is_file()
                and (not args.science_diagnostics or (directory / 'science_episodes.jsonl').is_file()))
    if complete:
        if args.skip_completed:
            print(f'Bereits abgeschlossen – übersprungen: {directory}', flush=True)
            return True
        raise ValueError(f'Run is already completed: {directory}; use --skip-completed or a new --out-dir.')
    if not args.restart_incomplete:
        return False
    saved = read_json(config_path) if config_path.exists() else {}
    leagues = [saved.get('league_dir'), args.league_dir]
    if args.mode == 'selfplay' and any(p and Path(p).expanduser().resolve() != directory for p in leagues):
        raise ValueError('Restart requires the self-play league inside --out-dir; use a fresh run directory and league.')
    archive = directory.parent / 'archive'
    archive.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = archive / f'{directory.name}-{stamp}-{uuid.uuid4().hex[:8]}'
    directory.rename(destination)
    print(f'Unvollständigen Lauf archiviert: {destination}\nFrischer Neustart: {directory}', flush=True)
    return False
