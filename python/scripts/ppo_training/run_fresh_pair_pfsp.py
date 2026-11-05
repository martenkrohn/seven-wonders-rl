"""Start a reproducible learner AND own-snapshot league from random weights."""
import argparse
import json
import os
from pathlib import Path

import torch

from train_unified import main as train, parse_args as training_args
from sevenwonders_rl.selfplay.opponent_pool import read_population
from sevenwonders_rl.utils.checkpoints import file_hash, load_model
from sevenwonders_rl.utils.runtime import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/fresh_pair_pfsp_v1.json')
    parser.add_argument('--out-dir')
    parser.add_argument('--total-timesteps', type=int)
    parser.add_argument('--eval-blocks', type=int)
    parser.add_argument('--eval-workers', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--snapshot-every', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--auto-resume', action='store_true', help='Start once, then resume the existing checkpoint on subsequent invocations')
    parser.add_argument('--stop', action='store_true', help='Request a clean stop of a continuous run after its current PPO update/evaluation')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print the resolved trainer arguments without training')
    args = parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parents[2])
    args.out_dir = args.out_dir or ('runs/continuous-pair-pfsp-v1-seed331' if args.continuous else
                                    'runs/fresh-pair-pfsp-v1-seed331')
    output = Path(args.out_dir).expanduser().resolve()
    if args.stop:
        if args.dry_run or args.resume or args.auto_resume:
            parser.error('--stop cannot be combined with start/resume/dry-run')
        if not (output / 'config.json').is_file() or not json.loads((output / 'config.json').read_text()).get('continuous'):
            parser.error('No configured continuous run at this output directory')
        atomic_json(output / 'stop_requested.json', dict(request='stop_after_current_update'))
        print(f'Stopp angefordert: {output}. Das aktuelle Update bzw. die Evaluation wird noch abgeschlossen.', flush=True)
        return output
    torch.set_num_threads(1)
    recipe = json.loads(Path(args.config).read_text())
    if recipe['version'] != 1:
        parser.error('Unsupported recipe version')
    settings = dict(recipe['trainer'])
    if args.continuous:
        if args.total_timesteps is not None:
            parser.error('--continuous has no total-timesteps budget')
        settings.pop('total_timesteps', None)
        settings.update(continuous=True, eval_every=262_144, save_every=512)
    if (settings.get('init_checkpoint') or settings.get('frozen_opponents') or
            settings.get('opponent_population') or settings.get('historical_checkpoints') or
            settings.get('mixed_league') or settings.get('league_dir') or
            not settings.get('pair_pfsp') or not settings.get('observation_memory') or
            settings.get('mode') != 'selfplay' or not settings.get('milestone_evaluations')):
        parser.error('Fresh recipe requires memory, own-snapshot pair-PFSP and milestone evaluation without pretrained training sources')
    for key, value in [('total_timesteps', args.total_timesteps), ('eval_games_per_matchup', args.eval_blocks),
                       ('eval_workers', args.eval_workers), ('seed', args.seed),
                       ('snapshot_every', args.snapshot_every), ('eval_every', args.eval_every),
                       ('save_every', args.save_every)]:
        if value is not None:
            settings[key] = value
    resume = args.resume or (args.auto_resume and (output / 'latest.pt').is_file())
    if args.auto_resume and output.exists() and any(output.iterdir()) and not (output / 'latest.pt').is_file():
        parser.error('Existing directory has no recoverable latest.pt; preserve it and select another --out-dir')
    settings.update(out_dir=args.out_dir, resume=resume)
    evaluation_paths = {p for group in read_population(settings['evaluation_population']) for p in group['checkpoints']}
    hashes = {str(Path(p).resolve()): h for p, h in recipe['evaluation_source_hashes'].items()}
    if evaluation_paths != set(hashes):
        parser.error('Recipe hashes must cover exactly the evaluation-only population')
    for path, digest in hashes.items():
        if file_hash(path) != digest:
            raise ValueError(f'Historical evaluation checkpoint changed: {path}')
        load_model(path)
    command = []
    for key, value in settings.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif value is not None:
            command.extend([flag, str(value)])
    training_args(command)  # Validate before creating a run directory.
    if args.dry_run:
        print(json.dumps(dict(recipe=args.config, trainer_arguments=command,
                              evaluation_source_hashes=hashes), indent=2))
        return command
    return train(command)


if __name__ == '__main__':
    main()
