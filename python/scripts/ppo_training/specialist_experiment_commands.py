"""Print specialist training/holdout shell commands without starting jobs."""
import argparse
from pathlib import Path
import shlex


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-checkpoint', default='runs/science-diagnostic-v1-baseline-seed31/latest.pt')
    parser.add_argument('--reference', nargs='+', default=[
        'runs/science-diagnostic-v1-baseline-seed31/checkpoint_131072.pt',
        'runs/science-diagnostic-v1-baseline-seed31/checkpoint_262144.pt'])
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--total-timesteps', type=int, default=524288)
    parser.add_argument('--specialist-weight', type=float, default=1.0)
    parser.add_argument('--prefix', default='runs/specialists-v2-frozen')
    parser.add_argument('--training-opponent', nargs='+', help='Fixed training checkpoints; defaults to the initial standard model')
    parser.add_argument('--phase', choices=['train', 'evaluate', 'all'], default='train')
    parser.add_argument('--include-control', action='store_true')
    args = parser.parse_args(argv)
    import math
    if (args.seed < 0 or args.total_timesteps < 512 or args.total_timesteps % 512 or
            not math.isfinite(args.specialist_weight) or args.specialist_weight < 0):
        parser.error('Use a nonnegative seed/weight and a positive step budget divisible by 512')
    for path in [args.init_checkpoint, *args.reference, *(args.training_opponent or [])]:
        if not Path(path).is_file():
            parser.error(f'Checkpoint missing: {path}')
    print('#!/usr/bin/env bash\nset -euo pipefail')
    print('# Run from python/. Runs execute sequentially; no exact resume.')
    for phase in (['train', 'evaluate'] if args.phase == 'all' else [args.phase]):
        for profile in ['science', 'military'] + (['standard'] if args.include_control else []):
            output = f'{args.prefix}-{profile}-seed{args.seed}'
            if phase == 'train':
                command = ['.venv/bin/python', 'scripts/ppo_training/train_unified.py',
                    '--mode', 'selfplay', '--init-checkpoint', args.init_checkpoint,
                    '--frozen-opponents', *(args.training_opponent or [args.init_checkpoint]),
                    '--reward-profile', profile, '--specialist-weight', str(args.specialist_weight),
                    '--throughput-preset', '--seed', str(args.seed), '--total-timesteps', str(args.total_timesteps),
                    '--eval-every', '65536', '--eval-games-per-matchup', '12',
                    '--snapshot-every', '4096', '--reward-scale', '10',
                    '--restart-incomplete', '--skip-completed', '--out-dir', output]
            else:
                command = ['.venv/bin/python', 'scripts/eval_selfplay.py', '--run-dir', output,
                    '--reference', *args.reference, '--blocks', '40', '--seed', '79001', '--workers', '2',
                    '--out-dir', output + '-holdout']
            print(shlex.join(command))


if __name__ == '__main__':
    main()
