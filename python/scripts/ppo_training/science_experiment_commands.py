"""Print a reproducible experiment shell script. Never launches training."""
import argparse
import shlex
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[31, 47, 59])
    parser.add_argument('--total-timesteps', type=int, default=524288)
    parser.add_argument('--prefix', default='runs/science-diagnostic-v1')
    args = parser.parse_args(argv)
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be distinct and nonnegative')
    if args.total_timesteps < 262144 or args.total_timesteps % 512:
        parser.error('budget must be at least 262144 and divisible by 512')
    print('#!/usr/bin/env bash\nset -euo pipefail')
    print('# Run from python/. Completed runs are skipped; incomplete runs are archived and restarted.')
    outputs = []
    for seed in args.seeds:
        for name, entropy, lam in [('baseline', '.01', '.95'), ('entropy', '.03', '.95'),
                                   ('credit', '.01', '1.0'), ('combined', '.03', '1.0')]:
            output = f'{args.prefix}-{name}-seed{seed}'
            outputs.append(output)
            command = [sys.executable, 'scripts/ppo_training/train_unified.py',
                       '--restart-incomplete', '--skip-completed',
                       '--mode', 'selfplay', '--science-diagnostics', '--seed', str(seed),
                       '--ent-coef', entropy, '--gae-lambda', lam,
                       '--total-timesteps', str(args.total_timesteps),
                       '--n-envs', '2', '--rollout-horizon', '256',
                       '--eval-every', '65536', '--eval-games-per-matchup', '12',
                       '--out-dir', output]
            print(shlex.join(command))
    print(shlex.join([sys.executable, 'scripts/ppo_training/analyze_science.py', *outputs,
                      '--out', f'{args.prefix}-summary.json']))


if __name__ == '__main__':
    main()
