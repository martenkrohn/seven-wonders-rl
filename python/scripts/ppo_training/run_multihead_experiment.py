"""Run one multi-head experiment, then evaluate before deciding on another run."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

from sevenwonders_rl.selfplay.opponent_pool import read_population
from sevenwonders_rl.utils.checkpoints import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-timesteps', type=int, default=262144)
    parser.add_argument('--blocks', type=int, default=8)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--eval-seed', type=int, default=89001)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--prior-strength', type=float, default=0.0)
    parser.add_argument('--init-checkpoint', default='runs/science-diagnostic-v1-baseline-seed31/latest.pt')
    parser.add_argument('--population', default='configs/multihead_v1_opponents.json')
    parser.add_argument('--out-dir', help='Defaults to runs/multihead-v1-seed<seed>')
    parser.add_argument('--phase', choices=['all', 'train', 'evaluate'], default='all')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print commands without running')
    args = parser.parse_args(argv)
    if args.total_timesteps < 512 or args.total_timesteps % 512:
        parser.error('total-timesteps must be a positive multiple of 512')
    if min(args.blocks, args.workers) < 1 or min(args.seed, args.eval_seed) < 0 or args.num_heads < 2:
        parser.error('Use positive blocks/workers, nonnegative seeds and at least two heads')
    if not math.isfinite(args.prior_strength) or args.prior_strength < 0:
        parser.error('prior-strength must be finite and nonnegative')
    root = Path(__file__).resolve().parents[2]
    # Stable relative paths even if called outside python/.
    def resolve(path):
        value = Path(path).expanduser()
        return value.resolve() if value.is_absolute() else (root / value).resolve()
    output = resolve(args.out_dir or f'runs/multihead-v1-seed{args.seed}')
    holdout = output.with_name(output.name + '-holdout')
    commands = []
    if args.phase in ('all', 'train'):
        initial, population = resolve(args.init_checkpoint), resolve(args.population)
        if output.exists() and any(output.iterdir()):
            parser.error(f'Training output is not empty: {output}; use a new directory or --phase evaluate')
        # Population paths follow the trainer's working directory convention.
        import os
        previous = Path.cwd()
        try:
            os.chdir(root)
            groups = read_population(population)
        finally:
            os.chdir(previous)
        load_model(initial)
        for group in groups:
            for path in group['checkpoints']:
                load_model(path)
        commands.append([sys.executable, 'scripts/ppo_training/train_unified.py',
            '--init-checkpoint', str(initial), '--opponent-population', str(population),
            '--num-heads', str(args.num_heads), '--prior-strength', str(args.prior_strength),
            '--reward-profile', 'standard', '--throughput-preset', '--seed', str(args.seed),
            '--total-timesteps', str(args.total_timesteps), '--eval-every', '65536',
            '--eval-games-per-matchup', '1', '--eval-workers', str(args.workers),
            '--snapshot-every', '16384', '--save-every', '16384', '--out-dir', str(output)])
    if args.phase in ('all', 'evaluate'):
        if holdout.exists() and any(holdout.iterdir()):
            parser.error(f'Holdout output is not empty: {holdout}')
        commands.append([sys.executable, 'scripts/eval_selfplay.py', '--run-dir', str(output),
            '--population-from-run', '--all-opponent-pairs', '--evaluate-heads',
            '--blocks', str(args.blocks), '--seed', str(args.eval_seed),
            '--workers', str(args.workers), '--out-dir', str(holdout)])
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
