"""One staged 2.1M-transition run with public information and live milestone reports."""
import argparse
from pathlib import Path
import os
import json
import torch

from train_unified import main as train
from sevenwonders_rl.utils.checkpoints import file_hash, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='runs/long-mixed-wonders-v1-seed131')
    parser.add_argument('--total-timesteps', type=int, default=2_097_152)
    parser.add_argument('--eval-blocks', type=int, default=8)
    parser.add_argument('--eval-workers', type=int, default=2)
    parser.add_argument('--snapshot-every', type=int, default=131_072)
    parser.add_argument('--pool-lag-steps', type=int, default=65_536)
    parser.add_argument('--resume', action='store_true', help='Continue latest complete update; restart unfinished games')
    args = parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parents[2])
    torch.set_num_threads(1)
    sources = json.loads(Path('configs/long_mixed_v1_source_hashes.json').read_text())
    for path, expected in sources.items():
        if file_hash(path) != expected:
            raise ValueError(f'Prepared source checkpoint changed: {path}')
        load_model(path)  # Validate every checkpoint before creating a training directory.
    return train([
        '--mode', 'selfplay', '--init-checkpoint', 'runs/multihead-v2-priors-seed31/latest.pt',
        '--num-heads', '4', '--prior-strength', '32', '--public-information',
        '--opponent-population', 'configs/long_mixed_v1_opponents.json', '--mixed-league',
        '--milestone-evaluations', '--eval-panel', 'configs/long_mixed_v1_eval_panel.json',
        '--seed', '131', '--eval-seed', '119001', '--throughput-preset',
        '--total-timesteps', str(args.total_timesteps), '--out-dir', args.out_dir,
        '--eval-games-per-matchup', str(args.eval_blocks), '--eval-workers', str(args.eval_workers),
        '--snapshot-every', str(args.snapshot_every), '--pool-lag-steps', str(args.pool_lag_steps),
        '--save-every', '8192', '--log-every', '8192', '--pool-max', '80']
        + (['--resume'] if args.resume else []))


if __name__ == '__main__':
    main()
