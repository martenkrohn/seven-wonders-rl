"""Second 2.1M phase: 60/20/20 own-policy pool and separate evaluation opponents."""
import argparse
import json
import os
from pathlib import Path
import torch

from train_unified import main as train
from sevenwonders_rl.utils.checkpoints import file_hash, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='runs/long-mixed-wonders-v2-seed231')
    parser.add_argument('--total-timesteps', type=int, default=2_097_152)
    parser.add_argument('--eval-blocks', type=int, default=8)
    parser.add_argument('--eval-workers', type=int, default=2)
    parser.add_argument('--snapshot-every', type=int, default=131_072)
    parser.add_argument('--pool-lag-steps', type=int, default=65_536)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parents[2])
    torch.set_num_threads(1)
    sources = json.loads(Path('configs/long_mixed_v2_sources.json').read_text())
    for path, digest in sources['hashes'].items():
        if file_hash(path) != digest:
            raise ValueError(f'Prepared checkpoint changed: {path}')
        load_model(path)
    return train([
        '--init-checkpoint', 'runs/long-mixed-wonders-v1-seed131/evaluations/step_2097152/model.pt',
        '--num-heads', '4', '--prior-strength', '32', '--public-information',
        '--opponent-population', 'configs/long_mixed_v2_opponents.json', '--mixed-league',
        '--mixed-recent-weight', '.60', '--mixed-archive-weight', '.20',
        '--historical-checkpoints', *sources['archive'],
        '--evaluation-population', 'configs/long_mixed_v2_evaluation.json',
        '--milestone-evaluations', '--eval-panel', 'configs/long_mixed_v2_eval_panel.json',
        '--seed', '231', '--eval-seed', '149001', '--throughput-preset',
        '--total-timesteps', str(args.total_timesteps), '--out-dir', args.out_dir,
        '--eval-games-per-matchup', str(args.eval_blocks), '--eval-workers', str(args.eval_workers),
        '--snapshot-every', str(args.snapshot_every), '--pool-lag-steps', str(args.pool_lag_steps),
        '--save-every', '8192', '--log-every', '8192', '--pool-max', '80']
        + (['--resume'] if args.resume else []))


if __name__ == '__main__':
    main()
