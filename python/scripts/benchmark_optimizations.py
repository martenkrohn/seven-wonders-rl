"""Paired before/after self-play, evaluation, feature and transport measurements.

Point --before-root at an immutable source tree containing python/src,
python/scripts and build/7Wonders. Both trees use this interpreter and the same
fixtures/checkpoints. Trials run sequentially, with warm-up and alternating order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest_weights(path):
    import torch
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    digest = hashlib.sha256()
    for key, value in sorted(checkpoint['state_dict'].items()):
        digest.update(key.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def worker(task):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root, output = Path(task['root']), Path(task['output'])
    mode = task['mode']
    if mode == 'features':
        from sevenwonders_rl.features import obs_builder
        paths = sorted((Path(task['fixture_root']) / 'tests/fixtures').rglob('*.json'))
        if not paths:
            raise ValueError('No fixture states')
        count = min(task.get('fixture_count', 512), len(paths))
        paths = [paths[i * (len(paths) - 1) // max(count - 1, 1)] for i in range(count)]
        states = [json.loads(p.read_text()) for p in paths]
        def observe(state):
            if hasattr(obs_builder, 'ObservationContext'):
                context = obs_builder.ObservationContext(state)
                return [context.observation(seat) for seat in range(3)]
            return [obs_builder.build_obs(state, seat) for seat in range(3)]
        for state in states[:8]:
            observe(state)
        start = time.perf_counter()
        digest = hashlib.sha256()
        for state in states:
            for obs in observe(state):
                for key, array in sorted(obs.items()):
                    digest.update(key.encode())
                    digest.update(array.dtype.str.encode())
                    digest.update(array.tobytes())
        elapsed = time.perf_counter() - start
        result = dict(seconds=elapsed, observations=3 * count,
                      fingerprint=digest.hexdigest(), units_per_second=3 * count / elapsed)
    elif mode == 'transport':
        import importlib.util
        from sevenwonders_rl.utils.runtime import exchange_io_root
        import shutil
        spec = importlib.util.spec_from_file_location('transport_benchmark', root / 'python/scripts/benchmark_runtime.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        io = exchange_io_root(None, output / 'exchange', 'transport-benchmark')
        try:
            module.transport_trial(io, 2, 2, 123, 1000)
            result = module.transport_trial(io, 20, 4, task['seed'], 1000)
            result['units_per_second'] = result['steps_per_second']
            result['seconds'] = result['wall_seconds']
        finally:
            shutil.rmtree(io)
    elif mode == 'evaluation':
        from sevenwonders_rl.evaluation.runner import evaluate_model
        from sevenwonders_rl.evaluation.selfplay import reference, evaluation_plan
        from sevenwonders_rl.utils.checkpoints import load_model
        from sevenwonders_rl.utils.runtime import exchange_io_root
        import shutil
        model = load_model(task['checkpoint'])
        plan = evaluation_plan([reference(path) for path in task['references']], task['seed'], task['blocks'])
        io = exchange_io_root(None, output / 'exchange', 'evaluation-benchmark')
        kwargs = dict(action_mode='stochastic', detailed=True)
        if task['workers'] != 1:
            kwargs['workers'] = task['workers']
        try:
            # Warm up engine, model and feature caches. Worker startup remains
            # included in the parallel evaluation's measured cost.
            evaluate_model(model, 'cpu', plan[:2], root, io / 'warmup', output / 'warmup.jsonl',
                           action_mode='stochastic', detailed=True)
            start = time.perf_counter()
            summary = evaluate_model(model, 'cpu', plan, root, io / 'measured', output / 'games.jsonl', **kwargs)
            elapsed = time.perf_counter() - start
            records = [json.loads(line) for line in (output / 'games.jsonl').read_text().splitlines()]
            for record in records:
                record.pop('duration_seconds', None)
            fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            result = dict(seconds=elapsed, units_per_second=len(plan) / elapsed,
                          fingerprint=fingerprint, summary=summary)
            if not summary['valid_for_selection']:
                raise RuntimeError('Incomplete evaluation')
        finally:
            shutil.rmtree(io)
    else:
        raise ValueError(mode)
    result['parent_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(output / 'measurement.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-root', type=Path)
    parser.add_argument('--after-root', type=Path, default=ROOT)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--n-envs', type=int, nargs='+', default=[2, 8])
    parser.add_argument('--modes', nargs='+', choices=['training', 'features', 'transport', 'evaluation'],
                        default=['training', 'features', 'transport', 'evaluation'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--references', type=Path, nargs='+')
    parser.add_argument('--blocks', type=int, default=8)
    parser.add_argument('--eval-workers', type=int, default=2)
    parser.add_argument('--worker-task', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_task:
        return worker(json.loads(args.worker_task.read_text()))
    if not args.before_root or not args.out_dir:
        parser.error('--before-root and --out-dir are required')
    if min(args.repeats, args.steps, args.blocks, args.eval_workers, *args.n_envs) < 1:
        parser.error('Counts must be positive')
    if any(512 % n or args.steps % n for n in args.n_envs):
        parser.error('Environment counts must divide 512 and the step budget')
    if 'evaluation' in args.modes and (not args.checkpoint or not args.references):
        parser.error('Evaluation requires common --checkpoint and --references')
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = {'before': args.before_root.resolve(), 'after': args.after_root.resolve()}
    write_json(output / 'config.json', {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                         if k != 'references'}, 'references': [str(p.resolve()) for p in args.references or []]})
    records = []
    def trial(mode, label, repeat, n=2, workers=1, warmup=False):
        root = roots[label]
        target = output / f'{mode}-{label}-n{n}-w{workers}-r{repeat}'
        target.mkdir()
        env = {**os.environ, 'PYTHONPATH': str(root / 'python/src'),
               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        if mode == 'training':
            training = target / 'training'
            cmd = [sys.executable, str(root / 'python/scripts/ppo_training/train_unified.py'),
                   '--mode', 'selfplay', '--seed', str(173 + max(repeat, 0)),
                   '--total-timesteps', str(512 if warmup else args.steps),
                   '--n-envs', str(n), '--rollout-horizon', str(512 // n),
                   '--snapshot-every', '512', '--eval-every', '100000',
                   '--save-every', '100000', '--eval-games-per-matchup', '1', '--out-dir', str(training)]
        else:
            task = dict(mode=mode, root=str(root), output=str(target), seed=901 + max(repeat, 0),
                        fixture_root=str(args.after_root.resolve() / 'python'), blocks=args.blocks,
                        workers=workers, checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                        references=[str(p.resolve()) for p in args.references or []])
            write_json(target / 'task.json', task)
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker-task', str(target / 'task.json')]
        start = time.perf_counter()
        with (target / 'process.log').open('w') as log:
            subprocess.run(cmd, cwd=root / 'python', env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        external = time.perf_counter() - start
        if mode == 'training':
            result = json.loads((training / 'result.json').read_text())
            timing = json.loads((training / 'timing.json').read_text())
            updates = [json.loads(line) for line in (training / 'updates.jsonl').read_text().splitlines()]
            result = dict(seconds=result['elapsed_seconds'], units_per_second=result['steps_per_second'],
                          fingerprint=digest_weights(training / 'latest.pt'),
                          trajectory_hashes=[u['trajectory_sha256'] for u in updates], timing=timing,
                          status=result['status'])
            if result['status'] != 'completed':
                raise RuntimeError('Training failed')
        else:
            result = json.loads((target / 'measurement.json').read_text())
        row = dict(mode=mode, label=label, repeat=repeat, n_envs=n, workers=workers,
                   external_seconds=external, **result)
        if not warmup:
            records.append(row)
            write_json(output / 'trials.json', records)
        print(f'{target.name}: {result["units_per_second"]:.2f}/s', flush=True)
    if 'training' in args.modes:
        for label in roots:
            trial('training', label, -1, warmup=True)
    for mode in args.modes:
        configs = ([(label, n, 1) for n in args.n_envs for label in roots] if mode == 'training' else
                   [('before', 2, 1), ('after', 2, 1)] +
                   ([('after', 2, args.eval_workers)] if mode == 'evaluation' and args.eval_workers > 1 else []))
        for repeat in range(args.repeats):
            for label, n, workers in (configs if repeat % 2 == 0 else configs[::-1]):
                trial(mode, label, repeat, n, workers)
    summaries = []
    for mode, label, n, workers in dict.fromkeys((r['mode'], r['label'], r['n_envs'], r['workers']) for r in records):
        group = [r for r in records if (r['mode'], r['label'], r['n_envs'], r['workers']) == (mode, label, n, workers)]
        equivalent = True
        for row in group:
            baseline = next(r for r in records if r['mode'] == mode and r['label'] == 'before'
                            and r['repeat'] == row['repeat'] and r['n_envs'] == n)
            equivalent &= baseline['fingerprint'] == row['fingerprint']
            if mode == 'training':
                equivalent &= baseline['trajectory_hashes'] == row['trajectory_hashes']
        summaries.append(dict(mode=mode, label=label, n_envs=n, workers=workers,
                              median_units_per_second=statistics.median(r['units_per_second'] for r in group),
                              median_external_seconds=statistics.median(r['external_seconds'] for r in group),
                              equivalent_to_before=equivalent))
    write_json(output / 'summary.json', summaries)
    print(json.dumps(summaries, indent=2))
    if not all(s['equivalent_to_before'] for s in summaries):
        raise SystemExit('Behavior fingerprints differ; inspect before accepting timings')


if __name__ == '__main__':
    main()
