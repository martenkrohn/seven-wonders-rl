"""Paired evaluation of one frozen model with and without legal-green priority."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from sevenwonders_rl.evaluation.model_labels import model_label
from sevenwonders_rl.evaluation.report import write_report
from sevenwonders_rl.evaluation.runner import evaluate_model
from sevenwonders_rl.evaluation.selfplay import evaluation_plan, paired_progress, plan_identity, read_records, reference
from sevenwonders_rl.utils.checkpoints import file_hash, load_model, manifest
from sevenwonders_rl.utils.runtime import atomic_json, exchange_io_root, run_io_root
from sevenwonders_rl.utils.training_console import HoldoutConsole


def report_existing(output):
    saved = json.loads((output / 'summary.json').read_text())
    if saved.get('experiment') != 'green_priority':
        raise ValueError('Expected a green-priority experiment summary')
    records = {name: read_records(output / desc['jsonl']) for name, desc in saved['models'].items()}
    paired = {'comparison_1': paired_progress(output / 'candidate.jsonl', output / 'comparison_1.jsonl')}
    metadata = {'seed': saved['seed'], 'references': saved['references'],
                'model_descriptors': saved['models'], 'manifest_available': (output / 'manifest.json').exists(),
                'experiment': 'green_priority', 'plan_id': saved['plan_id']}
    rendered = write_report(output, records, metadata, paired)
    print(rendered, flush=True)
    print(f'Ergebnisse: {output}\nDashboard: {output / "dashboard.html"}', flush=True)
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--reference', nargs=2, metavar='CHECKPOINT')
    parser.add_argument('--blocks', type=int, default=40, help='12 games per block, per variant; default 480 each')
    parser.add_argument('--seed', type=int, default=69001)
    parser.add_argument('--out-dir', help='Must be empty; default is a fresh runs/green-priority-* directory')
    parser.add_argument('--io-root')
    parser.add_argument('--workers', type=int, default=2,
                        help='Independent CPU evaluation workers; 1 runs serially')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    if args.blocks <= 0 or args.seed < 0 or args.workers <= 0:
        parser.error('blocks must be positive and seed nonnegative')
    if args.report_only:
        if not args.out_dir:
            parser.error('--report-only requires --out-dir')
        return report_existing(Path(args.out_dir).resolve())
    if not args.checkpoint or not args.reference:
        parser.error('--checkpoint and two --reference checkpoints are required')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    checkpoint_hash = file_hash(checkpoint)
    model = load_model(checkpoint)
    step = int(torch.load(checkpoint, map_location='cpu', weights_only=True)['timestep'])
    references = [reference(Path(path).expanduser()) for path in args.reference]
    plan = evaluation_plan(references, args.seed, args.blocks)
    plan_id = plan_identity(plan, 'stochastic')
    output = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_io_root('runs', 'green-priority')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Evaluation output directory must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    descriptors = {key: {'checkpoint': str(checkpoint), 'sha256': checkpoint_hash,
        'timestep': step, 'jsonl': key + '.jsonl', 'action_constraint': mode}
        for key, mode in [('comparison_1', 'none'), ('candidate', 'green_priority')]}
    atomic_json(output / 'manifest.json', {**manifest(root, vars(args)),
        'experiment': 'green_priority', 'plan_id': plan_id, 'models': descriptors,
        'references': references, 'action_mode': 'stochastic'})
    io_root = exchange_io_root(args.io_root, output / 'exchange', 'green-priority')
    console = HoldoutConsole(args.seed, args.blocks, len(plan), checkpoint, references, output,
        comparison_label='Grün-Priorität gegen freie Policy – identische Modellgewichte', baseline_key='comparison_1')
    summaries = {}
    try:
        for key, descriptor in descriptors.items():
            console.evaluation_start(model_label(key, descriptor), len(plan))
            summaries[key] = evaluate_model(model, 'cpu', plan, root, io_root / key,
                output / descriptor['jsonl'], action_mode='stochastic', checkpoint_id=checkpoint_hash,
                action_constraint=descriptor['action_constraint'], detailed=True, progress_callback=console.progress,
                workers=args.workers)
            console.evaluation_end(summaries[key])
        if file_hash(checkpoint) != checkpoint_hash:
            raise ValueError('Model checkpoint changed during evaluation')
        summaries.update(seed=args.seed, seed_blocks=args.blocks, references=references,
            models=descriptors, plan_id=plan_id, experiment='green_priority',
            progress=paired_progress(output / 'candidate.jsonl', output / 'comparison_1.jsonl'))
        atomic_json(output / 'summary.json', summaries)
        console.finish(summaries, output)
        report_existing(output)
        if not all(summaries[key]['valid_for_selection'] for key in descriptors):
            raise RuntimeError(f'Incomplete evaluation; inspect {output}')
        if summaries['candidate']['constraint_statistics']['violations']:
            raise RuntimeError('Green-priority rule was violated')
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        shutil.rmtree(io_root)
    return output


if __name__ == '__main__':
    main()
