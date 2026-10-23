"""Neural-only holdout evaluation, paired against the run's untrained policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from sevenwonders_rl.evaluation.runner import evaluate_model
from sevenwonders_rl.evaluation.selfplay import evaluation_plan, paired_progress, reference
from sevenwonders_rl.utils.checkpoints import load_model, file_hash
from sevenwonders_rl.utils.runtime import atomic_json, exchange_io_root
from sevenwonders_rl.utils.training_console import HoldoutConsole
from sevenwonders_rl.evaluation.selfplay import read_records
from sevenwonders_rl.evaluation.report import write_report
from sevenwonders_rl.utils.checkpoints import manifest
from sevenwonders_rl.evaluation.model_labels import model_label


def report_existing(output):
    saved = json.loads((output / 'summary.json').read_text())
    descriptors = saved.get('models') or {
        'initial': {'jsonl': 'initial.jsonl'},
        'candidate': {'jsonl': 'candidate.jsonl', **{k: v for k, v in saved.get('candidate', {}).items()
                                                   if k in ('checkpoint', 'sha256')}}}
    for name, desc in descriptors.items():
        path = Path(desc.get('checkpoint', ''))
        result = path.parent / 'result.json'
        if name == 'candidate' and desc.get('timestep') is None and path.name == 'latest.pt' and result.exists():
            state = json.loads(result.read_text())
            if state.get('status') == 'completed' and path.exists() and file_hash(path) == desc.get('sha256'):
                desc['timestep'] = state['timestep']
    records = {name: read_records(output / desc['jsonl']) for name, desc in descriptors.items()}
    paired = {name: paired_progress(output / descriptors['candidate']['jsonl'], output / desc['jsonl'])
              for name, desc in descriptors.items() if name != 'candidate'}
    metadata = {'seed': saved.get('seed'), 'references': saved.get('references', []),
                'manifest_available': (output / 'manifest.json').exists(),
                'model_descriptors': {k: v for k, v in descriptors.items() if 'checkpoint' in v}}
    rendered = write_report(output, records, metadata, paired)
    print(rendered, flush=True)
    print(f"Berichte gespeichert: {output}\n"
          'dashboard.html · report.txt · report.md · analysis.json · categories.csv · cards.csv · games.jsonl', flush=True)
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint", help="Defaults to the run's latest.pt")
    parser.add_argument("--reference", nargs="+", help="Optional frozen trained opponents from this run")
    parser.add_argument("--blocks", type=int, default=20, help="Independent seeds; 12 games per block with two references")
    parser.add_argument("--seed", type=int, default=9001, help="Separate from model-selection seeds")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--io-root")
    parser.add_argument("--action-mode", choices=["stochastic", "greedy"], default="stochastic")
    parser.add_argument("--opponent-inference", choices=["batched", "serial"], default="batched")
    parser.add_argument('--compare', nargs='+', help='Additional checkpoints evaluated on the identical plan')
    parser.add_argument('--report-only', action='store_true', help='Render existing results without running games')
    args = parser.parse_args(argv)
    if args.blocks <= 0 or args.seed < 0:
        parser.error("blocks must be positive and seed nonnegative")
    output = Path(args.out_dir).resolve()
    if args.report_only:
        return report_existing(output)
    if not args.run_dir:
        parser.error('--run-dir is required unless --report-only is used')
    torch.set_num_threads(1)
    run_dir = Path(args.run_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    stored = json.loads((run_dir / "references.json").read_text())["references"]
    references = [reference(path) for path in args.reference] if args.reference else stored
    plan = evaluation_plan(references, args.seed, args.blocks)
    candidate = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "latest.pt"
    initial = Path(stored[0]["checkpoint"])
    if file_hash(initial) != stored[0]["sha256"]:
        raise ValueError("Initial reference changed")
    io_root = exchange_io_root(args.io_root, output / "exchange", "holdout")
    root = Path(__file__).resolve().parents[2]
    summaries = {}
    models = {'initial': {'checkpoint': str(initial), 'jsonl': 'initial.jsonl', 'sha256': file_hash(initial)}}
    for i, path in enumerate(args.compare or []):
        path = Path(path).expanduser().resolve()
        models[f'comparison_{i + 1}'] = {'checkpoint': str(path), 'jsonl': f'comparison_{i + 1}.jsonl', 'sha256': file_hash(path)}
    models['candidate'] = {'checkpoint': str(candidate), 'jsonl': 'candidate.jsonl', 'sha256': file_hash(candidate)}
    atomic_json(output / 'manifest.json', {**manifest(root, vars(args)),
                'telemetry_schema_version': 1,
                'card_database_sha256': file_hash(root / 'python/src/sevenwonders_rl/data/card_db.json')})
    console = HoldoutConsole(args.seed, args.blocks, len(plan), candidate, references, output)
    try:
        for name, descriptor in models.items():
            path = Path(descriptor['checkpoint'])
            descriptor['timestep'] = int(torch.load(path, map_location='cpu', weights_only=True)['timestep'])
            console.evaluation_start(model_label(name, descriptor), len(plan))
            summaries[name] = evaluate_model(load_model(path), "cpu", plan, root,
                io_root / name, output / descriptor['jsonl'], action_mode=args.action_mode,
                checkpoint_id=descriptor['sha256'], batch_opponents=args.opponent_inference == "batched",
                detailed=True, progress_callback=console.progress)
            console.evaluation_end(summaries[name])
        summaries["progress"] = paired_progress(output / "candidate.jsonl", output / "initial.jsonl")
        summaries.update(seed=args.seed, seed_blocks=args.blocks, references=references,
                         models=models,
                         io_root=str(io_root), opponent_inference=args.opponent_inference,
                         candidate= {**summaries["candidate"], "checkpoint": str(candidate),
                                     "sha256": file_hash(candidate)})
        atomic_json(output / "summary.json", summaries)
        console.finish(summaries, output)
        report_existing(output)
    finally:
        shutil.rmtree(io_root)
    if not all(summaries[name]["valid_for_selection"] for name in models):
        raise RuntimeError("Holdout evaluation contains incomplete games; inspect records")
    return summaries


if __name__ == "__main__":
    main()
