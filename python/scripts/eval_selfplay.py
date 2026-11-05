"""Neural-only holdout evaluation, paired against the run's starting policy."""
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
    parser.add_argument('--evaluate-heads', action='store_true', help='Evaluate each head on the same plan in addition to per-game selection')
    parser.add_argument('--all-opponent-pairs', action='store_true', help='Cover every ordered pair of references')
    parser.add_argument('--population-from-run', action='store_true', help='Use the fixed opponent copies recorded in run config')
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint", help="Defaults to the run's latest.pt")
    parser.add_argument("--reference", nargs="+", help="Optional frozen trained opponents from this run")
    parser.add_argument("--blocks", type=int, default=20, help="Independent seeds; 12 games per block with two references")
    parser.add_argument("--seed", type=int, default=9001, help="Separate from model-selection seeds")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--io-root")
    parser.add_argument("--workers", type=int, default=2,
                        help="Independent CPU evaluation workers; use 1 for serial execution")
    parser.add_argument("--action-mode", choices=["stochastic", "greedy"], default="stochastic")
    parser.add_argument("--opponent-inference", choices=["batched", "serial"], default="batched")
    parser.add_argument('--compare', nargs='+', help='Additional checkpoints evaluated on the identical plan')
    parser.add_argument('--report-only', action='store_true', help='Render existing results without running games')
    args = parser.parse_args(argv)
    if args.blocks <= 0 or args.seed < 0 or args.workers <= 0:
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
    stored_record = json.loads((run_dir / "references.json").read_text())
    stored = stored_record['references']
    initialization = stored_record.get('initialization', {})
    pretrained = initialization.get('kind') == 'checkpoint'
    references = [reference(path) for path in args.reference] if args.reference else stored
    if args.population_from_run:
        if args.reference:
            parser.error('--population-from-run cannot be combined with --reference')
        references = json.loads((run_dir / 'config.json').read_text()).get('frozen_opponent_references', [])
        if not references:
            parser.error('Run has no frozen opponent population')
    plan = evaluation_plan(references, args.seed, args.blocks, all_pairs=args.all_opponent_pairs)
    candidate = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "latest.pt"
    initial = Path(stored[0]["checkpoint"])
    if file_hash(initial) != stored[0]["sha256"]:
        raise ValueError("Initial reference changed")
    io_root = exchange_io_root(args.io_root, output / "exchange", "holdout")
    root = Path(__file__).resolve().parents[2]
    summaries = {}
    models = {'initial': {'checkpoint': str(initial), 'jsonl': 'initial.jsonl', 'sha256': file_hash(initial),
                          'initialization': initialization}}
    for i, path in enumerate(args.compare or []):
        path = Path(path).expanduser().resolve()
        models[f'comparison_{i + 1}'] = {'checkpoint': str(path), 'jsonl': f'comparison_{i + 1}.jsonl', 'sha256': file_hash(path)}
    models['candidate'] = {'checkpoint': str(candidate), 'jsonl': 'candidate.jsonl', 'sha256': file_hash(candidate)}
    candidate_model = load_model(candidate)
    if args.evaluate_heads:
        for h in range(candidate_model.num_heads):
            models[f'head_{h}'] = {**models['candidate'], 'jsonl': f'head_{h}.jsonl', 'head_id': h}
    atomic_json(output / 'manifest.json', {**manifest(root, {**vars(args), 'num_heads': candidate_model.num_heads}),
                'telemetry_schema_version': 1,
                'card_database_sha256': file_hash(root / 'python/src/sevenwonders_rl/data/card_db.json')})
    console = HoldoutConsole(args.seed, args.blocks, len(plan), candidate, references, output,
        comparison_label=('Vortrainierter Ausgangspunkt und weitertrainiertes Modell auf gleichem Spielplan' if pretrained
                          else 'Untrainiertes und trainiertes Modell auf gleichem Spielplan'),
        baseline_pretrained=pretrained)
    try:
        for name, descriptor in models.items():
            path = Path(descriptor['checkpoint'])
            payload = torch.load(path, map_location='cpu', weights_only=True)
            descriptor['num_heads'] = payload['config'].get('num_heads', 1)
            descriptor['prior_strength'] = payload['config'].get('prior_strength', 0.0)
            descriptor['timestep'] = int(payload['timestep'])
            descriptor['reward_profile'] = payload['config'].get('reward_profile', 'standard')
            descriptor['initialization'] = payload['config'].get('initialization', {})
            console.evaluation_start(model_label(name, descriptor), len(plan))
            summaries[name] = evaluate_model(load_model(path), "cpu", plan, root,
                io_root / name, output / descriptor['jsonl'], action_mode=args.action_mode,
                checkpoint_id=descriptor['sha256'], batch_opponents=args.opponent_inference == "batched",
                detailed=True, progress_callback=console.progress, workers=args.workers,
                head_id=descriptor.get('head_id'))
            console.evaluation_end(summaries[name])
        if args.evaluate_heads:
            from sevenwonders_rl.evaluation.exploration import exploration_summary
            exploration = {}
            for name, descriptor in models.items():
                rows = read_records(output / descriptor['jsonl'])
                exploration[name] = exploration_summary(rows)
                if name != 'initial':
                    exploration[name]['paired_against_initial'] = paired_progress(
                        output / descriptor['jsonl'], output / 'initial.jsonl')
            atomic_json(output / 'exploration.json', exploration)
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
