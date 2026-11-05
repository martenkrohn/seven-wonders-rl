"""Compare start, quarter and final long-run checkpoints on one fresh fixed panel."""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from compare_multihead_runs import paired
from sevenwonders_rl.evaluation.exploration import exploration_summary
from sevenwonders_rl.evaluation.milestones import fixed_panel
from sevenwonders_rl.evaluation.report import write_report
from sevenwonders_rl.evaluation.runner import evaluate_model
from sevenwonders_rl.evaluation.selfplay import plan_identity, read_records
from sevenwonders_rl.utils.checkpoints import file_hash, load_model, manifest
from sevenwonders_rl.utils.runtime import atomic_json, exchange_io_root
from sevenwonders_rl.utils.training_restart import training_directory_lock
from sevenwonders_rl.utils.seeding import derive_seed


def direct_plans(models, seed, blocks):
    """Each focal policy meets the other two, in both orders and all seats."""
    plans = {}
    for name in models:
        others = [dict(checkpoint=v['checkpoint'], sha256=v['sha256'], group=k)
                  for k, v in models.items() if k != name]
        plans[name] = [dict(left=left, right=right, agent_seat=seat,
                           engine_seed=derive_seed(seed, 'evaluation', block))
            for block in range(blocks) for left, right in (others, others[::-1]) for seat in range(3)]
    return plans


def fresh_seed_audit(runs, proposed):
    """Stream prior training/evaluation records, excluding bulky episode states."""
    seen, checked = set(), []
    names = {'opponents.jsonl', 'candidate.jsonl', 'initial.jsonl', 'v1.jsonl',
             'head_0.jsonl', 'head_1.jsonl', 'head_2.jsonl', 'head_3.jsonl', 'games.jsonl'}
    for path in sorted(runs.rglob('*.jsonl')):
        if path.name not in names and not path.name.startswith('eval_'):
            continue
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if 'engine_seed' in row:
                    seen.add(int(row['engine_seed']))
        checked.append(str(path))
    if proposed.intersection(seen):
        raise ValueError('Proposed seeds overlap existing training/evaluation records; choose another --seed')
    return dict(overlap=0, previous_seed_count=len(seen), checked_files=checked)


def run(args):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    source = args.run_dir.resolve()
    config = json.loads((source / 'config.json').read_text())
    result = json.loads((source / 'result.json').read_text())
    if result['status'] != 'completed':
        raise ValueError('Source training must be complete')
    budget = int(config['total_timesteps'])
    steps = [('initial', 0, 'Start · 0%'), ('quarter', budget // 4, 'Zwischenstand · 25%'),
             ('candidate', budget, 'Endmodell · 100%')]
    original_refs = config.get('evaluation_opponent_references', config['frozen_opponent_references'])
    panel = Path(config['eval_panel'])
    plan = fixed_panel(original_refs, panel, args.seed, args.blocks)
    proposed = {s['engine_seed'] for s in plan}
    if args.include_direct:
        proposed.update(derive_seed(args.seed + 1, 'evaluation', b) for b in range(args.blocks))
    print('Prüfe neue Seeds gegen gespeicherte Trainings- und Evaluationsspiele …', flush=True)
    audit = fresh_seed_audit(root / 'python/runs', proposed)
    # Validate every source before making output files or launching games.
    sources = {name: source / 'evaluations' / f'step_{step:07d}' / 'model.pt'
               for name, step, _ in steps}
    hashes = {}
    for path in [*sources.values(), *(Path(r['checkpoint']) for r in original_refs)]:
        hashes[str(path)] = file_hash(path)
        load_model(path)
    for ref in original_refs:
        if hashes[ref['checkpoint']] != ref['sha256']:
            raise ValueError('Fixed reference changed')
    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; existing results are preserved')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'references').mkdir()
    (output / 'models').mkdir()
    def copy_verified(src, target):
        digest = hashes[str(src)]
        shutil.copyfile(src, target)
        if file_hash(target) != digest or file_hash(src) != digest:
            raise ValueError(f'Checkpoint changed during copying: {src}')
        return digest
    refs = []
    for i, ref in enumerate(original_refs):
        target = output / 'references' / f'opponent_{i:03d}.pt'
        copy_verified(Path(ref['checkpoint']), target)
        refs.append({**ref, 'checkpoint': str(target)})
    models = {}
    for name, step, label in steps:
        target = output / 'models' / f'{name}.pt'
        digest = copy_verified(sources[name], target)
        models[name] = dict(checkpoint=str(target), sha256=digest, label=label,
                            source_checkpoint=str(sources[name]), timestep=step, jsonl=f'{name}.jsonl')
    plan = fixed_panel(refs, panel, args.seed, args.blocks)
    plan_id = plan_identity(plan, 'stochastic')
    atomic_json(output / 'plan.json', dict(seed=args.seed, blocks=args.blocks, games=plan, plan_id=plan_id))
    atomic_json(output / 'manifest.json', {**manifest(root, dict(config, evaluation_seed=args.seed)),
        'models': models, 'references': refs, 'fresh_seed_check': audit, 'plan_id': plan_id,
        'role': 'Fresh-seed checkpoint comparison against familiar frozen opponents'})
    io = exchange_io_root(None, output / 'exchange', 'long-fresh')
    started = time.perf_counter()
    records, summaries = {}, {}
    try:
        for name, descriptor in models.items():
            print(f"{descriptor['label']}: {len(plan)} Spiele, {args.blocks} neue Seed-Blöcke", flush=True)
            summaries[name] = evaluate_model(load_model(descriptor['checkpoint']), 'cpu', plan, root,
                io / name, output / descriptor['jsonl'], action_mode='stochastic',
                checkpoint_id=descriptor['sha256'], detailed=True, workers=args.workers,
                progress_callback=lambda done, total: print(f'{name}: {done}/{total}', flush=True))
            if not summaries[name]['valid_for_selection']:
                raise RuntimeError('Incomplete evaluation; inspect raw records')
            records[name] = read_records(output / descriptor['jsonl'])
        comparisons = {}
        for left, right in [('candidate', 'initial'), ('quarter', 'initial'), ('candidate', 'quarter')]:
            groups = {}
            for group in ('all', 'without_science', 'with_science', 'anchors_only'):
                def selected(r):
                    values = list(r['opponents'].values())
                    return (group == 'all' or
                        group == 'without_science' and all(v['group'] != 'science' for v in values) or
                        group == 'with_science' and any(v['group'] == 'science' for v in values) or
                        group == 'anchors_only' and all(v['group'] == 'anchors' for v in values))
                groups[group] = paired([r for r in records[left] if selected(r)],
                                       [r for r in records[right] if selected(r)])
            comparisons[f'{left}_vs_{right}'] = groups
        def report_progress(p):
            return dict(paired_valid=True, seed_blocks=p['seed_blocks'], vp_diff_gain=p['margin_gain'],
                vp_diff_gain_ci95=p['margin_ci95'], win_share_gain=p['win_share_gain'],
                win_share_gain_ci95=p['win_share_ci95'])
        write_report(output, records, dict(seed=args.seed, references=refs, model_descriptors=models,
            plan_id=plan_id, evaluation_role='Fresh seeds, familiar opponents'),
            {name: report_progress(comparisons[f'candidate_vs_{name}']['all']) for name in ('initial', 'quarter')})
        atomic_json(output / 'exploration.json', {name: exploration_summary(rr) for name, rr in records.items()})
        atomic_json(output / 'summary.json', dict(models=models, performance=summaries, comparisons=comparisons,
            seed=args.seed, blocks=args.blocks, plan_id=plan_id, elapsed_seconds=time.perf_counter() - started))
        if args.include_direct:
            direct = output / 'direct'
            direct.mkdir()
            plans = direct_plans(models, args.seed + 1, args.blocks)
            atomic_json(direct / 'plan.json', dict(seed=args.seed + 1, plans=plans,
                interpretation='One seat uses the focal-policy RNG stream; opponents use opponent RNG/head streams. '
                               'These are balanced focal-policy evaluations, not duplicated records of one tournament game.'))
            direct_records, direct_summaries = {}, {}
            for name, descriptor in models.items():
                print(f"Direktvergleich · {descriptor['label']}: {len(plans[name])} Spiele", flush=True)
                direct_summaries[name] = evaluate_model(load_model(descriptor['checkpoint']), 'cpu',
                    plans[name], root, io / 'direct' / name, direct / f'{name}.jsonl',
                    action_mode='stochastic', checkpoint_id=descriptor['sha256'], detailed=True,
                    workers=args.workers)
                if not direct_summaries[name]['valid_for_selection']:
                    raise RuntimeError('Incomplete direct comparison; inspect records')
                direct_records[name] = read_records(direct / f'{name}.jsonl')
            write_report(direct, direct_records, dict(seed=args.seed + 1, model_descriptors=models,
                references=[], evaluation_role='Direct start/quarter/final competition; balanced seat assignments'), {})
            atomic_json(direct / 'summary.json', direct_summaries)
            print(f"Direktvergleich-Dashboard: {direct / 'dashboard.html'}", flush=True)
        print(json.dumps(comparisons, indent=2), flush=True)
        print(f"Dashboard: {output / 'dashboard.html'}", flush=True)
    finally:
        shutil.rmtree(io)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/long-mixed-wonders-v1-seed131'))
    parser.add_argument('--out-dir', type=Path, default=Path('runs/long-fresh-seed130001-blocks32'))
    parser.add_argument('--seed', type=int, default=130001)
    parser.add_argument('--blocks', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--include-direct', action='store_true',
                        help='Also evaluate start/quarter/final together, both opponent orders and every learner seat')
    args = parser.parse_args(argv)
    if min(args.blocks, args.workers) < 1 or args.seed < 0:
        parser.error('Positive blocks/workers and nonnegative seed required')
    with training_directory_lock(args.out_dir.resolve()):
        run(args)


if __name__ == '__main__':
    main()
