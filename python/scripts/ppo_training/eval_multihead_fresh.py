"""Paired fresh-seed evaluation of the two completed multi-head policies only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import torch

from compare_multihead_runs import paired
from sevenwonders_rl.evaluation.exploration import exploration_summary
from sevenwonders_rl.evaluation.report import write_report
from sevenwonders_rl.evaluation.runner import evaluate_model
from sevenwonders_rl.evaluation.selfplay import evaluation_plan, plan_identity, read_records
from sevenwonders_rl.utils.checkpoints import file_hash, load_model, manifest
from sevenwonders_rl.utils.runtime import atomic_json, exchange_io_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1', type=Path, default=Path('runs/multihead-v1-seed31'))
    parser.add_argument('--v2', type=Path, default=Path('runs/multihead-v2-priors-seed31'))
    parser.add_argument('--seed', type=int, default=99001)
    parser.add_argument('--blocks', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out-dir', type=Path, default=Path('runs/multihead-fresh-seed99001-blocks32'))
    args = parser.parse_args(argv)
    if min(args.blocks,args.workers) < 1 or args.seed < 0:
        parser.error('Use positive blocks/workers and a nonnegative seed')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    configs = [json.loads((run/'config.json').read_text()) for run in (args.v1,args.v2)]
    def population(config):
        return sorted((r['sha256'],r['group'],r['sampling_weight'])
                      for r in config['frozen_opponent_references'])
    if population(configs[0]) != population(configs[1]):
        raise ValueError('Opponent populations differ')
    original_refs = configs[0]['frozen_opponent_references']
    # Precommit the new plan and reject overlap with training, validation,
    # inspected holdouts and prior calibration before launching any games.
    plan = evaluation_plan(original_refs,args.seed,args.blocks,all_pairs=True)
    seeds = {spec['engine_seed'] for spec in plan}
    seen, checked = set(), []
    source_runs = {args.v1.resolve(),args.v2.resolve()}
    source_runs.update(Path(c['initialization']['checkpoint']).parent for c in configs)
    for run in sorted(source_runs):
        files = [run/'opponents.jsonl',*run.glob('eval_*.jsonl')]
        holdout = run.with_name(run.name+'-holdout')
        files.extend(holdout.glob('*.jsonl'))
        for path in files:
            if not path.exists():
                continue
            for record in read_records(path):
                if 'engine_seed' in record:
                    seen.add(record['engine_seed'])
            checked.append(str(path))
    calibration = Path('runs/multihead-prior-calibration-20261026/games.jsonl')
    if calibration.exists():
        seen.update(r['engine_seed'] for r in read_records(calibration))
        checked.append(str(calibration))
    if seeds.intersection(seen):
        raise ValueError('Proposed evaluation overlaps previously used seeds')
    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True,exist_ok=True)
    copies = output/'references';copies.mkdir()
    references = []
    for i, ref in enumerate(original_refs):
        if file_hash(ref['checkpoint']) != ref['sha256']:
            raise ValueError('Reference checkpoint changed')
        path = copies/f'opponent_{i:03d}.pt'
        shutil.copyfile(ref['checkpoint'],path)
        if file_hash(path) != ref['sha256']:
            raise ValueError('Reference copy differs')
        references.append({**ref,'checkpoint':str(path)})
    plan = evaluation_plan(references,args.seed,args.blocks,all_pairs=True)
    models = {}
    weights = output/'models';weights.mkdir()
    for name,run,label in [('v1',args.v1,'v1 (Priors: 0)'),('candidate',args.v2,'v2 (Priors: 32)')]:
        source = run/'latest.pt';digest=file_hash(source)
        path = weights/f'{name}.pt';shutil.copyfile(source,path)
        if file_hash(path) != digest or file_hash(source) != digest:
            raise ValueError('Model changed during copying')
        payload = torch.load(path,map_location='cpu',weights_only=True)
        models[name] = dict(checkpoint=str(path),source_checkpoint=str(source.resolve()),
            sha256=digest,jsonl=f'{name}.jsonl',label=label,timestep=int(payload['timestep']),
            num_heads=payload['config']['num_heads'],prior_strength=payload['config']['prior_strength'],
            reward_profile=payload['config']['reward_profile'],initialization=payload['config']['initialization'])
        load_model(path)
    if (models['v1']['prior_strength'],models['candidate']['prior_strength']) != (0,32):
        raise ValueError('Expected v1 without priors and v2 with strength 32')
    atomic_json(output/'plan.json',dict(seed=args.seed,blocks=args.blocks,games=plan))
    atomic_json(output/'manifest.json',{**manifest(root,{**vars(args),'v1':str(args.v1),
        'v2':str(args.v2),'out_dir':str(output),'num_heads':4}),
        'fresh_seed_check':dict(overlap=0,checked_files=checked,previous_seed_count=len(seen)),
        'models':models,'references':references,'plan_id':plan_identity(plan,'stochastic')})
    io_root = exchange_io_root(None,output/'exchange','multihead-fresh')
    start = time.perf_counter()
    summaries, records = {}, {}
    try:
        for name,descriptor in models.items():
            print(f"{descriptor['label']}: {len(plan)} games, {args.blocks} fresh blocks",flush=True)
            def progress(done,total):
                print(f'{name}: {done}/{total}',flush=True)
            summaries[name] = evaluate_model(load_model(descriptor['checkpoint']),'cpu',plan,root,
                io_root/name,output/descriptor['jsonl'],action_mode='stochastic',
                checkpoint_id=descriptor['sha256'],detailed=True,workers=args.workers,
                progress_callback=progress)
            if not summaries[name]['valid_for_selection']:
                raise RuntimeError('Incomplete games; inspect records')
            records[name] = read_records(output/descriptor['jsonl'])
            print(f"{name}: share={summaries[name]['win_share']:.4f}, margin={summaries[name]['vp_diff_mean']:.3f}",flush=True)
        progress = paired(records['candidate'],records['v1'])
        groups = {}
        for left in ('standard','science','military'):
            for right in ('standard','science','military'):
                def selected(row):
                    return row['opponents']['left']['group']==left and row['opponents']['right']['group']==right
                groups[f'{left} / {right}'] = paired(
                    [r for r in records['candidate'] if selected(r)],
                    [r for r in records['v1'] if selected(r)])
        exploration = {name:exploration_summary(rr) for name,rr in records.items()}
        exploration['paired_v2_vs_v1'] = progress
        exploration['paired_opponent_groups'] = groups
        atomic_json(output/'exploration.json',exploration)
        summaries.update(models=models,seed=args.seed,seed_blocks=args.blocks,references=references,
            plan_id=plan_identity(plan,'stochastic'),paired_v2_vs_v1=progress,
            paired_opponent_groups=groups,elapsed_seconds=time.perf_counter()-start,
            evaluation_policy='Uniform random head selected once per game; stochastic actions',
            head_counts={name:dict(Counter(str(r['head_id']) for r in rr)) for name,rr in records.items()})
        atomic_json(output/'summary.json',summaries)
        report_progress = dict(paired_valid=True,seed_blocks=progress['seed_blocks'],
            vp_diff_gain=progress['margin_gain'],vp_diff_gain_ci95=progress['margin_ci95'],
            win_share_gain=progress['win_share_gain'],win_share_gain_ci95=progress['win_share_ci95'])
        write_report(output,records,dict(seed=args.seed,references=references,
            manifest_available=True,model_descriptors=models),{'v1':report_progress})
        print(json.dumps(progress,indent=2),flush=True)
        print(f'Reports: {output}',flush=True)
    finally:
        shutil.rmtree(io_root)
    return summaries


if __name__ == '__main__':
    main()
