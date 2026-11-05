"""Compare completed experiments on identical games using opponent hashes.

Copied reference paths differ between runs; opponent weights and game plans must
match. This reuses saved games and never launches simulations or training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from sevenwonders_rl.utils.runtime import atomic_json


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def game_key(row):
    return (row['engine_seed'], row['agent_seat'], row['action_selection_mode'],
            row.get('action_constraint', 'none'),
            *(row['opponents'][side]['sha256'] for side in ('left', 'right')))


def paired(candidate, baseline):
    previous = {game_key(row): row for row in baseline}
    if len(previous) != len(baseline) or {game_key(r) for r in candidate} != set(previous) or len(candidate) != len(baseline):
        raise ValueError('Game plans differ or contain duplicate games')
    if any(r['status'] != 'completed' for r in candidate+baseline):
        raise ValueError('Comparison requires complete games')
    blocks = defaultdict(list)
    for row in candidate:
        old = previous[game_key(row)]
        blocks[row['engine_seed']].append([row['vp_diff']-old['vp_diff'], row['win_share']-old['win_share']])
    values = np.asarray([np.mean(v,axis=0) for v in blocks.values()])
    intervals = None
    if len(values) > 1:
        samples = values[np.random.default_rng(0).integers(len(values),size=(10000,len(values)))].mean(1)
        intervals = np.quantile(samples,[.025,.975],axis=0).T.tolist()
    return dict(games=len(candidate), seed_blocks=len(values),
        margin_gain=float(values[:,0].mean()), win_share_gain=float(values[:,1].mean()),
        margin_ci95=intervals[0] if intervals else None,
        win_share_ci95=intervals[1] if intervals else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run', type=Path, default=Path('runs/multihead-v1-seed31'))
    parser.add_argument('--current-run', type=Path, required=True)
    args = parser.parse_args(argv)
    old_config, new_config = [read(p/'config.json') for p in (args.previous_run,args.current_run)]
    if old_config['initialization']['sha256'] != new_config['initialization']['sha256']:
        raise ValueError('Runs do not share the same source checkpoint')
    if old_config['prior_strength'] != 0:
        raise ValueError('Previous run must have priors disabled to supply the original baseline')
    for key in ('seed','num_heads','total_timesteps','reward_profile'):
        if old_config[key] != new_config[key]:
            raise ValueError(f'Run configuration differs: {key}')
    def population(config):
        return sorted((r['sha256'],r['group'],r['sampling_weight']) for r in config['frozen_opponent_references'])
    if population(old_config) != population(new_config):
        raise ValueError('Opponent populations differ')
    before = args.previous_run.with_name(args.previous_run.name+'-holdout')
    after = args.current_run.with_name(args.current_run.name+'-holdout')
    old_summary, new_summary = read(before/'summary.json'), read(after/'summary.json')
    source = records(before/old_summary['models']['initial']['jsonl'])
    previous = records(before/old_summary['models']['candidate']['jsonl'])
    result = dict(previous_run=str(args.previous_run), current_run=str(args.current_run),
        source_sha256=old_config['initialization']['sha256'], prior_strength=new_config['prior_strength'],
        limitations='Exploratory paired comparisons on reused holdout seeds and familiar opponents. '
                    'One training seed; no multiple-comparison correction; no new simulations.', models={})
    for name, descriptor in new_summary['models'].items():
        rows = records(after/descriptor['jsonl'])
        standard = lambda r: all(r['opponents'][s]['group']=='standard' for s in ('left','right'))
        result['models'][name] = dict(
            against_original_source=paired(rows,source),
            against_previous_random_head_policy=paired(rows,previous),
            standard_only_against_source=paired([r for r in rows if standard(r)],[r for r in source if standard(r)]),
            standard_only_against_previous=paired([r for r in rows if standard(r)],[r for r in previous if standard(r)]))
    atomic_json(after/'comparison-to-v1.json',result)
    lines=['# Comparison with the first multi-head experiment','',
        'Uses saved games; no extra evaluation or training. Opponents are matched by SHA-256,',
        'with identical seeds, learner seats, action mode and source checkpoint.','',
        '| Current policy | Margin gain vs original source | Margin gain vs v1 random policy |',
        '| --- | ---: | ---: |']
    for name, metrics in result['models'].items():
        def format_gain(key):
            p=metrics[key]; bounds=p['margin_ci95']
            return f"{p['margin_gain']:+.2f}"+(f" [{bounds[0]:+.2f}, {bounds[1]:+.2f}]" if bounds else '')
        lines.append(f"| {name} | {format_gain('against_original_source')} | {format_gain('against_previous_random_head_policy')} |")
    lines.extend(['','Brackets are paired 95% bootstrap intervals over whole engine-seed blocks.',
        'The current `initial` policy includes priors before training. The original source',
        'is the prior-free baseline from v1. Win-share and standard-only comparisons are',
        'in comparison-to-v1.json. Reused holdout seeds are not fresh confirmation.',''])
    (after/'comparison-to-v1.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)
    return result


if __name__ == '__main__':
    main()
