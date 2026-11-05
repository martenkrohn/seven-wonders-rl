"""Offline discard analysis from acknowledged actions, grouped by age and plan."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def plan_key(row):
    return row['engine_seed'], row['agent_seat'], json.dumps(row['opponents'], sort_keys=True)


def summarize(rows):
    completed = [r for r in rows if r['status'] == 'completed']
    if not completed:
        raise ValueError('No completed games')
    blocks = defaultdict(list)
    ages = {age: Counter() for age in (1, 2, 3)}
    discarded_cards = {age: Counter() for age in ages}
    rounds = {age: Counter() for age in ages}
    game_counts = []
    for row in completed:
        if 'actions' not in row:
            raise ValueError('Detailed action traces are required; aggregate discard counts have no age information')
        per_game = Counter()
        per_age_actions = Counter()
        trace_discards = 0
        for event in row['actions']:
            if event['execution'] not in ('applied', 'fallback_discard'):
                raise ValueError('Trace contains unexecuted action')
            age = event['era']
            if age not in ages:
                raise ValueError(f'Invalid age {age}')
            ages[age]['actions'] += 1
            per_age_actions[age] += 1
            if event['execution'] == 'fallback_discard':
                ages[age]['engine_fallbacks'] += 1
                continue  # never call an engine replacement a policy discard
            if event['action'] != 'discard':
                continue
            trace_discards += 1
            per_game[age] += 1
            ages[age]['discards'] += 1
            alternatives = event.get('legal_build_cards')
            if alternatives is not None:
                ages[age]['legal_alternatives_known'] += 1
                ages[age]['has_build_alternative'] += int(bool(alternatives))
                ages[age]['discarded_card_itself_buildable'] += int(event['card'] in alternatives)
                ages[age]['no_structure_build_alternative'] += int(not alternatives)
            coins = event.get('coins_before')
            if coins is not None:
                ages[age]['coins_known'] += 1
                ages[age]['coins_sum'] += coins
                ages[age]['coins_0_to_2'] += int(coins <= 2)
                ages[age]['coins_10_or_more'] += int(coins >= 10)
            discarded_cards[age][event.get('color', 'unknown')] += 1
            rounds[age][event['turn'] % 7 + 1] += 1
        assert trace_discards == row['action_counts'].get('discard', 0)
        for age in ages:
            if per_age_actions[age] == 0:
                raise ValueError('Completed game has missing age actions')
            ages[age]['games_with_discard'] += int(per_game[age] > 0)
        counts = [per_game[age] for age in (1, 2, 3)]
        blocks[row['engine_seed']].append(counts)
        game_counts.append(counts)
    n = len(completed)
    data = np.asarray(game_counts)
    block_data = np.asarray([np.mean(values, axis=0) for _, values in sorted(blocks.items())])
    ci = None
    if len(block_data) >= 2:
        rng = np.random.default_rng(0)
        boot = block_data[rng.integers(len(block_data), size=(2000, len(block_data)))].mean(axis=1)
        ci = np.quantile(boot, [.025, .975], axis=0)
    total = int(data.sum())
    result = dict(attempted=len(rows), completed=n, failed=sum(r['status']=='failed' for r in rows),
                  truncated=sum(r['status']=='truncated' for r in rows), seed_blocks=len(blocks),
                  total_discards=total, discards_per_game=total/n,
                  games_without_discards=int((data.sum(axis=1)==0).sum()), ages={})
    for i, age in enumerate((1, 2, 3)):
        c = ages[age]
        count = c['discards']
        result['ages'][str(age)] = dict(**c, per_game=count/n,
            per_game_ci95=ci[:,i].tolist() if ci is not None else None,
            fraction_of_actions=count/c['actions'],
            fraction_of_all_discards=count/total if total else 0,
            fraction_games_with_discard=c['games_with_discard']/n,
            fraction_discards_with_build_alternative=c['has_build_alternative']/c['legal_alternatives_known'] if c['legal_alternatives_known'] else None,
            fraction_discarded_card_itself_buildable=c['discarded_card_itself_buildable']/c['legal_alternatives_known'] if c['legal_alternatives_known'] else None,
            mean_coins_before_discard=c['coins_sum']/c['coins_known'] if c['coins_known'] else None,
            discarded_colors=dict(discarded_cards[age]), rounds=dict(sorted(rounds[age].items())))
    return result


def analyze(directories):
    result = []; seen = set()
    for directory in directories:
        saved = json.loads((directory/'summary.json').read_text())
        analyses = json.loads((directory/'analysis.json').read_text())
        descriptors = saved['models']
        for name, desc in descriptors.items():
            initialization = desc.get('initialization', {})
            pretrained = name == 'initial' and initialization.get('kind') == 'checkpoint'
            step = desc.get('timestep', 0)
            if step < 65536 and not pretrained:
                continue
            if desc.get('action_constraint', 'none') != 'none':
                continue
            path = directory / desc['jsonl']
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            plan = hashlib.sha256(json.dumps(sorted(plan_key(r) for r in rows),sort_keys=True).encode()).hexdigest()
            identity = initialization['sha256'] if pretrained else desc['sha256']
            dedup = identity, plan, rows[0]['action_selection_mode'], desc.get('head_id')
            if dedup in seen:
                continue
            seen.add(dedup)
            if any(r.get('action_constraint', 'none') != 'none' for r in rows):
                raise ValueError('Constrained evaluation must not be mixed with normal policy behavior')
            summary = summarize(rows)
            existing = analyses['models'][name]
            for age, metrics in summary['ages'].items():
                expected = existing['age_actions'][age].get('discard', 0)
                assert abs(metrics['per_game']-expected)<1e-12
            assert abs(summary['discards_per_game']-existing['action_statistics']['discard']['mean'])<1e-12
            checkpoint = Path(desc['checkpoint'])
            label = checkpoint.parent.name
            if desc.get('head_id') is not None:
                label += f" / head {desc['head_id']}"
            if pretrained:
                label = Path(initialization['checkpoint']).parent.name
                step = initialization['source_timestep']
            result.append(dict(label=label,model_key=name,checkpoint=str(checkpoint),
                training_step=step,pretrained_initial=pretrained,evaluation_dir=str(directory),
                evaluation_seed=saved['seed'],plan_id=plan,checkpoint_id=identity,
                source_jsonl_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**summary))
    return dict(schema_version=1, models=result,
                methodology='Executed voluntary discards only. Ages use acknowledged action era. '
                'Rates use actual executed actions, not 21 protocol ticks. '
                'Bootstrap resamples whole engine-seed blocks. '
                'Deduplicate identical source models on the identical plan, excluding repeated pretrained baselines. '
                'Available structure builds do not prove discarding was suboptimal. '
                'Missing structure alternatives do not exclude a possible wonder action.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluations', nargs='+',type=Path)
    parser.add_argument('--out-dir',required=True,type=Path)
    args=parser.parse_args(argv)
    output=args.out_dir
    output.mkdir(parents=True,exist_ok=False)
    result=analyze(args.evaluations)
    (output/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    fields=['model','step','evaluation_seed','plan_id','games','age','discards','discards_per_game',
            'fraction_of_actions','fraction_of_all_discards','games_with_discard',
            'fraction_discards_with_build_alternative','fraction_discarded_card_itself_buildable']
    with (output/'by_age.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for model in result['models']:
            for age,metrics in model['ages'].items():
                writer.writerow(dict(model=model['label'],step=model['training_step'],
                    evaluation_seed=model['evaluation_seed'],plan_id=model['plan_id'],games=model['completed'],
                    age=age,discards=metrics['discards'],discards_per_game=metrics['per_game'],
                    fraction_of_actions=metrics['fraction_of_actions'],
                    fraction_of_all_discards=metrics['fraction_of_all_discards'],
                    games_with_discard=metrics['games_with_discard'],
                    fraction_discards_with_build_alternative=metrics['fraction_discards_with_build_alternative'],
                    fraction_discarded_card_itself_buildable=metrics['fraction_discarded_card_itself_buildable']))
    for model in result['models']:
        print(model['label'],model['training_step'],'seed',model['evaluation_seed'],
              'I/II/III',*[round(model['ages'][str(a)]['per_game'],3) for a in (1,2,3)],
              'total',round(model['discards_per_game'],3))
    print('Saved:',output)


if __name__=='__main__':
    main()
