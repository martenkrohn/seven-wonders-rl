"""Summarize completed training science trajectories, without running games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sevenwonders_rl.evaluation.science_diagnostics import summarize_games
from sevenwonders_rl.evaluation.dashboard import write_science_dashboard


def analyze_run(path, window):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text())
    source = path / 'science_episodes.jsonl'
    if not config.get('science_diagnostics') or not source.exists():
        raise ValueError(f'{path}: no science diagnostics; use --science-diagnostics in a fresh run')
    groups = {}
    count = 0
    discoveries = {}
    # Read one JSON line at a time, grouping compact trajectories by game start.
    with source.open() as stream:
        for line in stream:
            game = json.loads(line)
            if game['schema_version'] != 1 or game['status'] != 'completed':
                raise ValueError(f'{source}: unsupported/incomplete record')
            start = game['start_transition'] // window * window
            groups.setdefault(start, []).append(game)
            early = any(d['era'] == 1 and d['green_built'] for d in game['decisions'])
            if early and game['vp_margin'] >= 0:
                for threshold in (10, 20):
                    if game['science_points'] >= threshold:
                        discoveries.setdefault(str(threshold), {
                            key: game[key] for key in ('env', 'episode', 'engine_seed',
                                'start_transition', 'end_transition', 'science_points', 'vp_margin')})
            count += 1
    result_file = path / 'result.json'
    return {'run': str(path), 'seed': config['seed'], 'ent_coef': config['ent_coef'],
            'gae_lambda': config['gae_lambda'], 'window': window, 'completed_games': count,
            'first_early_green_nonloss_discoveries': discoveries,
            'training_result': json.loads(result_file.read_text()) if result_file.exists() else None,
            'windows': [{'start': start, 'end_exclusive': start + window,
                         **summarize_games(games)} for start, games in sorted(groups.items())]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--window', type=int, default=65536)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.window <= 0:
        parser.error('--window must be positive')
    runs = [analyze_run(path, args.window) for path in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {'schema_version': 1,
        'interpretation': 'Training associations only; changing opponents and state distributions. '
                          'Windows grouped by game start, not completion. '
                          'High science thresholds are diagnostics, not optimality criteria. '
                          'Compare playing strength on a shared frozen holdout plan.',
        'runs': runs}
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    dashboard = write_science_dashboard(args.out.with_suffix('.html'), summary)
    print('Run / Start / Spiele / Grün I: gebaut, legal, Policy-P / Wissenschaft >=10, >=20 / frühes Grün+>=10+Nichtverlust')
    for run in runs:
        for row in run['windows']:
            age = row['ages']['1']
            probability = age['green_probability_when_available']
            label = f'{probability:.1%}' if probability is not None else 'n/a'
            print(f"{run['run']} / {row['start']} / {row['games']} / "
                  f"{age['green_builds']}, {age['green_opportunities']}, {label} / "
                  f"{row['science_ge10']}, {row['science_ge20']} / "
                  f"{row['early_green_science_ge10_nonloss']}")
    print(f'Details: {args.out}')
    print(f'Dashboard: {dashboard}')


if __name__ == '__main__':
    main()
