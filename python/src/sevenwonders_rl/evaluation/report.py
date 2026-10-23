"""Reusable, offline evaluation reports from completed game records."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from sevenwonders_rl.evaluation.runner import aggregate
from sevenwonders_rl.evaluation.telemetry import CATEGORIES, LABELS, COLORS, COLOR_LABELS, distribution, card_color
from sevenwonders_rl.utils.runtime import atomic_json
from sevenwonders_rl.utils.training_console import category_lines, decimal
from sevenwonders_rl.evaluation.model_labels import model_label
from sevenwonders_rl.evaluation.dashboard import write_dashboard


def block_intervals(records):
    """Absolute strength intervals; correlated seat/matchup rotations stay together."""
    groups = defaultdict(list)
    for r in records:
        if r['status'] == 'completed':
            groups[r['engine_seed']].append([r['win_share'], r['vp_diff']])
    if len(groups) < 2:
        return {'seed_blocks': len(groups), 'win_share_ci95': None, 'vp_diff_ci95': None}
    data = np.array([np.mean(v, axis=0) for v in groups.values()])
    rng = np.random.default_rng(0)
    samples = data[rng.integers(len(data), size=(2000, len(data)))].mean(axis=1)
    bounds = np.quantile(samples, [.025, .975], axis=0).T.tolist()
    return {'seed_blocks': len(groups), 'win_share_ci95': bounds[0], 'vp_diff_ci95': bounds[1]}


def group_label(value):
    return Path(value['checkpoint']).stem if isinstance(value, dict) else str(value)


def summarize_model(records):
    summary = aggregate(records)
    summary.update(block_intervals(records))
    completed = [r for r in records if r['status'] == 'completed']
    groups = {k: defaultdict(list) for k in ('opponents', 'seat', 'wonder', 'outcome')}
    age_points = defaultdict(lambda: defaultdict(list))
    age_economy = defaultdict(lambda: defaultdict(list))
    initial_points = defaultdict(list)
    age_actions = {age: Counter() for age in (1, 2, 3)}
    cards = defaultdict(lambda: {'built': 0, 'available': 0, 'games': set()})
    opportunities = Counter()
    category_margins = {k: [] for k in CATEGORIES}
    final_count = age_count = trace_count = 0
    snapshot_issues = score_issues = unknown_cards = 0
    for index, r in enumerate(completed):
        seat = str(r['agent_seat'])
        pair = ' / '.join(group_label(r['opponents'][side]) for side in ('left', 'right'))
        groups['opponents'][pair].append(r)
        groups['seat'][seat].append(r)
        groups['outcome']['Sieg' if r['win'] else 'Gleichstand' if r['vp_diff'] == 0 else 'Niederlage'].append(r)
        players = r.get('final_state', {}).get('players', {})
        final = players.get(seat)
        if final:
            final_count += 1
            groups['wonder'][f"{final.get('wonder_name')} (ID {final.get('wonder_id')})"].append(r)
            others = [s for s in players if s != seat]
            if others:
                best = max(others, key=lambda s: (players[s]['points'].get('total') or 0, -int(s)))
                for k in CATEGORIES:
                    own, opp = final['points'].get(k), players[best]['points'].get(k)
                    if own is not None and opp is not None:
                        category_margins[k].append(own - opp)
            for player in players.values():
                p = player['points']
                if all(p.get(k) is not None for k in (*CATEGORIES, 'total')):
                    score_issues += int(sum(p[k] for k in CATEGORIES) != p['total'])
        states = r.get('age_states', [])
        initial = r.get('initial_state', {}).get('players', {}).get(seat, {}).get('points', {})
        for k in (*CATEGORIES, 'total'):
            if initial.get(k) is not None:
                initial_points[k].append(initial[k])
        ages = [s['completed_age'] for s in states]
        if ages == [1, 2, 3]:
            age_count += 1
        elif states:
            snapshot_issues += 1
        for state in states:
            age = state['completed_age']
            player = state['players'].get(seat)
            if not player:
                continue
            for k in (*CATEGORIES, 'total'):
                if player['points'].get(k) is not None:
                    age_points[str(age)][k].append(player['points'][k])
            for k in ('coins', 'shields', 'wood', 'ore', 'clay', 'stone', 'loom', 'glass', 'papyrus',
                      'gear', 'tablet', 'compass'):
                if player['resources'].get(k) is not None:
                    age_economy[str(age)][k].append(player['resources'][k])
            if player.get('wonder_stage') is not None:
                age_economy[str(age)]['wonder_stage'].append(player['wonder_stage'])
        if 'actions' in r:
            trace_count += 1
            for event in r['actions']:
                age = event['era']
                if age not in age_actions:
                    continue
                action = event['action']
                builds = action in ('build_structure', 'build_hand_free')
                bucket = event['color'] if builds else 'wonder' if action == 'build_wonder' else 'discard' if action == 'discard' else 'unknown'
                age_actions[age][bucket] += 1
                unknown_cards += int(builds and event['color'] == 'unknown')
                for name in event['legal_build_cards']:
                    cards[(name, age)]['available'] += 1
                for color in event['legal_build_colors']:
                    opportunities[color] += 1
                if builds and event['card']:
                    card = cards[(event['card'], age)]
                    card['built'] += 1
                    card['games'].add(index)
    n = len(completed)
    def subgroups(mapping):
        return {key: {'games': len(rows), 'win_share': float(np.mean([r['win_share'] for r in rows])),
                      'vp_diff_mean': float(np.mean([r['vp_diff'] for r in rows])),
                      'categories': aggregate(rows)['categories']} for key, rows in sorted(mapping.items())}
    summary.update(
        coverage={'completed': n, 'final_states': final_count, 'complete_age_sequences': age_count,
                  'action_traces': trace_count, 'invalid_age_sequences': snapshot_issues,
                  'score_sum_mismatches': score_issues, 'unknown_built_cards': unknown_cards},
        groups={k: subgroups(v) for k, v in groups.items()},
        age_points={age: {k: distribution(v) for k, v in keys.items()} for age, keys in age_points.items()},
        initial_points={k: distribution(v) for k, v in initial_points.items()},
        age_economy={age: {k: distribution(v) for k, v in keys.items()} for age, keys in age_economy.items()},
        age_actions={str(age): {k: count / trace_count for k, count in counts.items()} for age, counts in age_actions.items()} if trace_count else {},
        cards=[{'card': name, 'color': card_color(name), 'era': age, 'built': v['built'], 'legal_opportunities': v['available'],
                'games_built': len(v['games']), 'build_game_fraction': len(v['games']) / trace_count if trace_count else None,
                'selection_fraction': v['built'] / v['available'] if v['available'] else None}
               for (name, age), v in sorted(cards.items(), key=lambda item: (item[0][1], item[0][0]))],
        category_margin_to_best={k: distribution(v) for k, v in category_margins.items()},
        action_statistics={k: distribution([r['action_counts'].get(k, 0) for r in completed
                                           if 'action_counts' in r])
                           for k in ('discard', 'build_wonder', 'prod_build', 'other')},
        color_legal_opportunities=dict(opportunities))
    return summary


def section(title):
    return ['', '=' * 88, ' ' + title, '=' * 88]


def render_report(report):
    models = report['models']
    candidate = models['candidate']
    def model_title(name, short=False):
        return model_label(name, report.get('model_descriptors', {}).get(name), short=short)
    lines = section('AUSFÜHRLICHE SELF-PLAY-AUSWERTUNG')
    lines.extend([f" Test-Seed: {report.get('seed', 'nicht erfasst')}",
                  ' Gleicher Spielplan für alle Modellstände; geteilte Siege zählen anteilig.'])
    for name, descriptor in report.get('model_descriptors', {}).items():
        lines.append(f" {model_title(name)}: {descriptor['checkpoint']}")
    for i, ref in enumerate(report.get('references', []), 1):
        lines.append(f" Gegner {i}: {ref['checkpoint']}")
    lines.extend(section('1. DATENQUALITÄT'))
    lines.append(f" {'Modellstand':43} {'Versucht':>9} {'Abschluss':>10} {'Fehler':>8} {'Abbruch':>8}")
    for name, m in models.items():
        lines.append(f" {model_title(name):43} {m['attempted']:9d} {m['completed']:10d} {m['failed']:8d} {m['truncated']:8d}")
    c = candidate['coverage']
    for key, label in (('final_states', 'Endzustände aller Spieler'), ('complete_age_sequences', 'Vollständige Zeitalterfolgen'),
                       ('action_traces', 'Aktionsprotokolle')):
        lines.append(f" {label:35} {c[key]}/{c['completed']} Spiele")
    lines.append(f" Fehlerhafte Punktesummen: {c['score_sum_mismatches']} | "
                 f"Zeitalterfolgen: {c['invalid_age_sequences']} | Unbekannte gebaute Karten: {c['unknown_built_cards']}")
    lines.append(' Nicht erfasste Angaben werden nicht als Null interpretiert.')
    lines.extend(section('2. SPIELSTÄRKE UND FORTSCHRITT'))
    lines.extend([' A. LEISTUNG AUF DEMSELBEN TESTPLAN',
                  f" {'Modellstand':43} {'Sieganteil':>11} {'Punkte Ø':>10} {'Vorsprung Ø':>12}"])
    for name, m in models.items():
        share = decimal(100 * m['win_share'], 1) + ' %' if m['win_share'] is not None else 'nicht erfasst'
        lines.append(f" {model_title(name):43} {share:>11} {decimal(m['agent_vp_mean']):>10} {decimal(m['vp_diff_mean'], signed=True):>12}")
    lines.extend([' Vorsprung = eigene Punkte minus Punkte des stärksten Gegners.',
                  ' Minus bedeutet Rückstand; Sieganteil zählt geteilte Siege anteilig.',
                  f"\n B. UNSICHERHEIT FÜR: {model_title('candidate')}",
                  ' 95%-Intervalle der geschätzten Leistung:'])
    for key, title, scale, unit in (('win_share_ci95', 'Sieganteil', 100, '%'),
                                    ('vp_diff_ci95', 'Punktevorsprung', 1, 'Punkte')):
        ci = candidate.get(key)
        lines.append(f" {title:25} " +
                     (f"{decimal(ci[0]*scale)} {unit} bis {decimal(ci[1]*scale)} {unit}"
                      if ci else 'nicht verfügbar (zu wenige Seed-Blöcke)'))
    lines.append(' Intervalle: Bootstrap ganzer Engine-Seed-Blöcke, keine unabhängigen Sitzrotationen.')
    lines.append('\n C. DIREKTER FORTSCHRITTSVERGLEICH')
    for name, progress in report.get('paired_progress', {}).items():
        if progress.get('paired_valid'):
            lines.extend([f"\n {model_title('candidate')}", f" gegenüber {model_title(name)}", '-' * 88,
                          f" {'Kennzahl':25} {'Veränderung':>15}    95%-Intervall"])
            for value_key, ci_key, title, scale, unit in (
                ('vp_diff_gain', 'vp_diff_gain_ci95', 'Punktevorsprung', 1, 'Punkte'),
                ('win_share_gain', 'win_share_gain_ci95', 'Sieganteil', 100, 'Prozentpunkte')):
                ci = progress.get(ci_key)
                interval = f"{decimal(ci[0]*scale, signed=True)} bis {decimal(ci[1]*scale, signed=True)} {unit}" if ci else 'nicht verfügbar'
                lines.append(f" {title:25} {decimal(progress[value_key]*scale, signed=True):>15}    {interval}")
            ci = progress.get('vp_diff_gain_ci95')
            if ci:
                meaning = ('Hinweis auf Verbesserung beim Punktevorsprung' if ci[0] > 0 else
                           'Hinweis auf Verschlechterung beim Punktevorsprung' if ci[1] < 0 else
                           'Unterschied beim Punktevorsprung noch nicht eindeutig (Intervall enthält 0)')
                lines.append(' Einordnung: ' + meaning + '.')
    lines.append(' Positive Veränderungen sind besser. Die Intervalle beschreiben die Unsicherheit.')
    lines.extend(section('3. PUNKTEHERKUNFT UND VERTEILUNG'))
    lines.append(' Durchschnitt je Spiel auf identischem Testplan:')
    lines.append(f" {'Kategorie':25}" + ''.join(f' {model_title(name, True):>18}' for name in models))
    for k in (*CATEGORIES, 'commercial_combined', 'total'):
        if k == 'commercial_combined' and not any(m['categories'].get(k) for m in models.values()):
            continue
        category_label = LABELS.get(k, 'Handel + Münzen' if k == 'commercial_combined' else 'Gesamt')
        lines.append(f" {category_label:25}" + ''.join(f" {decimal(m['categories'].get(k, {}).get('mean') if m['categories'].get(k) else None):>18}" for m in models.values()))
    lines.extend(category_lines(candidate['categories'], detailed=True))
    lines.append(' Durchschnittlicher Kategorieabstand zum selben besten Gegner jeder Partie:')
    for k, stat in candidate['category_margin_to_best'].items():
        lines.append(f" {LABELS[k]:25} {decimal(stat['mean'] if stat else None, signed=True)}")
    lines.append(' Bei gegnerischem Punktegleichstand wird die niedrigere Sitznummer verwendet.')
    lines.extend(section('4. ENTWICKLUNG INNERHALB DES SPIELS — finales Modell'))
    lines.append(f" {'Wertung des aktuellen Zustands':32} {'Ende I':>13} {'Ende II':>13} {'Ende III':>13}")
    def age_row(label, key, source):
        vals = [source.get(str(a), {}).get(key) for a in (1, 2, 3)]
        return f" {label:32}" + ''.join(f" {decimal(v['mean'] if v else None):>13}" for v in vals)
    for k in (*CATEGORIES, 'total'):
        lines.append(age_row(LABELS.get(k, 'Gesamt'), k, candidate['age_points']))
    lines.append('\n Wirtschaft und Aufbau:')
    for k, label in (('coins', 'Münzbestand'), ('shields', 'Militärstärke / Schilde'),
                     ('wonder_stage', 'Vollendete Wunderabschnitte'), ('wood', 'Produktion Holz'),
                     ('ore', 'Produktion Erz'), ('clay', 'Produktion Lehm'), ('stone', 'Produktion Stein'),
                     ('loom', 'Produktion Stoff'), ('glass', 'Produktion Glas'), ('papyrus', 'Produktion Papyrus')):
        lines.append(age_row(label, k, candidate['age_economy']))
    for k in ('total', 'military'):
        vals = [candidate['age_points'].get(str(a), {}).get(k) for a in (1, 2, 3)]
        if all(vals) and c['complete_age_sequences'] == c['completed']:
            initial = candidate['initial_points'].get(k)
            if initial is None:
                continue
            previous = initial['mean']
            deltas = []
            for stat in vals:
                deltas.append(stat['mean'] - previous)
                previous = stat['mean']
            lines.append(f" {'Wertungsdelta ' + LABELS.get(k, 'Gesamt'):32}" + ''.join(f' {decimal(v, signed=True):>13}' for v in deltas))
    lines.extend([' Nach Militärwertung; Ende III zusätzlich nach finalen Gildeneffekten.',
                  ' Zwischenstände sind keine fest verdienten Endpunkte und keine kausalen Aktionswerte.',
                  ' Produktion: feste Statusressourcen; Wahlproduktion nicht als feste Ressource addiert.'])
    lines.extend(section('5. GEBAUTE KARTEN UND AKTIONEN JE ZEITALTER'))
    lines.append(f" {'Aktionen je Spiel':32} {'I':>13} {'II':>13} {'III':>13} {'Gesamt':>13}")
    for key in (*COLORS, 'wonder', 'discard', 'unknown'):
        values = [candidate['age_actions'].get(str(a), {}).get(key, 0) for a in (1, 2, 3)] if c['action_traces'] else None
        if key == 'unknown' and values and not sum(values):
            continue
        label = COLOR_LABELS.get(key, {'wonder': 'Wunderbauaktionen', 'discard': 'Verkaufte Karten', 'unknown': 'Unbekannt'}.get(key))
        lines.append(f" {label:32}" + (''.join(f' {decimal(v):>13}' for v in (*values, sum(values))) if values else ' nicht erfasst'))
    lines.append(' Nur bestätigte Aktionen: Wunderkarten und Verkäufe zählen nicht als Gebäude.')
    lines.extend(section('6. EINZELNE KARTEN UND LEGALE ALTERNATIVEN'))
    lines.append(f" {'Karte':26} {'Farbe':8} {'Alter':>5} {'Gebaut':>7} {'Spiele mit Bau':>16} {'Auswahlquote':>14}")
    if not candidate['cards']:
        lines.append(' Nicht erfasst. Für Kartendetails eine neue detaillierte Evaluation ausführen.')
    for card in candidate['cards']:
        lines.append(f" {card['card']:26} {card['color']:8} {card['era']:5d} {card['built']:7d} "
                     f"{decimal(card['build_game_fraction']*100, 1):>14} % "
                     f"{decimal(card['selection_fraction']*100 if card['selection_fraction'] is not None else None, 1):>12} %")
    lines.extend([' Auswahlquote: gebaut / Entscheidungen, bei denen diese Karte legal baubar war.',
                  ' Wiederholte Verfügbarkeit zählt wiederholt. Auch verfügbare, nie gebaute Karten erscheinen.',
                  ' Gebäudefarbe und Punkteherkunft sind verschieden; Wissenschaft/Militär können Wunderbeiträge enthalten.'])
    lines.append('\n Auswahl nach Farbe bei legaler Verfügbarkeit (Entscheidungen):')
    for color in COLORS:
        available = candidate['color_legal_opportunities'].get(color, 0)
        built = sum(card['built'] for card in candidate['cards'] if card['color'] == color)
        lines.append(f" {COLOR_LABELS[color]:25} " +
                     (f"{available} Möglichkeiten | {built} gewählt | {decimal(100*built/available, 1)} %"
                      if available else '0 legale Möglichkeiten' if c['action_traces'] else 'nicht erfasst'))
    lines.extend(section('7. UNTERSCHIEDE ZWISCHEN SPIELEN'))
    for group, label in (('opponents', 'Gegnerpaar'), ('seat', 'Spielerposition'), ('wonder', 'Wunder'), ('outcome', 'Spielausgang')):
        lines.append(f"\n {label:43} {'Spiele':>7} {'Sieganteil':>12} {'Vorsprung':>12}")
        if not candidate['groups'][group]:
            lines.append(' Nicht erfasst.')
        for name, stats in candidate['groups'][group].items():
            lines.append(f" {name:43} {stats['games']:7d} {decimal(stats['win_share']*100, 1):>10} % {decimal(stats['vp_diff_mean'], signed=True):>12}")
    lines.append('\n Punkteprofile der Untergruppen stehen vollständig in analysis.json und categories.csv.')
    lines.extend(section('8. BEFUNDE UND GRENZEN'))
    sci = candidate['categories'].get('science')
    if sci:
        lines.append(f" Wissenschaft: {decimal(sci['zero_fraction']*100, 1)} % der Spiele ohne Punkte.")
    for name, stats in models.items():
        discarded = stats['action_statistics']['discard']
        if discarded:
            lines.append(f" Verkaufte Karten ({model_label(name, report.get('model_descriptors', {}).get(name))}): Ø {decimal(discarded['mean'])} je Spiel.")
    lines.extend([' Ein Drittel Sieganteil ist ein Orientierungswert für gleich starke Dreispieler-Policies.',
                  ' Farbanteile und Zusammenhänge bei Siegen belegen keine optimalen Einzelentscheidungen.',
                  ' Ein Trainings-Seed und Gegner derselben Linie belegen noch keine allgemeine Spielstärke.',
                  ' Wunder/Edition entsprechen der verwendeten Simulatorvariante.',
                  ' Simulator-/Kartendatenstand: ' + ('siehe manifest.json.' if report.get('manifest_available') else 'nicht erfasst.')])
    return '\n'.join(lines) + '\n'


def write_report(output, records_by_model, metadata, paired):
    output = Path(output)
    report = {**metadata, 'models': {k: summarize_model(rows) for k, rows in records_by_model.items()},
              'paired_progress': paired}
    atomic_json(output / 'analysis.json', report)
    rendered = render_report(report)
    (output / 'report.txt').write_text(rendered, encoding='utf-8')
    (output / 'report.md').write_text('```text\n' + rendered + '```\n', encoding='utf-8')
    with (output / 'games.jsonl').open('w', encoding='utf-8') as f:
        for name, rows in records_by_model.items():
            for r in rows:
                f.write(json.dumps({'model': name, **r}, ensure_ascii=False, allow_nan=False) + '\n')
    with (output / 'categories.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'group_type', 'group', 'age', 'category', 'n', 'mean', 'median', 'p10', 'p90', 'max'])
        for name, m in report['models'].items():
            datasets = [('all', 'all', 'final', m['categories'])]
            datasets += [('age', 'all', age, stats) for age, stats in m['age_points'].items()]
            datasets += [(kind, group, 'final', stats['categories']) for kind, groups in m['groups'].items() for group, stats in groups.items()]
            for kind, group, age, stats in datasets:
                for key, stat in stats.items():
                    if stat:
                        writer.writerow([name, kind, group, age, key, *[stat[k] for k in ('n', 'mean', 'median', 'p10', 'p90', 'max')]])
    with (output / 'cards.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['model', 'card', 'color', 'era', 'built', 'legal_opportunities', 'games_built', 'build_game_fraction', 'selection_fraction']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, m in report['models'].items():
            for card in m['cards']:
                writer.writerow({'model': name, **card})
    write_dashboard(output, report)
    return rendered
