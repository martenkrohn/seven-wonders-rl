"""Compact exploration diagnostics from executed, detailed evaluation traces."""
from collections import Counter, defaultdict

from sevenwonders_rl.evaluation.runner import aggregate


def exploration_summary(records):
    completed = [r for r in records if r['status'] == 'completed']
    pairs = defaultdict(list)
    ages = {a: Counter() for a in (1, 2, 3)}
    for row in completed:
        labels = [row['opponents'][side] for side in ('left', 'right')]
        key = ' / '.join(value.get('group', value['checkpoint']) if isinstance(value, dict)
                         else str(value) for value in labels)
        pairs[key].append(row)
        for action in row.get('actions', []):
            if action['execution'] != 'applied':
                continue
            age = ages[action['era']]
            age['actions'] += 1
            age['decisions_with_legal_card_build'] += bool(action['legal_build_cards'])
            if action['action'] == 'discard':
                age['discards'] += 1
                age['discards_without_legal_card_build'] += not action['legal_build_cards']
            if action['action'] in ('build_structure', 'build_hand_free'):
                age['built_' + action['color']] += 1
    n = len(completed)
    return {
        'overall': aggregate(records),
        'head_games': dict(Counter(str(r.get('head_id', 0)) for r in completed)),
        'opponent_groups': {key: aggregate(rows) for key, rows in sorted(pairs.items())},
        'ages': {str(a): {**dict(counts),
            'grey_builds_per_game': counts['built_grey'] / n if n else None,
            'discards_per_game': counts['discards'] / n if n else None,
            'discards_without_legal_card_build_per_game': counts['discards_without_legal_card_build'] / n if n else None,
            'fraction_decisions_with_legal_card_build': counts['decisions_with_legal_card_build'] / counts['actions']
                if counts['actions'] else None} for a, counts in ages.items()},
        'interpretation': 'Missing legal card builds does not exclude wonder actions. '
                          'Different behaviour alone does not establish stronger play. '
                          'All comparisons share training opponents; unseen seeds are not unseen strategies.'}
