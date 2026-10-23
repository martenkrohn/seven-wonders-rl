"""Passive engine telemetry. Never adds information to policy observations."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from sevenwonders_rl.data.cards import by_name

CATEGORIES = ('blue', 'military', 'science', 'yellow', 'guild', 'wonder', 'treasury')
LABELS = dict(zip(CATEGORIES, ('Blau / Zivilbauten', 'Rot / Militär', 'Grün / Wissenschaft',
                             'Gelb / Handel', 'Violett / Gilden', 'Wunder', 'Münzen')))
COLORS = ('brown', 'grey', 'blue', 'red', 'green', 'yellow', 'purple')
COLOR_LABELS = dict(zip(COLORS, ('Braun / Rohstoffe', 'Grau / Manufaktur', 'Blau / Zivilbauten',
                               'Rot / Militär', 'Grün / Wissenschaft', 'Gelb / Handel', 'Violett / Gilden')))


def card_color(name):
    card = by_name(name) if name else None
    return card.get('color', 'unknown') if card else 'unknown'


def player_snapshot(player):
    raw = deepcopy(player.get('points', {}))
    resources = deepcopy(player.get('resources', {}))
    coins = resources.get('coins')
    points = {key: raw.get(source) for key, source in (
        ('blue', 'civilian'), ('military', 'military'), ('science', 'scientific'),
        ('guild', 'guild'), ('wonder', 'wonder'))}
    points['treasury'] = int(coins) // 3 if coins is not None else None
    points['yellow'] = (raw['commercial'] - points['treasury']
                        if raw.get('commercial') is not None and coins is not None else None)
    points['total'] = raw.get('total')
    if all(points.get(k) is not None for k in (*CATEGORIES, 'total')):
        if sum(points[k] for k in CATEGORIES) != points['total']:
            raise ValueError('Engine score breakdown does not sum to total')
    cards = deepcopy(player.get('cards_played'))
    return {'raw_points': raw, 'points': points, 'resources': resources,
            'wonder_id': player.get('wonder_id'), 'wonder_name': player.get('wonder_name'),
            'wonder_stage': player.get('wonder_stage'), 'cards': cards,
            'card_colors': dict(Counter(card_color(c) for c in cards)) if cards is not None else None}


def status_snapshot(status, completed_age=None):
    return {'completed_age': completed_age, 'game': deepcopy(status.get('game', {})),
            'players': {s: player_snapshot(p) for s, p in status.get('players', {}).items()}}


def action_event(before, after, obs, action, seat):
    """Only acknowledged actions count; skipped seventh-card ticks do not."""
    ack = after.get('last_actions', {}).get(str(seat), {})
    if ack.get('status') not in ('applied', 'fallback_discard'):
        return None
    player = before['players'][str(seat)]
    hand = player.get('cards_hand', [])
    slot = int(action) % 7
    name = hand[slot] if slot < len(hand) else None
    mask = obs['action_mask']
    legal = sorted({hand[i] for i in range(min(7, len(hand))) if mask[i] or mask[14 + i]})
    executed = ack.get('executed')
    return {'turn': before['game']['turn'], 'era': before['game']['era'],
            'seat': seat, 'action_index': int(action), 'execution': ack['status'],
            'action': executed, 'card': name, 'color': card_color(name),
            'legal_build_cards': legal,
            'legal_build_colors': sorted({card_color(c) for c in legal}),
            'coins_before': player.get('resources', {}).get('coins'),
            'coins_after': after['players'][str(seat)].get('resources', {}).get('coins'),
            'points_before': player_snapshot(player)['points'],
            'points_after': player_snapshot(after['players'][str(seat)])['points']}


def distribution(values):
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    positive = data[data > 0]
    return {'n': len(data), 'mean': float(data.mean()), 'median': float(np.median(data)),
            'std': float(data.std()), 'p10': float(np.quantile(data, .1)),
            'p90': float(np.quantile(data, .9)), 'min': float(data.min()), 'max': float(data.max()),
            'zero_fraction': float(np.mean(data == 0)), 'positive_fraction': float(np.mean(data > 0)),
            'positive_mean': float(positive.mean()) if len(positive) else None}


def category_statistics(records):
    completed = [r for r in records if r.get('status') == 'completed']
    values = {k: [] for k in (*CATEGORIES, 'total', 'commercial_combined')}
    for r in completed:
        final = r.get('final_state', {}).get('players', {}).get(str(r.get('agent_seat', 0)))
        if final:
            points = final['points']
        else:
            raw = r.get('points_breakdown', {})
            points = {k: raw.get(source) for k, source in (
                ('blue', 'civilian'), ('military', 'military'), ('science', 'scientific'),
                ('guild', 'guild'), ('wonder', 'wonder'), ('total', 'total'))}
            if raw.get('commercial') is not None:
                values['commercial_combined'].append(raw['commercial'])
        for k in (*CATEGORIES, 'total'):
            if points.get(k) is not None:
                values[k].append(points[k])
    return {k: distribution(v) for k, v in values.items()}
