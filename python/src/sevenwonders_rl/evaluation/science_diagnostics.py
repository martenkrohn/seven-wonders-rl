"""Passive science trajectories; never affects rewards, observations or RNGs."""
from __future__ import annotations

import json
from pathlib import Path

from sevenwonders_rl.evaluation.telemetry import action_event, card_color
from sevenwonders_rl.features.player_state import science_state


class ScienceDiagnostics:
    def __init__(self, path):
        self.path = Path(path)
        self.games = {}

    def start(self, env, metadata, transition_step):
        self.games[env] = {**metadata, 'schema_version': 1,
                           'start_transition': transition_step, 'decisions': [], 'ages': []}

    def observe(self, env, before, after, obs, action, probabilities, policy_step):
        game = self.games[env]
        seat = game['agent_seat']
        event = action_event(before, after, obs, action, seat)
        if event is not None:
            hand = before['players'][str(seat)]['cards_hand']
            green_indices = [kind * 7 + slot for kind in (0, 2)
                             for slot, name in enumerate(hand)
                             if card_color(name) == 'green' and obs['action_mask'][kind * 7 + slot]]
            # Sum action probability, not distinct-card probability: a card can
            # have both paid and free build actions. Skipped ticks never enter.
            game['decisions'].append({
                'turn': event['turn'], 'era': event['era'], 'policy_step': policy_step,
                'action': event['action'], 'card': event['card'],
                'execution': event['execution'],
                'green_available': bool(green_indices),
                'green_probability': float(sum(float(probabilities[j]) for j in green_indices)),
                'green_built': event['color'] == 'green' and event['action'] in
                               ('build_structure', 'build_hand_free'),
            })
        age = before['game']['era']
        if age in (1, 2, 3) and after['game']['era'] != age:
            player = after['players'][str(seat)]
            symbols, jokers = science_state(player)
            game['ages'].append({'age': age, 'symbols': symbols, 'jokers': jokers,
                                'green_built_total': sum(card_color(c) == 'green'
                                                         for c in player['cards_played'])})

    def finish(self, env, status, transition_step):
        game = self.games.pop(env)
        seat = game['agent_seat']
        totals = [status['players'][str(s)]['points']['total'] for s in range(3)]
        own = totals[seat]
        best = max(totals)
        game.update(status='completed', end_transition=transition_step, totals=totals,
                    science_points=status['players'][str(seat)]['points']['scientific'],
                    vp_margin=own - max(t for s, t in enumerate(totals) if s != seat),
                    win_share=(1 / totals.count(best) if own == best else 0.0))
        with self.path.open('a') as stream:
            stream.write(json.dumps(game, allow_nan=False) + '\n')


def summarize_games(games):
    """Descriptive associations, never estimates of causal action value."""
    def mean(values):
        return sum(values) / len(values) if values else None

    def outcomes(rows):
        return {'games': len(rows), 'science_mean': mean([g['science_points'] for g in rows]),
                'science_ge10': sum(g['science_points'] >= 10 for g in rows),
                'science_ge20': sum(g['science_points'] >= 20 for g in rows),
                'margin_mean': mean([g['vp_margin'] for g in rows]),
                'win_share': mean([g['win_share'] for g in rows])}

    result = outcomes(games)
    result['ages'] = {}
    for age in (1, 2, 3):
        decisions = [d for g in games for d in g['decisions'] if d['era'] == age]
        opportunities = [d for d in decisions if d['green_available']]
        built = sum(d['green_built'] for d in decisions)
        result['ages'][str(age)] = {
            'applied_decisions': len(decisions), 'green_opportunities': len(opportunities),
            'green_builds': built, 'green_choice_rate': built / len(opportunities) if opportunities else None,
            'green_probability_when_available': mean([d['green_probability'] for d in opportunities]),
        }
    early = lambda g: any(d['era'] == 1 and d['green_built'] for d in g['decisions'])
    result['early_green'] = outcomes([g for g in games if early(g)])
    result['no_early_green'] = outcomes([g for g in games if not early(g)])
    result['early_green_science_ge10_nonloss'] = sum(
        early(g) and g['science_points'] >= 10 and g['vp_margin'] >= 0 for g in games)
    result['early_green_science_ge20_nonloss'] = sum(
        early(g) and g['science_points'] >= 20 and g['vp_margin'] >= 0 for g in games)
    return result
