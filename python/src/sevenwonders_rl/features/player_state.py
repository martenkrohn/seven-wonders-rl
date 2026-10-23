"""Shared public player-state interpretation for actor and critic.

Uses the current simulator's cards and wonder effects, not a new rule edition.
"""
from sevenwonders_rl.data.cards import by_id, by_name_strict
from sevenwonders_rl.data.wonders import wonder_by_id

SCIENCE_SYMBOLS = ('gear', 'compass', 'tablet')
DISCOUNTS = ('raw_cheap_east', 'raw_cheap_west', 'manuf_cheap')


def built_ids(player):
    return [by_name_strict(name)['id'] for name in player.get('cards_played', [])]


def wonder_effects(player):
    wonder = wonder_by_id(player.get('wonder_id'))
    if wonder:
        for stage in wonder['stages'][:int(player.get('wonder_stage', 0))]:
            yield from stage.get('effects', [])


def science_state(player):
    counts = dict.fromkeys(SCIENCE_SYMBOLS, 0)
    jokers = 0
    for card_id in built_ids(player):
        symbols = by_id(card_id).get('effects', {}).get('science') or []
        for symbol in symbols:
            if symbol in counts:
                counts[symbol] += 1
            elif symbol == 'sci_extra':
                jokers += 1
    jokers += sum(effect.get('type') == 'add_science' for effect in wonder_effects(player))
    return counts, jokers


def discount_flags(player):
    flags = set()
    for card_id in built_ids(player):
        flags.update(by_id(card_id).get('effects', {}).get('discount_flags') or [])
    if any(effect.get('type') == 'buy_raw_cheap' for effect in wonder_effects(player)):
        flags.update(('raw_cheap_east', 'raw_cheap_west'))
    return flags.intersection(DISCOUNTS)


def chain_free(card, built_card_ids):
    """A prerequisite must actually be built; affordability is not a chain."""
    built = set(built_card_ids)
    if card['id'] in built:
        return False
    prerequisites = []
    if card.get('chain_from') is not None:
        prerequisites.append(card['chain_from'])
    prerequisites.extend(card.get('chain_from_any') or [])
    for prerequisite in prerequisites:
        identifier = prerequisite if isinstance(prerequisite, int) else by_name_strict(prerequisite)['id']
        if identifier in built:
            return True
    return False
