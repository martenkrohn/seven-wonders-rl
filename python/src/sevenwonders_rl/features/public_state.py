"""Append-only public information contract; never inspect another player's hand.

The original 102/99 inputs remain unchanged for historical opponents and priors.
The schema includes database contents so edited costs/effects cannot load silently.
"""
from functools import lru_cache
import hashlib
import json

import numpy as np

from sevenwonders_rl.data.cards import get_all_ids, load_card_db
from sevenwonders_rl.data.wonders import load_wonder_db, wonder_by_id
from sevenwonders_rl.features.player_state import built_ids

PUBLIC_STATE_VERSION = 1
RESOURCES = ('wood', 'ore', 'clay', 'stone', 'loom', 'glass', 'papyrus')
ABILITIES = ('add_science', 'hand_build_for_free', 'discard_build_for_free',
             'choose_raw_nontradable', 'choose_manuf_nontradable',
             'play_seventh_card', 'buy_raw_cheap', 'copy_guild')
CARD_IDS = tuple(sorted(get_all_ids()))
WONDER_IDS = tuple(sorted(map(int, load_wonder_db())))
STAGE_NAMES = tuple('cost_' + r for r in RESOURCES) + ('vp', 'coins', 'shields') + ABILITIES
PLAYER_NAMES = (tuple('wonder_' + str(i) for i in WONDER_IDS)
                + tuple('starting_' + r for r in RESOURCES) + ('stage_count',)
                + tuple(f'stage_{i}_{name}' for i in range(4)
                        for name in ('exists', 'built', 'next') + STAGE_NAMES)
                + tuple('built_card_' + str(i) for i in CARD_IDS))
GLOBAL_NAMES = tuple(seat + '_' + name for seat in ('self', 'left', 'right') for name in PLAYER_NAMES)
ACTION_NAMES = ('next_stage',) + STAGE_NAMES + ('est_trade_left', 'est_trade_right', 'coins_after_est_trade')
PUBLIC_GLOBAL_DIM, PUBLIC_ACTION_DIM = len(GLOBAL_NAMES), len(ACTION_NAMES)
_STAGE_OFFSET = len(WONDER_IDS) + len(RESOURCES) + 1
_STAGE_WIDTH = 3 + len(STAGE_NAMES)
_CARD_OFFSET = _STAGE_OFFSET + 4 * _STAGE_WIDTH
_CARD_INDEX = {card: i for i, card in enumerate(CARD_IDS)}


@lru_cache(maxsize=1)
def public_schema_hash():
    payload = dict(version=PUBLIC_STATE_VERSION, global_names=GLOBAL_NAMES,
                   action_names=ACTION_NAMES, cards=load_card_db(), wonders=load_wonder_db())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def stage_vector(stage):
    effects = stage.get('effects', [])
    types = {e['type'] for e in effects}
    # Halikarnassos B's first two stages bundle VP and a discard-pile build.
    if 'add_vp_discard' in types:
        types.add('discard_build_for_free')
    return np.asarray([stage.get('cost', {}).get(r, 0) for r in RESOURCES]
        + [stage.get('vp', 0),
           sum(e.get('amount', 0) for e in effects if e['type'] == 'add_coins'),
           sum(e.get('amount', 0) for e in effects if e['type'] == 'add_shields')]
        + [int(ability in types) for ability in ABILITIES], dtype=np.float32)


@lru_cache(maxsize=14)
def _board(wonder_id):
    wonder = wonder_by_id(wonder_id)
    if wonder is None:
        raise ValueError(f'Unknown public wonder identity: {wonder_id}')
    identity = [float(wonder_id == i) for i in WONDER_IDS]
    starting = [float(wonder['starting_resource'] == r) for r in RESOURCES]
    rows = [np.asarray(identity + starting + [len(wonder['stages'])], dtype=np.float32)]
    for i in range(4):
        rows.append(np.asarray([i < len(wonder['stages']), 0, 0], dtype=np.float32))
        rows.append(stage_vector(wonder['stages'][i]) if i < len(wonder['stages']) else
                    np.zeros(len(STAGE_NAMES), dtype=np.float32))
    rows.append(np.zeros(len(CARD_IDS), dtype=np.float32))
    return np.concatenate(rows)


def player_public(player, card_ids=None):
    wid, built = int(player['wonder_id']), int(player['wonder_stage'])
    row = _board(wid).copy()
    count = len(wonder_by_id(wid)['stages'])
    for i in range(4):
        row[_STAGE_OFFSET + i * _STAGE_WIDTH + 1] = i < built
        row[_STAGE_OFFSET + i * _STAGE_WIDTH + 2] = i == built and i < count
    for card_id in built_ids(player) if card_ids is None else card_ids:
        row[_CARD_OFFSET + _CARD_INDEX[card_id]] = 1
    return row


def wonder_actions(status, seat, mask, public_players):
    from sevenwonders_rl.features.action_features import _build_player_context
    from sevenwonders_rl.features.trade_estimator import estimate_trade_cost
    result = np.zeros((28, PUBLIC_ACTION_DIM), dtype=np.float32)
    player = status['players'][str(seat)]
    stage = int(player['wonder_stage'])
    stages = wonder_by_id(player['wonder_id'])['stages']
    if stage >= len(stages) or not mask[7:14].any():
        return result
    ctx = _build_player_context(status, seat, player_idx=0, public_players=public_players)
    next_stage = stages[stage]
    trade = estimate_trade_cost(required=next_stage['cost'], own_production=ctx.own_production,
        own_choice_resources=ctx.own_choice_resources, left_supply=ctx.left_supply,
        left_choice_supply=ctx.left_choice_supply, right_supply=ctx.right_supply,
        right_choice_supply=ctx.right_choice_supply, player_flags=ctx.discount_flags)
    row = np.concatenate(([stage + 1], stage_vector(next_stage),
                          [trade.cost_left, trade.cost_right, ctx.current_coins - trade.total_cost]))
    for slot in range(7):
        if mask[7 + slot]:
            result[7 + slot] = row
    return result
