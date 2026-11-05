"""Seat-local records of observations, never reconstructed opponent hands.

Frames occupy their original engine-turn slots. Own action fields are completed
only after acknowledgement; previously emitted payloads are immutable copies.
No raw engine status, opponent private hand or opponent action argument is stored.
"""
from functools import lru_cache
import hashlib
import json

import numpy as np

from sevenwonders_rl.features.public_state import (
    CARD_IDS, PLAYER_NAMES, PUBLIC_GLOBAL_DIM, PUBLIC_ACTION_DIM, public_schema_hash)
from sevenwonders_rl.features.player_state import wonder_effects
from sevenwonders_rl.features.action_feature_schema import FEATURE_NAMES

MEMORY_VERSION = 1
MEMORY_KEY = 'observation_memory'
HISTORY_LENGTH = 21
BASE_GLOBAL_DIM = 102 + PUBLIC_GLOBAL_DIM
ACTION_DIM = 99 + PUBLIC_ACTION_DIM
N_ACTIONS = 28
CARD_COUNT = len(CARD_IDS)
_CARD_INDEX = {card: index for index, card in enumerate(CARD_IDS)}
_PUBLIC_CARD_COLUMNS = [PLAYER_NAMES.index('built_card_' + str(card)) for card in CARD_IDS]

# Each frame records the information visible before its action. Hand absence at
# a skipped seventh-card exchange is explicitly distinguished from an empty hand.
HAND = slice(0, CARD_COUNT)
HAND_KNOWN = CARD_COUNT
PUBLIC = slice(HAND_KNOWN + 1, HAND_KNOWN + 1 + 102)
BUILT_CARDS = slice(PUBLIC.stop, PUBLIC.stop + 3 * CARD_COUNT)
OWN_CARD = slice(BUILT_CARDS.stop, BUILT_CARDS.stop + CARD_COUNT)
OWN_KIND = slice(OWN_CARD.stop, OWN_CARD.stop + 4)
ACK = slice(OWN_KIND.stop, OWN_KIND.stop + 3)
FRAME_DIM = ACK.stop

CURRENT_HAND = slice(0, CARD_COUNT + 1)
MASK = slice(CURRENT_HAND.stop, CURRENT_HAND.stop + N_ACTIONS)
ACTION_FEATURES = slice(MASK.stop, MASK.stop + N_ACTIONS * ACTION_DIM)
FRAMES = slice(ACTION_FEATURES.stop, ACTION_FEATURES.stop + HISTORY_LENGTH * FRAME_DIM)
VALID = slice(FRAMES.stop, FRAMES.stop + HISTORY_LENGTH)
MEMORY_DIM = VALID.stop


@lru_cache(maxsize=1)
def memory_schema_hash():
    schema = dict(version=MEMORY_VERSION, length=HISTORY_LENGTH, cards=CARD_IDS,
        frame_dim=FRAME_DIM, memory_dim=MEMORY_DIM, public_hash=public_schema_hash(),
        action_feature_names=FEATURE_NAMES,
        fields=('observed_hand_counts', 'hand_known', 'base_public_state',
                'relative_public_built_cards', 'acknowledged_own_card',
                'acknowledged_own_action_type', 'applied_skipped_fallback'),
        packing=('current_hand_and_known', 'legal_mask', 'action_features', 'frames', 'valid'))
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def card_counts(ids):
    counts = np.zeros(CARD_COUNT, dtype=np.float32)
    for card in ids:
        if int(card) == -1:
            continue
        if int(card) not in _CARD_INDEX:
            raise ValueError(f'Unknown observed hand card: {card}')
        counts[_CARD_INDEX[int(card)]] += 1
    return counts


def hand_is_observable(status, seat):
    """The engine's extra exchange must not reveal an unplayable seventh hand."""
    game = status['game']
    if game.get('phase') == 'finished':
        return False
    if int(game['turn']) % 7 != 6:
        return True
    return any(effect.get('type') == 'play_seventh_card'
               for effect in wonder_effects(status['players'][str(seat)]))


def hide_skipped_hand(observation, *, terminal=False):
    """Provide a neutral no-op choice for an exchange with no actual decision."""
    obs = dict(observation)
    obs['hand_ids'] = np.full(7, -1, dtype=np.int32)
    obs['action_feats'] = np.zeros_like(obs['action_feats'])
    obs['wonder_action_feats'] = np.zeros_like(obs['wonder_action_feats'])
    obs['action_mask'] = np.zeros_like(obs['action_mask'])
    if not terminal:
        obs['action_mask'][21] = 1  # Engine skips this discard command.
    return obs


class ObservationHistory:
    """One instance per game and seat; never attached to a shared policy."""

    def __init__(self):
        self.frames = np.zeros((HISTORY_LENGTH, FRAME_DIM), dtype=np.float32)
        self.valid = np.zeros(HISTORY_LENGTH, dtype=np.float32)
        self.turn = -1

    def observe(self, turn, observation, *, hand_known=True):
        if turn == HISTORY_LENGTH:  # Finished state has no next decision.
            return
        if not 0 <= turn < HISTORY_LENGTH or turn < self.turn:
            raise ValueError('Observation history requires increasing game-local turns')
        if self.valid[turn]:
            return  # Repeated inference/feature requests are read-only.
        if turn != self.turn + 1:
            raise ValueError('Observation history is missing an earlier engine turn')
        from sevenwonders_rl.utils.ppo_utils import flatten_global
        row = self.frames[turn]
        row[HAND] = card_counts(observation['hand_ids'])
        row[HAND_KNOWN] = hand_known
        row[PUBLIC] = flatten_global(observation)
        public = observation['public_state'].reshape(3, len(PLAYER_NAMES))
        row[BUILT_CARDS] = public[:, _PUBLIC_CARD_COLUMNS].reshape(-1)
        self.valid[turn] = 1
        self.turn = turn

    def acknowledge(self, turn, card, kind, outcome):
        if not 0 <= turn < HISTORY_LENGTH or not self.valid[turn]:
            raise ValueError('Own action has no preceding observed frame')
        if self.frames[turn, ACK].any():
            raise ValueError('Own action was already acknowledged')
        outcomes = ('applied', 'skipped', 'fallback_discard')
        if outcome not in outcomes:
            raise ValueError(f'Unknown own action acknowledgement: {outcome}')
        self.frames[turn, ACK.start + outcomes.index(outcome)] = 1
        if outcome != 'skipped':
            if card is None or kind is None:
                raise ValueError('Applied action is missing its own observed card')
            self.frames[turn, OWN_CARD] = card_counts([card])
            self.frames[turn, OWN_KIND.start + (3 if outcome == 'fallback_discard' else kind)] = 1

    def payload(self, observation, *, hand_known=True):
        result = np.zeros(MEMORY_DIM, dtype=np.float32)
        result[CURRENT_HAND] = np.concatenate((card_counts(observation['hand_ids']), [hand_known]))
        result[MASK] = observation['action_mask']
        result[ACTION_FEATURES] = np.concatenate(
            (observation['action_feats'], observation['wonder_action_feats']), axis=-1).reshape(-1)
        result[FRAMES] = self.frames.reshape(-1)
        result[VALID] = self.valid
        result.flags.writeable = False
        return result
