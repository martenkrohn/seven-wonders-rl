"""Evaluation-only action restrictions; preserve the model's state inputs."""
import numpy as np

from sevenwonders_rl.data.cards import by_id

ACTION_CONSTRAINTS = ('none', 'green_priority')


def green_build_mask(obs):
    mask = np.zeros_like(obs['action_mask'])
    for slot, card_id in enumerate(obs['hand_ids'][:7]):
        card = by_id(int(card_id))
        if card and card.get('color') == 'green':
            for kind in (0, 2):  # paid and free building, never wonder or discard
                index = kind * 7 + slot
                mask[index] = obs['action_mask'][index]
    return mask


def constrained_mask(obs, mode='none'):
    if mode not in ACTION_CONSTRAINTS:
        raise ValueError(f'Unknown action constraint: {mode}')
    if mode == 'green_priority':
        green = green_build_mask(obs)
        if green.any():
            return green
    return obs['action_mask']
