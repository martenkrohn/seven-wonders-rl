"""Action feature schema V2 for per-action feature extraction.

This module defines the NEW feature schema with:
- Per-resource breakdowns (no aggregation)
- Full game state context
- Opponent modeling
- Current science state

Philosophy: No lossy aggregation. Preserve all information.
"""

from typing import Dict, List

# Resource ordering (matches RESOURCE_ORDER in cards.py)
RESOURCE_ORDER = ['wood', 'ore', 'clay', 'stone', 'loom', 'glass', 'papyrus']
SCIENCE_SYMBOLS = ['compass', 'gear', 'tablet']
CARD_COLORS = ['brown', 'grey', 'blue', 'yellow', 'red', 'green', 'purple']
ACTION_TYPES = ['build_structure', 'build_wonder', 'build_hand_free', 'discard']
CARD_ERAS = [1, 2, 3]

# Known discount flags
DISCOUNT_FLAGS = ['raw_cheap_east', 'raw_cheap_west', 'manuf_cheap']


def _build_feature_names() -> tuple[List[str], Dict[str, int]]:
    """Build feature names list and index mapping.
    
    Returns:
        (feature_names, index_dict)
    """
    names = []
    idx = {}
    
    # ========================================================================
    # CATEGORY 1: UNIVERSAL HEADER (2 features)
    # ========================================================================
    idx['is_legal'] = len(names)
    names.append('is_legal')
    
    idx['slot_is_empty'] = len(names)
    names.append('slot_is_empty')
    
    # ========================================================================
    # CATEGORY 2: ACTION & CARD IDENTITY (15 features)
    # ========================================================================
    
    # Action type one-hot (4 features)
    idx['action_type_start'] = len(names)
    for i, at in enumerate(ACTION_TYPES):
        idx[f'action_type_{i}'] = len(names)
        names.append(f'action_type_{i}')
    
    # Card ID normalized
    idx['card_id_norm'] = len(names)
    names.append('card_id_norm')
    
    # Card color one-hot (7 features)
    idx['card_color_start'] = len(names)
    for color in CARD_COLORS:
        idx[f'card_color_{color}'] = len(names)
        names.append(f'card_color_{color}')
    
    # Card era one-hot (3 features)
    idx['card_era_start'] = len(names)
    for era in CARD_ERAS:
        idx[f'card_era_{era}'] = len(names)
        names.append(f'card_era_{era}')
    
    # ========================================================================
    # CATEGORY 3: CARD COSTS - PER RESOURCE (8 features)
    # ========================================================================
    idx['card_cost_start'] = len(names)
    for res in RESOURCE_ORDER:
        idx[f'card_cost_{res}'] = len(names)
        names.append(f'card_cost_{res}')
    
    idx['card_cost_coins'] = len(names)
    names.append('card_cost_coins')
    
    # ========================================================================
    # CATEGORY 4: OWN PRODUCTION - PER RESOURCE (7 features)
    # ========================================================================
    idx['own_prod_start'] = len(names)
    for res in RESOURCE_ORDER:
        idx[f'own_prod_{res}'] = len(names)
        names.append(f'own_prod_{res}')
    
    # ========================================================================
    # CATEGORY 5: NEIGHBOR SUPPLY - PER RESOURCE, PER NEIGHBOR (14 features)
    # ========================================================================
    idx['left_supply_start'] = len(names)
    for res in RESOURCE_ORDER:
        idx[f'left_supply_{res}'] = len(names)
        names.append(f'left_supply_{res}')
    
    idx['right_supply_start'] = len(names)
    for res in RESOURCE_ORDER:
        idx[f'right_supply_{res}'] = len(names)
        names.append(f'right_supply_{res}')
    
    # ========================================================================
    # CATEGORY 6: TRADE ESTIMATION (4 features)
    # ========================================================================
    idx['est_trade_cost_left'] = len(names)
    names.append('est_trade_cost_left')
    
    idx['est_trade_cost_right'] = len(names)
    names.append('est_trade_cost_right')
    
    idx['coins_after_card_cost'] = len(names)
    names.append('coins_after_card_cost')
    
    idx['coins_after_est_total_cost'] = len(names)
    names.append('coins_after_est_total_cost')
    
    # ========================================================================
    # CATEGORY 7: CURRENT GAME STATE (7 features)
    # ========================================================================
    idx['age_1'] = len(names)
    names.append('age_1')
    
    idx['age_2'] = len(names)
    names.append('age_2')
    
    idx['age_3'] = len(names)
    names.append('age_3')
    
    idx['round_number_norm'] = len(names)
    names.append('round_number_norm')
    
    idx['cards_remaining_norm'] = len(names)
    names.append('cards_remaining_norm')
    
    idx['military_balance_left'] = len(names)
    names.append('military_balance_left')
    
    idx['military_balance_right'] = len(names)
    names.append('military_balance_right')
    
    # ========================================================================
    # CATEGORY 8: CURRENT SCIENCE STATE - SELF (3 features)
    # ========================================================================
    idx['own_science_start'] = len(names)
    for sym in SCIENCE_SYMBOLS:
        idx[f'own_science_{sym}'] = len(names)
        names.append(f'own_science_{sym}')
    
    # ========================================================================
    # CATEGORY 9: OPPONENT STATE - PER OPPONENT (10 features)
    # ========================================================================
    
    # Left opponent (5 features)
    idx['left_military_shields'] = len(names)
    names.append('left_military_shields')
    
    idx['left_science_start'] = len(names)
    for sym in SCIENCE_SYMBOLS:
        idx[f'left_science_{sym}'] = len(names)
        names.append(f'left_science_{sym}')
    
    idx['left_wonder_progress'] = len(names)
    names.append('left_wonder_progress')
    
    # Right opponent (5 features)
    idx['right_military_shields'] = len(names)
    names.append('right_military_shields')
    
    idx['right_science_start'] = len(names)
    for sym in SCIENCE_SYMBOLS:
        idx[f'right_science_{sym}'] = len(names)
        names.append(f'right_science_{sym}')
    
    idx['right_wonder_progress'] = len(names)
    names.append('right_wonder_progress')
    
    # ========================================================================
    # CATEGORY 10: CARD EFFECTS - DELTAS (16 features)
    # ========================================================================
    
    # Military & VP
    idx['delta_shields'] = len(names)
    names.append('delta_shields')
    
    idx['delta_vp_fixed'] = len(names)
    names.append('delta_vp_fixed')
    
    # Science deltas (3 features)
    idx['delta_science_start'] = len(names)
    for sym in SCIENCE_SYMBOLS:
        idx[f'delta_science_{sym}'] = len(names)
        names.append(f'delta_science_{sym}')
    
    # Economics
    idx['delta_gain_coins'] = len(names)
    names.append('delta_gain_coins')
    
    # Production deltas (7 resources)
    idx['delta_prod_start'] = len(names)
    for res in RESOURCE_ORDER:
        idx[f'delta_prod_{res}'] = len(names)
        names.append(f'delta_prod_{res}')
    
    # Choice production (3 features)
    idx['delta_prod_choice_raw'] = len(names)
    names.append('delta_prod_choice_raw')
    
    idx['delta_prod_choice_manuf'] = len(names)
    names.append('delta_prod_choice_manuf')
    
    idx['delta_choice_tradable'] = len(names)
    names.append('delta_choice_tradable')
    
    # ========================================================================
    # CATEGORY 11: STRATEGIC FLAGS (3 features)
    # ========================================================================
    idx['adds_discount_start'] = len(names)
    for flag in DISCOUNT_FLAGS:
        idx[f'adds_discount_{flag}'] = len(names)
        names.append(f'adds_discount_{flag}')
    
    # ========================================================================
    # CATEGORY 12: CHAIN FEATURES (2 features)
    # ========================================================================
    idx['is_chain_free_now'] = len(names)
    names.append('is_chain_free_now')
    
    idx['enables_chain_to_count'] = len(names)
    names.append('enables_chain_to_count')
    
    # ========================================================================
    # CATEGORY 13: DISCARD (1 feature)
    # ========================================================================
    idx['discard_delta_coins'] = len(names)
    names.append('discard_delta_coins')
    
    return names, idx


# Build the schema once at module load time
FEATURE_NAMES, IDX = _build_feature_names()
K = len(FEATURE_NAMES)

# Define header column indices (always filled, even for illegal actions)
HEADER_IDXS = frozenset({
    IDX['is_legal'],
    IDX['slot_is_empty'],
    IDX['action_type_0'],
    IDX['action_type_1'],
    IDX['action_type_2'],
    IDX['action_type_3'],
    IDX['card_id_norm'],
    IDX['card_color_brown'],
    IDX['card_color_grey'],
    IDX['card_color_blue'],
    IDX['card_color_yellow'],
    IDX['card_color_red'],
    IDX['card_color_green'],
    IDX['card_color_purple'],
    IDX['card_era_1'],
    IDX['card_era_2'],
    IDX['card_era_3'],
})

# Export convenience constants
__all__ = [
    'K',
    'FEATURE_NAMES',
    'IDX',
    'HEADER_IDXS',
    'RESOURCE_ORDER',
    'SCIENCE_SYMBOLS',
    'CARD_COLORS',
    'ACTION_TYPES',
    'CARD_ERAS',
    'DISCOUNT_FLAGS',
]
