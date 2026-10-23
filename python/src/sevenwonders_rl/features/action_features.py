"""Per-action feature builder V2 for 7 Wonders.

This module computes a (28, K) feature matrix with:
- Per-resource breakdowns (no aggregation)
- Full game state context
- Opponent modeling
- Current science state

Philosophy: No lossy aggregation. Preserve all information.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
import numpy as np
from sevenwonders_rl.data.cards import by_id, by_name_strict, get_chain_to_count
from sevenwonders_rl.data.wonders import wonder_by_id
from sevenwonders_rl.data.mechanics import resource_kind
from sevenwonders_rl.features.action_feature_schema import (
    FEATURE_NAMES, IDX, K, RESOURCE_ORDER, SCIENCE_SYMBOLS,
    CARD_COLORS, ACTION_TYPES, CARD_ERAS, DISCOUNT_FLAGS
)
from sevenwonders_rl.features.trade_estimator import estimate_trade_cost, TradeEstimate

N_SLOTS = 7
N_ACTIONS = 28


class PlayerContext:
    """Precomputed context for a single player."""
    
    def __init__(
        self,
        current_coins: int,
        discount_flags: Set[str],
        built_card_ids: List[int],
        own_production: Dict[str, int],
        own_choice_resources: List[Set[str]],
        left_supply: Dict[str, int],
        left_choice_supply: List[Set[str]],
        right_supply: Dict[str, int],
        right_choice_supply: List[Set[str]],
        wonder_id: int,
        wonder_stage: int,
        # New V2 fields
        own_shields: int,
        own_science: Dict[str, int],
        left_shields: int,
        left_science: Dict[str, int],
        left_wonder_stages: int,
        left_wonder_max_stages: int,
        right_shields: int,
        right_science: Dict[str, int],
        right_wonder_stages: int,
        right_wonder_max_stages: int,
    ):
        self.current_coins = current_coins
        self.discount_flags = discount_flags
        self.built_card_ids = built_card_ids
        self.own_production = own_production
        self.own_choice_resources = own_choice_resources
        self.left_supply = left_supply
        self.left_choice_supply = left_choice_supply
        self.right_supply = right_supply
        self.right_choice_supply = right_choice_supply
        self.wonder_id = wonder_id
        self.wonder_stage = wonder_stage
        # V2 additions
        self.own_shields = own_shields
        self.own_science = own_science
        self.left_shields = left_shields
        self.left_science = left_science
        self.left_wonder_stages = left_wonder_stages
        self.left_wonder_max_stages = left_wonder_max_stages
        self.right_shields = right_shields
        self.right_science = right_science
        self.right_wonder_stages = right_wonder_stages
        self.right_wonder_max_stages = right_wonder_max_stages


def _seat_order(agent_seat: int) -> List[int]:
    """Return player seats in order: [self, left, right]."""
    return [agent_seat, (agent_seat - 1) % 3, (agent_seat + 1) % 3]


def _resolve_built_card_ids(status_player: Dict[str, Any]) -> List[int]:
    """Resolve card IDs from played card names."""
    card_ids = []
    for card_name in status_player.get('cards_played', []):
        card = by_name_strict(card_name)
        card_ids.append(card['id'])
    return card_ids


def _get_card_production(card_id: int) -> Dict[str, int]:
    """Get FIXED production from a card.
    
    Returns only fixed production (from produces_fixed field).
    Choice production is handled separately via neighbor choice supply.
    """
    card = by_id(card_id)
    if not card:
        return {}
    
    prod = {}
    effects = card.get('effects', {})
    
    # Get fixed production
    produces_fixed = effects.get('produces_fixed')
    if produces_fixed:
        if isinstance(produces_fixed, str):
            # Single resource
            prod[produces_fixed] = 1
        elif isinstance(produces_fixed, dict):
            # Multiple resources
            for res, amount in produces_fixed.items():
                if res in RESOURCE_ORDER:
                    prod[res] = amount
    
    return prod


def _compute_neighbor_choice_supply(built_card_ids: List[int]) -> List[Set[str]]:
    """Compute available choice production from neighbor's brown/grey cards.
    
    Returns list of choice sets (each set is options from one card).
    For example: [{'stone', 'wood'}, {'clay', 'ore'}] if neighbor has 2 choice cards.
    Only includes choice production from brown/grey cards (tradable).
    """
    choice_sets = []
    
    for card_id in built_card_ids:
        card = by_id(card_id)
        if not card:
            continue
        
        # Only brown (raw) and grey (manufactured) cards are tradable
        if card['color'] not in ['brown', 'grey']:
            continue
        
        effects = card.get('effects', {})
        produces_choice = effects.get('produces_choice')
        is_tradable = effects.get('choice_tradable', False)
        
        if not produces_choice or not is_tradable:
            continue
        
        # Create a set of available choice options for this card
        if isinstance(produces_choice, list):
            choice_set = set(res for res in produces_choice if res in RESOURCE_ORDER)
            if choice_set:
                choice_sets.append(choice_set)
    
    return choice_sets


def _extract_wonder_choice_resources(wonder_id: int, wonder_stages: int) -> List[Set[str]]:
    """Extract choice production from built wonder stages.
    
    Returns list of choice sets from wonder stage effects like:
    - choose_raw_nontradable: {'wood', 'ore', 'clay', 'stone'}
    - choose_manuf_nontradable: {'loom', 'glass', 'papyrus'}
    """
    choice_sets = []
    
    if not wonder_id or wonder_stages <= 0:
        return choice_sets
    
    wonder = wonder_by_id(wonder_id)
    if not wonder:
        return choice_sets
    
    stages = wonder.get('stages', [])
    for stage_idx in range(min(wonder_stages, len(stages))):
        stage = stages[stage_idx]
        effects = stage.get('effects', [])
        
        for effect in effects:
            if not isinstance(effect, dict):
                continue
            
            effect_type = effect.get('type')
            if effect_type == 'choose_raw_nontradable':
                # Can choose from all raw resources
                choice_sets.append({'wood', 'ore', 'clay', 'stone'})
            elif effect_type == 'choose_manuf_nontradable':
                # Can choose from all manufactured resources
                choice_sets.append({'loom', 'glass', 'papyrus'})
    
    return choice_sets


def _extract_own_choice_resources(built_card_ids: List[int]) -> List[Set[str]]:
    """Extract choice production from player's own built cards (all colors).
    
    Returns list of choice sets (each set is options from one card).
    Example: [{'stone', 'wood'}] if player has Timber Yard.
    Includes ALL card colors (brown, grey, yellow, etc).
    """
    choice_sets = []
    
    for card_id in built_card_ids:
        card = by_id(card_id)
        if not card:
            continue
        
        effects = card.get('effects', {})
        produces_choice = effects.get('produces_choice')
        
        if not produces_choice:
            continue
        
        # Create a set of available choice options for this card
        if isinstance(produces_choice, list):
            choice_set = set(res for res in produces_choice if res in RESOURCE_ORDER)
            if choice_set:
                choice_sets.append(choice_set)
    
    return choice_sets


def _compute_tradable_supply(
    player: Dict[str, Any],
    built_card_ids: List[int],
    choice_resources: List[Set[str]],
) -> Dict[str, int]:
    """Compute tradable resource supply (brown/grey cards + wonder).
    
    Tradable resources include:
    - Brown/grey card production
    - Wonder starting resource (always available)
    - Wonder stage production (based on stages_built)
    """
    supply = {res: 0 for res in RESOURCE_ORDER}
    
    # Add brown/grey card production
    for card_id in built_card_ids:
        card = by_id(card_id)
        if not card:
            continue
        
        # Only brown (raw) and grey (manufactured) cards are tradable
        if card['color'] not in ['brown', 'grey']:
            continue
        
        # Get production from this card
        card_prod = _get_card_production(card_id)
        for res, amount in card_prod.items():
            supply[res] += amount
    
    # Add wonder starting resource (always available, tradable)
    wonder_id = player.get('wonder_id')
    if wonder_id:
        wonder = wonder_by_id(wonder_id)
        if wonder:
            starting_res = wonder.get('starting_resource')
            if starting_res and starting_res in RESOURCE_ORDER:
                supply[starting_res] += 1
    
    # Add wonder stage production (for each built stage)
    wonder_stages = player.get('wonder_stage', 0)
    if wonder_id and wonder_stages > 0:
        wonder = wonder_by_id(wonder_id)
        if wonder:
            stages = wonder.get('stages', [])
            # Each stage built might produce resources (check effects)
            for stage_idx in range(min(wonder_stages, len(stages))):
                stage = stages[stage_idx]
                effects = stage.get('effects', [])
                # Check for production effects (if any)
                for effect in effects:
                    if isinstance(effect, dict) and effect.get('type') == 'add_production':
                        # Some wonders give production as effect
                        for res, amount in effect.get('resources', {}).items():
                            if res in RESOURCE_ORDER:
                                supply[res] += amount
    
    # Add choice production (from brown/grey cards)
    # Choice resources are already filtered to only tradable ones
    for choice_set in choice_resources:
        for res in choice_set:
            if res in RESOURCE_ORDER:
                supply[res] += 1
    
    return supply


def _extract_card_effects(card_id: int) -> Dict[str, int]:
    """Extract immediate effect deltas from a card."""
    card = by_id(card_id)
    if not card:
        return {}
    
    effects = card.get('effects', {}) or {}
    vp = card.get('vp', {}) or {}
    
    deltas = {
        'delta_shields': effects.get('shields', 0),
        'delta_gain_coins': effects.get('gain_coins', 0),
        'delta_vp_fixed': vp.get('fixed', 0),
    }
    
    # Science symbols (stored as list in DB, e.g. ['compass'])
    science = effects.get('science', []) or []
    for sym in SCIENCE_SYMBOLS:
        deltas[f'delta_science_{sym}'] = 1 if sym in science else 0
    
    # Production deltas
    for res in RESOURCE_ORDER:
        prod = _get_card_production(card_id)
        deltas[f'delta_prod_{res}'] = prod.get(res, 0)
    
    # Choice production
    choice_prod = effects.get('choice_production', {}) or {}
    if choice_prod:
        # Check if it's raw or manufactured
        options = choice_prod.get('options', [])
        raw_options = sum(1 for r in options if resource_kind(r) == 'raw')
        manuf_options = sum(1 for r in options if resource_kind(r) == 'manuf')
        
        if raw_options > 0:
            deltas['delta_prod_choice_raw'] = 1
        if manuf_options > 0:
            deltas['delta_prod_choice_manuf'] = 1
        
        deltas['delta_choice_tradable'] = 1 if choice_prod.get('tradable', False) else 0
    else:
        deltas['delta_prod_choice_raw'] = 0
        deltas['delta_prod_choice_manuf'] = 0
        deltas['delta_choice_tradable'] = 0
    
    return deltas


def _extract_science_counts(built_card_ids: List[int]) -> Dict[str, int]:
    """Extract current science symbol counts from built cards."""
    counts = {sym: 0 for sym in SCIENCE_SYMBOLS}
    
    for card_id in built_card_ids:
        card = by_id(card_id)
        if not card:
            continue
        
        effects = card.get('effects', {}) or {}
        science = effects.get('science', []) or []  # List of symbols
        
        for sym in SCIENCE_SYMBOLS:
            if sym in science:
                counts[sym] += 1
    
    return counts


def _build_player_context(
    status: Dict[str, Any],
    agent_seat: int,
    player_idx: int,
) -> PlayerContext:
    """Build player context for the agent (player_idx=0) or neighbors.
    
    Args:
        status: game_status.json
        agent_seat: Which seat is the agent
        player_idx: 0=agent, 1=left neighbor, 2=right neighbor
    """
    order = _seat_order(agent_seat)
    seat_id = order[player_idx]
    p = status['players'][str(seat_id)]
    
    # Get built cards
    built_card_ids = _resolve_built_card_ids(p)
    
    # Basic info
    current_coins = p.get('resources', {}).get('coins', 0)
    
    # Discount flags
    discount_flags = set()
    if p.get('flags', {}).get('raw_cheap_east', False):
        discount_flags.add('raw_cheap_east')
    if p.get('flags', {}).get('raw_cheap_west', False):
        discount_flags.add('raw_cheap_west')
    if p.get('flags', {}).get('manuf_cheap', False):
        discount_flags.add('manuf_cheap')
    
    # Own production (all production, including non-tradable)
    own_production = {}
    for res in RESOURCE_ORDER:
        own_production[res] = p.get('resources', {}).get(res, 0)
    
    # Wonder info (get early for choice resource extraction)
    wonder_id = p.get('wonder_id', 0)
    wonder_stage = p.get('wonder_stage', 0)
    
    # Choice resources from player's built cards AND wonder stages (for trade planning)
    own_choice_resources = _extract_own_choice_resources(built_card_ids)
    own_choice_resources.extend(_extract_wonder_choice_resources(wonder_id, wonder_stage))
    
    # Shields
    own_shields = p.get('resources', {}).get('shields', 0)
    
    # Science
    own_science = _extract_science_counts(built_card_ids)
    
    # Neighbor supply (tradable only)
    if player_idx == 0:
        # For agent, get left and right neighbor supply
        left_seat = order[1]
        right_seat = order[2]
        
        left_p = status['players'][str(left_seat)]
        right_p = status['players'][str(right_seat)]
        
        left_built = _resolve_built_card_ids(left_p)
        right_built = _resolve_built_card_ids(right_p)
        
        left_choice_supply = _compute_neighbor_choice_supply(left_built)
        right_choice_supply = _compute_neighbor_choice_supply(right_built)
        
        left_supply = _compute_tradable_supply(left_p, left_built, [])
        right_supply = _compute_tradable_supply(right_p, right_built, [])
        
        left_shields = left_p.get('resources', {}).get('shields', 0)
        right_shields = right_p.get('resources', {}).get('shields', 0)
        
        left_science = _extract_science_counts(left_built)
        right_science = _extract_science_counts(right_built)
        
        # Wonder progress
        left_wonder = wonder_by_id(left_p.get('wonder_id', 0))
        right_wonder = wonder_by_id(right_p.get('wonder_id', 0))
        
        left_wonder_stages = left_p.get('wonder_stage', 0)
        left_wonder_max_stages = len(left_wonder['stages']) if left_wonder else 1
        
        right_wonder_stages = right_p.get('wonder_stage', 0)
        right_wonder_max_stages = len(right_wonder['stages']) if right_wonder else 1
        
    else:
        # For neighbors, we don't need their neighbor info
        left_supply = {res: 0 for res in RESOURCE_ORDER}
        right_supply = {res: 0 for res in RESOURCE_ORDER}
        left_shields = 0
        right_shields = 0
        left_science = {sym: 0 for sym in SCIENCE_SYMBOLS}
        right_science = {sym: 0 for sym in SCIENCE_SYMBOLS}
        left_wonder_stages = 0
        left_wonder_max_stages = 1
        right_wonder_stages = 0
        right_wonder_max_stages = 1
        left_choice_supply = []
        right_choice_supply = []
    
    return PlayerContext(
        current_coins=current_coins,
        discount_flags=discount_flags,
        built_card_ids=built_card_ids,
        own_production=own_production,
        own_choice_resources=own_choice_resources,
        left_supply=left_supply,
        left_choice_supply=left_choice_supply,
        right_supply=right_supply,
        right_choice_supply=right_choice_supply,
        wonder_id=wonder_id,
        wonder_stage=wonder_stage,
        own_shields=own_shields,
        own_science=own_science,
        left_shields=left_shields,
        left_science=left_science,
        left_wonder_stages=left_wonder_stages,
        left_wonder_max_stages=left_wonder_max_stages,
        right_shields=right_shields,
        right_science=right_science,
        right_wonder_stages=right_wonder_stages,
        right_wonder_max_stages=right_wonder_max_stages,
    )


def _fill_game_state_feats(
    feats: np.ndarray,
    action_idx: int,
    status: Dict[str, Any],
    ctx: PlayerContext,
    agent_seat: int,
) -> None:
    """Fill game state features (age, round, cards remaining, military balance)."""
    # Age one-hot
    current_era = status['game']['era']
    feats[action_idx, IDX['age_1']] = 1.0 if current_era == 1 else 0.0
    feats[action_idx, IDX['age_2']] = 1.0 if current_era == 2 else 0.0
    feats[action_idx, IDX['age_3']] = 1.0 if current_era == 3 else 0.0
    
    # Round number (normalized 0-1, typically 6 rounds per age)
    current_round = status['game']['turn'] % 7
    max_rounds = 6  # Typically 6 rounds per age
    feats[action_idx, IDX['round_number_norm']] = min(current_round / max_rounds, 1.0)
    
    # Cards remaining in hand (normalized)
    hand_size = len(status['players'][str(agent_seat)].get('cards_hand', []))
    feats[action_idx, IDX['cards_remaining_norm']] = hand_size / 7.0
    
    # Military balance (your shields - neighbor's shields)
    feats[action_idx, IDX['military_balance_left']] = float(ctx.own_shields - ctx.left_shields)
    feats[action_idx, IDX['military_balance_right']] = float(ctx.own_shields - ctx.right_shields)


def _fill_science_state_feats(
    feats: np.ndarray,
    action_idx: int,
    ctx: PlayerContext,
) -> None:
    """Fill current science state features."""
    for sym in SCIENCE_SYMBOLS:
        feats[action_idx, IDX[f'own_science_{sym}']] = float(ctx.own_science.get(sym, 0))


def _fill_opponent_state_feats(
    feats: np.ndarray,
    action_idx: int,
    ctx: PlayerContext,
) -> None:
    """Fill opponent state features (military, science, wonder progress)."""
    # Left opponent
    feats[action_idx, IDX['left_military_shields']] = float(ctx.left_shields)
    for sym in SCIENCE_SYMBOLS:
        feats[action_idx, IDX[f'left_science_{sym}']] = float(ctx.left_science.get(sym, 0))
    feats[action_idx, IDX['left_wonder_progress']] = ctx.left_wonder_stages / max(ctx.left_wonder_max_stages, 1)
    
    # Right opponent
    feats[action_idx, IDX['right_military_shields']] = float(ctx.right_shields)
    for sym in SCIENCE_SYMBOLS:
        feats[action_idx, IDX[f'right_science_{sym}']] = float(ctx.right_science.get(sym, 0))
    feats[action_idx, IDX['right_wonder_progress']] = ctx.right_wonder_stages / max(ctx.right_wonder_max_stages, 1)


def _fill_card_identity_feats(
    feats: np.ndarray,
    action_idx: int,
    card: Dict[str, Any],
) -> None:
    """Fill card identity features (ID, color, era)."""
    card_id = card['id']
    max_card_id = 70  # Approximate max card ID
    feats[action_idx, IDX['card_id_norm']] = card_id / max_card_id
    
    # Color one-hot
    color = card.get('color', '')
    if color in CARD_COLORS:
        feats[action_idx, IDX[f'card_color_{color}']] = 1.0
    
    # Era one-hot
    era = card.get('era', 1)
    if era in CARD_ERAS:
        feats[action_idx, IDX[f'card_era_{era}']] = 1.0


def _fill_build_structure_feats(
    feats: np.ndarray,
    action_idx: int,
    card: Dict[str, Any],
    card_name: str,
    ctx: PlayerContext,
    playable_set: Set[str],
) -> None:
    """Fill features for building a structure (card from hand)."""
    # Card identity
    _fill_card_identity_feats(feats, action_idx, card)
    
    # Card costs - PER RESOURCE (not aggregated!)
    cost_dict = card.get('cost', {})
    coins_cost = cost_dict.get('coins', 0)
    
    for res in RESOURCE_ORDER:
        feats[action_idx, IDX[f'card_cost_{res}']] = float(cost_dict.get(res, 0))
    
    feats[action_idx, IDX['card_cost_coins']] = float(coins_cost)
    
    # Own production - PER RESOURCE (not aggregated!)
    for res in RESOURCE_ORDER:
        feats[action_idx, IDX[f'own_prod_{res}']] = float(ctx.own_production.get(res, 0))
    
    # Neighbor supply - PER RESOURCE, PER NEIGHBOR (not aggregated!)
    for res in RESOURCE_ORDER:
        feats[action_idx, IDX[f'left_supply_{res}']] = float(ctx.left_supply.get(res, 0))
        feats[action_idx, IDX[f'right_supply_{res}']] = float(ctx.right_supply.get(res, 0))
    
    # Trade estimation
    is_playable = (card_name in playable_set)
    if is_playable:
        trade_est = estimate_trade_cost(
            required=cost_dict,
            own_production=ctx.own_production,
            own_choice_resources=ctx.own_choice_resources,
            left_supply=ctx.left_supply,
            left_choice_supply=ctx.left_choice_supply,
            right_supply=ctx.right_supply,
            right_choice_supply=ctx.right_choice_supply,
            player_flags=ctx.discount_flags,
        )
    else:
        trade_est = TradeEstimate(0, 0, 0, 0, 0, {}, {}, True)
    
    feats[action_idx, IDX['est_trade_cost_left']] = float(trade_est.cost_left)
    feats[action_idx, IDX['est_trade_cost_right']] = float(trade_est.cost_right)
    feats[action_idx, IDX['coins_after_card_cost']] = float(ctx.current_coins - coins_cost)
    feats[action_idx, IDX['coins_after_est_total_cost']] = float(
        ctx.current_coins - coins_cost - trade_est.total_cost
    )
    
    # Card effect deltas
    card_deltas = _extract_card_effects(card['id'])
    for key, val in card_deltas.items():
        if key in IDX:
            feats[action_idx, IDX[key]] = float(val)
    
    # Strategic flags (discounts)
    for flag in DISCOUNT_FLAGS:
        flag_key = f'adds_discount_{flag}'
        if flag_key in IDX:
            card_flags = card.get('flags', {})
            feats[action_idx, IDX[flag_key]] = 1.0 if card_flags.get(flag, False) else 0.0
    
    # Chain features
    chain_prereq = card.get('chain_prerequisite', '')
    if chain_prereq:
        feats[action_idx, IDX['is_chain_free_now']] = 1.0 if card_name in playable_set else 0.0
    
    chain_to_count = get_chain_to_count(card['id'])
    feats[action_idx, IDX['enables_chain_to_count']] = float(chain_to_count)


def _fill_discard_feats(
    feats: np.ndarray,
    action_idx: int,
    status: Dict[str, Any],
) -> None:
    """Fill discard features (coins gained)."""
    current_era = status['game']['era']
    discard_coins = 3  # Always 3 coins for discarding
    feats[action_idx, IDX['discard_delta_coins']] = float(discard_coins)


def build_action_feats(
    status: Dict[str, Any],
    agent_seat: int,
    action_mask: np.ndarray,
) -> np.ndarray:
    """Build per-action feature matrix V2.
    
    Args:
        status: game_status.json dict
        agent_seat: Which seat is the agent (0, 1, or 2)
        action_mask: Legal action mask (28,)
    
    Returns:
        Feature matrix of shape (28, K) with dtype float32
    """
    feats = np.zeros((N_ACTIONS, K), dtype=np.float32)
    
    # Build player context
    ctx = _build_player_context(status, agent_seat, player_idx=0)
    
    # Get hand
    p = status['players'][str(agent_seat)]
    hand_names = p['cards_hand']
    hand_ids = p.get('cards_hand_id', [])
    
    # Pad to 7 slots
    hand_names_padded = list(hand_names) + [''] * (N_SLOTS - len(hand_names))
    hand_ids_padded = list(hand_ids) + [-1] * (N_SLOTS - len(hand_ids))
    
    # Get playable cards set
    playable_set = set(p['cards_playable'])
    
    # For each action
    for action_idx in range(N_ACTIONS):
        is_legal = (action_mask[action_idx] == 1)
        
        # Decode action
        action_type = action_idx // N_SLOTS
        slot = action_idx % N_SLOTS
        
        # Check if slot is empty
        slot_empty = (hand_ids_padded[slot] == -1)
        
        # Fill universal header
        feats[action_idx, IDX['is_legal']] = 1.0 if is_legal else 0.0
        feats[action_idx, IDX['slot_is_empty']] = 1.0 if slot_empty else 0.0
        
        # Action type one-hot
        feats[action_idx, IDX[f'action_type_{action_type}']] = 1.0
        
        # If slot empty or action illegal, stop here (don't fill non-header features)
        if slot_empty or not is_legal:
            continue
        
        # Fill game state features (same for all legal actions)
        _fill_game_state_feats(feats, action_idx, status, ctx, agent_seat)
        _fill_science_state_feats(feats, action_idx, ctx)
        _fill_opponent_state_feats(feats, action_idx, ctx)
        
        # Get card info
        card_id = hand_ids_padded[slot]
        card = by_id(card_id) if card_id != -1 else None
        
        if not card:
            continue
        
        card_name = hand_names_padded[slot]
        
        # Fill action-specific features
        if action_type == 0:  # build_structure
            _fill_build_structure_feats(feats, action_idx, card, card_name, ctx, playable_set)
        
        elif action_type == 1:  # build_wonder
            # TODO: Wonder features deferred for now
            pass
        
        elif action_type == 2:  # build_hand_free
            # Same as build_structure but free
            _fill_build_structure_feats(feats, action_idx, card, card_name, ctx, playable_set)
        
        elif action_type == 3:  # discard
            _fill_discard_feats(feats, action_idx, status)
    
    return feats
