"""Base class for feature-based policies with shared scoring infrastructure.

This module provides:
- Context dataclass for game state
- Helper functions for science scoring, military conflicts, VP-equivalent scoring
- Base FeaturePolicy class that all specialist policies inherit from
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from sevenwonders_rl.features.action_feature_schema import IDX
from sevenwonders_rl.data.cards import by_id

# Production targets (same as heuristic)
RAW_TARGET = np.array([2, 2, 2, 2])  # wood, ore, clay, stone
MANUF_TARGET = np.array([1, 1, 1])   # loom, glass, papyrus


def token_value(era: int) -> int:
    """Military token VP value by era."""
    return 1 if era == 1 else 3 if era == 2 else 5


def conflict_value(my_shields: int, neighbor_shields: int, era: int) -> int:
    """VP value of conflict outcome with one neighbor.
    
    Args:
        my_shields: Player's shield count
        neighbor_shields: Neighbor's shield count
        era: Current era (1, 2, or 3)
    
    Returns:
        VP value: +token for win, -1 for loss, 0 for tie
    """
    if my_shields > neighbor_shields:
        return token_value(era)
    if my_shields < neighbor_shields:
        return -1
    return 0


def science_score(gear: int, compass: int, tablet: int) -> int:
    """Total science VP from symbol counts.
    
    Formula: gear² + compass² + tablet² + 7 * min(gear, compass, tablet)
    
    Args:
        gear: Number of gear symbols
        compass: Number of compass symbols
        tablet: Number of tablet symbols
    
    Returns:
        Total science VP
    """
    return gear * gear + compass * compass + tablet * tablet + 7 * min(gear, compass, tablet)


def science_marginal(
    gear: int, compass: int, tablet: int,
    delta_gear: int, delta_compass: int, delta_tablet: int
) -> int:
    """Marginal science VP from adding symbols.
    
    Computes exact marginal: science_score(current + delta) - science_score(current)
    
    Args:
        gear, compass, tablet: Current symbol counts
        delta_gear, delta_compass, delta_tablet: Symbols being added
    
    Returns:
        Marginal VP gain
    """
    before = science_score(gear, compass, tablet)
    after = science_score(gear + delta_gear, compass + delta_compass, tablet + delta_tablet)
    return after - before


# ============================================================================
# VP-EQUIVALENT HELPERS (shared baseline for all policies)
# ============================================================================

def coin_costs_vp(f: np.ndarray, action_type: int) -> float:
    """VP-equivalent cost of coins spent (card + trade).
    
    Args:
        f: Feature vector
        action_type: 0=build, 1=wonder, 2=free, 3=discard
    
    Returns:
        VP cost (coins / 3.0)
    """
    if action_type in (0, 2):  # build_structure / build_hand_free
        return float(f[IDX["card_cost_coins"]] + f[IDX["est_trade_cost_coins"]]) / 3.0
    elif action_type == 1:  # build_wonder
        return float(f[IDX["wonder_est_trade_cost_coins"]]) / 3.0
    return 0.0


def coin_gain_vp(f: np.ndarray, action_type: int) -> float:
    """VP-equivalent value of coins gained.
    
    Args:
        f: Feature vector
        action_type: 0=build, 1=wonder, 2=free, 3=discard
    
    Returns:
        VP value (coins / 3.0)
    """
    if action_type == 3:  # discard
        return float(f[IDX["discard_delta_coins"]]) / 3.0
    return float(f[IDX["delta_gain_coins"]]) / 3.0


def military_swing_vp(f: np.ndarray, ctx: 'PolicyContext', action_type: int) -> float:
    """VP swing from military conflicts.
    
    Computes exact VP change from gaining shields:
    - Considers both left and right neighbors
    - Accounts for era-dependent token values (1/3/5)
    - Handles win/loss/tie transitions
    
    Args:
        f: Feature vector
        ctx: Game state context
        action_type: 0=build, 1=wonder, 2=free, 3=discard
    
    Returns:
        VP swing (can be negative if losing)
    """
    if action_type in (0, 2):  # Card builds
        dsh = int(round(f[IDX["delta_shields"]]))
    elif action_type == 1:  # Wonder
        dsh = int(round(f[IDX["wonder_delta_shields"]]))
    else:
        dsh = 0
    
    if dsh == 0:
        return 0.0
    
    my_after = ctx.my_shields + dsh
    era = ctx.era
    
    before = (
        conflict_value(ctx.my_shields, ctx.left_shields, era) +
        conflict_value(ctx.my_shields, ctx.right_shields, era)
    )
    after = (
        conflict_value(my_after, ctx.left_shields, era) +
        conflict_value(my_after, ctx.right_shields, era)
    )
    
    return float(after - before)


def science_marginal_vp(f: np.ndarray, ctx: 'PolicyContext', action_type: int) -> float:
    """Marginal science VP from this action.
    
    Uses exact science formula, not approximation.
    
    Args:
        f: Feature vector
        ctx: Game state context
        action_type: 0=build, 1=wonder, 2=free, 3=discard
    
    Returns:
        Marginal science VP
    """
    if action_type in (0, 2):  # Card builds
        dg = int(round(f[IDX["delta_science_gear"]]))
        dc = int(round(f[IDX["delta_science_compass"]]))
        dt = int(round(f[IDX["delta_science_tablet"]]))
    elif action_type == 1:  # Wonder
        dg = int(round(f[IDX["wonder_delta_science_gear"]]))
        dc = int(round(f[IDX["wonder_delta_science_compass"]]))
        dt = int(round(f[IDX["wonder_delta_science_tablet"]]))
    else:
        dg = dc = dt = 0
    
    if dg == dc == dt == 0:
        return 0.0
    
    return float(science_marginal(ctx.my_gear, ctx.my_compass, ctx.my_tablet, dg, dc, dt))


def estimate_guild_vp(f: np.ndarray, obs: Dict, action_type: int) -> float:
    """Estimate guild (purple card) VP.
    
    Uses neighbor card counts to estimate conditional VP.
    Only relevant in Age 3.
    
    Args:
        f: Feature vector
        obs: Full observation (for neighbor amounts)
        action_type: 0=build, 1=wonder, 2=free, 3=discard
    
    Returns:
        Estimated guild VP (0 if not Age 3 or not a guild)
    """
    if action_type == 3 or action_type == 1:  # Discard or wonder
        return 0.0
    
    game_meta = obs.get("game_meta")
    if game_meta is None or len(game_meta) < 1:
        return 0.0
    
    era = int(game_meta[0])
    if era != 3:
        return 0.0
    
    table_amount = obs.get("table_amount")
    if table_amount is None or len(table_amount) < 3:
        return 0.0
    
    # Get card ID from features (denormalize)
    card_id = int(round(f[IDX["card_id_norm"]] * 70))
    if card_id <= 0:
        return 0.0
    
    card = by_id(card_id)
    if not card:
        return 0.0
    
    vp = card.get("vp", {})
    conditional = vp.get("conditional")
    if not conditional:
        return 0.0
    
    try:
        cond_type = conditional.get("type")
        vp_per = int(conditional.get("vp_per", 0))
        
        if cond_type == "per_card_type" and vp_per > 0:
            card_type = conditional.get("card_type")
            
            # Map card type to table_amount index
            # table_amount shape: (3, 7) = [civilian, commercial, guild, military, scientific, raw, manuf]
            amount_idx = {
                "materials": 5,      # raw materials (brown)
                "manufactured": 6,   # manufactured goods (grey)
                "military": 3,       # red cards
                "scientific": 4,     # green cards
            }.get(card_type)
            
            if amount_idx is None:
                return 0.0
            
            # Neighbor amounts (seats 1 and 2 = left and right)
            left_count = int(table_amount[1][amount_idx])
            right_count = int(table_amount[2][amount_idx])
            
            return float(vp_per * (left_count + right_count))
    except Exception:
        return 0.0
    
    return 0.0


def prod_deficit_bonus(f: np.ndarray, obs: Dict, era: int) -> float:
    """Production deficit reduction bonus.
    
    Rewards actions that fill production gaps toward targets:
    - Raw: 2 of each (wood, ore, clay, stone)
    - Manuf: 1 of each (loom, glass, papyrus)
    
    Only active in Ages 1-2.
    
    Args:
        f: Feature vector
        obs: Full observation (for current production)
        era: Current era (1, 2, or 3)
    
    Returns:
        Bonus value (0 in Age 3)
    """
    if era == 3:
        return 0.0
    
    table_prod = obs.get("table_prod")
    if table_prod is None or len(table_prod) < 1:
        return 0.0
    
    my_prod = table_prod[0].astype(int)  # [wood, ore, clay, stone, loom, glass, papyrus]
    
    raw_have = my_prod[:4]
    manuf_have = my_prod[4:7]
    
    # Deficits
    raw_deficit = np.maximum(0, RAW_TARGET - raw_have)
    manuf_deficit = np.maximum(0, MANUF_TARGET - manuf_have)
    
    # Deltas from action
    raw_delta = np.array([
        f[IDX["delta_prod_wood"]],
        f[IDX["delta_prod_ore"]],
        f[IDX["delta_prod_clay"]],
        f[IDX["delta_prod_stone"]],
    ], dtype=float)
    
    manuf_delta = np.array([
        f[IDX["delta_prod_loom"]],
        f[IDX["delta_prod_glass"]],
        f[IDX["delta_prod_papyrus"]],
    ], dtype=float)
    
    # Reward only what helps deficits (avoid overcapping)
    raw_help = float(np.minimum(raw_delta, raw_deficit).sum())
    manuf_help = float(np.minimum(manuf_delta, manuf_deficit).sum())
    
    # Choice production: approximate as filling largest deficit
    choice_raw = float(f[IDX["delta_prod_choice_raw"]])
    choice_manuf = float(f[IDX["delta_prod_choice_manuf"]])
    
    if choice_raw > 0 and raw_deficit.max() > 0:
        raw_help += min(choice_raw, float(raw_deficit.max()))
    if choice_manuf > 0 and manuf_deficit.max() > 0:
        manuf_help += min(choice_manuf, float(manuf_deficit.max()))
    
    # Weight raw more in Age 1, manuf more in Age 2
    if era == 1:
        return 2.0 * raw_help + 1.0 * manuf_help
    else:  # era == 2
        return 1.0 * raw_help + 2.0 * manuf_help


def vp_equiv_base(f: np.ndarray, obs: Dict, ctx: 'PolicyContext', action_type: int) -> float:
    """VP-equivalent baseline score (shared by all policies).
    
    This is the foundation: how many VP is this action worth?
    - Fixed VP (blue cards, wonder stages)
    - Guild VP estimate (purple cards in Age 3)
    - Science marginal VP (exact formula)
    - Military swing VP (conflict outcomes)
    - Coin gain/cost VP (at 3:1 ratio)
    
    All policies start with this baseline, then add strategy-specific biases.
    
    Args:
        f: Feature vector
        obs: Full observation
        ctx: Game state context
        action_type: 0=build, 1=wonder, 2=free, 3=discard
    
    Returns:
        VP-equivalent score
    """
    vp_fixed = float(f[IDX["delta_vp_fixed"]])
    wonder_vp = float(f[IDX["wonder_vp_gain"]]) if action_type == 1 else 0.0
    guild_vp = estimate_guild_vp(f, obs, action_type)
    science_vp = science_marginal_vp(f, ctx, action_type)
    mil_vp = military_swing_vp(f, ctx, action_type)
    coin_vp = coin_gain_vp(f, action_type) - coin_costs_vp(f, action_type)
    
    return vp_fixed + wonder_vp + guild_vp + science_vp + mil_vp + coin_vp


@dataclass
class PolicyContext:
    """Game state context for policy decision-making."""
    era: int  # 1, 2, or 3
    t_in_age: int  # 0-6 (turn within age)
    my_coins: int
    my_shields: int
    left_shields: int
    right_shields: int
    my_gear: int
    my_compass: int
    my_tablet: int


class FeaturePolicy:
    """Base class for feature-based policies.
    
    All specialist policies inherit from this and override the score() method.
    The select_action() method handles action selection via scoring all legal actions.
    
    New in V2: VP-equivalent baseline scoring
    - All policies now use vp_equiv_base() + strategy-specific biases
    - Guild estimation included
    - Production deficit bonus for economy
    - Score signature changed to score(obs, action_idx, f, ctx)
    """
    
    def __init__(self, seed: int = 0):
        """Initialize policy with RNG seed."""
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, obs: Dict, rng: Optional[np.random.Generator] = None) -> int:
        """Select action by scoring all legal actions and picking best.
        
        Args:
            obs: Observation dictionary with action_mask, action_feats, etc.
            rng: Optional RNG (for compatibility with eval framework)
        
        Returns:
            Action index (0-27)
        """
        if rng is None:
            rng = self.rng
        
        mask = obs["action_mask"].astype(bool)
        feats = obs["action_feats"]
        legal = np.flatnonzero(mask)
        
        if len(legal) == 0:
            return 0
        
        # Extract context from observation
        era = int(obs["game_meta"][0])
        turn = int(obs["game_meta"][1])
        t_in_age = turn % 7
        
        gt = obs["global_table"]
        ctx = PolicyContext(
            era=era,
            t_in_age=t_in_age,
            my_coins=int(gt[0, 0]),
            my_shields=int(gt[0, 1]),
            left_shields=int(gt[1, 1]),
            right_shields=int(gt[2, 1]),
            my_gear=int(gt[0, 2]),
            my_compass=int(gt[0, 3]),
            my_tablet=int(gt[0, 4]),
        )
        
        # Score all legal actions
        scores = np.full(len(feats), -np.inf, dtype=np.float32)
        
        for action_idx in legal:
            scores[action_idx] = self.score(obs, action_idx, feats[action_idx], ctx)
        
        # Find best score
        best_score = np.max(scores[legal])
        best_actions = legal[scores[legal] == best_score]
        
        # Tiny randomness for tie-breaking
        return int(rng.choice(best_actions))
    
    def score(self, obs: Dict, action_idx: int, f: np.ndarray, ctx: PolicyContext) -> float:
        """Score an action based on features and context.
        
        Must be overridden by subclasses.
        
        New signature (V2):
        - obs: Full observation (for guild estimation, production tracking)
        - action_idx: Action index 0-27 (for identifying card slot)
        - f: Feature vector for this action (80 features)
        - ctx: Game state context
        
        Returns:
            Score (higher is better, in VP-equivalent units)
        """
        raise NotImplementedError("Subclasses must implement score()")
    
    def reset(self, observation: Dict = None) -> None:
        """Reset policy state (optional, for compatibility with eval framework)."""
        pass


def get_action_type(features: np.ndarray) -> int:
    """Extract action type from feature vector.
    
    Args:
        features: Feature vector (80 features)
    
    Returns:
        Action type: 0=build_structure, 1=build_wonder, 2=build_hand_free, 3=discard
    """
    type_features = features[IDX["action_type_0"]:IDX["action_type_3"] + 1]
    return int(np.argmax(type_features))


def get_card_color(features: np.ndarray) -> str:
    """Extract card color from feature vector.
    
    Args:
        features: Feature vector (80 features)
    
    Returns:
        Color name or "none" if empty slot
    """
    colors = ["brown", "grey", "blue", "yellow", "red", "green", "purple"]
    color_start = IDX["card_color_brown"]
    color_end = IDX["card_color_purple"]
    
    for i, color in enumerate(colors):
        if features[color_start + i] > 0.5:
            return color
    
    return "none"
