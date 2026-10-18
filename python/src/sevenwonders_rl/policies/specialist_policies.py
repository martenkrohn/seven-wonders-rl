"""Specialist bot policies for 7 Wonders.

This module contains 3 distinct policy implementations:
1. BaselinePolicy - Production-focused raw VP optimizer (blue/purple + coins)
2. MilitaryTempoPolicy - Military margin layer on top of baseline (never takes science)
3. MadScientistPolicy - Science layer on top of baseline (never takes military)

All policies share the same baseline heuristic:
- Ages 1-2: strong bias toward meeting production targets (with caps)
- Otherwise: maximize raw VP-equivalent (fixed VP + guild est + coins net costs)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

from sevenwonders_rl.policies.feature_policy_base import (
    FeaturePolicy,
    PolicyContext,
    get_action_type,
    estimate_guild_vp,
)
from sevenwonders_rl.features.action_feature_schema import IDX


# -----------------------------
# Shared baseline scoring pieces
# -----------------------------

# Baseline wonder valuation knobs
WONDER_STAGE_BASE_VP_EQ = 1.5


def _wonder_effect_vp_equiv() -> float:
    """Baseline VP-equivalent value for wonder stage effects.

    Wonder-specific features not yet implemented in V2 schema.
    Intentionally simple: no timing/urgency, no opponent modeling.
    """
    return WONDER_STAGE_BASE_VP_EQ


def _raw_vp_equiv(f: np.ndarray, obs: Dict[str, Any], action_type: int) -> float:
    """Raw VP-equivalent scoring (no science/military value).

    Includes:
    - fixed VP
    - guild estimate (Age 3, if applicable)
    - coins net of costs (coins / 3)
    """
    # Discard: +3 coins ~ +1 VP-equivalent
    if action_type == 3:
        return 1.0

    # Wonder stage build: Use coins_after_est_total_cost as proxy for net value
    # (Wonder features not yet implemented, so use card cost structure)
    if action_type == 1:
        # For wonder: estimate based on card cost and available coins
        # Since wonder-specific features not yet available, use simplified scoring
        coins_avail = float(f[IDX["coins_after_est_total_cost"]])
        # Assume building wonder is worth coins saved / 3 + base wonder value
        return max(0.5, coins_avail / 3.0 + _wonder_effect_vp_equiv())

    # Card build / hand-free: fixed VP + guild estimate + net coins
    vp = float(f[IDX["delta_vp_fixed"]])
    vp += float(estimate_guild_vp(f, obs, action_type))

    coins_after = float(f[IDX["coins_after_est_total_cost"]])
    return float(vp + coins_after / 3.0)


def _get_my_prod(obs: Dict[str, Any]) -> np.ndarray | None:
    table_prod = obs.get("table_prod")
    if table_prod is None or len(table_prod) < 1:
        return None
    return table_prod[0].astype(int)


def _prod_caps() -> tuple[np.ndarray, np.ndarray]:
    # [wood, ore, clay, stone] and [loom, glass, papyrus]
    raw_caps = np.array([2, 2, 2, 2], dtype=int)
    manuf_caps = np.array([1, 1, 1], dtype=int)
    return raw_caps, manuf_caps


def _prod_deficits(my_prod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = my_prod[:4].astype(int)
    manuf = my_prod[4:7].astype(int)
    return raw, manuf


def _violates_prod_caps(f: np.ndarray, obs: Dict[str, Any]) -> bool:
    my_prod = _get_my_prod(obs)
    if my_prod is None:
        return False

    raw_caps, manuf_caps = _prod_caps()
    raw, manuf = _prod_deficits(my_prod)

    if float(f[IDX.get("delta_prod_wood", 0)]) > 0.5 and raw[0] >= raw_caps[0]:
        return True
    if float(f[IDX.get("delta_prod_ore", 0)]) > 0.5 and raw[1] >= raw_caps[1]:
        return True
    if float(f[IDX.get("delta_prod_clay", 0)]) > 0.5 and raw[2] >= raw_caps[2]:
        return True
    if float(f[IDX.get("delta_prod_stone", 0)]) > 0.5 and raw[3] >= raw_caps[3]:
        return True

    if float(f[IDX.get("delta_prod_loom", 0)]) > 0.5 and manuf[0] >= manuf_caps[0]:
        return True
    if float(f[IDX.get("delta_prod_glass", 0)]) > 0.5 and manuf[1] >= manuf_caps[1]:
        return True
    if float(f[IDX.get("delta_prod_papyrus", 0)]) > 0.5 and manuf[2] >= manuf_caps[2]:
        return True

    if float(f[IDX.get("delta_prod_choice_raw", 0)]) > 0.5 and np.all(raw >= raw_caps):
        return True
    if float(f[IDX.get("delta_prod_choice_manuf", 0)]) > 0.5 and np.all(manuf >= manuf_caps):
        return True

    return False


def _helps_prod_targets(f: np.ndarray, obs: Dict[str, Any]) -> bool:
    my_prod = _get_my_prod(obs)
    if my_prod is None:
        return False

    raw_caps, manuf_caps = _prod_caps()
    raw, manuf = _prod_deficits(my_prod)

    if raw[0] < raw_caps[0] and float(f[IDX.get("delta_prod_wood", 0)]) > 0.5:
        return True
    if raw[1] < raw_caps[1] and float(f[IDX.get("delta_prod_ore", 0)]) > 0.5:
        return True
    if raw[2] < raw_caps[2] and float(f[IDX.get("delta_prod_clay", 0)]) > 0.5:
        return True
    if raw[3] < raw_caps[3] and float(f[IDX.get("delta_prod_stone", 0)]) > 0.5:
        return True

    if manuf[0] < manuf_caps[0] and float(f[IDX.get("delta_prod_loom", 0)]) > 0.5:
        return True
    if manuf[1] < manuf_caps[1] and float(f[IDX.get("delta_prod_glass", 0)]) > 0.5:
        return True
    if manuf[2] < manuf_caps[2] and float(f[IDX.get("delta_prod_papyrus", 0)]) > 0.5:
        return True

    if float(f[IDX.get("delta_prod_choice_raw", 0)]) > 0.5 and np.any(raw < raw_caps):
        return True
    if float(f[IDX.get("delta_prod_choice_manuf", 0)]) > 0.5 and np.any(manuf < manuf_caps):
        return True

    return False


def _missing_any_prod_targets(obs: Dict[str, Any]) -> bool:
    my_prod = _get_my_prod(obs)
    if my_prod is None:
        return False
    raw_caps, manuf_caps = _prod_caps()
    raw, manuf = _prod_deficits(my_prod)
    return bool(np.any(raw < raw_caps) or np.any(manuf < manuf_caps))


def _production_bias(f: np.ndarray, obs: Dict[str, Any], era: int) -> float:
    """Strong pull toward production in Ages 1-2 until targets met, with cap avoidance."""
    if era > 2:
        return 0.0
    if not _missing_any_prod_targets(obs):
        return 0.0

    if _violates_prod_caps(f, obs):
        return -8.0
    if _helps_prod_targets(f, obs):
        return 8.0
    return -2.0


# -----------------------------
# Hard-ban detectors (feature-based)
# -----------------------------

def _is_science_action(f: np.ndarray, action_type: int) -> bool:
    # Science cards (build / hand-free)
    if action_type in (0, 2):
        ds = (
            float(f[IDX.get("delta_science_gear", 0)]) +
            float(f[IDX.get("delta_science_compass", 0)]) +
            float(f[IDX.get("delta_science_tablet", 0)])
        )
        return ds > 0.5

    # Wonder stages that add science (explicit deltas or "add science" effect)
    if action_type == 1:
        ds = (
            float(f[IDX.get("wonder_delta_science_gear", 0)]) +
            float(f[IDX.get("wonder_delta_science_compass", 0)]) +
            float(f[IDX.get("wonder_delta_science_tablet", 0)]) +
            float(f[IDX.get("wonder_effect_add_science", 0)])
        )
        return ds > 0.5

    return False


def _is_military_action(f: np.ndarray, action_type: int) -> bool:
    # Red cards (build / hand-free)
    if action_type in (0, 2):
        return float(f[IDX.get("delta_shields", 0)]) > 0.5

    # Wonder stages that add shields
    if action_type == 1:
        return float(f[IDX.get("wonder_delta_shields", 0)]) > 0.5

    return False


# -----------------------------
# Policies
# -----------------------------

class BaselinePolicy(FeaturePolicy):
    """Production-first, then raw VP (fixed VP + guild estimate + coins net costs).

    No explicit value for military or science; those only get chosen if they
    incidentally score well under raw VP-equivalent (rare).
    """

    def score(self, obs: Dict[str, Any], action_idx: int, f: np.ndarray, ctx: PolicyContext) -> float:
        action_type = get_action_type(f)
        s = _raw_vp_equiv(f, obs, action_type)
        s += _production_bias(f, obs, ctx.era)
        return s


class MilitaryTempoPolicy(FeaturePolicy):
    """Military layer on top of the same baseline heuristic.

    - HARD BAN: never takes science
    - Goal: build shields until safely ahead vs BOTH neighbors, then revert to baseline
    - Still uses baseline production-first bias in Ages 1-2 and raw VP otherwise
    """

    @staticmethod
    def _military_threshold(era: int) -> int:
        return 2 if era == 1 else 3 if era == 2 else 4

    @staticmethod
    def _coin_costs(f: np.ndarray, action_type: int) -> float:
        # Return the estimated total coin cost (card cost + trade cost)
        # coins_after_est_total_cost = current_coins - (card_cost + trade_cost)
        # So: card_cost + trade_cost = current_coins - coins_after_est_total_cost
        # We approximate by using the delta from card cost
        if action_type in (0, 2):
            # For card builds: estimate from coins_after_est_total_cost
            # If we have coins_after_est_total_cost, the cost was the difference
            coins_after = float(f[IDX["coins_after_est_total_cost"]])
            # Negative means coins deficit, positive means coins remaining
            return max(0.0, -coins_after) if coins_after < 0 else 0.0
        if action_type == 1:
            # For wonder: similar estimation
            coins_after = float(f[IDX["coins_after_est_total_cost"]])
            return max(0.0, -coins_after) if coins_after < 0 else 0.0
        return 0.0

    @staticmethod
    def _delta_shields(f: np.ndarray, action_type: int) -> float:
        if action_type in (0, 2):
            return float(f[IDX.get("delta_shields", 0)])
        if action_type == 1:
            return float(f[IDX.get("wonder_delta_shields", 0)])
        return 0.0

    def score(self, obs: Dict[str, Any], action_idx: int, f: np.ndarray, ctx: PolicyContext) -> float:
        action_type = get_action_type(f)
        era = ctx.era
        late = ctx.t_in_age / 6.0  # 0..1

        # HARD BAN: never pick science
        if _is_science_action(f, action_type):
            return -np.inf

        # Baseline heuristic
        base = _raw_vp_equiv(f, obs, action_type)
        base += _production_bias(f, obs, era)

        # Military goal layer
        dsh = self._delta_shields(f, action_type)
        coin_costs = self._coin_costs(f, action_type)

        lead_left = ctx.my_shields - ctx.left_shields
        lead_right = ctx.my_shields - ctx.right_shields
        safe_lead = min(lead_left, lead_right)
        threshold = self._military_threshold(era)

        mil_bias = 0.0
        if dsh > 0 and safe_lead < threshold:
            cheap = 1.0 / (1.0 + coin_costs)
            mil_bias = (6.0 + 6.0 * late) * cheap * dsh
        elif dsh > 0 and safe_lead >= threshold:
            mil_bias = -0.5 * dsh  # tiny discouragement for "wasted" shields

        # Wonder free-build value is now handled in baseline (_raw_vp_equiv)
        freebuild_value = 0.0

        score = base + mil_bias + freebuild_value

        # Slight discard penalty (keeps it from discarding too eagerly)
        if action_type == 3:
            score -= 1.0

        return float(score)


class MadScientistPolicy(FeaturePolicy):
    """Science layer on top of the same baseline heuristic.

    - HARD BAN: never takes military
    - Goal: accumulate science symbols until a per-age target, then revert to baseline
    - Still uses baseline production-first bias in Ages 1-2 and raw VP otherwise
    """

    @staticmethod
    def _science_target(era: int) -> int:
        # Tune freely; this is intentionally simple and stable.
        return 2 if era == 1 else 4 if era == 2 else 6

    @staticmethod
    def _delta_science_symbols(f: np.ndarray, action_type: int) -> float:
        if action_type in (0, 2):
            return float(
                float(f[IDX.get("delta_science_gear", 0)]) +
                float(f[IDX.get("delta_science_compass", 0)]) +
                float(f[IDX.get("delta_science_tablet", 0)])
            )

        if action_type == 1:
            ds = float(
                float(f[IDX.get("wonder_delta_science_gear", 0)]) +
                float(f[IDX.get("wonder_delta_science_compass", 0)]) +
                float(f[IDX.get("wonder_delta_science_tablet", 0)])
            )
            if ds <= 0.0:
                ds = float(f[IDX.get("wonder_effect_add_science", 0)])
            return ds

        return 0.0

    def score(self, obs: Dict[str, Any], action_idx: int, f: np.ndarray, ctx: PolicyContext) -> float:
        action_type = get_action_type(f)
        era = ctx.era

        # HARD BAN: never pick military
        if _is_military_action(f, action_type):
            return -np.inf

        # Baseline heuristic
        s = _raw_vp_equiv(f, obs, action_type)
        s += _production_bias(f, obs, era)

        # Science goal layer (active until target reached)
        cur_science = int(ctx.my_gear + ctx.my_compass + ctx.my_tablet)
        target = self._science_target(era)

        ds = self._delta_science_symbols(f, action_type)
        if cur_science < target and ds > 0.0:
            s += 10.0 * ds  # strong pull while below target

        # Slightly stronger discard penalty than baseline
        if action_type == 3:
            s -= 1.2

        return float(s)


__all__ = [
    "BaselinePolicy",
    "MilitaryTempoPolicy",
    "MadScientistPolicy",
]
