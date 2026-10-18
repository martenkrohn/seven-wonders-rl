"""Action analysis utilities for policy evaluation.

Provides detailed descriptive statistics about agent/bot behavior:
- Action type frequencies (discard/wonder/card colors)
- Resource/military/science gains
- VP breakdowns
- Hard ban adherence checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

from sevenwonders_rl.policies.feature_policy_base import get_action_type
from sevenwonders_rl.policies.specialist_policies import _is_science_action, _is_military_action
from sevenwonders_rl.features.action_feature_schema import IDX


# Card color mapping
CARD_COLORS = {
    "brown": "card_color_brown",
    "grey": "card_color_grey",
    "blue": "card_color_blue",
    "yellow": "card_color_yellow",
    "red": "card_color_red",
    "green": "card_color_green",
    "purple": "card_color_purple",
}

# Exclusive bucket categories for analysis
BUCKETS = ("wonder", "discard", "brown", "grey", "blue", "yellow", "red", "green", "purple", "unknown")


def get_card_color_from_feats(f: np.ndarray) -> str:
    """Extract card color from feature vector using one-hot encoding."""
    best = None
    best_val = 0.0
    for color, key in CARD_COLORS.items():
        idx = IDX.get(key, None)
        if idx is None:
            continue
        val = float(f[idx])
        if val > best_val:
            best_val = val
            best = color

    if best is None or best_val < 0.5:
        return "unknown"
    return best


@dataclass
class ActionEvent:
    """Single action event from a game."""
    game_idx: int
    seat: int
    policy_name: str
    era: int
    turn: int
    action_idx: int
    action_type: int
    is_discard: bool
    is_wonder: bool
    is_science: bool
    is_military: bool
    prod_gain: float
    shields_gain: float
    science_gain: float
    card_color: str


def log_action_from_obs_feats(
    obs: Dict[str, Any],
    action_idx: int,
    game_idx: int,
    seat: int,
    policy_name: str,
) -> ActionEvent:
    """Create ActionEvent from observation and selected action."""
    feats = np.asarray(obs["action_feats"], dtype=np.float32)
    
    if 0 <= action_idx < feats.shape[0]:
        f = feats[action_idx]
    else:
        f = feats[0]  # defensive fallback

    at = int(get_action_type(f))
    era = int(obs["game_meta"][0])
    turn = int(obs["game_meta"][1])

    is_discard = (at == 3)
    is_wonder = (at == 1)
    is_science = _is_science_action(f, at)
    is_military = _is_military_action(f, at)

    # Production gains
    prod_keys = [
        "delta_prod_wood", "delta_prod_ore", "delta_prod_clay", "delta_prod_stone",
        "delta_prod_loom", "delta_prod_glass", "delta_prod_papyrus",
        "delta_prod_choice_raw", "delta_prod_choice_manuf",
    ]
    prod_gain = float(sum(float(f[IDX.get(k, 0)]) for k in prod_keys))

    shields_gain = float(f[IDX.get("delta_shields", 0)]) + float(f[IDX.get("wonder_delta_shields", 0)])
    science_gain = float(
        f[IDX.get("delta_science_gear", 0)] +
        f[IDX.get("delta_science_compass", 0)] +
        f[IDX.get("delta_science_tablet", 0)] +
        f[IDX.get("wonder_delta_science_gear", 0)] +
        f[IDX.get("wonder_delta_science_compass", 0)] +
        f[IDX.get("wonder_delta_science_tablet", 0)] +
        f[IDX.get("wonder_effect_add_science", 0)]
    )

    card_color = get_card_color_from_feats(f)

    return ActionEvent(
        game_idx=game_idx,
        seat=seat,
        policy_name=policy_name,
        era=era,
        turn=turn,
        action_idx=action_idx,
        action_type=at,
        is_discard=is_discard,
        is_wonder=is_wonder,
        is_science=is_science,
        is_military=is_military,
        prod_gain=prod_gain,
        shields_gain=shields_gain,
        science_gain=science_gain,
        card_color=card_color,
    )


def categorize_action(e: ActionEvent) -> str:
    """Exclusive category: wonder > discard > card_color."""
    if e.is_wonder:
        return "wonder"
    if e.is_discard:
        return "discard"
    return e.card_color


def mean_std(xs: List[float]) -> Tuple[float, float]:
    """Calculate mean and standard deviation."""
    if not xs:
        return 0.0, 0.0
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=0)) if len(xs) > 1 else 0.0
    return m, s


def analyze_actions(
    events: List[ActionEvent],
    policy_names: Optional[List[str]] = None,
) -> str:
    """Generate detailed action analysis report.
    
    Args:
        events: List of action events to analyze
        policy_names: Optional list of policy names to analyze (default: auto-detect)
        
    Returns:
        Formatted analysis report as string
    """
    if not events:
        return "No actions to analyze."

    # Auto-detect policies if not provided
    if policy_names is None:
        policy_names = sorted(set(e.policy_name for e in events))

    ages = (1, 2, 3)
    
    # Index events by (game_idx, policy_name) and (game_idx, policy_name, age)
    ev_by_gp: Dict[Tuple[int, str], List[ActionEvent]] = defaultdict(list)
    ev_by_gpa: Dict[Tuple[int, str, int], List[ActionEvent]] = defaultdict(list)

    for e in events:
        ev_by_gp[(e.game_idx, e.policy_name)].append(e)
        ev_by_gpa[(e.game_idx, e.policy_name, e.era)].append(e)

    # Get unique game indices
    game_indices = sorted(set(e.game_idx for e in events))
    
    # Build output
    lines = []
    lines.append("=" * 80)
    lines.append("ACTION ANALYSIS (mean ± std over games)")
    lines.append("=" * 80)

    # Collect per-game metrics for each policy
    per_policy_game_metrics: Dict[str, Dict[str, List[float]]] = {
        pol: defaultdict(list) for pol in policy_names
    }

    for gidx in game_indices:
        for pol in policy_names:
            evs = ev_by_gp.get((gidx, pol), [])
            n = len(evs)
            
            # Initialize metrics
            metrics = {"n": float(n)}
            
            # Initialize all buckets
            for b in BUCKETS:
                metrics[f"turns_{b}"] = 0.0
                metrics[f"share_{b}"] = 0.0
            
            if n > 0:
                # Categorize actions
                cats = defaultdict(int)
                for e in evs:
                    cats[categorize_action(e)] += 1

                # Compute shares
                for b in BUCKETS:
                    ct = float(cats[b])
                    metrics[f"turns_{b}"] = ct
                    metrics[f"share_{b}"] = ct / n

                # Aggregate gains
                metrics["prod_gain_sum"] = float(np.sum([e.prod_gain for e in evs]))
                metrics["shields_gain_sum"] = float(np.sum([e.shields_gain for e in evs]))
                metrics["science_gain_sum"] = float(np.sum([e.science_gain for e in evs]))
            else:
                metrics["prod_gain_sum"] = 0.0
                metrics["shields_gain_sum"] = 0.0
                metrics["science_gain_sum"] = 0.0

            # Collect metrics
            per_policy_game_metrics[pol]["n_actions"].append(metrics["n"])
            for b in BUCKETS:
                per_policy_game_metrics[pol][f"turns_{b}"].append(metrics[f"turns_{b}"])
                per_policy_game_metrics[pol][f"share_{b}"].append(metrics[f"share_{b}"])
            per_policy_game_metrics[pol]["prod_gain_sum"].append(metrics["prod_gain_sum"])
            per_policy_game_metrics[pol]["shields_gain_sum"].append(metrics["shields_gain_sum"])
            per_policy_game_metrics[pol]["science_gain_sum"].append(metrics["science_gain_sum"])

    # Print per-policy summaries
    for pol in policy_names:
        d = per_policy_game_metrics[pol]
        n_games = len(d["n_actions"])
        
        lines.append(f"\n[{pol.upper()}] (n={n_games} games)")
        
        m, s = mean_std(d["n_actions"])
        lines.append(f"  actions/game:   {m:6.2f} ± {s:6.2f}")
        
        for b in BUCKETS:
            m, s = mean_std(d[f"turns_{b}"])
            lines.append(f"  {b:8s} turns: {m:6.2f} ± {s:6.2f}")

        m, s = mean_std(d["prod_gain_sum"])
        lines.append(f"  prod_gain sum:    {m:7.1f} ± {s:7.1f}")
        m, s = mean_std(d["shields_gain_sum"])
        lines.append(f"  shields_gain sum: {m:7.1f} ± {s:7.1f}")
        m, s = mean_std(d["science_gain_sum"])
        lines.append(f"  science_gain sum: {m:7.1f} ± {s:7.1f}")

    # Per-age breakdown
    lines.append("\n" + "=" * 80)
    lines.append("PER-AGE ACTION MIX (mean ± std over games)")
    lines.append("=" * 80)

    per_policy_age: Dict[Tuple[str, int], Dict[str, List[float]]] = {
        (pol, age): defaultdict(list) for pol in policy_names for age in ages
    }

    for gidx in game_indices:
        for pol in policy_names:
            for age in ages:
                evs = ev_by_gpa.get((gidx, pol, age), [])
                n = len(evs)
                
                if n == 0:
                    for b in BUCKETS:
                        per_policy_age[(pol, age)][b].append(0.0)
                    per_policy_age[(pol, age)]["n_actions"].append(0.0)
                    continue

                cats = defaultdict(int)
                for e in evs:
                    cats[categorize_action(e)] += 1

                for b in BUCKETS:
                    per_policy_age[(pol, age)][b].append(cats[b] / n)
                per_policy_age[(pol, age)]["n_actions"].append(float(n))

    for pol in policy_names:
        lines.append(f"\n{'='*30} POLICY: {pol.upper():<10} {'='*30}")
        
        for age in ages:
            d = per_policy_age[(pol, age)]
            mN, sN = mean_std(d["n_actions"])

            lines.append(f"\n[ Age {age} ] (Avg Total Actions: {mN:.2f} ± {sN:.2f})")
            lines.append("-" * 40)

            display_lines = []
            for b in BUCKETS:
                m, s = mean_std(d[b])
                c = m * mN
                c_std = s * mN + m * sN
                pct = m * 100
                display_lines.append(f"{b.capitalize():<12}: {c:>5.2f} ± {c_std:<5.2f} ({pct:>5.1f}%)")

            for i in range(0, len(display_lines), 2):
                left = display_lines[i]
                right = display_lines[i+1] if i+1 < len(display_lines) else ""
                lines.append(f"  {left}    {right}")

    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


__all__ = [
    "ActionEvent",
    "log_action_from_obs_feats",
    "categorize_action",
    "analyze_actions",
]
