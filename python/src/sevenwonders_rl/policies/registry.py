"""Policy registry for string-based policy instantiation.

Maps policy names to constructors for dynamic opponent selection.
"""

from __future__ import annotations
from typing import Any


def make_policy(name: str, seed: int) -> Any:
    """Create a policy instance from its name.
    
    Args:
        name: Policy name (case-insensitive)
        seed: Random seed for policy initialization
    
    Returns:
        Policy instance with select_action(obs, rng) method
    
    Raises:
        ValueError: If policy name is unknown
    
    Supported policies:
        - baseline: Production-focused VP optimizer
        - mad_scientist: Hyper-aggressive science strategy
        - military: Military tempo with shield caps
        - random: Uniform random selection
    """
    name = name.lower()
    
    if name == "baseline":
        from sevenwonders_rl.policies.specialist_policies import BaselinePolicy
        return BaselinePolicy()
    
    if name == "mad_scientist":
        from sevenwonders_rl.policies.specialist_policies import MadScientistPolicy
        return MadScientistPolicy(seed=seed)
    
    if name == "military":
        from sevenwonders_rl.policies.specialist_policies import MilitaryTempoPolicy
        return MilitaryTempoPolicy(seed=seed)
    
    if name == "random":
        from sevenwonders_rl.policies.random_policy import RandomPolicy
        return RandomPolicy(seed=seed)
    
    raise ValueError(f"Unknown policy: '{name}'. Supported: baseline, mad_scientist, military, random")


# Convenience mapping for validation
KNOWN_POLICIES = ["baseline", "mad_scientist", "military", "random"]
