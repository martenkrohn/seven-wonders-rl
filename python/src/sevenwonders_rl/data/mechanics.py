"""Mechanics database loader and utilities."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any


@lru_cache(maxsize=1)
def load_mechanics() -> Dict[str, Any]:
    """Load mechanics configuration from JSON file (cached)."""
    mech_path = Path(__file__).parent / "mechanics.json"
    with open(mech_path, 'r') as f:
        return json.load(f)


def get_trade_costs() -> Dict[str, int]:
    """Get base trade costs for resources.
    
    Returns:
        Dict with 'raw' and 'manuf' keys mapping to coin costs.
    """
    mech = load_mechanics()
    trade = mech.get('trade', {})
    return {
        'raw': trade.get('base_cost_per_resource', 2),
        'manuf': trade.get('base_cost_per_resource', 2),
        'raw_discount': trade.get('discount_cost_per_resource', 1),
        'manuf_discount': trade.get('discount_cost_per_resource', 1),
    }


def get_manufactured_discount_flag() -> str:
    """Get the discount flag for manufactured goods."""
    mech = load_mechanics()
    trade = mech.get('trade', {})
    return trade.get('manufactured_discount_flag', 'manuf_cheap')


def get_raw_discount_flags() -> list[str]:
    """Get discount flags for raw materials."""
    mech = load_mechanics()
    trade = mech.get('trade', {})
    return trade.get('raw_discount_flags', ['raw_cheap_east', 'raw_cheap_west', 'wonder_raw_cheap'])


def resource_kind(resource: str) -> str:
    """Determine if a resource is 'raw' or 'manuf'.
    
    Args:
        resource: Resource name (e.g., 'wood', 'glass')
    
    Returns:
        'raw' or 'manuf'
    """
    raw_resources = {'wood', 'ore', 'clay', 'stone'}
    manuf_resources = {'loom', 'glass', 'papyrus'}
    
    if resource in raw_resources:
        return 'raw'
    elif resource in manuf_resources:
        return 'manuf'
    else:
        raise ValueError(f"Unknown resource: {resource}")


__all__ = [
    'load_mechanics',
    'get_trade_costs',
    'get_manufactured_discount_flag',
    'get_raw_discount_flags',
    'resource_kind',
]
