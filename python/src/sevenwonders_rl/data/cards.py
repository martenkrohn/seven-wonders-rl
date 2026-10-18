"""Card database loader and utilities."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional

# Constants
RESOURCE_ORDER = ['wood', 'ore', 'clay', 'stone', 'loom', 'glass', 'papyrus', 'coins']
SCIENCE_SYMBOLS = ['gear', 'compass', 'tablet']

TYPE_TO_COLOR = {
    'materials': 'brown',
    'manufactured': 'grey',
    'civilian': 'blue',
    'commercial': 'yellow',
    'military': 'red',
    'scientific': 'green',
    'guild': 'purple',
}

COLOR_TO_TYPE = {v: k for k, v in TYPE_TO_COLOR.items()}


@lru_cache(maxsize=1)
def load_card_db() -> Dict[str, Dict[str, Any]]:
    """Load card database from JSON file (cached)."""
    db_path = Path(__file__).parent / "card_db.json"
    with open(db_path, 'r') as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_name_index() -> Dict[str, int]:
    """Build name -> id mapping (cached)."""
    db = load_card_db()
    name_to_id = {}
    for card_id_str, card in db.items():
        name = card['name'].lower()
        name_to_id[name] = card['id']
        # Also index by id_name for robustness
        id_name = card['id_name'].lower()
        name_to_id[id_name] = card['id']
    return name_to_id


def by_id(card_id: int) -> Optional[Dict[str, Any]]:
    """Get card by numeric ID."""
    db = load_card_db()
    return db.get(str(card_id))


def by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get card by name (case-insensitive)."""
    name_index = _build_name_index()
    card_id = name_index.get(name.lower())
    if card_id is not None:
        return by_id(card_id)
    return None


def by_name_strict(name: str) -> Dict[str, Any]:
    """Get card by name, raising with helpful error if not found.
    
    Args:
        name: Card name (normalized in database)
    
    Returns:
        Card dictionary
    
    Raises:
        ValueError: If name not found, with suggestions for similar names
    """
    card = by_name(name)
    if card is not None:
        return card
    
    # Name not found; provide suggestions
    name_index = _build_name_index()
    all_names = sorted(set(
        c.get('name', '')
        for c in load_card_db().values()
    ))
    
    # Find similar names (very basic heuristic)
    name_lower = name.lower()
    similar = [
        n for n in all_names
        if name_lower in n.lower() or n.lower() in name_lower
    ]
    
    if similar:
        suggestions = ', '.join(f'"{s}"' for s in similar[:5])
        raise ValueError(
            f"Card name '{name}' not found in database. "
            f"Did you mean one of: {suggestions}?"
        )
    else:
        raise ValueError(
            f"Card name '{name}' not found in database. "
            f"Total cards in DB: {len(all_names)}"
        )


def get_all_ids() -> list[int]:
    """Get list of all card IDs."""
    db = load_card_db()
    return [int(card_id) for card_id in db.keys()]


def get_cards_by_type(card_type: str) -> list[Dict[str, Any]]:
    """Get all cards of a specific type."""
    db = load_card_db()
    return [card for card in db.values() if card['type'] == card_type]


def get_cards_by_color(color: str) -> list[Dict[str, Any]]:
    """Get all cards of a specific color."""
    db = load_card_db()
    return [card for card in db.values() if card['color'] == color]


def get_cards_by_era(era: int) -> list[Dict[str, Any]]:
    """Get all cards in a specific era."""
    db = load_card_db()
    return [card for card in db.values() if era in card['eras']]


@lru_cache(maxsize=1)
def _build_reverse_chain_map() -> Dict[str, list[int]]:
    """Build reverse chain map: prerequisite id_name -> list of card ids that chain to it (cached)."""
    db = load_card_db()
    chain_to = {}
    
    for card_id_str, card in db.items():
        card_id = card['id']
        
        # Handle chain_from (single prerequisite)
        if card.get('chain_from') is not None:
            prereq_id = card['chain_from']
            prereq_card = by_id(prereq_id)
            if prereq_card:
                prereq_id_name = prereq_card['id_name']
                if prereq_id_name not in chain_to:
                    chain_to[prereq_id_name] = []
                chain_to[prereq_id_name].append(card_id)
        
        # Handle chain_from_any (multiple possible prerequisites)
        if card.get('chain_from_any') is not None and card['chain_from_any']:
            for prereq_id_name in card['chain_from_any']:
                if prereq_id_name not in chain_to:
                    chain_to[prereq_id_name] = []
                chain_to[prereq_id_name].append(card_id)
    
    return chain_to


def get_chain_to_count(card_id_name: str) -> int:
    """Get count of cards that can be built for free from a given card.
    
    Args:
        card_id_name: The id_name of the prerequisite card
    
    Returns:
        Count of cards that have this card as a prerequisite
    """
    chain_to_map = _build_reverse_chain_map()
    return len(chain_to_map.get(card_id_name, []))


def get_chain_to_ids(card_id_name: str) -> list[int]:
    """Get list of card ids that can be built for free from a given card.
    
    Args:
        card_id_name: The id_name of the prerequisite card
    
    Returns:
        List of card ids
    """
    chain_to_map = _build_reverse_chain_map()
    return chain_to_map.get(card_id_name, [])
