"""Wonder database loader and utilities."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional


@lru_cache(maxsize=1)
def load_wonder_db() -> Dict[str, Dict[str, Any]]:
    """Load wonder database from JSON file (cached)."""
    db_path = Path(__file__).parent / "wonder_db.json"
    with open(db_path, 'r') as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_name_index() -> Dict[str, int]:
    """Build name -> id and id_name -> id mapping (cached)."""
    db = load_wonder_db()
    name_to_id = {}
    for wonder_id_str, wonder in db.items():
        name = wonder['name'].lower()
        name_to_id[name] = wonder['id']
        # Also index by id_name
        id_name = wonder['id_name'].lower()
        name_to_id[id_name] = wonder['id']
    return name_to_id


def wonder_by_id(wonder_id: int) -> Optional[Dict[str, Any]]:
    """Get wonder by numeric ID."""
    db = load_wonder_db()
    return db.get(str(wonder_id))


def wonder_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get wonder by name (case-insensitive)."""
    name_index = _build_name_index()
    wonder_id = name_index.get(name.lower())
    if wonder_id is not None:
        return wonder_by_id(wonder_id)
    return None


def get_all_wonder_ids() -> list[int]:
    """Get list of all wonder IDs."""
    db = load_wonder_db()
    return [int(wonder_id) for wonder_id in db.keys()]


def get_wonder_stage_count(wonder_id: int) -> int:
    """Get number of stages for a wonder."""
    wonder = wonder_by_id(wonder_id)
    if wonder:
        return len(wonder['stages'])
    return 0
