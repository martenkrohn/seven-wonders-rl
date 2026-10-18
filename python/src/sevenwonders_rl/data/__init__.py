"""Data module for 7 Wonders card and wonder databases."""

from .cards import load_card_db, by_id as card_by_id, by_name as card_by_name
from .wonders import load_wonder_db, wonder_by_id

__all__ = [
    "load_card_db",
    "card_by_id",
    "card_by_name",
    "load_wonder_db",
    "wonder_by_id",
]
