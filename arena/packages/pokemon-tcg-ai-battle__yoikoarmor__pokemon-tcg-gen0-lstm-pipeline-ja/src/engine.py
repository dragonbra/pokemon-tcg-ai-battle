"""Thin wrapper around the bundled `cg` battle engine.

The engine ships inside `sample_submission/cg` (a ctypes shim over cg.dll /
libcg.so). This module makes it importable from anywhere in the project and
caches the static card / attack data.
"""
import os
import sys
import functools

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAMPLE = os.path.join(_ROOT, "sample_submission")
if _SAMPLE not in sys.path:
    sys.path.insert(0, _SAMPLE)

# Re-export the engine surface.
from cg.game import battle_start, battle_finish, battle_select, visualize_data  # noqa: E402
from cg.api import (  # noqa: E402
    to_observation_class,
    all_card_data,
    all_attack,
    Observation,
)

ROOT = _ROOT


@functools.lru_cache(maxsize=1)
def card_db() -> dict:
    """cardId -> CardData."""
    return {c.cardId: c for c in all_card_data()}


@functools.lru_cache(maxsize=1)
def attack_db() -> dict:
    """attackId -> Attack."""
    return {a.attackId: a for a in all_attack()}


def read_deck_csv(path: str) -> list[int]:
    """Read a 60-card deck list from a one-id-per-line csv."""
    with open(path, "r") as f:
        rows = f.read().split("\n")
    return [int(rows[i]) for i in range(60)]


__all__ = [
    "battle_start",
    "battle_finish",
    "battle_select",
    "visualize_data",
    "to_observation_class",
    "all_card_data",
    "all_attack",
    "Observation",
    "card_db",
    "attack_db",
    "read_deck_csv",
    "ROOT",
]
