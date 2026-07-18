"""The original official water starter agent.

This module is intentionally self-contained so it can remain a regression
opponent while other deck agents evolve independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _submission_root() -> Path:
    """Resolve the directory where Kaggle unpacked the submission bundle."""
    source_file = globals().get("__file__")
    if isinstance(source_file, str):
        source_root = Path(source_file).resolve().parent
        if (source_root / "deck.csv").exists():
            return source_root

    # Kaggle's simulation loader may execute main.py with ``exec`` and omit
    # __file__. The official FAQ says submission files are then available in
    # /kaggle_simulations/agent, while cwd remains /kaggle/working.
    kaggle_root = Path("/kaggle_simulations/agent")
    if (kaggle_root / "deck.csv").exists():
        return kaggle_root

    return Path.cwd()


ROOT = _submission_root()
DECK_PATH = ROOT / "deck.csv"


def read_deck_csv() -> list[int]:
    values = [line.strip() for line in DECK_PATH.read_text(encoding="utf-8").splitlines()]
    deck = [int(value) for value in values if value]
    if len(deck) != 60:
        raise ValueError(f"{DECK_PATH.name} must contain 60 cards, got {len(deck)}")
    return deck


def _option_priority(option: dict[str, Any], index: int) -> tuple[int, int]:
    priorities = {
        13: 0,  # ATTACK
        8: 1,   # ATTACH
        9: 2,   # EVOLVE
        7: 3,   # PLAY
        10: 4,  # ABILITY
        12: 5,  # RETREAT
        14: 99, # END
    }
    return priorities.get(option.get("type"), 10), index


def _select_indices(select: dict[str, Any]) -> list[int]:
    options = select.get("option") or []
    min_count = int(select.get("minCount", 0))
    max_count = int(select.get("maxCount", len(options)))
    if not options:
        if min_count:
            raise ValueError("simulator returned a required selection with no options")
        return []
    if min_count == 0 and select.get("type") != 0:
        return []
    count = min_count if min_count > 0 else 1
    count = min(count, max_count, len(options))
    ranked = sorted(enumerate(options), key=lambda item: _option_priority(item[1], item[0]))
    return [index for index, _ in ranked[:count]]


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Return the legal action selected by the original deterministic baseline."""
    if obs_dict.get("select") is None:
        return read_deck_csv()
    return _select_indices(obs_dict["select"])
