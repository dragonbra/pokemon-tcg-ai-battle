"""Actor-visible view classification used by offline and online causal state."""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any


class ViewKind(str, Enum):
    NONE = "none"
    ELIGIBLE_SUBSET = "eligible_subset"
    FULL_MEMBERSHIP = "full_membership"
    ORDERED_VIEW = "ordered_view"


_KNOWN_LOG_TYPES = frozenset({
    "draw", "move", "shuffle", "hand_view", "deck_view", "prize_take", "prize_place",
    "prize_swap", "discard", "recover", "attach", "evolve", "retreat", "attack", "damage",
    "ko", "reveal", "search", "turn", "ability", "supporter", "stadium", "item", "status",
})


def classify_deck_view(select: Mapping[str, Any]) -> ViewKind:
    deck = select.get("deck")
    options = select.get("option", [])
    if deck is None:
        return ViewKind.NONE
    if not isinstance(deck, (tuple, list)):
        raise ValueError("unclassified deck view payload")
    if not isinstance(options, (tuple, list)):
        raise ValueError("unclassified deck view options")
    # Engine contract: select.deck is the complete current deck in physical order; options carry eligibility.
    return ViewKind.ORDERED_VIEW


def classify_log(log: Mapping[str, Any]) -> str:
    value = log.get("type")
    if isinstance(value, int) and not isinstance(value, bool):
        return f"engine_log_{value}"
    if isinstance(value, str) and value in _KNOWN_LOG_TYPES:
        return value
    raise ValueError(f"unclassified actor-visible log type: {value!r}")


__all__ = ["ViewKind", "classify_deck_view", "classify_log"]
