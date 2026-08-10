"""Stable Search hypotheses built without opponent-private evaluation state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HiddenHypothesis:
    your_deck: tuple[int, ...]
    your_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    opponent_active: tuple[int, ...]


def _visible_own_ids(observation: Mapping[str, Any], actor: int) -> Counter[int]:
    current = observation["current"]
    own = current["players"][actor]
    counts: Counter[int] = Counter()
    serials: set[int] = set()

    def visit(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        serial = value.get("serial")
        if isinstance(serial, int) and not isinstance(serial, bool):
            if serial in serials:
                return
            serials.add(serial)
        card_id = value.get("id")
        if isinstance(card_id, int) and not isinstance(card_id, bool):
            counts[card_id] += 1
        for field in ("energyCards", "tools", "preEvolution"):
            children = value.get(field)
            if isinstance(children, Sequence):
                for child in children:
                    visit(child)

    for zone in ("active", "bench", "hand", "discard"):
        values = own.get(zone)
        if isinstance(values, Sequence):
            for value in values:
                visit(value)
    for value in current.get("stadium") or ():
        if isinstance(value, Mapping) and value.get("playerIndex") == actor:
            visit(value)
    for value in current.get("looking") or ():
        visit(value)
    select = observation.get("select")
    if isinstance(select, Mapping):
        for field in ("contextCard", "effect"):
            value = select.get(field)
            if isinstance(value, Mapping) and value.get("playerIndex") == actor:
                visit(value)
    return counts


def build_nonprivileged_hypothesis(
    observation: Mapping[str, Any], registered_deck: Sequence[int]
) -> HiddenHypothesis:
    """Use focal-visible data plus a fixed public opponent placeholder prior."""

    current = observation["current"]
    actor = int(current["yourIndex"])
    own = current["players"][actor]
    opponent = current["players"][1 - actor]
    remaining = Counter(int(card_id) for card_id in registered_deck)
    remaining.subtract(_visible_own_ids(observation, actor))
    if any(count < 0 for count in remaining.values()):
        raise ValueError("visible focal cards exceed registered deck counts")
    hidden = sorted(card_id for card_id, count in remaining.items() for _ in range(count))
    deck_count = int(own["deckCount"])
    prize_count = len(own.get("prize") or ())
    if len(hidden) != deck_count + prize_count:
        raise ValueError("focal hidden-zone counts do not match registered deck ledger")

    opponent_deck_count = int(opponent["deckCount"])
    opponent_prize_count = len(opponent.get("prize") or ())
    opponent_hand_count = int(opponent["handCount"])
    # 119 is a public valid Basic Pokemon; 2 is a public valid Basic Energy.
    opponent_deck = ([119] if opponent_deck_count else []) + [2] * max(
        0, opponent_deck_count - 1
    )
    facedown_active = bool(opponent.get("active")) and opponent["active"][0] is None
    return HiddenHypothesis(
        your_deck=tuple(hidden[:deck_count]),
        your_prize=tuple(hidden[deck_count:]),
        opponent_deck=tuple(opponent_deck),
        opponent_prize=(2,) * opponent_prize_count,
        opponent_hand=(2,) * opponent_hand_count,
        opponent_active=(119,) if facedown_active else (),
    )


__all__ = ["HiddenHypothesis", "build_nonprivileged_hypothesis"]
