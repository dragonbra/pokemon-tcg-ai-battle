from __future__ import annotations

from collections.abc import Sequence

from ..cards import (
    ABRA,
    ALAKAZAM,
    ATTACK_LINE,
    BASIC_PSYCHIC,
    DUNSPARCE,
    DUDUNSPARCE,
    KADABRA,
    RARE_CANDY,
)
from ..model import SemanticOption, TurnFacts
from ..profiles import RecoveryPolicy


def _take_by_id(
    options: Sequence[SemanticOption],
    priorities: Sequence[int],
    limit: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    used: set[int] = set()
    for card_id in priorities:
        option = next(
            (
                candidate
                for candidate in options
                if candidate.index not in used and candidate.card_id == card_id
            ),
            None,
        )
        if option is None:
            continue
        selected.append(option.index)
        used.add(option.index)
        if len(selected) >= limit:
            break
    return tuple(selected)


def choose_lanas_aid(
    options: Sequence[SemanticOption], facts: TurnFacts, *, max_count: int
) -> tuple[int, ...]:
    priorities: list[int] = []
    for card_id in (ABRA, KADABRA, ALAKAZAM, ABRA, KADABRA, ALAKAZAM):
        priorities.append(card_id)
    priorities.extend((DUNSPARCE, DUDUNSPARCE, BASIC_PSYCHIC))
    chosen = _take_by_id(options, priorities, max_count)
    if chosen:
        return chosen
    del facts
    return ()


def choose_sacred_ash(
    options: Sequence[SemanticOption], facts: TurnFacts, *, max_count: int,
    recovery_policy: RecoveryPolicy = RecoveryPolicy.COMPLETE_LINES,
) -> tuple[int, ...]:
    if recovery_policy == RecoveryPolicy.STAGE_WEIGHTED or RARE_CANDY in facts.yours.hand_ids:
        priorities = (
            ABRA,
            ALAKAZAM,
            KADABRA,
            ABRA,
            ALAKAZAM,
            KADABRA,
            DUNSPARCE,
            DUDUNSPARCE,
        )
    else:
        priorities = (
            ABRA,
            KADABRA,
            ALAKAZAM,
            ABRA,
            KADABRA,
            ALAKAZAM,
            DUNSPARCE,
            DUDUNSPARCE,
        )
    selected = list(_take_by_id(options, priorities, max_count))
    if len(selected) < max_count:
        selected.extend(
            option.index
            for option in options
            if option.index not in selected
            and option.card_id in ATTACK_LINE | {DUNSPARCE, DUDUNSPARCE}
        )
    return tuple(selected[:max_count])
