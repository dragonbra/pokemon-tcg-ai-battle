from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .rating import RatingState


@dataclass(frozen=True)
class Pairing:
    first: str
    second: str
    sequence: int = 0

    @property
    def players(self) -> tuple[str, str]:
        return self.first, self.second


def build_smoke_pairs(deck_ids: Sequence[str]) -> tuple[Pairing, ...]:
    return tuple(Pairing(first, second) for first, second in itertools.combinations(deck_ids, 2))


def build_rating_pairs(
    states: Sequence[RatingState],
    completed_counts: Mapping[tuple[str, str], int],
    seed: int,
    limit: int,
) -> tuple[Pairing, ...]:
    if limit < 1:
        return ()
    randomizer = random.Random(seed)
    candidates: list[tuple[tuple[float, int, float, str, str], Pairing]] = []
    for first_state, second_state in itertools.combinations(states, 2):
        if first_state.status != "active" or second_state.status != "active":
            continue
        first, second = sorted((first_state.name, second_state.name))
        count = completed_counts.get((first, second), completed_counts.get((second, first), 0))
        key = (
            abs(first_state.mu - second_state.mu),
            count,
            randomizer.random(),
            first,
            second,
        )
        candidates.append((key, Pairing(first, second)))
    candidates.sort(key=lambda item: item[0])
    return tuple(pairing for _, pairing in candidates[:limit])
