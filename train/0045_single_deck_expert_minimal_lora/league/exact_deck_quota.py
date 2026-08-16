"""Reproducible exact-deck base quotas plus a seeded random remainder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import random

from ..own_archetype import DeckMapping
from .meta_balanced import MetaDeckSlot


def exact_deck_quota_schedule(
    *,
    random_seed: int,
    shuffle_seed: int,
    mappings: Sequence[DeckMapping],
    lanes: int,
    fixed_deck_quotas: Mapping[str, int],
    random_deck_ids: Sequence[str],
) -> tuple[MetaDeckSlot, ...]:
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    archetype_by_deck = {row.deck_id: int(row.archetype_id) for row in mappings}
    if len(archetype_by_deck) != len(mappings):
        raise ValueError("deck mappings must be unique")
    fixed = {str(deck_id): int(games) for deck_id, games in fixed_deck_quotas.items()}
    random_pool = tuple(map(str, random_deck_ids))
    if not fixed or any(games <= 0 for games in fixed.values()):
        raise ValueError("fixed deck quotas must be positive")
    if not random_pool or len(set(random_pool)) != len(random_pool):
        raise ValueError("random deck pool must be non-empty and unique")
    unknown = (set(fixed) | set(random_pool)) - set(archetype_by_deck)
    if unknown:
        raise ValueError(f"quota schedule contains unknown decks: {sorted(unknown)}")
    remaining = lanes - sum(fixed.values())
    if remaining < 0:
        raise ValueError("fixed deck quotas exceed lane count")

    rows = [
        (archetype_by_deck[deck_id], deck_id)
        for deck_id, games in sorted(fixed.items())
        for _ in range(games)
    ]
    chooser = random.Random(int(random_seed))
    rows.extend(
        (archetype_by_deck[deck_id], deck_id)
        for deck_id in (chooser.choice(random_pool) for _ in range(remaining))
    )
    random.Random(int(shuffle_seed)).shuffle(rows)
    if len(rows) != lanes:
        raise AssertionError("exact-deck quota schedule accounting mismatch")
    return tuple(
        MetaDeckSlot(lane_id=index, archetype_id=meta_id, deck_id=deck_id)
        for index, (meta_id, deck_id) in enumerate(rows)
    )


__all__ = ["exact_deck_quota_schedule"]
