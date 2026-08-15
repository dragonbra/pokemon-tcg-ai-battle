"""Deterministic exact-quota Meta-first rollout scheduling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import random

from ..own_archetype import DeckMapping
from .meta_balanced import MetaDeckSlot


def aggressive_meta_quota_schedule(
    *,
    quota_seed: int,
    shuffle_seed: int,
    mappings: Sequence[DeckMapping],
    lanes: int,
    fixed_meta_quotas: Mapping[int, int],
) -> tuple[MetaDeckSlot, ...]:
    """Allocate fixed Meta quotas and balance the remainder over other Meta."""
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    deck_ids = tuple(row.deck_id for row in mappings)
    expected = tuple(f"{value:03d}" for value in range(1, len(deck_ids) + 1))
    if tuple(sorted(deck_ids)) != expected or len(set(deck_ids)) != len(deck_ids):
        raise ValueError("exact-quota schedule requires contiguous unique deck IDs")
    members: dict[int, list[str]] = defaultdict(list)
    for row in mappings:
        members[int(row.archetype_id)].append(row.deck_id)
    active = tuple(sorted(members))
    fixed = {int(meta_id): int(value) for meta_id, value in fixed_meta_quotas.items()}
    if not fixed or set(fixed) - set(active):
        raise ValueError("fixed Meta quotas contain an inactive class")
    if any(value <= 0 for value in fixed.values()):
        raise ValueError("fixed Meta quotas must be positive integers")
    remaining_lanes = lanes - sum(fixed.values())
    remaining_meta = tuple(meta_id for meta_id in active if meta_id not in fixed)
    if remaining_lanes < len(remaining_meta):
        raise ValueError("remaining quota cannot cover every other active Meta")

    base, extra = divmod(remaining_lanes, len(remaining_meta))
    order = list(remaining_meta)
    random.Random(int(quota_seed)).shuffle(order)
    extras = set(order[:extra])
    quotas = {
        **fixed,
        **{
            meta_id: base + int(meta_id in extras)
            for meta_id in remaining_meta
        },
    }
    rows: list[tuple[int, str]] = []
    for meta_id in active:
        class_decks = tuple(sorted(members[meta_id]))
        deck_base, deck_extra = divmod(quotas[meta_id], len(class_decks))
        deck_order = list(class_decks)
        random.Random((int(shuffle_seed) << 8) ^ meta_id).shuffle(deck_order)
        extra_decks = set(deck_order[:deck_extra])
        for deck_id in class_decks:
            rows.extend(
                (meta_id, deck_id)
                for _ in range(deck_base + int(deck_id in extra_decks))
            )
    random.Random(int(shuffle_seed)).shuffle(rows)
    if len(rows) != lanes:
        raise AssertionError("exact-quota schedule accounting mismatch")
    return tuple(
        MetaDeckSlot(lane_id=index, archetype_id=meta_id, deck_id=deck_id)
        for index, (meta_id, deck_id) in enumerate(rows)
    )


__all__ = ["aggressive_meta_quota_schedule"]
