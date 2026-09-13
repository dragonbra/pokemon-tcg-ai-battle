"""Deterministic focal-deck scheduling for batched self-play."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True, slots=True)
class FocalLane:
    lane_id: int
    deck_id: str


def balanced_focal_schedule(
    seed: int, *, deck_ids: tuple[str, ...], lanes: int = 256,
) -> tuple[FocalLane, ...]:
    """Return a seeded schedule whose per-deck counts differ by at most one."""
    if not deck_ids or len(set(deck_ids)) != len(deck_ids):
        raise ValueError("focal deck IDs must be non-empty and unique")
    if lanes < len(deck_ids):
        raise ValueError("focal lanes must cover every configured deck")
    quotient, remainder = divmod(lanes, len(deck_ids))
    offset = seed % len(deck_ids)
    extra = {deck_ids[(offset + index) % len(deck_ids)] for index in range(remainder)}
    values = [
        deck_id
        for deck_id in deck_ids
        for _ in range(quotient + int(deck_id in extra))
    ]
    random.Random(seed).shuffle(values)
    return tuple(FocalLane(index, deck_id) for index, deck_id in enumerate(values))


__all__ = ["FocalLane", "balanced_focal_schedule"]
