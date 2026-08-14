"""Deterministic Meta-first, exact-deck-balanced rollout schedules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random
from pathlib import Path

from ..own_archetype import DeckMapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class MetaDeckSlot:
    lane_id: int
    archetype_id: int
    deck_id: str


def _validated_members(mappings: Sequence[DeckMapping]) -> dict[int, tuple[str, ...]]:
    deck_ids = tuple(row.deck_id for row in mappings)
    expected_deck_ids = tuple(
        f"{value:03d}" for value in range(1, len(deck_ids) + 1)
    )
    if tuple(sorted(deck_ids)) != expected_deck_ids or len(set(deck_ids)) != len(deck_ids):
        raise ValueError(
            "Meta-first schedule requires each contiguous exact deck identity exactly once"
        )
    members: dict[int, list[str]] = defaultdict(list)
    for row in mappings:
        members[int(row.archetype_id)].append(row.deck_id)
    if not members or any(not values for values in members.values()):
        raise ValueError("Meta-first schedule requires non-empty active Meta classes")
    return {meta_id: tuple(sorted(values)) for meta_id, values in members.items()}


def balanced_meta_deck_schedule(
    *,
    quota_seed: int,
    shuffle_seed: int,
    mappings: Sequence[DeckMapping],
    lanes: int = 512,
    meta_weights: Mapping[int, float] | None = None,
) -> tuple[MetaDeckSlot, ...]:
    """Balance lanes over active Meta first, then over member exact decks.

    Reusing ``quota_seed`` gives focal and opponent the same Meta marginal.
    A distinct ``shuffle_seed`` independently chooses member-deck remainders and
    lane ordering, so the two actors are not coupled into artificial matchups.
    """

    if lanes <= 0:
        raise ValueError("lanes must be positive")
    members = _validated_members(mappings)
    meta_ids = tuple(sorted(members))

    if meta_weights is None:
        base_meta, extra_meta = divmod(lanes, len(meta_ids))
        quota_order = list(meta_ids)
        random.Random(int(quota_seed)).shuffle(quota_order)
        meta_quota = {
            meta_id: base_meta + int(meta_id in set(quota_order[:extra_meta]))
            for meta_id in meta_ids
        }
    else:
        unknown = set(map(int, meta_weights)) - set(meta_ids)
        if unknown:
            raise ValueError(f"Meta weights contain inactive class IDs: {sorted(unknown)}")
        weights = {
            meta_id: float(meta_weights.get(meta_id, 1.0)) for meta_id in meta_ids
        }
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError("Meta weights must all be positive")
        total_weight = sum(weights.values())
        exact = {
            meta_id: lanes * weights[meta_id] / total_weight for meta_id in meta_ids
        }
        meta_quota = {meta_id: int(exact[meta_id]) for meta_id in meta_ids}
        remainder = lanes - sum(meta_quota.values())
        tie_order = list(meta_ids)
        random.Random(int(quota_seed)).shuffle(tie_order)
        tie_rank = {meta_id: rank for rank, meta_id in enumerate(tie_order)}
        remainder_order = sorted(
            meta_ids,
            key=lambda meta_id: (
                -(exact[meta_id] - meta_quota[meta_id]), tie_rank[meta_id]
            ),
        )
        for meta_id in remainder_order[:remainder]:
            meta_quota[meta_id] += 1

    rows: list[tuple[int, str]] = []
    for meta_id in meta_ids:
        deck_ids = members[meta_id]
        base_deck, extra_deck = divmod(meta_quota[meta_id], len(deck_ids))
        deck_order = list(deck_ids)
        random.Random((int(shuffle_seed) << 8) ^ meta_id).shuffle(deck_order)
        extra_members = set(deck_order[:extra_deck])
        for deck_id in deck_ids:
            rows.extend(
                (meta_id, deck_id)
                for _ in range(base_deck + int(deck_id in extra_members))
            )

    random.Random(int(shuffle_seed)).shuffle(rows)
    if len(rows) != lanes:
        raise AssertionError("internal Meta-first quota accounting mismatch")
    return tuple(
        MetaDeckSlot(lane_id=lane_id, archetype_id=meta_id, deck_id=deck_id)
        for lane_id, (meta_id, deck_id) in enumerate(rows)
    )


__all__ = ["MetaDeckSlot", "PROJECT_ROOT", "balanced_meta_deck_schedule"]
