"""Canonical Phantom Dive allocations; no engine or model dependency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


PHANTOM_DIVE_ATTACK_ID = 154
PHANTOM_DIVE_COUNTERS = 6
PHANTOM_DIVE_CONTEXTS = frozenset({14, "DamageCounterAny"})
PHANTOM_DIVE_MAX_TARGETS = 8


@dataclass(frozen=True, order=True, slots=True)
class StableTargetIdentity:
    player_index: int
    serial: int
    card_id: int
    initial_bench_slot: int


@dataclass(frozen=True, slots=True)
class DragapultDamageAllocation:
    target_ids: tuple[StableTargetIdentity, ...]
    counters: tuple[int, ...]
    total: int = PHANTOM_DIVE_COUNTERS

    def __post_init__(self) -> None:
        if not self.target_ids or len(self.target_ids) != len(self.counters):
            raise ValueError("allocation targets and counters must be non-empty and aligned")
        if tuple(sorted(self.target_ids)) != self.target_ids:
            raise ValueError("allocation targets must use canonical stable-identity order")
        if any(value < 0 for value in self.counters) or sum(self.counters) != self.total:
            raise ValueError("allocation counters must be non-negative and sum to total")

    def primitive_target_serials(self) -> tuple[int, ...]:
        return tuple(
            target.serial
            for target, count in zip(self.target_ids, self.counters, strict=True)
            for _ in range(count)
        )


def _weak_compositions(total: int, length: int, prefix: tuple[int, ...] = ()):
    if length == 1:
        yield prefix + (total,)
        return
    for value in range(total + 1):
        yield from _weak_compositions(total - value, length - 1, prefix + (value,))


def enumerate_allocations(
    targets: Sequence[StableTargetIdentity], total: int = PHANTOM_DIVE_COUNTERS
) -> tuple[DragapultDamageAllocation, ...]:
    canonical = tuple(sorted(targets))
    if (
        not 1 <= len(canonical) <= PHANTOM_DIVE_MAX_TARGETS
        or len(set(canonical)) != len(canonical)
    ):
        raise ValueError(
            "Phantom Dive requires one to eight distinct stable Bench targets"
        )
    return tuple(
        DragapultDamageAllocation(canonical, counters, total)
        for counters in _weak_compositions(total, len(canonical))
    )


def opponent_bench_targets(observation: Mapping[str, Any], actor: int) -> tuple[StableTargetIdentity, ...]:
    current = observation.get("current")
    players = current.get("players") if isinstance(current, Mapping) else None
    opponent = 1 - actor
    if not isinstance(players, Sequence) or len(players) != 2:
        raise ValueError("official observation has no two-player state")
    bench = players[opponent].get("bench") if isinstance(players[opponent], Mapping) else None
    if not isinstance(bench, Sequence):
        raise ValueError("official observation has no opponent Bench")
    targets = []
    for slot, raw in enumerate(bench):
        if not isinstance(raw, Mapping):
            raise ValueError("opponent Bench entry is not a mapping")
        serial, card_id = raw.get("serial"), raw.get("id")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                   for value in (serial, card_id)):
            raise ValueError("Bench target lacks a stable serial/card identity")
        targets.append(StableTargetIdentity(opponent, serial, card_id, slot))
    return tuple(sorted(targets))


__all__ = [
    "DragapultDamageAllocation", "PHANTOM_DIVE_ATTACK_ID", "PHANTOM_DIVE_CONTEXTS",
    "PHANTOM_DIVE_MAX_TARGETS",
    "StableTargetIdentity", "enumerate_allocations", "opponent_bench_targets",
]
