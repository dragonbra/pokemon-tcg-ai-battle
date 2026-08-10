"""Serializable Zero-Shot sequential-teacher labels for the allocation head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True, slots=True)
class AllocationBCSample:
    battle_id: str
    seed: int
    opponent_id: str
    focal_first: bool
    turn: int
    pre_action_features: dict[str, Tensor]
    root_index: int
    target_ids: tuple[dict[str, int], ...]
    counters: tuple[int, ...]
    primitive_order: tuple[int, ...]
    immediate_ko_targets: int
    immediate_prizes: int
    search_seed: int = 0
    primitive_action_prefix: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if sum(self.counters) != 6 or not 1 <= len(self.counters) <= 5:
            raise ValueError("allocation BC label must be a complete six-counter allocation")


def split_for_battle(battle_id: str, *, validation_modulus: int = 10) -> str:
    """Stable battle-level split; no macro from one battle can cross splits."""
    import hashlib
    value = int.from_bytes(hashlib.sha256(battle_id.encode()).digest()[:8], "big")
    return "validation" if value % validation_modulus == 0 else "train"


__all__ = ["AllocationBCSample", "split_for_battle"]
