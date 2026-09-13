"""Causal player-local identity and functional resource ledgers."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class KnowledgeStage(str, Enum):
    UNOBSERVED = "unobserved"
    VISIBLE_NOW = "visible_now"
    REMEMBERED = "remembered"
    INFERRED_EXACT = "inferred_exact"
    BOUNDED = "bounded"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class KnownCount:
    value: int | None
    lower: int
    upper: int
    state: KnowledgeStage
    source_event: int | None = None
    age: int = 0
    valid: bool = True

    @classmethod
    def unknown(cls, upper: int) -> KnownCount:
        return cls(None, 0, upper, KnowledgeStage.UNKNOWN)

    @classmethod
    def exact(cls, value: int, state: KnowledgeStage, source_event: int) -> KnownCount:
        return cls(value, value, value, state, source_event)


@dataclass(frozen=True, slots=True)
class IdentityLedgerEntry:
    card_id: int
    initial: int
    visible: Mapping[str, int]
    deck: KnownCount
    prize: KnownCount


def immutable_entries(entries: dict[int, IdentityLedgerEntry]) -> Mapping[int, IdentityLedgerEntry]:
    return MappingProxyType(dict(entries))


__all__ = ["IdentityLedgerEntry", "KnowledgeStage", "KnownCount", "immutable_entries"]
