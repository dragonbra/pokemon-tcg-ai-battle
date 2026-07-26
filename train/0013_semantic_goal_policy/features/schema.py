"""Versioned typed feature contracts for policy inputs."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "semantic_goal_typed_input_v1"


class EpistemicState(str, Enum):
    OBSERVED = "observed"
    REMEMBERED = "remembered"
    INFERRED_EXACT = "inferred_exact"
    BOUNDED = "bounded"
    UNKNOWN = "unknown"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    PADDING = "padding"


@dataclass(frozen=True, slots=True)
class NumericFeature:
    value: float
    epistemic: EpistemicState
    padding: bool = False
    overflow: bool = False

    @classmethod
    def observed(cls, value: float, *, cap: float | None = None) -> NumericFeature:
        overflow = cap is not None and abs(value) > cap
        clipped = max(-cap, min(cap, value)) if cap is not None else value
        return cls(float(clipped), EpistemicState.OBSERVED, False, overflow)

    @classmethod
    def unknown(cls) -> NumericFeature:
        return cls(0.0, EpistemicState.UNKNOWN)

    @classmethod
    def missing(cls) -> NumericFeature:
        return cls(0.0, EpistemicState.MISSING)

    @classmethod
    def not_applicable(cls) -> NumericFeature:
        return cls(0.0, EpistemicState.NOT_APPLICABLE)

    @classmethod
    def padding_value(cls) -> NumericFeature:
        return cls(0.0, EpistemicState.PADDING, True)


class RelationType(str, Enum):
    OWNED_BY = "owned_by"
    LOCATED_IN = "located_in"
    ATTACHED_TO = "attached_to"
    EVOLVES_FROM = "evolves_from"
    TARGETS = "targets"
    MOVES_FROM = "moves_from"
    MOVES_TO = "moves_to"
    OBSERVED_IN = "observed_in"
    SUMMARIZES = "summarizes"
    PREVIOUS_EVENT = "previous_event"


@dataclass(frozen=True, slots=True)
class Relation:
    kind: RelationType
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class StateToken:
    token_id: str
    categorical: Mapping[str, str | int]
    numeric: Mapping[str, NumericFeature]


@dataclass(frozen=True, slots=True)
class EntityToken:
    token_id: str
    owner: int
    zone: str
    card_id: int | None
    instance_key: int | None
    categorical: Mapping[str, str | int]
    numeric: Mapping[str, NumericFeature]


@dataclass(frozen=True, slots=True)
class DeckToken:
    token_id: str
    card_id: int
    multiplicity: int


@dataclass(frozen=True, slots=True)
class LedgerToken:
    token_id: str
    card_id: int | None
    capability: str | None
    counts: Mapping[str, NumericFeature]


@dataclass(frozen=True, slots=True)
class EventToken:
    token_id: str
    event_type: str
    actor: int | None
    card_id: int | None
    relative_age: int
    visibility_source: str


@dataclass(frozen=True, slots=True)
class OptionToken:
    token_id: str
    option_type: int | str
    card_id: int | None
    source_entity: str | None
    target_entity: str | None
    categorical: Mapping[str, str | int]
    numeric: Mapping[str, NumericFeature]


@dataclass(slots=True)
class TypedPolicyInput:
    state: StateToken
    entities: tuple[EntityToken, ...]
    registered_deck: tuple[DeckToken, ...]
    ledgers: tuple[LedgerToken, ...]
    events: tuple[EventToken, ...]
    options: tuple[OptionToken, ...]
    relations: tuple[Relation, ...]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported typed schema version")
        tokens = [self.state.token_id]
        for group in (self.entities, self.registered_deck, self.ledgers, self.events, self.options):
            tokens.extend(item.token_id for item in group)
        if len(tokens) != len(set(tokens)):
            raise ValueError("typed token IDs must be unique")
        endpoints = set(tokens)
        for relation in self.relations:
            if relation.source not in endpoints or relation.target not in endpoints:
                raise ValueError("invalid relation endpoint")
        seen_cards: set[int] = set()
        for token in self.registered_deck:
            if token.card_id in seen_cards or token.multiplicity <= 0:
                raise ValueError("invalid registered deck multiplicity")
            seen_cards.add(token.card_id)
        if sum(token.multiplicity for token in self.registered_deck) != 60:
            raise ValueError("registered deck must contain 60 cards")


__all__ = ["DeckToken", "EntityToken", "EpistemicState", "EventToken", "LedgerToken", "NumericFeature", "OptionToken", "Relation", "RelationType", "SCHEMA_VERSION", "StateToken", "TypedPolicyInput"]
