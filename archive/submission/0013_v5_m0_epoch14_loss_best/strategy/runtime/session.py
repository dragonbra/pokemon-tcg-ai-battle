"""Shared causal session for offline replay and online inference."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..features.card_semantics import CardSemanticRegistry
from ..features.compiler import CompiledFeatures, compile_features, tensorize_compiled
from ..features.schema import SCHEMA_VERSION, TypedPolicyInput
from ..knowledge.state import CausalKnowledgeState


def _plain(value: Any) -> Any:
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TypedInputEnvelope:
    schema_version: str
    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TypedInputEnvelope:
        if set(value) != {"schema_version", "payload"} or value["schema_version"] != SCHEMA_VERSION or not isinstance(value["payload"], Mapping):
            raise ValueError("typed input schema mismatch")
        return cls(value["schema_version"], value["payload"])


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    decision_index: int
    typed: TypedPolicyInput
    sha256: str
    tensors: Mapping[str, Any]


class PolicySession:
    def __init__(
        self,
        actor: int,
        deck: Sequence[int],
        *,
        registry: CardSemanticRegistry | None = None,
    ) -> None:
        self.actor = actor
        self.deck = tuple(deck)
        self.registry = registry
        self.knowledge = CausalKnowledgeState.new_game(actor, deck)
        self._last: EncodedDecision | None = None

    @classmethod
    def new_game(
        cls,
        actor: int,
        deck: Sequence[int],
        *,
        registry: CardSemanticRegistry | None = None,
    ) -> PolicySession:
        return cls(actor, deck, registry=registry)

    def observe(self, observation: Mapping[str, Any]) -> EncodedDecision:
        knowledge = self.knowledge.consume(observation)
        compiled = compile_features(
            observation,
            registered_deck=self.deck,
            knowledge=knowledge,
            registry=self.registry,
        )
        self._last = EncodedDecision(
            knowledge.decision_index,
            compiled.typed,
            compiled.digest,
            tensorize_compiled(compiled),
        )
        return self._last

    def record_action(self, action: Sequence[int]) -> None:
        if self._last is None:
            raise ValueError("cannot record action before observation")
        self.knowledge.record_pending(action, self._last.decision_index)


def replay_session(
    actor: int,
    deck: Sequence[int],
    observations: Sequence[Mapping[str, Any]],
    actions: Sequence[Sequence[int]],
    *,
    registry: CardSemanticRegistry | None = None,
) -> tuple[EncodedDecision, ...]:
    if len(observations) != len(actions):
        raise ValueError("offline replay observations/actions length mismatch")
    runtime = PolicySession.new_game(actor, deck, registry=registry)
    output: list[EncodedDecision] = []
    for observation, action in zip(observations, actions):
        output.append(runtime.observe(observation))
        runtime.record_action(action)
    return tuple(output)


__all__ = ["EncodedDecision", "PolicySession", "TypedInputEnvelope", "replay_session"]
