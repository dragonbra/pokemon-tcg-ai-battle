"""Deterministic typed-to-tensor compiler shared by offline batches and runtime."""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..knowledge.state import KnowledgeSnapshot
from .card_semantics import CardSemanticRegistry, FieldState
from .observation import encode_observation
from .schema import (
    DeckToken,
    EntityToken,
    EpistemicState,
    EventToken,
    LedgerToken,
    NumericFeature,
    OptionToken,
    Relation,
    RelationType,
    StateToken,
    TypedPolicyInput,
)

COMPILER_VERSION = "semantic_goal_feature_compiler_v2"
SEMANTIC_WIDTH = 64
_COMPILER_SOURCES = (
    Path(__file__),
    Path(__file__).with_name("observation.py"),
    Path(__file__).with_name("schema.py"),
    Path(__file__).with_name("card_semantics.py"),
    Path(__file__).parents[1] / "knowledge" / "state.py",
    Path(__file__).parents[1] / "knowledge" / "ledger.py",
    Path(__file__).parents[1] / "knowledge" / "visibility.py",
    Path(__file__).parents[1] / "training" / "batching.py",
)


def compiler_sha256() -> str:
    """Bind every source module that participates in causal feature materialization."""
    digest = hashlib.sha256()
    root = Path(__file__).parents[1]
    for path in sorted(_COMPILER_SOURCES, key=lambda item: str(item.relative_to(root))):
        digest.update(str(path.relative_to(root)).encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
STATE_CAT_WIDTH = 8
STATE_NUM_WIDTH = 16
ENTITY_CAT_WIDTH = 8
ENTITY_NUM_WIDTH = 12
OPTION_CAT_WIDTH = 12
OPTION_NUM_WIDTH = 8
LEDGER_CAT_WIDTH = 6
LEDGER_NUM_WIDTH = 12
EVENT_CAT_WIDTH = 6
EVENT_NUM_WIDTH = 8


def _code(value: object, *, width: int = 4095) -> int:
    if value is None:
        return 0
    payload = str(value).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % width + 1


def _category(value: object, *, width: int = 4095) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return min(value + 1, width)
    return _code(value, width=width)


def _number(value: object, cap: float = 1000.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(-cap, min(cap, float(value)))


def _semantic_vector(card_id: int | None, registry: CardSemanticRegistry | None) -> list[float]:
    vector = [0.0] * SEMANTIC_WIDTH
    if card_id is None:
        return vector
    card = registry.lookup(card_id) if registry is not None else None
    if card is None:
        vector[0] = 1.0
        return vector
    for offset, value in enumerate((card.card_kind, card.stage, card.category, card.pokemon_type.value)):
        vector[offset] = _code(value, width=255) / 255.0
    for offset, field in ((4, card.hp), (5, card.retreat)):
        if field.value is not None:
            vector[offset] = _number(field.value, 1000.0) / 1000.0
        vector[offset + 2] = _code(field.state.value, width=16) / 16.0
    for capability in sorted(card.capabilities):
        vector[8 + _code(capability, width=48) % (SEMANTIC_WIDTH - 8)] += 1.0
    for move in card.moves:
        vector[56 + (_code(move.name, width=8) - 1)] += 1.0
        for effect in move.effects:
            vector[48 + (_code(effect.kind, width=8) - 1)] += 1.0
    return vector


def _numeric(feature: NumericFeature) -> tuple[float, float, float, float]:
    state = list(EpistemicState).index(feature.epistemic) + 1
    return feature.value, float(state), float(feature.overflow), float(feature.padding)


def _entity_semantics(entity: EntityToken, registry: CardSemanticRegistry | None) -> list[float]:
    return _semantic_vector(entity.card_id, registry)


def _option_semantics(option: OptionToken, registry: CardSemanticRegistry | None) -> list[float]:
    return _semantic_vector(option.card_id, registry)


@dataclass(frozen=True, slots=True)
class CompiledFeatures:
    typed: TypedPolicyInput
    entities_semantic: tuple[tuple[float, ...], ...]
    options_semantic: tuple[tuple[float, ...], ...]
    deck_semantic: tuple[tuple[float, ...], ...]
    ledger_semantic: tuple[tuple[float, ...], ...]
    events_semantic: tuple[tuple[float, ...], ...]
    compiler_version: str = COMPILER_VERSION

    @property
    def digest(self) -> str:
        import json
        payload = {
            "compiler_version": self.compiler_version,
            "typed": _plain(self.typed),
            "entities_semantic": self.entities_semantic,
            "options_semantic": self.options_semantic,
            "deck_semantic": self.deck_semantic,
            "ledger_semantic": self.ledger_semantic,
            "events_semantic": self.events_semantic,
        }
        return hashlib.sha256((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
    return value


def _with_causal_channels(
    typed: TypedPolicyInput,
    raw: Mapping[str, Any],
    knowledge: KnowledgeSnapshot | None,
    registry: CardSemanticRegistry | None,
    *,
    max_events: int,
) -> CompiledFeatures:
    ledgers: list[LedgerToken] = []
    ledger_semantics: list[tuple[float, ...]] = []
    entries = knowledge.self_ledger if knowledge is not None else {}
    for deck in typed.registered_deck:
        entry = entries.get(deck.card_id)
        capability = None
        if registry is not None:
            capability = ",".join(sorted(registry.lookup(deck.card_id).capabilities)) or "none"
        counts: dict[str, NumericFeature] = {
            "registered": NumericFeature.observed(float(deck.multiplicity), cap=60),
        }
        if entry is not None:
            counts["deck"] = NumericFeature.observed(float(entry.deck.value), cap=60) if entry.deck.value is not None else NumericFeature.unknown()
            counts["prize"] = NumericFeature.observed(float(entry.prize.value), cap=60) if entry.prize.value is not None else NumericFeature.unknown()
            counts["deck_epistemic"] = NumericFeature.observed(float(_code(entry.deck.state.value, width=16)), cap=16)
            counts["prize_epistemic"] = NumericFeature.observed(float(_code(entry.prize.state.value, width=16)), cap=16)
        ledgers.append(LedgerToken(f"ledger:{deck.card_id}", deck.card_id, capability, counts))
        ledger_semantics.append(tuple(_semantic_vector(deck.card_id, registry)))

    raw_logs = raw.get("logs", [])
    logs = tuple(item for item in raw_logs if isinstance(item, Mapping)) if isinstance(raw_logs, Sequence) else ()
    logs = logs[-max_events:]
    events: list[EventToken] = []
    event_semantics: list[tuple[float, ...]] = []
    for index, log in enumerate(logs):
        card_id = log.get("cardId")
        card_id = card_id if isinstance(card_id, int) and not isinstance(card_id, bool) else None
        events.append(EventToken(
            f"event:{index}",
            str(log.get("type", "unknown")),
            log.get("playerIndex") if isinstance(log.get("playerIndex"), int) else None,
            card_id,
            len(logs) - index,
            "actor_visible_log",
        ))
        event_semantics.append(tuple(_semantic_vector(card_id, registry)))

    relations = list(typed.relations)
    for index in range(1, len(events)):
        relations.append(Relation(RelationType.PREVIOUS_EVENT, events[index].token_id, events[index - 1].token_id))
    typed = TypedPolicyInput(
        typed.state, typed.entities, typed.registered_deck, tuple(ledgers), tuple(events), typed.options,
        tuple(relations), typed.schema_version,
    )
    typed.validate()
    return CompiledFeatures(
        typed,
        tuple(tuple(_entity_semantics(entity, registry)) for entity in typed.entities),
        tuple(tuple(_option_semantics(option, registry)) for option in typed.options),
        tuple(tuple(_semantic_vector(deck.card_id, registry)) for deck in typed.registered_deck),
        tuple(ledger_semantics),
        tuple(event_semantics),
    )


def compile_features(
    raw: Mapping[str, Any],
    *,
    registered_deck: Sequence[int],
    knowledge: KnowledgeSnapshot | None = None,
    registry: CardSemanticRegistry | None = None,
    max_events: int = 64,
) -> CompiledFeatures:
    typed = encode_observation(raw, registered_deck=registered_deck)
    current = raw.get("current", {})
    players = current.get("players", []) if isinstance(current, Mapping) else []
    actor = current.get("yourIndex") if isinstance(current, Mapping) else None
    numeric = dict(typed.state.numeric)
    if isinstance(actor, int) and not isinstance(actor, bool) and isinstance(players, Sequence):
        own = players[actor] if 0 <= actor < len(players) and isinstance(players[actor], Mapping) else {}
        opponent_index = 1 - actor if len(players) == 2 else -1
        opponent = players[opponent_index] if 0 <= opponent_index < len(players) and isinstance(players[opponent_index], Mapping) else {}
        numeric.update({
            "own_deck_count": NumericFeature.observed(_number(own.get("deckCount"), 60), cap=60),
            "opponent_deck_count": NumericFeature.observed(_number(opponent.get("deckCount"), 60), cap=60),
            "own_hand_count": NumericFeature.observed(_number(own.get("handCount"), 60), cap=60),
            "opponent_hand_count": NumericFeature.observed(_number(opponent.get("handCount"), 60), cap=60),
            "own_prize_count": NumericFeature.observed(float(len(own.get("prize") or ())), cap=6),
            "opponent_prize_count": NumericFeature.observed(float(len(opponent.get("prize") or ())), cap=6),
        })
    typed.state = StateToken(typed.state.token_id, typed.state.categorical, numeric)
    return _with_causal_channels(typed, raw, knowledge, registry, max_events=max_events)


def _cat(values: Sequence[object], width: int) -> list[int]:
    output = [_category(value) for value in values[:width]]
    return output + [0] * (width - len(output))


def _numeric_map(values: Mapping[str, NumericFeature], names: Sequence[str], width: int) -> list[float]:
    output: list[float] = []
    for name in names:
        feature = values.get(name, NumericFeature.not_applicable())
        value, state, overflow, padding = _numeric(feature)
        output.extend((value, state + 16.0 * overflow + 32.0 * padding))
    return (output + [0.0] * width)[:width]


def _option_cat(option: OptionToken) -> list[int]:
    values = [
        option.option_type, option.card_id, option.categorical.get("attackId"),
        option.categorical.get("count"), option.categorical.get("energyIndex"),
        option.categorical.get("toolIndex"), option.categorical.get("index"),
        option.categorical.get("area"), option.categorical.get("playerIndex"),
        option.categorical.get("targetPlayerIndex"), option.categorical.get("inPlayIndex"),
        option.categorical.get("specialConditionType"),
    ]
    return _cat(values, OPTION_CAT_WIDTH)


def tensorize_compiled(compiled: CompiledFeatures, *, max_entities: int = 128, max_events: int = 64) -> dict[str, Any]:
    """Convert typed tokens into fixed-width Python arrays; batching owns padding."""
    typed = compiled.typed
    state = typed.state
    current = state.categorical
    state_cat = _cat([current.get("actor"), current.get("first_player"), current.get("select_type"), current.get("select_context"), current.get("select_effect")], STATE_CAT_WIDTH)
    state_num = _numeric_map(state.numeric, (
        "turn", "turn_action_count", "own_deck_count", "opponent_deck_count",
        "own_hand_count", "opponent_hand_count", "min_count", "max_count",
    ), STATE_NUM_WIDTH)
    entities = list(typed.entities[:max_entities])
    entity_cat = [_cat([entity.owner, entity.zone, entity.card_id, entity.instance_key, entity.categorical.get("slot")], ENTITY_CAT_WIDTH) for entity in entities]
    entity_num = [_numeric_map(entity.numeric, ("hp", "max_hp", "deck_count", "hand_count", "bench_max"), ENTITY_NUM_WIDTH) for entity in entities]
    options = list(typed.options)
    option_cat = [_option_cat(option) for option in options]
    option_num = [_numeric_map(option.numeric, (
        "count", "number", "energyIndex", "toolIndex",
    ), OPTION_NUM_WIDTH) for option in options]
    ledger_cat = [_cat([item.card_id, item.capability, item.token_id], LEDGER_CAT_WIDTH) for item in typed.ledgers]
    ledger_num = [_numeric_map(item.counts, ("registered", "deck", "prize", "deck_epistemic", "prize_epistemic"), LEDGER_NUM_WIDTH) for item in typed.ledgers]
    events = list(typed.events[-max_events:])
    entity_indices = {entity.token_id: index for index, entity in enumerate(entities)}
    state_cat[5] = min(max(0, len(typed.entities) - max_entities), 4095)
    state_cat[6] = min(max(0, len(typed.events) - max_events), 4095)
    relation_endpoint_overflow = sum(
        relation.source not in entity_indices or relation.target not in entity_indices
        for relation in typed.relations
        if relation.source.startswith("entity:") or relation.target.startswith("entity:")
    )
    state_cat[7] = min(relation_endpoint_overflow, 4095)
    if len(options) > 64:
        raise ValueError("legal option count exceeds decoder capacity 64")
    event_cat = [_cat([item.event_type, item.actor, item.card_id, item.visibility_source], EVENT_CAT_WIDTH) for item in events]
    event_num = [[float(item.relative_age)] + [0.0] * (EVENT_NUM_WIDTH - 1) for item in events]
    relations = [
        (entity_indices[relation.source], entity_indices[relation.target], _code(relation.kind, width=15))
        for relation in typed.relations
        if relation.source in entity_indices and relation.target in entity_indices
    ]
    return {
        "state_cat": state_cat, "state_num": state_num,
        "entities_cat": entity_cat, "entities_num": entity_num,
        "entity_semantic": list(compiled.entities_semantic[:max_entities]),
        "entity_mask": [True] * len(entities),
        "deck_card_ids": [item.card_id for item in typed.registered_deck],
        "deck_multiplicity": [item.multiplicity for item in typed.registered_deck],
        "deck_semantic": list(compiled.deck_semantic),
        "deck_mask": [True] * len(typed.registered_deck),
        "ledger_cat": ledger_cat, "ledger_num": ledger_num,
        "ledger_semantic": list(compiled.ledger_semantic), "ledger_mask": [True] * len(ledger_cat),
        "events_cat": event_cat, "events_num": event_num,
        "event_semantic": list(compiled.events_semantic), "event_mask": [True] * len(event_cat),
        "relations": relations,
        "options_cat": option_cat, "options_num": option_num,
        "option_semantic": list(compiled.options_semantic), "option_mask": [True] * len(options),
        "min_count": int(state.numeric["min_count"].value),
        "max_count": int(state.numeric["max_count"].value),
    }


__all__ = [
    "COMPILER_VERSION", "CompiledFeatures", "compile_features", "compiler_sha256",
    "tensorize_compiled",
]
