"""Lifetime-aware incremental canonical feature compilation.

The active compiler caches battle constants, card-entity fragments, and
append-only event payload fragments while rebuilding dynamic indices and ages.
The stateless compiler remains authoritative: any lifetime, classification, or
relation failure clears the session cache and executes a full rebuild.

The legacy dependency projection helpers remain public only as audit tools for
the rejected whole-layer experiment; the active path never freezes or compares
the complete observation tree.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from ..contracts.fields import SCHEMA_VERSION
from ..domain.prototypes import PrototypeIndex
from ..knowledge.state import CausalSnapshot
from .compiler import compile_canonical_row
from .layers import (
    CardLayer,
    EventLayer,
    GlobalLayer,
    OptionLayer,
    ResourceLayer,
    assemble_canonical_record,
    compile_global_layer,
    compile_resource_layer,
)
from .fragments import CardFragmentCompiler, EventFragmentCompiler
from .dynamic_fragments import OptionFragmentCompiler
from .relations import integer


LAYER_NAMES = ("cards", "resources", "events", "options", "globals")


class _DependencyError(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _battle_constants_signature(row: Mapping[str, Any], actor: int) -> tuple[Any, ...]:
    manifest = row.get("deck_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("incremental row has no deck manifest")
    counts = manifest.get("counts")
    if not isinstance(counts, (list, tuple)):
        raise ValueError("incremental deck manifest has no counts")
    compact: list[tuple[int, int]] = []
    for item in counts:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("incremental deck count is malformed")
        identity, count = item
        if (
            isinstance(identity, bool) or not isinstance(identity, int)
            or isinstance(count, bool) or not isinstance(count, int)
        ):
            raise ValueError("incremental deck count is not integral")
        compact.append((identity, count))
    return actor, SCHEMA_VERSION, tuple(compact)


@dataclass(frozen=True, slots=True)
class LayerDependencies:
    """Immutable, explicitly partitioned inputs for one canonical decision."""

    cards: tuple[Any, ...]
    resources: tuple[Any, ...]
    events: tuple[Any, ...]
    options: tuple[Any, ...]
    global_facts: tuple[Any, ...]
    decision_index: int
    actor: int
    schema_version: str


@dataclass(frozen=True, slots=True)
class _LayerBundle:
    cards: CardLayer
    resources: ResourceLayer
    events: EventLayer
    options: OptionLayer
    globals: GlobalLayer


class IncrementalCompileStats:
    """Monotonic counters suitable for logs and performance profiles."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def _increment(self, name: str, amount: int = 1) -> None:
        self._counts[name] += amount

    def snapshot(self) -> dict[str, int | float]:
        result: dict[str, int | float] = dict(sorted(self._counts.items()))
        for name in ("decisions", "fallbacks", "full_rebuilds", "resets"):
            result.setdefault(name, self._counts[name])
        decisions = self._counts["decisions"]
        for layer in LAYER_NAMES:
            hits = self._counts[f"layer/{layer}/hit"]
            misses = self._counts[f"layer/{layer}/miss"]
            result.setdefault(f"layer/{layer}/hit", hits)
            result.setdefault(f"layer/{layer}/miss", misses)
            result[f"layer/{layer}/hit_rate"] = (
                float(hits) / float(hits + misses) if hits + misses else 0.0
            )
        result["fallback_rate"] = (
            float(self._counts["fallbacks"]) / float(decisions) if decisions else 0.0
        )
        return result


def _freeze(value: Any) -> Any:
    """Make one explicitly selected dependency value deterministic and immutable."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Enum):
        return (type(value).__qualname__, _freeze(value.value))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple((field.name, _freeze(getattr(value, field.name))) for field in fields(value)),
        )
    if isinstance(value, Mapping):
        items = [(_freeze(key), _freeze(item)) for key, item in value.items()]
        return tuple(sorted(items, key=lambda pair: (type(pair[0]).__name__, repr(pair[0]))))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    raise TypeError(f"unsupported dependency value: {type(value).__qualname__}")


def _player(current: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    players = current.get("players")
    if (
        not isinstance(players, Sequence)
        or isinstance(players, (str, bytes))
        or len(players) != 2
        or not isinstance(players[index], Mapping)
    ):
        raise ValueError("incremental projection requires two player states")
    return players[index]


def _selected_card_facts(player: Mapping[str, Any], zones: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple((zone, _freeze(player.get(zone))) for zone in zones)


def _card_layout_projection(cards: tuple[Any, ...]) -> tuple[Any, ...]:
    """Compact identity/layout view used by relation-bearing layer dependencies.

    The complete cards projection still drives the dirty graph, so this token is
    diagnostic redundancy rather than the sole guard against relation staleness.
    """

    return ("card_layout_v1", cards)


def project_layer_dependencies(
    row: Mapping[str, Any], snapshot: CausalSnapshot
) -> LayerDependencies:
    """Enumerate every actor-visible source consumed by each pure layer."""

    observation = row.get("actor_observation")
    if not isinstance(observation, Mapping):
        raise ValueError("incremental row has no actor observation")
    current, select = observation.get("current"), observation.get("select")
    if not isinstance(current, Mapping) or not isinstance(select, Mapping):
        raise ValueError("incremental observation has no current/select state")
    actor = integer(current.get("yourIndex"), -1)
    if actor not in (0, 1) or snapshot.perspective_actor != actor:
        raise _DependencyError(
            "actor_mismatch", "incremental actor and causal snapshot disagree"
        )
    if not isinstance(snapshot.decision_index, int) or snapshot.decision_index < 0:
        raise _DependencyError(
            "invalid_decision_index",
            "incremental decision index must be a non-negative integer",
        )
    actor_schema = row.get("actor_schema_version", SCHEMA_VERSION)
    if actor_schema != SCHEMA_VERSION:
        raise _DependencyError(
            "schema_mismatch", f"incremental actor schema mismatch: {actor_schema!r}"
        )
    opponent = 1 - actor
    own, other = _player(current, actor), _player(current, opponent)
    status_names = ("asleep", "burned", "confused", "paralyzed", "poisoned")
    cards = (
        actor,
        _selected_card_facts(own, ("active", "bench", "hand", "discard")),
        _selected_card_facts(other, ("active", "bench", "discard")),
        ("own_status", tuple((name, _freeze(own.get(name))) for name in status_names)),
        ("other_status", tuple((name, _freeze(other.get(name))) for name in status_names)),
        ("stadium", _freeze(current.get("stadium"))),
        ("looking", _freeze(current.get("looking"))),
        ("select_deck", _freeze(select.get("deck"))),
        ("context_card", _freeze(select.get("contextCard"))),
        ("effect_card", _freeze(select.get("effect"))),
        ("deck_order_known", bool(snapshot.deck_order_known)),
        ("known_self_deck_order", _freeze(snapshot.known_self_deck_order)),
        ("known_opponent_hand", _freeze(snapshot.known_opponent_hand)),
        ("possible_opponent_hand", _freeze(snapshot.possible_opponent_hand)),
        ("remembered_opponent_cards", _freeze(snapshot.remembered_opponent_cards)),
    )
    layout = _card_layout_projection(cards)
    resources = (
        ("self_ledger", _freeze(snapshot.self_ledger)),
        ("deck_order_known", bool(snapshot.deck_order_known)),
    )
    events = (("recent_events", _freeze(snapshot.recent_events)), layout)
    options = (
        ("select_type", _freeze(select.get("type"))),
        ("select_context", _freeze(select.get("context"))),
        ("context_card", _freeze(select.get("contextCard"))),
        ("effect_card", _freeze(select.get("effect"))),
        ("legal_options", _freeze(select.get("option"))),
        ("min_count", _freeze(select.get("minCount"))),
        ("max_count", _freeze(select.get("maxCount"))),
        layout,
    )
    global_facts = (
        actor,
        tuple(
            (name, _freeze(current.get(name)))
            for name in (
                "firstPlayer", "turn", "turnActionCount", "supporterPlayed",
                "stadiumPlayed", "energyAttached", "retreated", "looking",
            )
        ),
        tuple(
            (
                side,
                tuple((name, _freeze(player.get(name))) for name in (
                    "deckCount", "handCount", "benchMax", *status_names,
                )),
                ("prize_count", len(player.get("prize")) if isinstance(player.get("prize"), (list, tuple)) else 0),
                ("bench_count", len(player.get("bench")) if isinstance(player.get("bench"), (list, tuple)) else 0),
            )
            for side, player in (("own", own), ("other", other))
        ),
        tuple(
            (name, _freeze(select.get(name)))
            for name in (
                "type", "context", "minCount", "maxCount",
                "remainDamageCounter", "remainEnergyCost",
            )
        ),
        ("option_count", len(select.get("option")) if isinstance(select.get("option"), (list, tuple)) else 0),
        ("deck_membership_known", bool(snapshot.deck_membership_known)),
        ("deck_order_known", bool(snapshot.deck_order_known)),
        ("known_opponent_hand_count", len(snapshot.known_opponent_hand)),
        ("unknown_opponent_hand", int(snapshot.unknown_opponent_hand)),
        ("possible_opponent_hand_count", len(snapshot.possible_opponent_hand)),
        ("possible_opponent_hand_known_lower", snapshot.possible_opponent_hand_known_lower),
        ("possible_opponent_hand_known_upper", snapshot.possible_opponent_hand_known_upper),
    )
    return LayerDependencies(
        cards=cards,
        resources=resources,
        events=events,
        options=options,
        global_facts=global_facts,
        decision_index=snapshot.decision_index,
        actor=actor,
        schema_version=SCHEMA_VERSION,
    )


def dirty_layers(
    previous: LayerDependencies | None, current: LayerDependencies
) -> frozenset[str]:
    """Return the dependency-closed set of layers that must be rebuilt."""

    if previous is None:
        return frozenset(LAYER_NAMES)
    if previous.actor != current.actor or previous.schema_version != current.schema_version:
        return frozenset(LAYER_NAMES)
    dirty: set[str] = set()
    if previous.cards != current.cards:
        dirty.update(("cards", "events", "options"))
    if previous.resources != current.resources:
        dirty.add("resources")
    if previous.events != current.events:
        dirty.add("events")
    if previous.options != current.options:
        dirty.add("options")
    if previous.global_facts != current.global_facts:
        dirty.add("globals")
    return frozenset(dirty)


def _validate_relations(bundle: _LayerBundle) -> None:
    card_count = len(bundle.cards.cat)
    for name, values in (
        ("card_parent", bundle.cards.parent),
        ("event_source", bundle.events.source),
        ("event_target", bundle.events.target),
        ("event_before", bundle.events.before),
        ("event_after", bundle.events.after),
        ("option_source", bundle.options.source),
        ("option_target", bundle.options.target),
        ("option_context", bundle.options.context),
        ("option_effect_card", bundle.options.effect_card),
    ):
        if any(index < -1 or index >= card_count for index in values):
            raise ValueError(f"incremental {name} relation endpoint outside card layer")
    option_count = len(bundle.options.cat)
    for name, values in (
        ("option_skill_parent", bundle.options.skill_parent),
        ("option_effect_parent", bundle.options.effect_parent),
    ):
        if any(index < 0 or index >= option_count for index in values):
            raise ValueError(f"incremental {name} relation endpoint outside option layer")


class IncrementalCanonicalCompiler:
    """One-session lifetime-aware fragment cache with fail-closed rebuilds.

    Battle identity and turn ownership are tracked explicitly.  Cards and
    events are always assembled in canonical order, while their expensive
    entity/payload rows are reused at a finer granularity than a whole layer.
    The remaining small layers stay stateless until they have an independently
    measured fragment boundary.
    """

    def __init__(self, prototypes: PrototypeIndex):
        self.prototypes = prototypes
        self.stats = IncrementalCompileStats()
        self._session_actor: int | None = None
        self._last_decision_index: int | None = None
        self._battle_signature: Any | None = None
        self._turn: int | None = None
        self._turn_epoch = -1
        self._card_fragments = CardFragmentCompiler()
        self._event_fragments = EventFragmentCompiler()
        self._option_fragments = OptionFragmentCompiler()

    def reset(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("incremental reset requires a non-empty reason")
        self._session_actor = None
        self._last_decision_index = None
        self._battle_signature = None
        self._turn = None
        self._turn_epoch = -1
        self._card_fragments.clear()
        self._event_fragments.clear()
        self._option_fragments.clear()
        self.stats._increment("resets")
        self.stats._increment(f"reset/{reason}")

    def _fallback(
        self, row: Mapping[str, Any], snapshot: CausalSnapshot, reason: str
    ) -> dict[str, Any]:
        self.reset(reason)
        self.stats._increment("fallbacks")
        self.stats._increment(f"fallback/{reason}")
        for layer in LAYER_NAMES:
            self.stats._increment(f"layer/{layer}/miss")
        record = compile_canonical_row(row, snapshot, self.prototypes)
        self.stats._increment("full_rebuilds")
        if isinstance(snapshot.decision_index, int):
            self._last_decision_index = snapshot.decision_index
        return record

    def compile(
        self, row: Mapping[str, Any], snapshot: CausalSnapshot
    ) -> dict[str, Any]:
        self.stats._increment("decisions")
        try:
            observation = row.get("actor_observation")
            if not isinstance(observation, Mapping):
                raise ValueError("incremental row has no actor observation")
            current_state = observation.get("current")
            select = observation.get("select")
            if not isinstance(current_state, Mapping) or not isinstance(select, Mapping):
                raise ValueError("incremental observation has no current/select state")
            actor = integer(current_state.get("yourIndex"), -1)
            if actor not in (0, 1) or snapshot.perspective_actor != actor:
                raise _DependencyError("actor_mismatch", "incremental actor and snapshot disagree")
            if row.get("actor_schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
                raise _DependencyError("schema_mismatch", "incremental actor schema mismatch")
            decision_index = snapshot.decision_index
            if not isinstance(decision_index, int) or decision_index < 0:
                raise _DependencyError("invalid_decision_index", "invalid decision index")
            battle_signature = _battle_constants_signature(row, actor)
            turn = integer(current_state.get("turn"), -1)
            if turn < 0:
                raise _DependencyError("invalid_turn", "incremental turn must be non-negative")
        except Exception as error:
            reason = (
                error.reason
                if isinstance(error, _DependencyError)
                else f"projection_{type(error).__name__}"
            )
            return self._fallback(row, snapshot, reason)

        if self._session_actor is None:
            self._session_actor = actor
            self._battle_signature = battle_signature
            self.stats._increment("lifetime/battle/start")
        elif actor != self._session_actor:
            return self._fallback(row, snapshot, "actor_mismatch")
        elif battle_signature != self._battle_signature:
            return self._fallback(row, snapshot, "battle_constants_changed")
        else:
            self.stats._increment("lifetime/battle/reuse")
        if (
            self._last_decision_index is not None
            and decision_index != self._last_decision_index + 1
        ):
            return self._fallback(row, snapshot, "non_chronological_decision")
        if self._turn is None or turn != self._turn:
            if self._turn is not None and turn < self._turn:
                return self._fallback(row, snapshot, "turn_rewound")
            self._turn = turn
            self._turn_epoch += 1
            self.stats._increment("lifetime/turn/start")
        else:
            self.stats._increment("lifetime/turn/reuse")

        card_hits = self._card_fragments.hits
        card_misses = self._card_fragments.misses
        event_hits = self._event_fragments.hits
        event_misses = self._event_fragments.misses
        option_hits = self._option_fragments.semantic_hits
        option_misses = self._option_fragments.semantic_misses
        try:
            cards = self._card_fragments.compile(row, snapshot)
            resources = compile_resource_layer(row, snapshot)
            events = self._event_fragments.compile(row, snapshot, cards)
            options = self._option_fragments.compile(
                row, snapshot, self.prototypes, cards
            )
            globals_ = compile_global_layer(row, snapshot)
            bundle = _LayerBundle(cards, resources, events, options, globals_)
            _validate_relations(bundle)
            record = assemble_canonical_record(
                row, cards, resources, events, options, globals_
            )
        except Exception as error:
            return self._fallback(row, snapshot, f"layer_{type(error).__name__}")
        new_card_hits = self._card_fragments.hits - card_hits
        new_card_misses = self._card_fragments.misses - card_misses
        new_event_hits = self._event_fragments.hits - event_hits
        new_event_misses = self._event_fragments.misses - event_misses
        new_option_hits = self._option_fragments.semantic_hits - option_hits
        new_option_misses = self._option_fragments.semantic_misses - option_misses
        self.stats._increment("fragment/card/hit", new_card_hits)
        self.stats._increment("fragment/card/miss", new_card_misses)
        self.stats._increment("fragment/event/hit", new_event_hits)
        self.stats._increment("fragment/event/miss", new_event_misses)
        self.stats._increment("fragment/option_semantic/hit", new_option_hits)
        self.stats._increment("fragment/option_semantic/miss", new_option_misses)
        self.stats._increment(
            "work/avoided_fragments",
            new_card_hits + new_event_hits + new_option_hits,
        )
        self.stats._increment(
            "work/rebuilt_fragments",
            new_card_misses + new_event_misses + new_option_misses,
        )
        # These counters retain the old public shape without claiming whole-layer
        # reuse.  Every decision assembles all five canonical layers.
        for layer in LAYER_NAMES:
            self.stats._increment(f"layer/{layer}/miss")
        self._last_decision_index = decision_index
        return record


__all__ = [
    "IncrementalCanonicalCompiler",
    "IncrementalCompileStats",
    "LAYER_NAMES",
    "LayerDependencies",
    "dirty_layers",
    "project_layer_dependencies",
]
