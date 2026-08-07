"""Auditable prototype for decoded-observation lifetime version tokens.

The official engine still returns a complete JSON observation.  This module
turns only the compiler-consumed top-level card segments into compact binary
commitments immediately after decoding.  V2 benchmarks rejected connecting
this tracker to the hot path because it duplicated traversal; the prototype is
retained to make that negative result and its lifecycle semantics reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import pickle
from typing import Any

from ..knowledge.state import CausalSnapshot
from .relations import integer


class ObservationVersionError(ValueError):
    """The observation cannot be assigned trustworthy incremental versions."""


@dataclass(frozen=True, slots=True)
class ObservationVersions:
    decision: int
    battle: int
    turn: int
    cards: int
    events: int
    resources: int
    selection: int
    globals: int
    card_segments: tuple[tuple[str, int], ...]

    def card_segment_map(self) -> dict[str, int]:
        return dict(self.card_segments)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservationVersionError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ObservationVersionError(f"{label} must be a sequence")
    return value


def _items(value: Any) -> Sequence[Any]:
    """Mirror the stateless compiler: absent/non-list card zones are empty."""

    return value if isinstance(value, (list, tuple)) else ()


def _commit(value: Any) -> bytes:
    """Return a type-exact, order-sensitive commitment without Python recursion."""

    try:
        return pickle.dumps(value, protocol=5)
    except Exception as error:
        raise ObservationVersionError(
            f"observation segment is not safely serializable: {type(error).__name__}"
        ) from error


class ObservationVersionTracker:
    """Own monotonic dirty versions for exactly one actor-local battle."""

    def __init__(self, actor: int, *, trusted_engine: bool = False):
        if actor not in (0, 1):
            raise ValueError("version tracker actor must be 0 or 1")
        self.actor = actor
        self.trusted_engine = bool(trusted_engine)
        self.reset()

    def reset(self) -> None:
        self._last_decision: int | None = None
        self._turn_value: int | None = None
        self._turn_version = 0
        self._cards_version = 0
        self._events_version = 0
        self._resources_version = 0
        self._selection_version = 0
        self._globals_version = 0
        self._segment_tokens: dict[str, Any] = {}
        self._segment_versions: dict[str, int] = {}
        self._event_token: tuple[tuple[int, int], ...] | None = None
        self._resource_token: Any = None
        self._selection_token: Any = None

    def _changed(self, previous: Any, value: Any) -> tuple[bool, Any]:
        if not self.trusted_engine:
            token = _commit(value)
            return token != previous, token
        if (
            previous is value
            and previous is not None
            and isinstance(value, (list, Mapping))
        ):
            # Official observations are freshly decoded.  Reusing a mutable
            # container crosses that contract, so do not trust self-equality.
            raise ObservationVersionError(
                "trusted engine observation reused a mutable segment object"
            )
        try:
            return previous is None or previous != value, value
        except Exception as error:
            raise ObservationVersionError(
                f"observation segment equality failed: {type(error).__name__}"
            ) from error

    def _segments(
        self, current: Mapping[str, Any], select: Mapping[str, Any], snapshot: CausalSnapshot
    ) -> tuple[tuple[str, Any], ...]:
        players = _sequence(current.get("players"), "current.players")
        if len(players) != 2:
            raise ObservationVersionError("current.players must contain two players")
        own = _mapping(players[self.actor], "own player")
        other = _mapping(players[1 - self.actor], "opponent player")
        values: list[tuple[str, Any]] = []
        for prefix, player, zones in (
            ("self", own, ("active", "bench", "hand", "discard")),
            ("opponent", other, ("active", "bench", "discard")),
        ):
            for zone in zones:
                values.append((f"{prefix}_{zone}", _items(player.get(zone))))
        values.extend((
            ("stadium", _items(current.get("stadium"))),
            ("looking", _items(current.get("looking"))),
            ("select_deck", _items(select.get("deck"))),
            ("known_self_deck_order", snapshot.known_self_deck_order),
            ("known_opponent_hand", snapshot.known_opponent_hand),
            ("possible_opponent_hand", snapshot.possible_opponent_hand),
            ("remembered_opponent_cards", snapshot.remembered_opponent_cards),
            ("context_card", select.get("contextCard")),
            ("effect_card", select.get("effect")),
        ))
        # Active status changes the root card categorical row.
        status_names = ("asleep", "burned", "confused", "paralyzed", "poisoned")
        values.append(("self_active_status", tuple(own.get(name) for name in status_names)))
        values.append(("opponent_active_status", tuple(other.get(name) for name in status_names)))
        values.append(("deck_order_known", bool(snapshot.deck_order_known)))
        return tuple(values)

    def observe(
        self, row: Mapping[str, Any], snapshot: CausalSnapshot
    ) -> ObservationVersions:
        observation = _mapping(row.get("actor_observation"), "actor_observation")
        current = _mapping(observation.get("current"), "observation.current")
        select = _mapping(observation.get("select"), "observation.select")
        actor = integer(current.get("yourIndex"), -1)
        if actor != self.actor or snapshot.perspective_actor != self.actor:
            raise ObservationVersionError("version tracker actor mismatch")
        decision = snapshot.decision_index
        if not isinstance(decision, int) or decision < 0:
            raise ObservationVersionError("invalid causal decision index")
        if self._last_decision is not None and decision != self._last_decision + 1:
            raise ObservationVersionError("version tracker requires strict chronology")

        dirty_card = False
        for name, value in self._segments(current, select, snapshot):
            changed, token = self._changed(self._segment_tokens.get(name), value)
            if changed:
                self._segment_versions[name] = self._segment_versions.get(name, 0) + 1
                dirty_card = True
            self._segment_tokens[name] = token
        if dirty_card:
            self._cards_version += 1

        event_token = tuple(
            (event.source_event, id(event)) for event in snapshot.recent_events
        )
        if event_token != self._event_token:
            self._event_token = event_token
            self._events_version += 1

        resource_token = (
            tuple(snapshot.self_ledger.items()), bool(snapshot.deck_order_known)
        )
        if resource_token != self._resource_token:
            self._resource_token = resource_token
            self._resources_version += 1

        selection_changed, selection_token = self._changed(
            self._selection_token, select
        )
        if selection_changed:
            self._selection_version += 1
        self._selection_token = selection_token

        turn = integer(current.get("turn"), -1)
        if turn < 0:
            raise ObservationVersionError("turn must be non-negative")
        if turn != self._turn_value:
            if self._turn_value is not None and turn < self._turn_value:
                raise ObservationVersionError("turn rewound")
            self._turn_value = turn
            self._turn_version += 1
        self._globals_version += 1
        self._last_decision = decision
        return ObservationVersions(
            decision=decision,
            battle=1,
            turn=self._turn_version,
            cards=self._cards_version,
            events=self._events_version,
            resources=self._resources_version,
            selection=self._selection_version,
            globals=self._globals_version,
            card_segments=tuple(sorted(self._segment_versions.items())),
        )


__all__ = [
    "ObservationVersionError",
    "ObservationVersions",
    "ObservationVersionTracker",
]
