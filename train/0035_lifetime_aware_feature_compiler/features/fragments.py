"""Fine-grained reusable fragments for chronological canonical compilation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..domain.prototypes import FieldState
from ..knowledge.state import CausalSnapshot, TypedEvent
from .layers import (
    CardLayer,
    EventLayer,
    ZONE,
    _categorical_bool,
    _categorical_int,
    _context,
    _items,
    _knowledge_code,
    _raw_int,
    _relative_owner,
    _state,
    _status_bits,
)
from .relations import card_id, integer


def primitive_signature(value: Any) -> Any:
    """Return a cheap deterministic signature made only from engine primitives."""

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, Mapping):
        items = [
            (primitive_signature(key), primitive_signature(item))
            for key, item in value.items()
        ]
        return tuple(sorted(items, key=lambda item: repr(item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(primitive_signature(item) for item in value)
    # Knowledge records are frozen/slotted dataclasses.  Their repr is stable,
    # but unknown runtime objects must fail closed rather than become cache keys.
    slots = getattr(type(value), "__slots__", ())
    if slots:
        return (type(value).__qualname__, tuple(
            (name, primitive_signature(getattr(value, name))) for name in slots
        ))
    raise TypeError(f"unsupported fragment signature value: {type(value).__qualname__}")


def _direct_card_projection(value: Any) -> tuple[Any, ...]:
    """Project the fields read by ``add`` without walking grandchildren.

    Attached cards are leaf rows in the canonical schema: their nested lists
    affect only the four numeric counts and are never recursively materialized.
    The old signature nevertheless descended through every nested value.  This
    projection mirrors the compiler's real one-level dependency boundary and
    retains the compiler's presence/type states so values such as ``True`` and
    ``1`` cannot collide in a cache key.
    """

    if not isinstance(value, Mapping):
        # Preserve fail-closed handling for unknown runtime objects.  Primitive
        # scalar cards have no dynamic fields in the canonical compiler.
        primitive_signature(value)
        return (card_id(value), None, None, None, None, None, 0, 0, 0, 0)
    serial, serial_state = _raw_int(value, "serial")
    hp, hp_state = _raw_int(value, "hp")
    maximum_hp, max_hp_state = _raw_int(value, "maxHp")
    player = value.get("playerIndex")
    player_index = player if isinstance(player, int) and not isinstance(player, bool) else None
    appeared = value.get("appearThisTurn", value.get("appear"))
    appeared_present = isinstance(appeared, bool)
    return (
        card_id(value),
        serial, serial_state,
        player_index,
        hp, hp_state,
        maximum_hp, max_hp_state,
        bool(appeared) if appeared_present else False,
        appeared_present,
        len(_items(value.get("energyCards"))),
        len(_items(value.get("energies"))),
        len(_items(value.get("tools"))),
        len(_items(value.get("preEvolution"))),
    )


def _card_projection(value: Any) -> tuple[Any, ...]:
    """Return the exact bounded semantic state of one root card entity."""

    direct = _direct_card_projection(value)
    if not isinstance(value, Mapping):
        return direct
    return (
        direct,
        tuple(_direct_card_projection(item) for item in _items(value.get("energyCards"))),
        tuple(
            (item if isinstance(item, int) and not isinstance(item, bool) else None)
            for item in _items(value.get("energies"))
        ),
        tuple(_direct_card_projection(item) for item in _items(value.get("tools"))),
        tuple(_direct_card_projection(item) for item in _items(value.get("preEvolution"))),
    )


def _same_nested_primitive_types(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Type-safe equality guard used only for the less-common attached trees."""

    for name in (
        "id", "cardId", "serial", "playerIndex", "hp", "maxHp",
        "appearThisTurn", "appear",
    ):
        if type(left.get(name)) is not type(right.get(name)):
            return False
    for name in ("energyCards", "tools", "preEvolution"):
        left_value, right_value = left.get(name), right.get(name)
        left_items = left_value if isinstance(left_value, (list, tuple)) else ()
        right_items = right_value if isinstance(right_value, (list, tuple)) else ()
        if len(left_items) != len(right_items):
            return False
        for left_item, right_item in zip(left_items, right_items):
            if type(left_item) is not type(right_item):
                return False
            if isinstance(left_item, Mapping) and not _same_nested_primitive_types(
                left_item, right_item
            ):
                return False
    left_value, right_value = left.get("energies"), right.get("energies")
    left_energy = left_value if isinstance(left_value, (list, tuple)) else ()
    right_energy = right_value if isinstance(right_value, (list, tuple)) else ()
    return len(left_energy) == len(right_energy) and all(
        type(left_item) is type(right_item)
        for left_item, right_item in zip(left_energy, right_energy)
    )


@dataclass(frozen=True, slots=True)
class _CardFragment:
    cat: tuple[tuple[int, ...], ...]
    num: tuple[tuple[float, ...], ...]
    state: tuple[tuple[int, ...], ...]
    parent: tuple[int, ...]
    serial_locations: tuple[tuple[int, int], ...]
    child_locations: tuple[tuple[int, str, int, int], ...]


@dataclass(slots=True)
class _CardEntityCache:
    """One stable entity state with independently keyed placement overlays."""

    projection: tuple[Any, ...]
    raw_snapshot: Any
    identity: int
    hp_type: type
    max_hp_type: type
    appeared_type: type
    leaf: bool
    overlays: dict[tuple[int, ...], _CardFragment]


class CardFragmentCompiler:
    """Materialize card entities once and rebase their symbolic relations."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], _CardEntityCache] = {}
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._cache.clear()

    def _fragment(
        self,
        raw: Any,
        *,
        actor: int,
        owner: int,
        zone: int,
        slot: int,
        kind: int,
        status: int = 0,
        identity_knowledge: int = 1,
    ) -> _CardFragment:
        identity = card_id(raw)
        item = raw if isinstance(raw, Mapping) else {}
        raw_serial = item.get("serial")
        entity_key = (
            (kind, raw_serial)
            if isinstance(raw_serial, int) and not isinstance(raw_serial, bool)
            else None
        )
        overlay_key = (actor, owner, zone, slot, status, identity_knowledge)
        cached = self._cache.get(entity_key) if entity_key is not None else None
        appeared = item.get("appearThisTurn", item.get("appear"))
        if (
            cached is not None
            and cached.raw_snapshot is not raw
            and cached.raw_snapshot == raw
            and cached.identity == identity
            and cached.hp_type is type(item.get("hp"))
            and cached.max_hp_type is type(item.get("maxHp"))
            and cached.appeared_type is type(appeared)
            and (
                cached.leaf
                or _same_nested_primitive_types(cached.raw_snapshot, item)
            )
        ):
            fragment = cached.overlays.get(overlay_key)
            if fragment is not None:
                self.hits += 1
                return fragment
        projection = _card_projection(raw)
        if identity <= 0:
            self.misses += 1
            return _CardFragment((), (), (), (), (), ())
        if not 0 <= slot < 256:
            raise ValueError(f"card zone slot outside exact categorical range: {slot}")
        if cached is not None and cached.projection == projection:
            fragment = cached.overlays.get(overlay_key)
            if fragment is not None:
                cached.raw_snapshot = raw
                self.hits += 1
                return fragment
        self.misses += 1
        cats: list[tuple[int, ...]] = []
        nums: list[tuple[float, ...]] = []
        states: list[tuple[int, ...]] = []
        parents: list[int] = []
        serial_locations: list[tuple[int, int]] = []
        child_locations: list[tuple[int, str, int, int]] = []

        def add(
            child: Any,
            *,
            child_owner: int,
            child_zone: int,
            child_slot: int,
            child_kind: int,
            parent: int = -1,
            child_status: int = 0,
            knowledge: int = 1,
        ) -> int:
            child_identity = card_id(child)
            if child_identity <= 0:
                return -1
            child_item = child if isinstance(child, Mapping) else {}
            serial, serial_state = _raw_int(child_item, "serial")
            hp, hp_state = _raw_int(child_item, "hp")
            maximum_hp, max_hp_state = _raw_int(child_item, "maxHp")
            energy_cards = _items(child_item.get("energyCards"))
            energies = _items(child_item.get("energies"))
            tools = _items(child_item.get("tools"))
            evolution = _items(child_item.get("preEvolution"))
            appeared = child_item.get("appearThisTurn", child_item.get("appear"))
            appeared_present = isinstance(appeared, bool)
            is_pokemon = child_kind == 2
            index = len(cats)
            cats.append((
                child_identity,
                int(serial) + 1 if serial_state == int(FieldState.PRESENT) else 0,
                child_owner, child_zone, child_slot + 1, child_kind, child_status, 0, knowledge,
            ))
            nums.append((
                hp, maximum_hp, float(len(energy_cards)), float(len(energies)),
                float(len(tools)), float(len(evolution)),
                float(bool(appeared)) if appeared_present else 0.0,
            ))
            states.append((
                hp_state if is_pokemon else _state(False, False),
                max_hp_state if is_pokemon else _state(False, False),
                _state(True) if is_pokemon else _state(False, False),
                _state(True) if is_pokemon else _state(False, False),
                _state(True) if is_pokemon else _state(False, False),
                _state(True) if is_pokemon else _state(False, False),
                _state(appeared_present) if is_pokemon else _state(False, False),
            ))
            parents.append(parent)
            if serial_state == int(FieldState.PRESENT):
                serial_locations.append((int(serial), index))
            return index

        root = add(
            raw, child_owner=owner, child_zone=zone, child_slot=slot,
            child_kind=kind, child_status=status, knowledge=identity_knowledge,
        )
        if root >= 0 and kind == 2:
            player_index = integer(item.get("playerIndex"), 0 if owner == 1 else 1)
            for child_slot, child in enumerate(_items(item.get("energyCards"))):
                child_player = integer(
                    child.get("playerIndex") if isinstance(child, Mapping) else None,
                    player_index,
                )
                child_owner = _relative_owner(child_player, actor)
                prefix = "self" if child_owner == 1 else "opponent"
                child_index = add(
                    child, child_owner=child_owner, child_zone=ZONE[f"{prefix}_energy"],
                    child_slot=child_slot, child_kind=3, parent=root,
                )
                child_locations.append((root, "energy", child_slot, child_index))
            for field, suffix, child_kind in (
                ("tools", "tool", 4), ("preEvolution", "evolution", 5),
            ):
                for child_slot, child in enumerate(_items(item.get(field))):
                    child_player = integer(
                        child.get("playerIndex") if isinstance(child, Mapping) else None,
                        player_index,
                    )
                    child_owner = _relative_owner(child_player, actor)
                    prefix = "self" if child_owner == 1 else "opponent"
                    child_index = add(
                        child, child_owner=child_owner,
                        child_zone=ZONE[f"{prefix}_{suffix}"], child_slot=child_slot,
                        child_kind=child_kind, parent=root,
                    )
                    child_locations.append((root, field, child_slot, child_index))
            for unit_slot, energy_type in enumerate(_items(item.get("energies"))):
                if not 0 <= integer(energy_type, -1) < 16:
                    raise ValueError(f"resolved EnergyTypeIndex outside audited range: {energy_type!r}")
                if not 0 <= unit_slot < 256:
                    raise ValueError(
                        f"resolved Energy slot outside exact categorical range: {unit_slot}"
                    )
                cats.append((
                    0, 0, owner,
                    ZONE["self_resolved_energy" if owner == 1 else "opponent_resolved_energy"],
                    unit_slot + 1, 9, 0, integer(energy_type) + 1, 4,
                ))
                nums.append((0.0,) * 7)
                states.append((int(FieldState.NOT_APPLICABLE),) * 7)
                parents.append(root)
        fragment = _CardFragment(
            tuple(cats), tuple(nums), tuple(states), tuple(parents),
            tuple(serial_locations), tuple(child_locations),
        )
        if entity_key is not None:
            if cached is None or cached.projection != projection:
                cached = _CardEntityCache(
                    projection,
                    raw,
                    identity,
                    type(item.get("hp")),
                    type(item.get("maxHp")),
                    type(appeared),
                    not any(_items(item.get(name)) for name in (
                        "energyCards", "energies", "tools", "preEvolution",
                    )),
                    {},
                )
                self._cache[entity_key] = cached
            cached.overlays[overlay_key] = fragment
        return fragment

    def compile(self, row: Mapping[str, Any], snapshot: CausalSnapshot) -> CardLayer:
        current, select, actor, opponent, _own, _other, _options, _minimum, _maximum = _context(row, snapshot)
        cats: list[tuple[int, ...]] = []
        nums: list[tuple[float, ...]] = []
        states: list[tuple[int, ...]] = []
        parents: list[int] = []
        locations: dict[tuple[int, int, int], int] = {}
        serial_locations: dict[int, int] = {}
        child_locations: dict[tuple[int, str, int], int] = {}

        def append(fragment: _CardFragment, key: tuple[int, int, int] | None = None) -> int:
            offset = len(cats)
            cats.extend(fragment.cat); nums.extend(fragment.num); states.extend(fragment.state)
            parents.extend(parent + offset if parent >= 0 else -1 for parent in fragment.parent)
            for serial, index in fragment.serial_locations:
                if serial in serial_locations:
                    raise ValueError(f"duplicate card serial in canonical observation: {serial}")
                serial_locations[serial] = index + offset
            for parent, field, slot, index in fragment.child_locations:
                child_locations[(parent + offset, field, slot)] = (
                    index + offset if index >= 0 else -1
                )
            if key is not None and fragment.cat:
                locations[key] = offset
            return offset if fragment.cat else -1

        def player_zone(player_index: int, name: str, area: int) -> None:
            player = current["players"][player_index]
            relative = _relative_owner(player_index, actor)
            prefix = "self" if relative == 1 else "opponent"
            for slot, raw in enumerate(_items(player.get(name))):
                append(self._fragment(
                    raw, actor=actor, owner=relative,
                    zone=ZONE[f"{prefix}_{name}"], slot=slot,
                    kind=2 if name in {"active", "bench"} else 1,
                    status=_status_bits(player) if name == "active" else 0,
                ), (player_index, area, slot))

        for player_index in (actor, opponent):
            player_zone(player_index, "active", 4)
            player_zone(player_index, "bench", 5)
            if player_index == actor:
                player_zone(player_index, "hand", 2)
            player_zone(player_index, "discard", 3)
        for slot, raw in enumerate(_items(current.get("stadium"))):
            item = raw if isinstance(raw, Mapping) else {}
            append(self._fragment(
                raw, actor=actor,
                owner=_relative_owner(integer(item.get("playerIndex"), -1), actor),
                zone=ZONE["stadium"], slot=slot, kind=6,
            ), (-1, 7, slot))
        for slot, raw in enumerate(_items(current.get("looking"))):
            append(self._fragment(raw, actor=actor, owner=1, zone=ZONE["looking"], slot=slot, kind=7), (actor, 12, slot))
        deck_items = _items(select.get("deck"))
        for slot, raw in enumerate(deck_items):
            append(self._fragment(raw, actor=actor, owner=1, zone=ZONE["select_deck"], slot=slot, kind=1), (actor, 1, slot))
        if not deck_items and snapshot.deck_order_known:
            for slot, known in enumerate(snapshot.known_self_deck_order):
                if known.serial not in serial_locations:
                    append(self._fragment(
                        {"id": known.card_id, "serial": known.serial}, actor=actor, owner=1,
                        zone=ZONE["known_self_deck_order"], slot=slot, kind=1,
                        identity_knowledge=2,
                    ))
        select_serials = {
            integer(value.get("serial"), -1) for value in
            (select.get("contextCard"), select.get("effect")) if isinstance(value, Mapping)
        }
        for slot, known in enumerate(snapshot.known_opponent_hand):
            if known.serial not in serial_locations:
                append(self._fragment(
                    {"id": known.card_id, "serial": known.serial}, actor=actor, owner=2,
                    zone=ZONE["known_opponent_hand"], slot=slot, kind=8,
                    identity_knowledge=2,
                ))
        for slot, known in enumerate(snapshot.possible_opponent_hand):
            if known.serial not in serial_locations:
                append(self._fragment(
                    {"id": known.card_id, "serial": known.serial}, actor=actor, owner=2,
                    zone=ZONE["known_opponent_hand"], slot=slot, kind=8,
                    identity_knowledge=3,
                ))
        for slot, known in enumerate(snapshot.remembered_opponent_cards):
            if known.serial not in serial_locations and known.serial not in select_serials:
                append(self._fragment(
                    {"id": known.card_id, "serial": known.serial}, actor=actor, owner=2,
                    zone=ZONE["remembered_opponent_hidden"], slot=slot, kind=10,
                    identity_knowledge=2,
                ))

        def ensure_select_card(raw: Any) -> int:
            if not isinstance(raw, Mapping) or card_id(raw) <= 0:
                return -1
            serial = integer(raw.get("serial"), -1)
            if serial in serial_locations:
                return serial_locations[serial]
            player_index = integer(raw.get("playerIndex"), actor)
            relative = _relative_owner(player_index, actor)
            return append(self._fragment(
                raw, actor=actor, owner=relative,
                zone=ZONE["self_playing" if relative == 1 else "opponent_playing"],
                slot=0, kind=7,
            ))

        context_index = ensure_select_card(select.get("contextCard"))
        effect_card_index = ensure_select_card(select.get("effect"))
        return CardLayer(
            tuple(cats), tuple(nums), tuple(states), tuple(parents),
            MappingProxyType(locations), MappingProxyType(serial_locations),
            MappingProxyType(child_locations), context_index, effect_card_index,
        )


@dataclass(frozen=True, slots=True)
class _EventFragment:
    cat: tuple[int, ...]
    numeric_tail: tuple[float, ...]
    state: tuple[int, ...]
    source_serial: int
    target_serial: int
    before_serial: int
    after_serial: int


class EventFragmentCompiler:
    """Cache immutable event rows; update only age and card-index relations."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], tuple[TypedEvent, _EventFragment]] = {}
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._cache.clear()

    def _fragment(self, event: TypedEvent, actor: int) -> _EventFragment:
        key = (actor, event.source_event)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is event:
            self.hits += 1
            return cached[1]
        if cached is not None:
            # CausalKnowledge owns immutable TypedEvent objects for the session;
            # a new object under an old source id violates append-only identity.
            raise ValueError("event source identity was reused")
        self.misses += 1
        p = event.payload
        relative = 3 if event.actor is None else _relative_owner(event.actor, actor)
        target_id = max(0, integer(p.get("cardIdTarget")))
        source_serial = integer(p.get("serial"), -1)
        target_serial = integer(p.get("serialTarget"), -1)
        if event.name == "switch":
            source_serial = integer(p.get("serialActive"), -1)
            target_serial = integer(p.get("serialBench"), -1)
        numeric = tuple(_raw_int(p, name) for name in ("value", "count", "number"))
        fragment = _EventFragment(
            (event.log_type + 1, relative, int(event.card_id or 0), target_id,
             max(0, integer(p.get("attackId"))), int(event.from_area if event.from_area is not None else -1) + 1,
             int(event.to_area if event.to_area is not None else -1) + 1, int(event.identity_visible) + 1,
             int(isinstance(p.get("serial"), int)) + 1,
             int(isinstance(p.get("serialTarget"), int) or target_id > 0) + 1,
             max(0, integer(p.get("cardIdActive"))), max(0, integer(p.get("cardIdBench"))),
             max(0, integer(p.get("cardIdBefore"))), max(0, integer(p.get("cardIdAfter"))),
             _categorical_int(p, "serial"), _categorical_int(p, "serialTarget"),
             _categorical_int(p, "serialActive"), _categorical_int(p, "serialBench"),
             _categorical_int(p, "serialBefore"), _categorical_int(p, "serialAfter"),
             _categorical_bool(p, "isRecover"), _categorical_bool(p, "hasBasicPokemon"),
             _categorical_bool(p, "head"), _categorical_bool(p, "putDamageCounter"),
             _categorical_int(p, "result"), _categorical_int(p, "reason"),
             _categorical_int(p, "index"), _categorical_int(p, "energyIndex"),
             _categorical_int(p, "toolIndex"), _categorical_int(p, "inPlayArea"),
             _categorical_int(p, "inPlayIndex")),
            tuple(value for value, _ in numeric),
            (int(FieldState.PRESENT), *(state for _, state in numeric)),
            source_serial, target_serial, integer(p.get("serialBefore"), -1),
            integer(p.get("serialAfter"), -1),
        )
        self._cache[key] = (event, fragment)
        return fragment

    def compile(self, row: Mapping[str, Any], snapshot: CausalSnapshot, cards: CardLayer) -> EventLayer:
        _current, _select, actor, _opponent, _own, _other, _options, _minimum, _maximum = _context(row, snapshot)
        newest = snapshot.recent_events[-1].source_event if snapshot.recent_events else -1
        cats: list[tuple[int, ...]] = []
        nums: list[tuple[float, ...]] = []
        states: list[tuple[int, ...]] = []
        source: list[int] = []; target: list[int] = []; before: list[int] = []; after: list[int] = []
        for event in snapshot.recent_events:
            fragment = self._fragment(event, actor)
            cats.append(fragment.cat)
            nums.append((float(newest - event.source_event), *fragment.numeric_tail))
            states.append(fragment.state)
            source.append(cards.serial_locations.get(fragment.source_serial, -1))
            target.append(cards.serial_locations.get(fragment.target_serial, -1))
            before.append(cards.serial_locations.get(fragment.before_serial, -1))
            after.append(cards.serial_locations.get(fragment.after_serial, -1))
        layer = EventLayer(
            tuple(cats), tuple(nums), tuple(states), tuple(source), tuple(target),
            tuple(before), tuple(after),
        )
        # Keep one recent-window worth of payloads without paying an eviction
        # scan on every action.  source_event is monotonic within a session.
        if len(self._cache) > 128:
            active = {(actor, event.source_event) for event in snapshot.recent_events}
            self._cache = {
                key: value for key, value in self._cache.items() if key in active
            }
        return layer


__all__ = ["CardFragmentCompiler", "EventFragmentCompiler", "primitive_signature"]
