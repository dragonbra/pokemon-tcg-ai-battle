"""Action-lifetime resource and legal-option fragment compilers.

The stateless functions in :mod:`.layers` remain the semantic authority.  The
classes here preserve their exact immutable layer contracts while avoiding two
forms of repeated work:

* resource identity/initial-count rows live for a battle, while unchanged
  visible/bounded-count fields are reused and only their ages are patched;
* option prototype expansions depend on semantic identities, not on the
  current card indices or option ordinal, so those expansions are cached while
  all observation-relative relations are rebuilt on every decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.prototypes import FieldState, PrototypeIndex
from ..knowledge.state import CausalSnapshot
from .layers import (
    CardLayer,
    OptionLayer,
    ResourceLayer,
    _context,
    _knowledge_code,
    _raw_int,
    _relative_owner,
    _state,
)
from .relations import card_id, integer


_VISIBLE_ZONES = ("active", "bench", "hand", "discard", "stadium", "playing")


@dataclass(frozen=True, slots=True)
class _ResourceCore:
    cat: tuple[int, ...]
    num: tuple[float, ...]
    state: tuple[int, ...]


class ResourceFragmentCompiler:
    """Compile exact resource layers with battle-static and age-patch reuse."""

    def __init__(self) -> None:
        self._layout: tuple[tuple[int, int], ...] | None = None
        self._cores: dict[int, tuple[tuple[Any, ...], _ResourceCore]] = {}
        self.compiles = 0
        self.battle_starts = 0
        self.layout_hits = 0
        self.layout_misses = 0
        self.core_hits = 0
        self.core_misses = 0

    def clear(self) -> None:
        """Forget battle-owned fragments without resetting diagnostic totals."""

        self._layout = None
        self._cores.clear()

    def snapshot(self) -> dict[str, int]:
        return {
            "compiles": self.compiles,
            "battle_starts": self.battle_starts,
            "layout_hits": self.layout_hits,
            "layout_misses": self.layout_misses,
            "core_hits": self.core_hits,
            "core_misses": self.core_misses,
        }

    def compile(
        self, row: Mapping[str, Any], snapshot: CausalSnapshot
    ) -> ResourceLayer:
        # Preserve the authoritative compiler's validation boundary even though
        # resource materialization itself consumes only the causal ledger.
        _context(row, snapshot)
        self.compiles += 1
        entries = tuple(sorted(snapshot.self_ledger.items()))
        layout = tuple((identity, int(entry.initial)) for identity, entry in entries)
        if layout != self._layout:
            self.layout_misses += 1
            self.battle_starts += 1
            self._layout = layout
            self._cores.clear()
        else:
            self.layout_hits += 1

        cats: list[tuple[int, ...]] = []
        nums: list[tuple[float, ...]] = []
        states: list[tuple[int, ...]] = []
        deck_order_code = int(snapshot.deck_order_known) + 1
        for identity, entry in entries:
            visible = entry.visible
            # Ages deliberately do not participate.  They advance at action
            # lifetime and occupy only the final two numeric columns.
            signature = (
                entry.initial,
                *(visible.get(name, 0) for name in _VISIBLE_ZONES),
                entry.deck.value,
                entry.deck.lower,
                entry.deck.upper,
                entry.deck.state,
                entry.prize.value,
                entry.prize.lower,
                entry.prize.upper,
                entry.prize.state,
                deck_order_code,
            )
            cached = self._cores.get(identity)
            if cached is not None and cached[0] == signature:
                self.core_hits += 1
                core = cached[1]
            else:
                self.core_misses += 1
                core = _ResourceCore(
                    cat=(
                        identity,
                        _knowledge_code(entry.deck.state),
                        _knowledge_code(entry.prize.state),
                        deck_order_code,
                    ),
                    num=(
                        float(entry.initial),
                        *(float(visible.get(name, 0)) for name in _VISIBLE_ZONES),
                        float(entry.deck.value or 0),
                        float(entry.deck.lower),
                        float(entry.deck.upper),
                        float(entry.prize.value or 0),
                        float(entry.prize.lower),
                        float(entry.prize.upper),
                    ),
                    state=(
                        int(FieldState.PRESENT),
                        *([int(FieldState.PRESENT)] * 6),
                        _state(entry.deck.value is not None),
                        int(FieldState.PRESENT),
                        int(FieldState.PRESENT),
                        _state(entry.prize.value is not None),
                        int(FieldState.PRESENT),
                        int(FieldState.PRESENT),
                        int(FieldState.PRESENT),
                        int(FieldState.PRESENT),
                    ),
                )
                self._cores[identity] = (signature, core)
            cats.append(core.cat)
            nums.append(core.num + (float(entry.deck.age), float(entry.prize.age)))
            states.append(core.state)
        return ResourceLayer(tuple(cats), tuple(nums), tuple(states))


@dataclass(frozen=True, slots=True)
class _OptionSemantics:
    skills: tuple[tuple[int, int], ...]
    effects: tuple[tuple[int, int], ...]


class OptionFragmentCompiler:
    """Cache prototype semantics and rebuild action-local option relations."""

    def __init__(self) -> None:
        self._prototype_owner: PrototypeIndex | None = None
        self._semantics: dict[tuple[int, int, int, int], _OptionSemantics] = {}
        self.compiles = 0
        self.semantic_hits = 0
        self.semantic_misses = 0
        self.prototype_resets = 0

    def clear(self) -> None:
        self._prototype_owner = None
        self._semantics.clear()

    def snapshot(self) -> dict[str, int]:
        return {
            "compiles": self.compiles,
            "semantic_hits": self.semantic_hits,
            "semantic_misses": self.semantic_misses,
            "prototype_resets": self.prototype_resets,
            "semantic_entries": len(self._semantics),
        }

    def _semantic_expansion(
        self,
        prototypes: PrototypeIndex,
        source_identity: int,
        context_card: int,
        effect_card: int,
        attack_id: int,
    ) -> _OptionSemantics:
        key = (source_identity, context_card, effect_card, attack_id)
        cached = self._semantics.get(key)
        if cached is not None:
            self.semantic_hits += 1
            return cached
        self.semantic_misses += 1

        candidates: list[tuple[int, int]] = []
        for relation_role, identity in (
            (1, source_identity), (2, context_card), (3, effect_card)
        ):
            card = prototypes.engine_cards.get(identity, {})
            for role, name in enumerate(
                ("ability_skill_id", "play_skill_id", "delay_skill_id"), 1
            ):
                skill_id = integer(card.get(name))
                if skill_id > 0:
                    candidates.append((skill_id, (relation_role - 1) * 3 + role))
        skills: list[tuple[int, int]] = []
        effects: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for skill_id, role in candidates:
            if (skill_id, role) in seen:
                continue
            seen.add((skill_id, role))
            skills.append((skill_id, role))
            effects.extend(
                (effect_ref, 1)
                for effect_ref in prototypes.skill_effect_refs.get(skill_id, ())
            )
        effects.extend(
            (effect_ref, 2)
            for effect_ref in prototypes.attack_effect_refs.get(attack_id, ())
        )
        result = _OptionSemantics(tuple(skills), tuple(effects))
        self._semantics[key] = result
        return result

    def compile(
        self,
        row: Mapping[str, Any],
        snapshot: CausalSnapshot,
        prototypes: PrototypeIndex,
        cards: CardLayer,
    ) -> OptionLayer:
        if prototypes is not self._prototype_owner:
            self._prototype_owner = prototypes
            self._semantics.clear()
            self.prototype_resets += 1
        self.compiles += 1
        _current, select, actor, _opponent, _own, _other, options, minimum, maximum = (
            _context(row, snapshot)
        )
        context_card = card_id(select.get("contextCard"))
        effect_card = card_id(select.get("effect"))
        cats: list[tuple[int, ...]] = []
        nums: list[tuple[float, ...]] = []
        states: list[tuple[int, ...]] = []
        source: list[int] = []
        target: list[int] = []
        contexts: list[int] = []
        effect_cards: list[int] = []
        skill_ids: list[int] = []
        skill_roles: list[int] = []
        skill_parents: list[int] = []
        effect_ids: list[int] = []
        effect_roles: list[int] = []
        effect_parents: list[int] = []

        for option_index, value in enumerate(options):
            option = value if isinstance(value, Mapping) else {}
            action_type = integer(option.get("type"), -1)
            source_player = integer(option.get("playerIndex"), actor)
            source_area = integer(option.get("area"), -1)
            if action_type == 7 and source_area < 0:
                source_area = 2
            source_slot = integer(option.get("index"), -1)
            source_index = cards.locations.get(
                (source_player, source_area, source_slot), -1
            )
            if source_area == 7:
                source_index = cards.locations.get((-1, 7, source_slot), source_index)
            if action_type == 15:
                source_index = cards.serial_locations.get(
                    integer(option.get("serial"), -1), source_index
                )
            parent_source = source_index
            energy_index = integer(option.get("energyIndex"), -1)
            tool_index = integer(option.get("toolIndex"), -1)
            if energy_index >= 0 and parent_source >= 0:
                source_index = cards.child_locations.get(
                    (parent_source, "energy", energy_index), source_index
                )
            elif tool_index >= 0 and parent_source >= 0:
                source_index = cards.child_locations.get(
                    (parent_source, "tools", tool_index), source_index
                )
            source_identity = card_id(option.get("cardId"))
            if source_index >= 0:
                source_identity = cards.cat[source_index][0]
            source_owner = (
                cards.cat[source_index][2]
                if source_index >= 0
                else _relative_owner(source_player, actor)
            )
            target_player = integer(
                option.get("inPlayPlayerIndex", option.get("targetPlayerIndex")), -1
            )
            if target_player < 0 and option.get("inPlayArea") is not None:
                target_player = actor
            target_area = integer(option.get("inPlayArea"), -1)
            target_slot = integer(option.get("inPlayIndex"), -1)
            target_index = cards.locations.get(
                (target_player, target_area, target_slot), -1
            )
            target_identity = cards.cat[target_index][0] if target_index >= 0 else 0
            target_owner = cards.cat[target_index][2] if target_index >= 0 else 3
            attack_id = max(0, integer(option.get("attackId")))
            number = _raw_int(option, "number")
            count = _raw_int(option, "count")
            cats.append((
                action_type + 1, source_owner, source_area + 1,
                target_owner, target_area + 1, source_identity, target_identity,
                attack_id, integer(option.get("specialConditionType"), -1) + 1,
                integer(select.get("type"), -1) + 1,
                integer(select.get("context"), -1) + 1,
                context_card, effect_card, option_index + 1, source_slot + 1,
                target_slot + 1, energy_index + 1, tool_index + 1,
                integer(option.get("serial"), -1) + 1,
            ))
            nums.append((number[0], count[0]))
            states.append((number[1], count[1]))
            source.append(source_index)
            target.append(target_index)
            contexts.append(cards.context_index)
            effect_cards.append(cards.effect_card_index)

            semantics = self._semantic_expansion(
                prototypes, source_identity, context_card, effect_card, attack_id
            )
            for skill_id, role in semantics.skills:
                skill_ids.append(skill_id)
                skill_roles.append(role)
                skill_parents.append(option_index)
            for effect_id, role in semantics.effects:
                effect_ids.append(effect_id)
                effect_roles.append(role)
                effect_parents.append(option_index)

        return OptionLayer(
            tuple(cats), tuple(nums), tuple(states), tuple(source), tuple(target),
            tuple(contexts), tuple(effect_cards), tuple(skill_ids), tuple(skill_roles),
            tuple(skill_parents), tuple(effect_ids), tuple(effect_roles),
            tuple(effect_parents), minimum, maximum,
        )


__all__ = ["OptionFragmentCompiler", "ResourceFragmentCompiler"]
