"""Alias-safe, lifetime-aware materialization of canonical Python records.

The canonical layers are immutable tuples.  Model collation, however, consumes the
historical dict/list record contract.  This component keeps private list templates
for layer fields whose tuple objects survive across decisions and copies those
templates into every returned record.  Consequently a caller may freely mutate a
record without corrupting the assembler or a later decision.

This module is deliberately independent of the active compiler path.  Admission is
handled separately after chronological parity and performance gates pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Callable

from ..contracts.fields import ACTOR_KEYS, SCHEMA_VERSION
from .layers import CardLayer, EventLayer, GlobalLayer, OptionLayer, ResourceLayer
from .relations import integer


@dataclass(slots=True)
class AssemblyStats:
    calls: int = 0
    elapsed_ns: int = 0
    materialization_cache_hits: int = 0
    materialization_cache_misses: int = 0
    deck_cache_hits: int = 0
    deck_cache_misses: int = 0

    def snapshot(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "elapsed_ns": self.elapsed_ns,
            "mean_elapsed_ns": self.elapsed_ns / self.calls if self.calls else 0.0,
            "materialization_cache_hits": self.materialization_cache_hits,
            "materialization_cache_misses": self.materialization_cache_misses,
            "deck_cache_hits": self.deck_cache_hits,
            "deck_cache_misses": self.deck_cache_misses,
        }


@dataclass(slots=True)
class _Materialization:
    source: object
    template: dict[str, Any]


class IncrementalRecordAssembler:
    """Materialize exact canonical records while reusing private conversion work.

    An instance is intended to live for one battle.  A changed registered-deck
    manifest is accepted safely and replaces the battle-level deck cache; callers
    may also call :meth:`reset_battle` explicitly at a session boundary.
    """

    def __init__(self) -> None:
        self.stats = AssemblyStats()
        self._materializations: dict[str, _Materialization] = {}
        self._deck_counts_source: object | None = None
        self._deck_signature: tuple[tuple[int, int], ...] | None = None
        self._deck_template: tuple[int, ...] | None = None

    def reset_battle(self) -> None:
        self._materializations.clear()
        self._deck_counts_source = None
        self._deck_signature = None
        self._deck_template = None

    def _layer_template(
        self, name: str, source: object, build: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        cached = self._materializations.get(name)
        if cached is not None and cached.source is source:
            self.stats.materialization_cache_hits += 1
            return cached.template
        self.stats.materialization_cache_misses += 1
        template = build()
        self._materializations[name] = _Materialization(source, template)
        return template

    @staticmethod
    def _copy_rows(rows: list[list[Any]]) -> list[list[Any]]:
        return [row.copy() for row in rows]

    def _registered_deck(self, row: Mapping[str, Any]) -> list[int]:
        manifest = row.get("deck_manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError("canonical row has no deck manifest")
        counts = manifest.get("counts")
        if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)):
            raise ValueError("canonical deck manifest counts must be a sequence")
        if counts is self._deck_counts_source and self._deck_template is not None:
            self.stats.deck_cache_hits += 1
            return list(self._deck_template)
        signature_items: list[tuple[int, int]] = []
        for item in counts:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ValueError("canonical deck manifest count entry must be a pair")
            identity, count = integer(item[0], -1), integer(item[1], -1)
            if identity <= 0 or count <= 0:
                raise ValueError("canonical deck manifest identities/counts must be positive")
            signature_items.append((identity, count))
        signature = tuple(signature_items)
        if signature == self._deck_signature and self._deck_template is not None:
            self.stats.deck_cache_hits += 1
            self._deck_counts_source = counts
            return list(self._deck_template)
        deck = tuple(identity for identity, count in signature for _ in range(count))
        if len(deck) != 60:
            raise ValueError("canonical deck manifest must contain exactly 60 cards")
        self.stats.deck_cache_misses += 1
        self._deck_counts_source = counts
        self._deck_signature = signature
        self._deck_template = deck
        return list(deck)

    def assemble(
        self,
        row: Mapping[str, Any],
        cards: CardLayer,
        resources: ResourceLayer,
        events: EventLayer,
        options: OptionLayer,
        globals: GlobalLayer,
    ) -> dict[str, Any]:
        started = perf_counter_ns()
        try:
            raw_action = row.get("ordered_action")
            values = list(raw_action) if isinstance(raw_action, (list, tuple)) else []
            action = [integer(value, -1) for value in values]
            if len(set(action)) != len(action) or any(
                value < 0 or value >= len(options.cat) for value in action
            ):
                raise ValueError("canonical target is not a unique legal option sequence")
            if not options.min_count <= len(action) <= options.max_count:
                raise ValueError("canonical target violates selection bounds")

            global_template = self._layer_template("globals", globals, lambda: {
                "global_cat": list(globals.cat),
                "global_num": list(globals.num),
                "global_state": list(globals.state),
            })
            card_template = self._layer_template("cards", cards, lambda: {
                "card_cat": [list(item) for item in cards.cat],
                "card_num": [list(item) for item in cards.num],
                "card_state": [list(item) for item in cards.state],
                "card_parent": [value + 1 for value in cards.parent],
            })
            resource_template = self._layer_template("resources", resources, lambda: {
                "resource_cat": [list(item) for item in resources.cat],
                "resource_num": [list(item) for item in resources.num],
                "resource_state": [list(item) for item in resources.state],
            })
            event_template = self._layer_template("events", events, lambda: {
                "event_cat": [list(item) for item in events.cat],
                "event_num": [list(item) for item in events.num],
                "event_state": [list(item) for item in events.state],
                "event_source": [value + 1 for value in events.source],
                "event_target": [value + 1 for value in events.target],
                "event_before": [value + 1 for value in events.before],
                "event_after": [value + 1 for value in events.after],
            })
            option_template = self._layer_template("options", options, lambda: {
                "option_cat": [list(item) for item in options.cat],
                "option_num": [list(item) for item in options.num],
                "option_state": [list(item) for item in options.state],
                "option_source": [value + 1 for value in options.source],
                "option_target": [value + 1 for value in options.target],
                "option_context": [value + 1 for value in options.context],
                "option_effect_card": [value + 1 for value in options.effect_card],
                "option_skill_id": list(options.skill_id),
                "option_skill_role": list(options.skill_role),
                "option_skill_parent": [value + 1 for value in options.skill_parent],
                "option_effect_id": list(options.effect_id),
                "option_effect_role": list(options.effect_role),
                "option_effect_parent": [value + 1 for value in options.effect_parent],
                "min_count": options.min_count,
                "max_count": options.max_count,
            })
            actor_record = {
                "global_cat": global_template["global_cat"].copy(),
                "global_num": global_template["global_num"].copy(),
                "global_state": global_template["global_state"].copy(),
                "card_cat": self._copy_rows(card_template["card_cat"]),
                "card_num": self._copy_rows(card_template["card_num"]),
                "card_state": self._copy_rows(card_template["card_state"]),
                "card_parent": card_template["card_parent"].copy(),
                "resource_cat": self._copy_rows(resource_template["resource_cat"]),
                "resource_num": self._copy_rows(resource_template["resource_num"]),
                "resource_state": self._copy_rows(resource_template["resource_state"]),
                "event_cat": self._copy_rows(event_template["event_cat"]),
                "event_num": self._copy_rows(event_template["event_num"]),
                "event_state": self._copy_rows(event_template["event_state"]),
                "event_source": event_template["event_source"].copy(),
                "event_target": event_template["event_target"].copy(),
                "event_before": event_template["event_before"].copy(),
                "event_after": event_template["event_after"].copy(),
                "option_cat": self._copy_rows(option_template["option_cat"]),
                "option_num": self._copy_rows(option_template["option_num"]),
                "option_state": self._copy_rows(option_template["option_state"]),
                "option_source": option_template["option_source"].copy(),
                "option_target": option_template["option_target"].copy(),
                "option_context": option_template["option_context"].copy(),
                "option_effect_card": option_template["option_effect_card"].copy(),
                "option_skill_id": option_template["option_skill_id"].copy(),
                "option_skill_role": option_template["option_skill_role"].copy(),
                "option_skill_parent": option_template["option_skill_parent"].copy(),
                "option_effect_id": option_template["option_effect_id"].copy(),
                "option_effect_role": option_template["option_effect_role"].copy(),
                "option_effect_parent": option_template["option_effect_parent"].copy(),
                "min_count": option_template["min_count"],
                "max_count": option_template["max_count"],
            }
            if set(actor_record) != ACTOR_KEYS:
                raise AssertionError(
                    f"canonical actor contract drift: {sorted(set(actor_record) ^ ACTOR_KEYS)}"
                )
            return {
                "schema_version": SCHEMA_VERSION,
                "actor": actor_record,
                "target": {
                    "ordered_action": action,
                    "termination": deepcopy(row.get("action_termination")),
                },
                "audit": {
                    "identity": deepcopy(row.get("identity")),
                    "split": deepcopy(row.get("split")),
                },
                "registered_deck": self._registered_deck(row),
            }
        finally:
            self.stats.calls += 1
            self.stats.elapsed_ns += perf_counter_ns() - started


__all__ = ["AssemblyStats", "IncrementalRecordAssembler"]
