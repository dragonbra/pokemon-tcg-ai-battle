"""Ordered full-action validation and explicit, collision-safe option alignment.

Occurrence identities are alignment metadata created from the original option list
exactly once. They are deliberately not exported by :mod:`features` and must not
become model features.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


SUPPORTED_OPTION_FIELDS = frozenset(
    {
        "type",
        "playerIndex",
        "area",
        "index",
        "cardId",
        "inPlayPlayerIndex",
        "targetPlayerIndex",
        "inPlayArea",
        "inPlayIndex",
        "number",
        "attackId",
        "count",
        "energyIndex",
        "toolIndex",
        "specialConditionType",
    }
)
# Observation metadata may align records but must never influence identity or features.
NON_SEMANTIC_METADATA_FIELDS = frozenset({"serial"})
_SEMANTIC_FIELDS = (
    "type",
    "playerIndex",
    "area",
    "index",
    "cardId",
    "inPlayArea",
    "inPlayIndex",
    "number",
    "attackId",
    "count",
    "energyIndex",
    "toolIndex",
)


def _exact_int(value: Any, field: str, *, allow_null: bool = True) -> int | None:
    if value is None and allow_null:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an exact integer or null")
    return value


def _count(value: Any, field: str) -> int:
    result = _exact_int(value, field, allow_null=False)
    assert result is not None
    return result


def unrecognized_option_fields(options: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return unknown option-field occurrence counts for the dataset schema gate.

    Call this across the real replay dataset before schema freeze. This contract
    intentionally keeps previously unseen fields inert per option; Task 11 must
    decide whether the corpus-level audit permits or rejects them.
    """
    counts: Counter[str] = Counter()
    known = SUPPORTED_OPTION_FIELDS | NON_SEMANTIC_METADATA_FIELDS
    for option_index, option in enumerate(options):
        if not isinstance(option, Mapping):
            raise ValueError("option must be a mapping")
        for key in option:
            if not isinstance(key, str):
                raise ValueError(f"option {option_index} has non-string key {key!r}")
            if key not in known:
                counts[key] += 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True, slots=True)
class OptionSemanticIdentity:
    """Actor-visible option content, excluding option-list position and occurrence."""

    type: int | None
    player_index: int | None
    area: int | None
    index: int | None
    card_id: int | None
    target_player_index: int | None
    in_play_area: int | None
    in_play_index: int | None
    number: int | None
    attack_id: int | None
    count: int | None
    energy_index: int | None
    tool_index: int | None

    @classmethod
    def from_option(cls, option: Mapping[str, Any]) -> OptionSemanticIdentity:
        if not isinstance(option, Mapping):
            raise ValueError("option must be a mapping")
        # Unknown fields are intentionally inert here. The dataset gate must inspect
        # unrecognized_option_fields() across the full corpus before freezing schema.
        canonical_target = _exact_int(option.get("inPlayPlayerIndex"), "inPlayPlayerIndex")
        alias_target = _exact_int(option.get("targetPlayerIndex"), "targetPlayerIndex")
        if canonical_target is not None and alias_target is not None and canonical_target != alias_target:
            raise ValueError("contradictory target player aliases")
        target_player = canonical_target if canonical_target is not None else alias_target
        values = {field: _exact_int(option.get(field), field) for field in _SEMANTIC_FIELDS}
        return cls(
            type=values["type"],
            player_index=values["playerIndex"],
            area=values["area"],
            index=values["index"],
            card_id=values["cardId"],
            target_player_index=target_player,
            in_play_area=values["inPlayArea"],
            in_play_index=values["inPlayIndex"],
            number=values["number"],
            attack_id=values["attackId"],
            count=values["count"],
            energy_index=values["energyIndex"],
            tool_index=values["toolIndex"],
        )


@dataclass(frozen=True, slots=True)
class _OptionOccurrenceIdentity:
    """Non-feature alignment identity assigned only in original observation order."""

    semantic: OptionSemanticIdentity
    occurrence: int


class ActionTermination(str, Enum):
    OPTIONAL_STOP = "optional_stop"
    FORCED_MAX = "forced_max"


@dataclass(frozen=True, slots=True)
class OrderedAction:
    indices: tuple[int, ...]
    termination: ActionTermination

    @property
    def has_explicit_stop(self) -> bool:
        return self.termination is ActionTermination.OPTIONAL_STOP


def validate_ordered_action(
    indices: Sequence[int],
    *,
    option_count: int,
    min_count: int,
    max_count: int,
    decoder_capacity: int,
) -> OrderedAction:
    """Validate an order-preserving action without sorting or deduplication."""
    option_count = _count(option_count, "option_count")
    min_count = _count(min_count, "min_count")
    max_count = _count(max_count, "max_count")
    decoder_capacity = _count(decoder_capacity, "decoder_capacity")
    if option_count < 0 or min_count < 0 or max_count < min_count or decoder_capacity < 0:
        raise ValueError("invalid non-negative action count bounds")
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise ValueError("action indices must be a sequence")
    ordered = tuple(indices)
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in ordered):
        raise ValueError("action indices must be integers")
    if len(ordered) > decoder_capacity:
        raise ValueError("action exceeds decoder capacity")
    if len(ordered) < min_count:
        raise ValueError("action is shorter than minimum count")
    if len(ordered) > max_count:
        raise ValueError("action exceeds maximum count")
    if len(set(ordered)) != len(ordered):
        raise ValueError("action indices must be distinct")
    if any(index < 0 or index >= option_count for index in ordered):
        raise ValueError("action index out of range")
    termination = ActionTermination.FORCED_MAX if len(ordered) == max_count else ActionTermination.OPTIONAL_STOP
    return OrderedAction(indices=ordered, termination=termination)


def build_occurrence_identities(
    original_options: Sequence[Mapping[str, Any]],
) -> tuple[_OptionOccurrenceIdentity, ...]:
    """Create alignment-only occurrence identities in original observation order."""
    occurrences: dict[OptionSemanticIdentity, int] = {}
    result: list[_OptionOccurrenceIdentity] = []
    for option in original_options:
        semantic = OptionSemanticIdentity.from_option(option)
        occurrence = occurrences.get(semantic, 0)
        result.append(_OptionOccurrenceIdentity(semantic, occurrence))
        occurrences[semantic] = occurrence + 1
    return tuple(result)


def _validate_permutation(old_to_new: Sequence[int], size: int) -> tuple[int, ...]:
    if not isinstance(old_to_new, Sequence) or isinstance(old_to_new, (str, bytes)):
        raise ValueError("permutation must be a sequence")
    permutation = tuple(old_to_new)
    if len(permutation) != size:
        raise ValueError("permutation must be a bijection of occurrence identities")
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in permutation):
        raise ValueError("permutation must contain exact integers")
    if set(permutation) != set(range(size)):
        raise ValueError("permutation must be a bijection of occurrence identities")
    return permutation


def remap_action(
    action: OrderedAction,
    *,
    old_to_new: Sequence[int],
    original_occurrences: Sequence[_OptionOccurrenceIdentity],
    target_options: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[OrderedAction, tuple[_OptionOccurrenceIdentity, ...]]:
    """Carry original occurrence identities through an explicit old-to-new permutation.

    `target_options`, if provided, validates only that each target semantic identity
    matches the identity carried into that position. It never assigns occurrences.
    """
    occurrences = tuple(original_occurrences)
    if not all(isinstance(item, _OptionOccurrenceIdentity) for item in occurrences):
        raise ValueError("original occurrences must be alignment identities")
    permutation = _validate_permutation(old_to_new, len(occurrences))
    if any(index < 0 or index >= len(occurrences) for index in action.indices):
        raise ValueError("action index is absent from original occurrences")
    carried: list[_OptionOccurrenceIdentity | None] = [None] * len(occurrences)
    for old_index, new_index in enumerate(permutation):
        carried[new_index] = occurrences[old_index]
    aligned = tuple(item for item in carried if item is not None)
    if target_options is not None:
        if len(target_options) != len(aligned):
            raise ValueError("target options length does not match permutation")
        for index, option in enumerate(target_options):
            if OptionSemanticIdentity.from_option(option) != aligned[index].semantic:
                raise ValueError("target options do not match carried semantic identities")
    return (
        OrderedAction(tuple(permutation[index] for index in action.indices), action.termination),
        aligned,
    )


def serialize_occurrence_identities(
    occurrences: Sequence[_OptionOccurrenceIdentity],
) -> tuple[dict[str, object], ...]:
    """Serialize alignment-only identities for audit storage, never model features."""
    result: list[dict[str, object]] = []
    for item in occurrences:
        if not isinstance(item, _OptionOccurrenceIdentity):
            raise ValueError("occurrences must contain alignment identities")
        semantic = item.semantic
        result.append(
            {
                "semantic": {
                    "type": semantic.type,
                    "player_index": semantic.player_index,
                    "area": semantic.area,
                    "index": semantic.index,
                    "card_id": semantic.card_id,
                    "target_player_index": semantic.target_player_index,
                    "in_play_area": semantic.in_play_area,
                    "in_play_index": semantic.in_play_index,
                    "number": semantic.number,
                    "attack_id": semantic.attack_id,
                    "count": semantic.count,
                    "energy_index": semantic.energy_index,
                    "tool_index": semantic.tool_index,
                },
                "ordinal": item.occurrence,
            }
        )
    return tuple(result)


__all__ = [
    "ActionTermination",
    "OptionSemanticIdentity",
    "OrderedAction",
    "NON_SEMANTIC_METADATA_FIELDS",
    "SUPPORTED_OPTION_FIELDS",
    "build_occurrence_identities",
    "remap_action",
    "serialize_occurrence_identities",
    "unrecognized_option_fields",
    "validate_ordered_action",
]
