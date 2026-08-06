"""Audited actor-visible fields for the 0032 semantic policy.

The contract keeps official observation facts, full-engine prototype facts, and explicit
relations. It excludes answers that require running a partial rules engine and removes values
that are exactly recoverable from another actor-visible field or relation.
"""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = "0033_effect_summary_semantic_decision_v1"
ENERGY_TYPE_COUNT = 12


@dataclass(frozen=True, slots=True)
class CategoricalGroup:
    names: tuple[str, ...]
    vocabularies: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.vocabularies):
            raise ValueError("categorical field names and vocabularies must align")
        if not self.names or min(self.vocabularies) < 1:
            raise ValueError("categorical groups require positive vocabularies")

    @property
    def width(self) -> int:
        return len(self.names)


GLOBAL_CATEGORICAL = CategoricalGroup(
    names=(
        "select_type",
        "select_context",
        "relative_first_player",
        "supporter_played",
        "stadium_played",
        "energy_attached",
        "retreated",
    ),
    vocabularies=(66, 130, 4, 3, 3, 3, 3),
)
GLOBAL_NUM_FIELDS = (
    "turn",
    "turn_action_count",
    "own_deck_count",
    "opponent_deck_count",
    "opponent_hand_count",
    "own_prize_count",
    "opponent_prize_count",
    "own_bench_max",
    "opponent_bench_max",
    "remaining_damage_counter",
    "remaining_energy_cost",
)

CARD_CATEGORICAL = CategoricalGroup(
    names=(
        "card_id",
        "relative_owner",
        "zone",
        "zone_slot",
        "status_bits",
        "appeared_this_turn_state",
    ),
    vocabularies=(2049, 4, 12, 130, 33, 5),
)
CARD_NUM_FIELDS = (
    "current_hp",
    "maximum_hp",
    *tuple(f"resolved_energy_type_{index}_count" for index in range(ENERGY_TYPE_COUNT)),
)

RESOURCE_CATEGORICAL = CategoricalGroup(
    names=("card_id", "deck_knowledge", "prize_knowledge"),
    vocabularies=(2049, 8, 8),
)
RESOURCE_NUM_FIELDS = (
    "initial_count",
    "visible_playing",
    "deck_value",
    "deck_lower",
    "deck_upper",
    "prize_value",
    "prize_lower",
    "prize_upper",
    "deck_information_age",
    "prize_information_age",
)

EVENT_CATEGORICAL = CategoricalGroup(
    names=(
        "log_type",
        "relative_actor",
        "card_id",
        "target_card_id",
        "attack_id",
        "from_area",
        "to_area",
        "active_card_id",
        "bench_card_id",
        "before_card_id",
        "after_card_id",
        "is_recover",
        "put_damage_counter",
        "coin_head",
        "special_condition_type",
        "result_type",
        "reason_type",
    ),
    vocabularies=(
        32, 4, 2049, 2049, 2049, 34, 34, 2049, 2049, 2049, 2049,
        3, 3, 3, 34, 130, 130,
    ),
)
EVENT_NUM_FIELDS = (
    "age",
    "value",
    "count",
    "number",
)

OPTION_CATEGORICAL = CategoricalGroup(
    names=(
        "action_type",
        "source_owner",
        "source_area",
        "target_owner",
        "target_area",
        "source_card_id",
        "target_card_id",
        "attack_id",
        "special_condition_type",
        "context_card_id",
        "effect_card_id",
    ),
    vocabularies=(66, 4, 34, 4, 34, 2049, 2049, 2049, 34, 2049, 2049),
)
OPTION_NUM_FIELDS = ("number", "count")

GLOBAL_CAT_FIELDS = GLOBAL_CATEGORICAL.names
GLOBAL_CAT_VOCABS = GLOBAL_CATEGORICAL.vocabularies
CARD_CAT_FIELDS = CARD_CATEGORICAL.names
CARD_CAT_VOCABS = CARD_CATEGORICAL.vocabularies
RESOURCE_CAT_FIELDS = RESOURCE_CATEGORICAL.names
RESOURCE_CAT_VOCABS = RESOURCE_CATEGORICAL.vocabularies
EVENT_CAT_FIELDS = EVENT_CATEGORICAL.names
EVENT_CAT_VOCABS = EVENT_CATEGORICAL.vocabularies
OPTION_CAT_FIELDS = OPTION_CATEGORICAL.names
OPTION_CAT_VOCABS = OPTION_CATEGORICAL.vocabularies

ACTOR_KEYS = frozenset(
    {
        "global_cat",
        "global_num",
        "global_state",
        "card_cat",
        "card_num",
        "card_state",
        "card_parent",
        "resource_cat",
        "resource_num",
        "resource_state",
        "event_cat",
        "event_num",
        "event_state",
        "event_source",
        "event_target",
        "option_cat",
        "option_num",
        "option_state",
        "option_source",
        "option_target",
        "min_count",
        "max_count",
    }
)
MASK_KEYS = frozenset(
    {
        "card_mask",
        "resource_mask",
        "event_mask",
        "option_mask",
    }
)
EXPECTED_BATCH_KEYS = ACTOR_KEYS | MASK_KEYS | {"targets"}


@dataclass(frozen=True, slots=True)
class FieldWidths:
    global_cat: int = len(GLOBAL_CAT_FIELDS)
    global_num: int = len(GLOBAL_NUM_FIELDS)
    card_cat: int = len(CARD_CAT_FIELDS)
    card_num: int = len(CARD_NUM_FIELDS)
    resource_cat: int = len(RESOURCE_CAT_FIELDS)
    resource_num: int = len(RESOURCE_NUM_FIELDS)
    event_cat: int = len(EVENT_CAT_FIELDS)
    event_num: int = len(EVENT_NUM_FIELDS)
    option_cat: int = len(OPTION_CAT_FIELDS)
    option_num: int = len(OPTION_NUM_FIELDS)
    option_state: int = len(OPTION_NUM_FIELDS)
    global_state: int = len(GLOBAL_NUM_FIELDS)
    card_state: int = len(CARD_NUM_FIELDS)
    resource_state: int = len(RESOURCE_NUM_FIELDS)
    event_state: int = len(EVENT_NUM_FIELDS)


WIDTHS = FieldWidths()


__all__ = [name for name in globals() if name.isupper()] + [
    "CategoricalGroup",
    "FieldWidths",
]
