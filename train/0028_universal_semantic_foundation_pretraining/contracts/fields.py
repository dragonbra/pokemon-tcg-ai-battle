"""Human-readable field definitions for every actor-visible tensor.

Categorical fields have independent vocabularies and embeddings. Numeric fields are
continuous/count facts; option numeric fields have a parallel explicit field-state tensor so
padding, present zero, unknown, and not-applicable cannot collapse into one value.
"""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = "0028_canonical_semantic_decision_v1"


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
        "own_status_bits",
        "opponent_status_bits",
        "deck_membership_known",
        "deck_order_known",
    ),
    vocabularies=(66, 130, 4, 3, 3, 3, 3, 33, 33, 3, 3),
)
GLOBAL_NUM_FIELDS = (
    "turn",
    "turn_action_count",
    "own_deck_count",
    "opponent_deck_count",
    "own_hand_count",
    "opponent_hand_count",
    "own_prize_count",
    "opponent_prize_count",
    "own_bench_count",
    "opponent_bench_count",
    "legal_option_count",
    "minimum_selection_count",
    "maximum_selection_count",
    "remaining_damage_counter",
    "remaining_energy_cost",
    "known_opponent_hand_count",
    "unknown_opponent_hand_count",
)

CARD_CATEGORICAL = CategoricalGroup(
    names=("card_id", "relative_owner", "zone", "kind", "status_bits"),
    vocabularies=(2049, 4, 18, 8, 33),
)
CARD_NUM_FIELDS = (
    "current_hp",
    "maximum_hp",
    "damage",
    "attached_energy_count",
    "tool_count",
    "pre_evolution_count",
    "appeared_this_turn",
)

RESOURCE_CATEGORICAL = CategoricalGroup(
    names=("card_id", "deck_knowledge", "prize_knowledge", "deck_order_known"),
    vocabularies=(2049, 8, 8, 3),
)
RESOURCE_NUM_FIELDS = (
    "initial_count",
    "visible_active",
    "visible_bench",
    "visible_hand",
    "visible_discard",
    "visible_stadium",
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
        "from_area",
        "to_area",
        "identity_visible",
        "has_serial",
        "has_target",
    ),
    vocabularies=(32, 4, 2049, 34, 34, 3, 3, 3),
)
EVENT_NUM_FIELDS = ("age", "value", "damage_counters", "coin_heads")

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
        "selected_energy_type",
        "special_condition_type",
        "select_type",
        "select_context",
        "context_card_id",
        "effect_card_id",
    ),
    vocabularies=(66, 4, 34, 4, 34, 2049, 2049, 4097, 14, 34, 66, 130, 2049, 2049),
)
OPTION_NUM_FIELDS = (
    "number",
    "count",
    "remaining_damage_counter",
    "remaining_energy_cost",
    "base_damage",
    "required_energy_count",
    "attached_energy_count",
    "exact_energy_matches",
    "typed_energy_deficit",
    "total_energy_deficit",
    "target_current_hp",
    "target_maximum_hp",
    "target_hp_after_base_damage",
    "base_damage_is_ko",
    "source_current_hp",
    "source_maximum_hp",
)

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

SKILL_ROLE_VOCAB = 11
EFFECT_ROLE_VOCAB = 8

ACTOR_KEYS = frozenset(
    {
        "global_cat",
        "global_num",
        "card_cat",
        "card_num",
        "card_parent",
        "resource_cat",
        "resource_num",
        "event_cat",
        "event_num",
        "option_cat",
        "option_num",
        "option_state",
        "option_source",
        "option_target",
        "option_skill_id",
        "option_skill_role",
        "option_skill_parent",
        "option_effect_id",
        "option_effect_role",
        "option_effect_parent",
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
        "option_skill_mask",
        "option_effect_mask",
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


WIDTHS = FieldWidths()


__all__ = [name for name in globals() if name.isupper()] + [
    "CategoricalGroup",
    "FieldWidths",
]
