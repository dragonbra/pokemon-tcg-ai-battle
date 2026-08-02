"""Named, typed tensor contract for the canonical 0025 actor."""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = "0025_canonical_semantic_decision_v2"

GLOBAL_CAT_FIELDS = (
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
)
GLOBAL_CAT_VOCABS = (66, 130, 4, 3, 3, 3, 3, 33, 33, 3, 3)
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

CARD_CAT_FIELDS = ("card_id", "relative_owner", "zone", "kind", "status_bits")
CARD_CAT_VOCABS = (2049, 4, 18, 8, 33)
CARD_NUM_FIELDS = (
    "current_hp",
    "maximum_hp",
    "damage",
    "attached_energy_count",
    "tool_count",
    "pre_evolution_count",
    "appeared_this_turn",
)

RESOURCE_CAT_FIELDS = ("card_id", "deck_knowledge", "prize_knowledge", "deck_order_known")
RESOURCE_CAT_VOCABS = (2049, 8, 8, 3)
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

EVENT_CAT_FIELDS = (
    "log_type",
    "relative_actor",
    "card_id",
    "from_area",
    "to_area",
    "identity_visible",
    "has_serial",
    "has_target",
)
EVENT_CAT_VOCABS = (32, 4, 2049, 34, 34, 3, 3, 3)
EVENT_NUM_FIELDS = ("age", "value", "damage_counters", "coin_heads")

OPTION_CAT_FIELDS = (
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
)
OPTION_CAT_VOCABS = (66, 4, 34, 4, 34, 2049, 2049, 4097, 14, 34, 66, 130, 2049, 2049)
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


__all__ = [name for name in globals() if name.isupper()]
