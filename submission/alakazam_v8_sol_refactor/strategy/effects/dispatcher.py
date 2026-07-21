from __future__ import annotations

from collections.abc import Sequence

from ..cards import (
    ALAKAZAM,
    BASIC_PSYCHIC,
    BOSS_ORDERS,
    DAWN,
    DUDUNSPARCE,
    ENHANCED_HAMMER,
    HILDA,
    LANAS_AID,
    NIGHT_STRETCHER,
    POKE_PAD,
    POFFIN,
    SACRED_ASH,
    TELEPATH_ENERGY,
)
from ..memory import GameMemory
from ..model import ActionKind, Decision, PlanKind, SemanticOption, TurnFacts, TurnPlan
from ..options import decode_options, required_effect_fallback
from ..profiles import StrategyProfile
from .recovery import choose_lanas_aid, choose_sacred_ash
from .search import choose_search
from .targeting import choose_hammer_target, choose_rare_candy_target, choose_switch_target


def _decision(indexes: Sequence[int], rule_id: str, plan: TurnPlan, purpose: str) -> Decision:
    return Decision(tuple(indexes), rule_id, plan.kind, purpose)


def _effect_identity(select: dict) -> tuple[int | None, int | None]:
    effect = select.get("effect") or {}
    context_card = select.get("contextCard") or {}
    effect_id = effect.get("id", context_card.get("id"))
    serial = effect.get("serial", context_card.get("serial"))
    return (
        int(effect_id) if effect_id is not None else None,
        int(serial) if serial is not None else None,
    )


def _choose_card_by_priority(
    options: Sequence[SemanticOption], priorities: Sequence[int]
) -> tuple[int, ...]:
    for card_id in priorities:
        option = next((candidate for candidate in options if candidate.card_id == card_id), None)
        if option:
            return (option.index,)
    return ()


def _yes_no(
    options: Sequence[SemanticOption], facts: TurnFacts, effect_id: int | None, plan: TurnPlan
) -> tuple[int, ...]:
    yes = next((option for option in options if option.action_kind == ActionKind.YES), None)
    no = next((option for option in options if option.action_kind == ActionKind.NO), None)
    accept = True
    if effect_id == DUDUNSPARCE:
        accept = any(
            pokemon.card_id == ALAKAZAM and pokemon.has_energy_type(BASIC_PSYCHIC)
            for pokemon in facts.yours.bench
        ) and (facts.yours.deck_count > 10 or plan.kind == PlanKind.VICTORY)
    chosen = yes if accept else no
    return (chosen.index,) if chosen else ()


def select_effect(
    obs: dict,
    facts: TurnFacts,
    plan: TurnPlan,
    memory: GameMemory,
    profile: StrategyProfile,
) -> Decision:
    select = obs.get("select") or {}
    options = decode_options(select, facts)
    effect_id, effect_serial = _effect_identity(select)
    min_count = int(select.get("minCount", 0))
    max_count = int(select.get("maxCount", len(options)))
    context = int(select.get("context", 0))
    select_type = int(select.get("type", 0))
    explicit_step = select.get("effectStep")
    explicit_step = int(explicit_step) if explicit_step is not None else None

    indexes: tuple[int, ...] = ()
    rule_id = "effect.required_fallback"
    purpose = "satisfy_required_effect"
    if context == 37:
        indexes = choose_rare_candy_target(options, facts)
        rule_id, purpose = "effect.rare_candy_target", "evolve_attack_base"
    elif effect_id == ENHANCED_HAMMER:
        indexes = choose_hammer_target(options)
        rule_id, purpose = "effect.hammer_target", "remove_special_energy"
    elif effect_id == LANAS_AID:
        indexes = choose_lanas_aid(options, facts, max_count=max_count)
        rule_id, purpose = "effect.lanas_aid", "recover_attack_line"
    elif effect_id == SACRED_ASH:
        indexes = choose_sacred_ash(
            options,
            facts,
            max_count=max_count,
            recovery_policy=profile.recovery,
        )
        rule_id, purpose = "effect.sacred_ash", "recover_complete_lines"
    elif effect_id in {POKE_PAD, DAWN, HILDA, POFFIN, TELEPATH_ENERGY}:
        indexes = choose_search(
            effect_id,
            options,
            facts,
            plan,
            memory,
            max_count=max_count,
            effect_serial=effect_serial,
            explicit_step=explicit_step,
        )
        rule_id, purpose = "effect.search", "advance_planned_route"
    elif effect_id == NIGHT_STRETCHER:
        indexes = _choose_card_by_priority(
            options, (ALAKAZAM, KADABRA, 741, BASIC_PSYCHIC)
        )
        rule_id, purpose = "effect.night_stretcher", "recover_route_gap"
    elif effect_id == BOSS_ORDERS or context in {3, 4}:
        indexes = choose_switch_target(options, facts)
        rule_id, purpose = "effect.switch_target", "promote_planned_target"
    elif select_type == 4:
        indexes = choose_hammer_target(options)
        rule_id, purpose = "effect.energy_target", "select_energy"
    elif select_type == 8:
        ranked = sorted(options, key=lambda option: -int(option.raw.get("number", 0)))
        indexes = (ranked[0].index,) if ranked else ()
        rule_id, purpose = "effect.count_max", "maximize_effect_value"
    elif select_type == 9 or all(
        option.action_kind in {ActionKind.YES, ActionKind.NO} for option in options
    ):
        indexes = _yes_no(options, facts, effect_id, plan)
        rule_id, purpose = "effect.optional", "accept_safe_effect"

    if len(indexes) < min_count:
        indexes = required_effect_fallback(select, options)
        rule_id, purpose = "fallback.effect_required", "satisfy_required_effect"
    indexes = tuple(indexes[:max_count])
    memory.advance_effect(effect_serial, effect_id)
    return _decision(indexes, rule_id, plan, purpose)
