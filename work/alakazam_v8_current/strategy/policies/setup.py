from __future__ import annotations

from collections.abc import Sequence

from ..cards import (
    ABRA,
    ALAKAZAM,
    BASIC_PSYCHIC,
    DUNSPARCE,
    KADABRA,
    POFFIN,
    POKE_PAD,
    TELEPATH_ENERGY,
)
from ..model import (
    ActionIntent,
    ActionKind,
    DecisionPhase,
    PlanKind,
    RouteCertainty,
    SemanticOption,
    TurnFacts,
    TurnPlan,
)
from ..profiles import StrategyProfile
from ..routes import RouteAnalysis


def _play(card_id: int, rule_id: str, purpose: str) -> ActionIntent:
    return ActionIntent(
        rule_id=rule_id,
        phase=DecisionPhase.OPTIONAL_PREPARE,
        action_kind=ActionKind.PLAY,
        purpose=purpose,
        card_id=card_id,
    )


def _setup_complete(facts: TurnFacts) -> bool:
    field_ids = [pokemon.card_id for pokemon in facts.yours.field]
    return field_ids.count(ABRA) >= 2 and DUNSPARCE in field_ids


def _has_visible_successor(facts: TurnFacts) -> bool:
    for pokemon in facts.yours.bench:
        if pokemon.card_id in {KADABRA, ALAKAZAM} and pokemon.has_energy_type(BASIC_PSYCHIC):
            return True
        if (
            pokemon.card_id == ABRA
            and pokemon.has_energy_type(BASIC_PSYCHIC)
            and (
                KADABRA in facts.yours.hand_ids
                or (
                    ALAKAZAM in facts.yours.hand_ids
                    and 1079 in facts.yours.hand_ids
                )
            )
        ):
            return True
    return False


def _bench_insurance_needed(facts: TurnFacts, plan: TurnPlan) -> bool:
    active = facts.yours.active
    if (
        plan.kind == PlanKind.VICTORY
        or plan.primary_attack.certainty != RouteCertainty.CONFIRMED
        or not active
        or active.card_id not in {KADABRA, ALAKAZAM}
        or not active.has_energy_type(BASIC_PSYCHIC)
    ):
        return False
    if len(facts.yours.bench) >= facts.yours.bench_max or _has_visible_successor(facts):
        return False
    return True


def _current_attacker_needs_psychic(
    facts: TurnFacts, options: Sequence[SemanticOption]
) -> bool:
    active = facts.yours.active
    if (
        not active
        or active.card_id not in {ABRA, KADABRA, ALAKAZAM}
        or active.has_energy_type(BASIC_PSYCHIC)
    ):
        return False
    return any(
        option.action_kind == ActionKind.ATTACH
        and option.card_id in {BASIC_PSYCHIC, TELEPATH_ENERGY}
        and option.target is not None
        and option.target.key == active.key
        for option in options
    )


def propose(
    facts: TurnFacts,
    plan: TurnPlan,
    routes: RouteAnalysis,
    options: Sequence[SemanticOption],
    profile: StrategyProfile,
) -> tuple[ActionIntent, ...]:
    del routes, profile
    if facts.item_lock:
        return ()
    play_ids = {option.card_id for option in options if option.action_kind == ActionKind.PLAY}
    intents: list[ActionIntent] = []
    if (
        POFFIN in play_ids
        and len(facts.yours.bench) < facts.yours.bench_max
        and not _current_attacker_needs_psychic(facts, options)
    ):
        field_ids = [pokemon.card_id for pokemon in facts.yours.field]
        if not _setup_complete(facts) or _bench_insurance_needed(facts, plan):
            intents.append(_play(POFFIN, "setup.poffin_anchor", "establish_attack_bench"))
    if ABRA in play_ids and len(facts.yours.bench) < facts.yours.bench_max:
        field_ids = [pokemon.card_id for pokemon in facts.yours.field]
        if field_ids.count(ABRA) < 2:
            intents.append(_play(ABRA, "setup.basic_abra", "establish_attack_base"))
    if DUNSPARCE in play_ids and len(facts.yours.bench) < facts.yours.bench_max:
        if DUNSPARCE not in {pokemon.card_id for pokemon in facts.yours.field}:
            intents.append(_play(DUNSPARCE, "setup.dunsparce_anchor", "establish_handoff_base"))
    if POKE_PAD in play_ids and not (facts.own_turn <= 1 and _setup_complete(facts)):
        intents.append(_play(POKE_PAD, "setup.poke_pad_route", "find_required_evolution"))
    return tuple(intents)
