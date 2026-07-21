from __future__ import annotations

from collections.abc import Sequence

from ..cards import ABRA, DUNSPARCE, POFFIN, POKE_PAD
from ..model import ActionIntent, ActionKind, DecisionPhase, SemanticOption, TurnFacts, TurnPlan
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


def propose(
    facts: TurnFacts,
    plan: TurnPlan,
    routes: RouteAnalysis,
    options: Sequence[SemanticOption],
    profile: StrategyProfile,
) -> tuple[ActionIntent, ...]:
    del plan, routes, profile
    if facts.item_lock:
        return ()
    play_ids = {option.card_id for option in options if option.action_kind == ActionKind.PLAY}
    intents: list[ActionIntent] = []
    if POFFIN in play_ids and len(facts.yours.bench) < facts.yours.bench_max:
        field_ids = [pokemon.card_id for pokemon in facts.yours.field]
        if not (field_ids.count(ABRA) >= 2 and DUNSPARCE in field_ids):
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
