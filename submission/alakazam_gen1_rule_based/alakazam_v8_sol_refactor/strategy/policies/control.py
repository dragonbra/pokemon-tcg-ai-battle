from __future__ import annotations

from collections.abc import Sequence

from ..cards import (
    BASIC_ENERGY_IDS,
    BOSS_ORDERS,
    ENHANCED_HAMMER,
    NIGHTTIME_MINE,
    PROTECTIVE_DAMAGE_ENERGIES,
)
from ..model import ActionIntent, ActionKind, DecisionPhase, SemanticOption, TurnFacts, TurnPlan
from ..profiles import HammerPolicy, StadiumPolicy, StrategyProfile
from ..routes import RouteAnalysis


def _play(card_id: int, rule_id: str, purpose: str) -> ActionIntent:
    return ActionIntent(
        rule_id,
        DecisionPhase.MUST_PREPARE,
        ActionKind.PLAY,
        purpose,
        card_id=card_id,
        required_before_attack=True,
    )


def propose(
    facts: TurnFacts,
    plan: TurnPlan,
    routes: RouteAnalysis,
    options: Sequence[SemanticOption],
    profile: StrategyProfile,
) -> tuple[ActionIntent, ...]:
    if facts.item_lock:
        return ()
    intents: list[ActionIntent] = []
    play_ids = {option.card_id for option in options if option.action_kind == ActionKind.PLAY}
    opponent_special = any(
        energy_id not in BASIC_ENERGY_IDS
        for pokemon in facts.opponent.field
        for energy_id in pokemon.attached_energy_ids
    )
    protective_special = any(
        energy_id in PROTECTIVE_DAMAGE_ENERGIES
        for pokemon in facts.opponent.field
        for energy_id in pokemon.attached_energy_ids
    )
    hammer_allowed = profile.hammer == HammerPolicy.ALWAYS_LEGAL_TARGET or protective_special
    if ENHANCED_HAMMER in play_ids and opponent_special and hammer_allowed:
        intents.append(_play(ENHANCED_HAMMER, "control.enhanced_hammer", "remove_special_energy"))
    stadium_allowed = profile.stadium == StadiumPolicy.IMMEDIATE_VALUE
    if (
        NIGHTTIME_MINE in play_ids
        and stadium_allowed
        and routes.attack.certainty.value == "confirmed"
        and routes.attack.target
        and routes.attack.expected_damage < routes.attack.target.hp
    ):
        intents.append(
            _play(
                NIGHTTIME_MINE,
                "control.nighttime_mine",
                "change_opponent_stadium_pressure",
            )
        )
    if (
        BOSS_ORDERS in play_ids
        and plan.supporter_purpose == "gust_confirmed_knockout"
        and routes.attack.target
        and routes.attack.expected_damage < routes.attack.target.hp
    ):
        if facts.opponent.bench:
            intents.append(_play(BOSS_ORDERS, "control.boss_knockout", "gust_confirmed_knockout"))
    return tuple(intents)
