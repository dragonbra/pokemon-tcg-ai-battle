from __future__ import annotations

from collections.abc import Sequence

from ..cards import ABRA, POWERFUL_HAND_ATTACK, TRADING_PLACES_ATTACK
from ..model import ActionIntent, ActionKind, DecisionPhase, SemanticOption, TurnFacts, TurnPlan
from ..profiles import StrategyProfile
from ..routes import RouteAnalysis


def propose(
    facts: TurnFacts,
    plan: TurnPlan,
    routes: RouteAnalysis,
    options: Sequence[SemanticOption],
    profile: StrategyProfile,
) -> tuple[ActionIntent, ...]:
    del facts, profile
    if plan.attack_blockers or routes.attack.certainty.value != "confirmed":
        return ()
    if routes.attack.attacker and routes.attack.attacker.card_id == ABRA:
        return ()
    intents: list[ActionIntent] = []
    for option in options:
        if option.action_kind != ActionKind.ATTACK:
            continue
        if option.attack_id == TRADING_PLACES_ATTACK:
            continue
        if routes.attack.attacker and routes.attack.attacker.card_id == 743:
            if option.attack_id != POWERFUL_HAND_ATTACK:
                continue
        intents.append(
            ActionIntent(
                rule_id="commit.attack",
                phase=DecisionPhase.ATTACK_COMMIT,
                action_kind=ActionKind.ATTACK,
                purpose="submit_current_attack",
                attack_id=option.attack_id,
            )
        )
    return tuple(intents)
