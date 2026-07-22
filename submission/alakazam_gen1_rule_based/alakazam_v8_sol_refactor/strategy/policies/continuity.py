from __future__ import annotations

from collections.abc import Sequence

from ..cards import ALAKAZAM, DUDUNSPARCE, FEZANDIPITI_EX, KADABRA
from ..model import (
    ActionIntent,
    ActionKind,
    DecisionPhase,
    PlanKind,
    SemanticOption,
    TurnFacts,
    TurnPlan,
)
from ..profiles import DeckSafetyPolicy, StrategyProfile
from ..routes import RouteAnalysis


def _intent(
    rule_id: str,
    action_kind: ActionKind,
    purpose: str,
    *,
    card_id: int | None = None,
    target_key: tuple[int, int, int | None] | None = None,
) -> ActionIntent:
    return ActionIntent(
        rule_id=rule_id,
        phase=DecisionPhase.CURRENT_ATTACKER,
        action_kind=action_kind,
        purpose=purpose,
        card_id=card_id,
        target_key=target_key,
        required_before_attack=True,
    )


def propose(
    facts: TurnFacts,
    plan: TurnPlan,
    routes: RouteAnalysis,
    options: Sequence[SemanticOption],
    profile: StrategyProfile,
) -> tuple[ActionIntent, ...]:
    draw_threshold = 5 if profile.deck_safety == DeckSafetyPolicy.AGGRESSIVE else 10
    intents: list[ActionIntent] = []
    for goal in plan.must_goals:
        if goal.rule_id == "evolution.bench_kadabra":
            target = next(
                (pokemon for pokemon in facts.yours.bench if pokemon.card_id == KADABRA),
                None,
            )
            if target:
                intents.append(
                    _intent(
                        goal.rule_id,
                        ActionKind.EVOLVE,
                        goal.purpose,
                        card_id=ALAKAZAM,
                        target_key=target.key,
                    )
                )
        elif goal.rule_id == "evolution.active_kadabra" and facts.yours.active:
            intents.append(
                _intent(
                    goal.rule_id,
                    ActionKind.EVOLVE,
                    goal.purpose,
                    card_id=ALAKAZAM,
                    target_key=facts.yours.active.key,
                )
            )
        elif goal.rule_id == "handoff.dudunsparce" and facts.yours.active:
            intents.append(
                _intent(
                    goal.rule_id,
                    ActionKind.EVOLVE,
                    goal.purpose,
                    card_id=DUDUNSPARCE,
                    target_key=facts.yours.active.key,
                )
            )

    if facts.previous_opponent_turn_had_ko and plan.kind != PlanKind.VICTORY:
        ability = next(
            (
                option
                for option in options
                if option.action_kind == ActionKind.ABILITY
                and option.card_id == FEZANDIPITI_EX
            ),
            None,
        )
        play = next(
            (
                option
                for option in options
                if option.action_kind == ActionKind.PLAY
                and option.card_id == FEZANDIPITI_EX
            ),
            None,
        )
        if ability:
            intents.insert(
                0,
                _intent(
                    "continuity.fezandipiti_draw",
                    ActionKind.ABILITY,
                    "refill_after_ko",
                    card_id=FEZANDIPITI_EX,
                ),
            )
        elif play:
            intents.insert(
                0,
                _intent(
                    "continuity.fezandipiti_play",
                    ActionKind.PLAY,
                    "refill_after_ko",
                    card_id=FEZANDIPITI_EX,
                ),
            )

    if (
        facts.yours.active
        and facts.yours.active.card_id == DUDUNSPARCE
        and routes.handoff.certainty.value == "confirmed"
    ):
        ability = next(
            (
                option
                for option in options
                if option.action_kind == ActionKind.ABILITY
                and option.card_id == DUDUNSPARCE
            ),
            None,
        )
        if ability:
            intents.insert(
                0,
                _intent(
                    "continuity.dudunsparce_handoff",
                    ActionKind.ABILITY,
                    "promote_ready_attacker",
                    card_id=DUDUNSPARCE,
                ),
            )

    for option in options:
        if option.action_kind != ActionKind.ABILITY:
            continue
        if option.card_id in {KADABRA, ALAKAZAM} and facts.yours.deck_count > draw_threshold:
            intents.append(
                _intent(
                    "continuity.psychic_draw",
                    ActionKind.ABILITY,
                    "prepare_next_attack",
                    card_id=option.card_id,
                )
            )
    return tuple(intents)
