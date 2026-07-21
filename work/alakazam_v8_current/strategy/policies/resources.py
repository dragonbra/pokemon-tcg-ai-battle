from __future__ import annotations

from collections.abc import Sequence

from ..cards import (
    ABRA,
    ALAKAZAM,
    BASIC_PSYCHIC,
    DAWN,
    DUDUNSPARCE,
    DUNSPARCE,
    ENRICHING_ENERGY,
    FEZANDIPITI_EX,
    HILDA,
    KADABRA,
    LANAS_AID,
    NIGHT_STRETCHER,
    RARE_CANDY,
    SACRED_ASH,
    SHAYMIN,
    TELEPATH_ENERGY,
    XEROSIC,
)
from ..model import (
    ActionIntent,
    ActionKind,
    DecisionPhase,
    PlanKind,
    SemanticOption,
    TurnFacts,
    TurnPlan,
)
from ..profiles import StrategyProfile
from ..routes import RouteAnalysis


def _intent(
    rule_id: str,
    action_kind: ActionKind,
    purpose: str,
    *,
    card_id: int | None = None,
    target_key=None,
) -> ActionIntent:
    return ActionIntent(
        rule_id=rule_id,
        phase=DecisionPhase.RESOURCE_ALLOCATION,
        action_kind=action_kind,
        purpose=purpose,
        card_id=card_id,
        target_key=target_key,
        required_before_attack=True,
    )


def _play(
    options: Sequence[SemanticOption], card_id: int, rule_id: str, purpose: str
) -> ActionIntent | None:
    if any(
        option.action_kind == ActionKind.PLAY and option.card_id == card_id
        for option in options
    ):
        return _intent(rule_id, ActionKind.PLAY, purpose, card_id=card_id)
    return None


def propose(
    facts: TurnFacts,
    plan: TurnPlan,
    routes: RouteAnalysis,
    options: Sequence[SemanticOption],
    profile: StrategyProfile,
) -> tuple[ActionIntent, ...]:
    del profile, routes
    intents: list[ActionIntent] = []
    if not facts.budget.supporter_used:
        if plan.supporter_purpose == "complete_evolution_chain":
            dawn = _play(
                options,
                DAWN,
                "resource.dawn_evolution",
                "complete_evolution_chain",
            )
            if dawn:
                intents.append(dawn)
        elif plan.supporter_purpose == "supply_evolution_and_energy":
            hilda = _play(
                options,
                HILDA,
                "resource.hilda_attack_line",
                "supply_evolution_and_energy",
            )
            if hilda:
                intents.append(hilda)
        elif plan.supporter_purpose == "supply_handoff_engine":
            hilda = _play(
                options,
                HILDA,
                "resource.hilda_handoff_engine",
                "supply_handoff_engine",
            )
            if hilda:
                intents.append(hilda)
        elif plan.supporter_purpose == "recover_attack_line":
            lana = _play(
                options,
                LANAS_AID,
                "resource.lana_successor",
                "recover_attack_line",
            )
            if lana:
                intents.append(lana)
        elif plan.supporter_purpose == "reduce_opponent_hand":
            xerosic = _play(
                options,
                XEROSIC,
                "resource.xerosic_disruption",
                "reduce_opponent_hand",
            )
            if xerosic:
                intents.append(xerosic)

    if not facts.budget.energy_used:
        for option in options:
            if option.action_kind != ActionKind.ATTACH or option.target is None:
                continue
            if (
                option.card_id == TELEPATH_ENERGY
                and option.target.card_id in {DUNSPARCE, DUDUNSPARCE}
                and not option.target.has_energy_type(BASIC_PSYCHIC)
            ):
                intents.append(
                    _intent(
                        "resource.attach_telepath_setup",
                        ActionKind.ATTACH,
                        "open_basic_psychic_search",
                        card_id=option.card_id,
                        target_key=option.target.key,
                    )
                )
                break
            if (
                option.card_id == ENRICHING_ENERGY
                and option.target.card_id == DUDUNSPARCE
                and facts.yours.active is not None
                and facts.yours.active.key == option.target.key
                and (
                    facts.yours.hand_count < 20
                    and facts.yours.deck_count > 14
                    or plan.kind.value == "victory"
                )
            ):
                intents.append(
                    _intent(
                        "resource.attach_enriching_dudunsparce",
                        ActionKind.ATTACH,
                        "enable_run_away_draw",
                        card_id=option.card_id,
                        target_key=option.target.key,
                    )
                )
                break
            if (
                option.card_id == ENRICHING_ENERGY
                and facts.yours.active is not None
                and option.target.key == facts.yours.active.key
                and option.target.card_id in {DUNSPARCE, DUDUNSPARCE, FEZANDIPITI_EX, SHAYMIN}
                and (facts.yours.deck_count > 14 or plan.kind == PlanKind.VICTORY)
            ):
                intents.append(
                    _intent(
                        "resource.attach_enriching_draw",
                        ActionKind.ATTACH,
                        "trigger_enriching_draw",
                        card_id=option.card_id,
                        target_key=option.target.key,
                    )
                )
                break
            if option.card_id == TELEPATH_ENERGY and option.target.card_id in {
                ABRA,
                KADABRA,
                ALAKAZAM,
            }:
                if not option.target.has_energy_type(BASIC_PSYCHIC):
                    intents.append(
                        _intent(
                            "resource.attach_psychic",
                            ActionKind.ATTACH,
                            "charge_attack_line",
                            card_id=option.card_id,
                            target_key=option.target.key,
                        )
                    )
                    break
            if option.card_id == BASIC_PSYCHIC and option.target.card_id in {
                ABRA,
                KADABRA,
                ALAKAZAM,
            }:
                if not option.target.has_energy_type(BASIC_PSYCHIC):
                    intents.append(
                        _intent(
                            "resource.attach_psychic",
                            ActionKind.ATTACH,
                            "charge_attack_line",
                            card_id=option.card_id,
                            target_key=option.target.key,
                        )
                    )
                    break
    if not facts.item_lock:
        for card_id, rule_id, purpose in (
            (RARE_CANDY, "resource.rare_candy_route", "complete_direct_alakazam"),
            (NIGHT_STRETCHER, "resource.night_stretcher", "recover_route_gap"),
            (SACRED_ASH, "resource.sacred_ash", "restore_attack_lines"),
        ):
            item = _play(options, card_id, rule_id, purpose)
            if item:
                intents.append(item)
    return tuple(intents)
