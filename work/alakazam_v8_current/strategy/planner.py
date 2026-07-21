from __future__ import annotations

import hashlib

from .cards import ABRA, ALAKAZAM, BASIC_PSYCHIC, DUDUNSPARCE, DUNSPARCE, HILDA, KADABRA
from .model import PlanGoal, PlanKind, RouteCertainty, TurnFacts, TurnPlan
from .profiles import AttackPreparation, StrategyProfile
from .routes import RouteAnalysis


def _plan_id(facts: TurnFacts, routes: RouteAnalysis) -> str:
    key = (
        facts.turn,
        facts.own_turn,
        facts.yours.active.key if facts.yours.active else None,
        routes.attack.attacker.key if routes.attack.attacker else None,
        routes.handoff.attacker.key if routes.handoff.attacker else None,
    )
    return hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:12]


def build_turn_plan(
    facts: TurnFacts,
    routes: RouteAnalysis,
    profile: StrategyProfile,
    previous: TurnPlan | None = None,
) -> TurnPlan:
    attack = routes.attack
    can_ko = (
        attack.certainty == RouteCertainty.CONFIRMED
        and attack.target is not None
        and attack.expected_damage >= attack.target.hp
    )
    victory = can_ko and facts.yours.prize_count <= attack.prize_value
    goals: list[PlanGoal] = []
    active = facts.yours.active
    prepare_before_lethal = (
        profile.attack_preparation == AttackPreparation.COMPLETE_BEFORE_ATTACK or not can_ko
    )
    if (
        not victory
        and prepare_before_lethal
        and active
        and active.card_id == ABRA
        and active.can_evolve
    ):
        active_evolutions = tuple(
            option
            for option in routes.evolution_options
            if option.target is not None
            and option.target.key == active.key
            and option.card_id in {KADABRA, ALAKAZAM}
        )
        if active_evolutions:
            target_card = (
                ALAKAZAM
                if any(option.card_id == ALAKAZAM for option in active_evolutions)
                else KADABRA
            )
            goals.append(
                PlanGoal(
                    rule_id=(
                        "evolution.active_abra_alakazam"
                        if target_card == ALAKAZAM
                        else "evolution.active_abra_kadabra"
                    ),
                    purpose="prepare_current_attacker",
                    required_before_attack=True,
                )
            )
    if not victory and prepare_before_lethal and active and active.card_id == ALAKAZAM:
        for bench in facts.yours.bench:
            if bench.card_id == KADABRA and bench.can_evolve and ALAKAZAM in facts.yours.hand_ids:
                goals.append(
                    PlanGoal(
                        rule_id="evolution.bench_kadabra",
                        purpose="preserve_next_attack_line",
                        required_before_attack=True,
                    )
                )
                break
        else:
            for bench in facts.yours.bench:
                if bench.card_id != ABRA or not bench.can_evolve:
                    continue
                if any(
                    option.target is not None
                    and option.target.key == bench.key
                    and option.card_id == KADABRA
                    for option in routes.evolution_options
                ):
                    goals.append(
                        PlanGoal(
                            rule_id="evolution.bench_abra_kadabra",
                            purpose="prepare_next_attack",
                            required_before_attack=True,
                        )
                    )
                    break
    if not victory and prepare_before_lethal and active and active.card_id == ALAKAZAM:
        for bench in facts.yours.bench:
            if bench.card_id != DUNSPARCE or not bench.can_evolve:
                continue
            if any(
                option.target is not None
                and option.target.key == bench.key
                and option.card_id == DUDUNSPARCE
                for option in routes.evolution_options
            ):
                goals.append(
                    PlanGoal(
                        rule_id="evolution.bench_dunsparce_dudunsparce",
                        purpose="prepare_draw_handoff_engine",
                        required_before_attack=True,
                    )
                )
                break
    if (
        not victory
        and prepare_before_lethal
        and active
        and active.card_id == KADABRA
        and active.can_evolve
    ):
        if ALAKAZAM in facts.yours.hand_ids and routes.evolution_options:
            goals.append(
                PlanGoal(
                    rule_id="evolution.active_kadabra",
                    purpose="prepare_current_attacker",
                    required_before_attack=True,
                )
            )
    if (
        not victory
        and prepare_before_lethal
        and active
        and active.card_id == KADABRA
        and attack.certainty == RouteCertainty.CONFIRMED
        and (
            not active.can_evolve
            or ALAKAZAM not in facts.yours.hand_ids
        )
    ):
        for bench in facts.yours.bench:
            if bench.card_id != ABRA or not bench.can_evolve:
                continue
            if any(
                option.target is not None
                and option.target.key == bench.key
                and option.card_id == KADABRA
                for option in routes.evolution_options
            ):
                goals.append(
                    PlanGoal(
                        rule_id="evolution.bench_abra_kadabra",
                        purpose="prepare_next_attack",
                        required_before_attack=True,
                    )
                )
                break
    if (
        not victory
        and prepare_before_lethal
        and active
        and active.card_id == DUNSPARCE
        and any(
            option.target is not None
            and option.target.key == active.key
            and option.card_id == DUDUNSPARCE
            and option.target.can_evolve
            for option in routes.evolution_options
        )
    ):
        goals.append(
            PlanGoal(
                rule_id="evolution.active_dudunsparce",
                purpose="prepare_draw_handoff_engine",
                required_before_attack=True,
            )
        )
    if (
        not victory
        and prepare_before_lethal
        and active
        and active.card_id == DUNSPARCE
    ):
        for bench in facts.yours.bench:
            if bench.card_id != ABRA or not bench.can_evolve:
                continue
            if any(
                option.target is not None
                and option.target.key == bench.key
                and option.card_id == KADABRA
                for option in routes.evolution_options
            ):
                goals.append(
                    PlanGoal(
                        rule_id="evolution.bench_abra_kadabra",
                        purpose="prepare_next_attack",
                        required_before_attack=True,
                    )
                )
                break
    if routes.handoff.via_dudunsparce and not victory:
        goals.append(
            PlanGoal(
                rule_id="handoff.dudunsparce",
                purpose="replace_active_with_ready_alakazam",
                required_before_attack=True,
            )
        )
    if victory:
        kind = PlanKind.VICTORY
    elif attack.certainty == RouteCertainty.CONFIRMED or goals:
        kind = PlanKind.ATTACK
    else:
        kind = PlanKind.BUILD_SURVIVE
    supporter_purpose = None
    if not victory:
        if (
            attack.certainty == RouteCertainty.CONFIRMED
            and attack.target
            and attack.expected_damage < attack.target.hp
            and any(attack.expected_damage >= pokemon.hp for pokemon in facts.opponent.bench)
        ):
            supporter_purpose = "gust_confirmed_knockout"
        elif any(
            pokemon.card_id == ABRA and pokemon.can_evolve
            for pokemon in facts.yours.bench
        ):
            supporter_purpose = "complete_evolution_chain"
        elif (
            active
            and active.card_id in {ALAKAZAM, DUNSPARCE}
            and DUNSPARCE in {pokemon.card_id for pokemon in facts.yours.field}
            and DUDUNSPARCE not in {pokemon.card_id for pokemon in facts.yours.field}
            and HILDA in facts.yours.hand_ids
        ):
            supporter_purpose = "supply_handoff_engine"
        elif not any(
            pokemon.card_id == ALAKAZAM
            and pokemon.has_energy_type(BASIC_PSYCHIC)
            for pokemon in facts.yours.bench
        ) and any(
            card_id in {ABRA, KADABRA, ALAKAZAM}
            for card_id in facts.yours.discard_ids
        ):
            supporter_purpose = "recover_attack_line"
        elif (
            active
            and (
                (
                    active.card_id in {ABRA, KADABRA}
                    and (
                        ALAKAZAM in facts.yours.hand_ids
                        or HILDA in facts.yours.hand_ids
                    )
                )
                or (
                    active.card_id == ALAKAZAM
                    and not active.has_energy_type(BASIC_PSYCHIC)
                    and HILDA in facts.yours.hand_ids
                )
            )
        ):
            supporter_purpose = "supply_evolution_and_energy"
        elif (
            active
            and active.card_id == ALAKAZAM
            and active.has_energy_type(BASIC_PSYCHIC)
            and active.hp < active.max_hp
            and facts.opponent.hand_count >= 6
            and attack.target
            and attack.expected_damage < attack.target.hp
        ):
            supporter_purpose = "reduce_opponent_hand"

    energy_purpose = None
    if active and active.card_id in {ABRA, KADABRA, ALAKAZAM}:
        if not active.has_energy_type(BASIC_PSYCHIC):
            energy_purpose = "charge_current_attacker"
        elif any(
            pokemon.card_id in {ABRA, KADABRA, ALAKAZAM}
            and not pokemon.has_energy_type(BASIC_PSYCHIC)
            for pokemon in facts.yours.bench
        ):
            energy_purpose = "charge_successor"
    if (
        active
        and active.card_id in {DUNSPARCE, DUDUNSPARCE}
        and routes.handoff.certainty == RouteCertainty.CONFIRMED
    ):
        energy_purpose = "enable_handoff"

    if victory:
        deck_budget = max(0, facts.yours.deck_count)
    elif facts.yours.deck_count > 15:
        deck_budget = facts.yours.deck_count - 15
    elif facts.yours.deck_count > 10:
        deck_budget = 1
    else:
        deck_budget = 0
    blockers = ("must_goals",) if goals and not victory else ()
    return TurnPlan(
        plan_id=_plan_id(facts, routes),
        revision=(previous.revision + 1 if previous else 0),
        kind=kind,
        primary_attack=attack,
        handoff=routes.handoff,
        must_goals=tuple(goals),
        supporter_purpose=supporter_purpose,
        energy_purpose=energy_purpose,
        deck_budget=deck_budget,
        attack_blockers=blockers,
    )
