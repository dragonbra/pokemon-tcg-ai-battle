from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .cards import (
    ALAKAZAM,
    ATTACK_DAMAGE,
    ATTACK_ENERGY_COUNT,
    BASIC_PSYCHIC,
    DUDUNSPARCE,
    DUNSPARCE,
    KADABRA,
    POWERFUL_HAND_ATTACK,
    RULE_BOX_POKEMON,
)
from .model import (
    ActionKind,
    AttackRoute,
    HandoffRoute,
    RouteCertainty,
    SemanticOption,
    TurnFacts,
)


@dataclass(frozen=True)
class RouteAnalysis:
    attack: AttackRoute
    handoff: HandoffRoute
    evolution_options: tuple[SemanticOption, ...] = ()
    ability_options: tuple[SemanticOption, ...] = ()


def _damage(facts: TurnFacts, card_id: int, *, after_draw: int = 0) -> int:
    if card_id == ALAKAZAM:
        return max(0, facts.yours.hand_count + after_draw) * 20
    return int(ATTACK_DAMAGE.get(card_id, 0))


def _has_attack_option(options: Sequence[SemanticOption], attack_id: int | None = None) -> bool:
    return any(
        option.action_kind == ActionKind.ATTACK
        and (attack_id is None or option.attack_id == attack_id)
        for option in options
    )


def _ready_attacker(facts: TurnFacts):
    for pokemon in facts.yours.bench:
        if pokemon.card_id == ALAKAZAM and pokemon.has_energy_type(BASIC_PSYCHIC):
            return pokemon
    return None


def analyze_routes(facts: TurnFacts, options: Sequence[SemanticOption]) -> RouteAnalysis:
    active = facts.yours.active
    target = facts.opponent.active
    attack = AttackRoute(
        attacker=active,
        target=target,
        expected_damage=0,
        prize_value=0,
        certainty=RouteCertainty.BLOCKED,
    )
    if active and target:
        attack_id = POWERFUL_HAND_ATTACK if active.card_id == ALAKAZAM else None
        legal_attack = _has_attack_option(options, attack_id)
        expected = _damage(facts, active.card_id)
        energy_ready = (
            active.card_id == ALAKAZAM and active.has_energy_type(BASIC_PSYCHIC)
        ) or (
            active.card_id != ALAKAZAM
            and len(active.energy_types) >= ATTACK_ENERGY_COUNT.get(active.card_id, 1)
        )
        certainty = (
            RouteCertainty.CONFIRMED
            if legal_attack and energy_ready
            else RouteCertainty.BLOCKED
        )
        prize_value = 2 if target.card_id in RULE_BOX_POKEMON else 1
        attack = AttackRoute(
            attacker=active,
            target=target,
            expected_damage=expected,
            prize_value=prize_value,
            certainty=certainty,
            needs_energy=not energy_ready,
        )

    ready = _ready_attacker(facts)
    evolution_options = tuple(
        option for option in options if option.action_kind == ActionKind.EVOLVE
    )
    ability_options = tuple(
        option for option in options if option.action_kind == ActionKind.ABILITY
    )
    via_dudunsparce = bool(
        active
        and active.card_id in {DUNSPARCE, DUDUNSPARCE}
        and ready
        and (
            active.card_id == DUDUNSPARCE
            or any(option.card_id == DUDUNSPARCE for option in evolution_options)
        )
    )
    if via_dudunsparce:
        handoff = HandoffRoute(
            attacker=ready,
            certainty=RouteCertainty.CONFIRMED,
            needs_evolution=active.card_id == DUNSPARCE,
            via_dudunsparce=True,
        )
    elif (
        active
        and active.card_id == DUNSPARCE
        and ready
        and (
            facts.resources.known_in_deck.get(DUDUNSPARCE, 0) > 0
            or facts.resources.unknown_deck_or_prize.get(DUDUNSPARCE, 0) > 0
        )
    ):
        handoff = HandoffRoute(
            attacker=ready,
            certainty=RouteCertainty.POSSIBLE,
            needs_evolution=True,
            needs_search=True,
            via_dudunsparce=True,
        )
    elif ready and any(option.action_kind == ActionKind.RETREAT for option in options):
        handoff = HandoffRoute(attacker=ready, certainty=RouteCertainty.CONFIRMED)
    elif active and active.card_id == DUNSPARCE:
        handoff = HandoffRoute(attacker=ready, certainty=RouteCertainty.BLOCKED)
    else:
        handoff = HandoffRoute(attacker=None, certainty=RouteCertainty.BLOCKED)
    return RouteAnalysis(
        attack=attack,
        handoff=handoff,
        evolution_options=evolution_options,
        ability_options=ability_options,
    )
