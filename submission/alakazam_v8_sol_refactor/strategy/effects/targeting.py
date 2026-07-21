from __future__ import annotations

from collections.abc import Sequence

from ..cards import ALAKAZAM, BASIC_PSYCHIC, PROTECTIVE_DAMAGE_ENERGIES
from ..model import Area, SemanticOption, TurnFacts


def choose_hammer_target(options: Sequence[SemanticOption]) -> tuple[int, ...]:
    def key(option: SemanticOption) -> tuple[int, int]:
        protective = option.energy_id in PROTECTIVE_DAMAGE_ENERGIES
        active = option.target is not None and option.target.area == Area.ACTIVE
        if active and protective:
            tier = 0
        elif protective:
            tier = 1
        elif active:
            tier = 2
        else:
            tier = 3
        return tier, option.index

    ranked = sorted(options, key=key)
    return (ranked[0].index,) if ranked else ()


def choose_switch_target(
    options: Sequence[SemanticOption], facts: TurnFacts
) -> tuple[int, ...]:
    own = [option for option in options if option.owner == facts.your_index]
    if own:
        ready = next(
            (
                option
                for option in own
                if option.target
                and option.target.card_id == ALAKAZAM
                and option.target.has_energy_type(BASIC_PSYCHIC)
            ),
            None,
        )
        return ((ready or own[0]).index,)

    damage = (
        facts.yours.hand_count * 20
        if facts.yours.active and facts.yours.active.card_id == ALAKAZAM
        else 0
    )
    opponent = [option for option in options if option.owner == facts.opponent_index]
    knockouts = [
        option for option in opponent if option.target and damage >= option.target.hp and damage > 0
    ]
    candidates = knockouts or opponent
    if not candidates:
        return ()
    chosen = max(
        candidates,
        key=lambda option: (
            option.target.hp if option.target else 0,
            -option.index,
        ),
    )
    return (chosen.index,)


def choose_rare_candy_target(
    options: Sequence[SemanticOption], facts: TurnFacts
) -> tuple[int, ...]:
    candidates = [
        option
        for option in options
        if option.target and option.target.card_id == 741 and option.target.can_evolve
    ]
    if not candidates:
        return ()
    candidates.sort(
        key=lambda option: (
            0 if option.target and option.target.area == Area.ACTIVE else 1,
            0 if option.target and option.target.has_energy_type(BASIC_PSYCHIC) else 1,
            option.index,
        )
    )
    return (candidates[0].index,)
