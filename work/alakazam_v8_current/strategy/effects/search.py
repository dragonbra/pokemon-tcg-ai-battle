from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..cards import (
    ABRA,
    ALAKAZAM,
    BASIC_PSYCHIC,
    DAWN,
    DUNSPARCE,
    DUDUNSPARCE,
    ENRICHING_ENERGY,
    HILDA,
    KADABRA,
    POKE_PAD,
    POFFIN,
    TELEPATH_ENERGY,
)
from ..memory import GameMemory
from ..model import SemanticOption, TurnFacts, TurnPlan


def _first_matching(
    options: Sequence[SemanticOption], priorities: Iterable[int]
) -> tuple[int, ...]:
    for card_id in priorities:
        for option in options:
            if option.card_id == card_id:
                return (option.index,)
    return ()


def _ready_bench_alakazam(facts: TurnFacts) -> bool:
    return any(
        pokemon.card_id == ALAKAZAM and pokemon.has_energy_type(BASIC_PSYCHIC)
        for pokemon in facts.yours.bench
    )


def choose_search(
    effect_id: int,
    options: Sequence[SemanticOption],
    facts: TurnFacts,
    plan: TurnPlan,
    memory: GameMemory,
    *,
    max_count: int,
    effect_serial: int | None,
    explicit_step: int | None,
) -> tuple[int, ...]:
    del plan
    if effect_id == POKE_PAD:
        active = facts.yours.active
        if active and active.card_id == DUNSPARCE and _ready_bench_alakazam(facts):
            return _first_matching(options, (DUDUNSPARCE, ALAKAZAM, KADABRA, ABRA))
        if active and active.card_id == KADABRA:
            return _first_matching(options, (ALAKAZAM, KADABRA, ABRA, DUNSPARCE))
        return _first_matching(options, (KADABRA, ALAKAZAM, ABRA, DUNSPARCE, DUDUNSPARCE))

    if effect_id == DAWN:
        step = (
            explicit_step
            if explicit_step is not None
            else memory.effect_step(effect_serial, effect_id)
        )
        if step == 0:
            field_ids = {pokemon.card_id for pokemon in facts.yours.field}
            priorities = (DUNSPARCE, ABRA) if DUNSPARCE not in field_ids else (ABRA, DUNSPARCE)
        elif step == 1:
            active = facts.yours.active
            priorities = (
                (DUDUNSPARCE, KADABRA)
                if active and active.card_id == DUNSPARCE and _ready_bench_alakazam(facts)
                else (KADABRA, DUDUNSPARCE)
            )
        else:
            priorities = (ALAKAZAM,)
        return _first_matching(options, priorities)

    if effect_id == HILDA:
        field_ids = {pokemon.card_id for pokemon in facts.yours.field}
        if (
            facts.yours.active
            and facts.yours.active.card_id == ALAKAZAM
            and ABRA not in field_ids
            and DUNSPARCE in field_ids
        ):
            return _first_matching(
                options,
                (DUDUNSPARCE, ENRICHING_ENERGY, ALAKAZAM, KADABRA),
            )
        attack_base = next(
            (pokemon for pokemon in facts.yours.field if pokemon.card_id in {ABRA, KADABRA}),
            None,
        )
        if any(option.card_id in {KADABRA, ALAKAZAM, DUDUNSPARCE} for option in options):
            if attack_base and attack_base.card_id == KADABRA:
                return _first_matching(options, (ALAKAZAM, KADABRA, DUDUNSPARCE))
            if attack_base and attack_base.card_id == ABRA:
                return _first_matching(options, (ALAKAZAM, KADABRA, DUDUNSPARCE))
            return _first_matching(options, (KADABRA, DUDUNSPARCE, ALAKAZAM))
        return _first_matching(options, (BASIC_PSYCHIC, TELEPATH_ENERGY))

    if effect_id == POFFIN:
        field_ids = [pokemon.card_id for pokemon in facts.yours.field]
        priorities = [ABRA, DUNSPARCE] if field_ids.count(ABRA) < 2 else [DUNSPARCE, ABRA]
        selected: list[int] = []
        used: set[int] = set()
        while len(selected) < max_count:
            found = None
            for card_id in priorities:
                found = next(
                    (
                        option
                        for option in options
                        if option.index not in used and option.card_id == card_id
                    ),
                    None,
                )
                if found:
                    break
            if not found:
                break
            selected.append(found.index)
            used.add(found.index)
        return tuple(selected)

    if effect_id == TELEPATH_ENERGY:
        return _first_matching(options, (ABRA,))
    return ()
