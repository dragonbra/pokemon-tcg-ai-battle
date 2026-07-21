from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterable, Sequence

from .model import ActionIntent, ActionKind, Area, PokemonRef, SemanticOption, TurnFacts


OPTION_KIND = {
    1: ActionKind.YES,
    2: ActionKind.NO,
    3: ActionKind.SELECT,
    4: ActionKind.SELECT,
    5: ActionKind.SELECT,
    6: ActionKind.SELECT,
    7: ActionKind.PLAY,
    8: ActionKind.ATTACH,
    9: ActionKind.EVOLVE,
    10: ActionKind.ABILITY,
    12: ActionKind.RETREAT,
    13: ActionKind.ATTACK,
    14: ActionKind.END,
}


def _first_int(option: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = option.get(key)
        if isinstance(value, int):
            return value
    return None


def _pokemon_at(
    facts: TurnFacts,
    *,
    owner: int,
    area: int | None,
    index: int | None,
) -> PokemonRef | None:
    if index is None:
        return None
    player = facts.yours if owner == facts.your_index else facts.opponent
    if area == Area.ACTIVE:
        return player.active if index == 0 else None
    if area == Area.BENCH and 0 <= index < len(player.bench):
        return player.bench[index]
    return None


def _selection_card_id(
    option: dict[str, Any], select: dict[str, Any], facts: TurnFacts
) -> int | None:
    if option.get("cardId") is not None:
        return int(option["cardId"])
    option_type = int(option.get("type", -1))
    owner = int(option.get("playerIndex", facts.your_index))
    area = _first_int(option, "area", "inPlayArea")
    index = _first_int(option, "indexInArea", "index")

    if option_type == 7:
        if owner != facts.your_index or index is None:
            return None
        return facts.yours.hand_ids[index] if 0 <= index < len(facts.yours.hand_ids) else None
    if area == Area.DECK:
        deck = select.get("deck") or []
        return (
            int(deck[index]["id"])
            if index is not None
            and 0 <= index < len(deck)
            and deck[index]
            and deck[index].get("id") is not None
            else None
        )
    if area == Area.DISCARD:
        player = facts.yours if owner == facts.your_index else facts.opponent
        return (
            player.discard_ids[index]
            if index is not None and 0 <= index < len(player.discard_ids)
            else None
        )
    pokemon = _pokemon_at(facts, owner=owner, area=area, index=index)
    return pokemon.card_id if pokemon else None


def _decode_option(
    index: int,
    option: dict[str, Any],
    select: dict[str, Any],
    facts: TurnFacts,
) -> SemanticOption:
    raw_type = int(option.get("type", -1))
    action_kind = OPTION_KIND.get(
        raw_type,
        ActionKind.COUNT if raw_type == 0 else ActionKind.UNKNOWN,
    )
    owner = int(option.get("playerIndex", facts.your_index))
    area = _first_int(option, "inPlayArea", "area")
    field_index = _first_int(option, "inPlayIndex", "indexInArea", "index")
    pokemon = _pokemon_at(facts, owner=owner, area=area, index=field_index)
    card_id = _selection_card_id(option, select, facts)
    source = pokemon if action_kind == ActionKind.ABILITY else None
    target = (
        pokemon
        if action_kind in {ActionKind.ATTACH, ActionKind.EVOLVE, ActionKind.SELECT}
        else None
    )

    energy_id = None
    energy_index = option.get("energyIndex")
    if isinstance(energy_index, int) and pokemon and 0 <= energy_index < len(
        pokemon.attached_energy_ids
    ):
        energy_id = pokemon.attached_energy_ids[energy_index]
    elif raw_type == 6:
        energy_id = card_id
    elif action_kind == ActionKind.ATTACH:
        energy_id = card_id

    effect = select.get("effect") or {}
    effect_id = effect.get("id")
    return SemanticOption(
        index=index,
        action_kind=action_kind,
        raw_type=raw_type,
        card_id=card_id,
        source=source,
        target=target,
        energy_id=energy_id,
        attack_id=(
            int(option["attackId"]) if option.get("attackId") is not None else None
        ),
        effect_id=int(effect_id) if effect_id is not None else None,
        owner=owner,
        area=int(area) if area is not None else None,
        raw=MappingProxyType(dict(option)),
    )


def decode_options(select: dict[str, Any], facts: TurnFacts) -> tuple[SemanticOption, ...]:
    return tuple(
        _decode_option(index, option, select, facts)
        for index, option in enumerate(select.get("option") or [])
    )


def match_intent(
    intent: ActionIntent, options: Sequence[SemanticOption]
) -> SemanticOption | None:
    for option in options:
        if option.action_kind != intent.action_kind:
            continue
        if intent.card_id is not None and option.card_id != intent.card_id:
            continue
        if intent.attack_id is not None and option.attack_id != intent.attack_id:
            continue
        option_target = option.target or option.source
        if intent.target_key is not None and (
            option_target is None or option_target.key != intent.target_key
        ):
            continue
        return option
    return None


def options_of_kind(
    options: Iterable[SemanticOption], kind: ActionKind
) -> tuple[SemanticOption, ...]:
    return tuple(option for option in options if option.action_kind == kind)


def required_effect_fallback(
    select: dict[str, Any], options: Sequence[SemanticOption]
) -> tuple[int, ...]:
    min_count = int(select.get("minCount", 0))
    max_count = int(select.get("maxCount", len(options)))
    count = min(max(min_count, 0), max_count, len(options))
    return tuple(option.index for option in options[:count])
