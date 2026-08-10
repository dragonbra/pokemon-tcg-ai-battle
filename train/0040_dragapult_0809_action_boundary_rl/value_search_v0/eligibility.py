"""Exact Phase 3 handler-shape whitelist and policy-conditioned grouping."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Family(str, Enum):
    BASIC_PLACEMENT = "basic_placement"
    ENERGY_TARGET = "energy_target"
    RETREAT_PAYMENT = "retreat_payment"
    RETREAT_SWITCH_TARGET = "retreat_switch_target"
    OPPONENT_TARGET = "opponent_target"
    ATTACHED_RESOURCE_DESTINATION = "attached_resource_destination"


@dataclass(frozen=True, slots=True)
class CandidateGroup:
    family: Family
    selections: tuple[tuple[int, ...], ...]
    policy_selection: tuple[int, ...]


def _one_policy_option(
    observation: Mapping[str, Any], policy_selection: Sequence[int]
) -> tuple[Mapping[str, Any], Mapping[str, Any], int] | None:
    select = observation.get("select")
    if not isinstance(select, Mapping) or len(policy_selection) != 1:
        return None
    options = select.get("option")
    index = policy_selection[0]
    if (
        not isinstance(options, Sequence)
        or isinstance(options, (str, bytes))
        or isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(options)
        or not isinstance(options[index], Mapping)
    ):
        return None
    return select, options[index], index


def policy_conditioned_group(
    observation: Mapping[str, Any],
    policy_selection: Sequence[int],
    *,
    is_basic_card: Callable[[int], bool],
) -> tuple[CandidateGroup | None, str]:
    """Return only alternatives in the actor-selected audited semantic family."""

    unpacked = _one_policy_option(observation, policy_selection)
    if unpacked is None:
        return None, "UNKNOWN_SELECTION_SHAPE"
    select, chosen, chosen_index = unpacked
    options = select["option"]
    current = observation.get("current")
    if not isinstance(current, Mapping) or current.get("yourIndex") not in (0, 1):
        return None, "UNKNOWN_ACTOR"
    focal = int(current["yourIndex"])
    if select.get("minCount") != 1 or select.get("maxCount") != 1:
        return None, "UNKNOWN_SELECTION_CARDINALITY"

    family: Family | None = None
    candidate_indices: list[int] = []
    select_type = select.get("type")
    context = select.get("context")
    effect = select.get("effect")
    context_card = select.get("contextCard")

    if select_type == 0 and context == 0 and chosen.get("type") == 7:
        players = current.get("players")
        own = players[focal] if isinstance(players, Sequence) and len(players) == 2 else None
        hand = own.get("hand") if isinstance(own, Mapping) else None
        chosen_hand_index = chosen.get("index")
        chosen_card = (
            hand[chosen_hand_index]
            if isinstance(hand, Sequence)
            and isinstance(chosen_hand_index, int)
            and not isinstance(chosen_hand_index, bool)
            and 0 <= chosen_hand_index < len(hand)
            else None
        )
        chosen_id = chosen_card.get("id") if isinstance(chosen_card, Mapping) else None
        if not isinstance(chosen_id, int) or not is_basic_card(chosen_id):
            return None, "POLICY_PLAY_NOT_BASIC"
        family = Family.BASIC_PLACEMENT
        for index, option in enumerate(options):
            if not isinstance(option, Mapping) or option.get("type") != 7:
                continue
            hand_index = option.get("index")
            card = (
                hand[hand_index]
                if isinstance(hand, Sequence)
                and isinstance(hand_index, int)
                and not isinstance(hand_index, bool)
                and 0 <= hand_index < len(hand)
                else None
            )
            card_id = card.get("id") if isinstance(card, Mapping) else None
            if isinstance(card_id, int) and is_basic_card(card_id):
                candidate_indices.append(index)
    elif select_type == 0 and context == 0 and chosen.get("type") == 8:
        family = Family.ENERGY_TARGET
        source = (chosen.get("area"), chosen.get("index"))
        candidate_indices = [
            index
            for index, option in enumerate(options)
            if isinstance(option, Mapping)
            and option.get("type") == 8
            and (option.get("area"), option.get("index")) == source
        ]
    elif (
        select_type == 4
        and context == 30
        and effect is None
        and context_card is None
        and chosen.get("type") == 6
        and current.get("retreated") is True
    ):
        family = Family.RETREAT_PAYMENT
        candidate_indices = [
            index for index, option in enumerate(options)
            if isinstance(option, Mapping)
            and option.get("type") == 6
            and option.get("playerIndex") == focal
        ]
    elif select_type == 1 and context == 3 and chosen.get("type") == 3:
        owners = {
            option.get("playerIndex")
            for option in options if isinstance(option, Mapping)
        }
        if effect is None and context_card is None and owners == {focal}:
            family = Family.RETREAT_SWITCH_TARGET
        elif (
            isinstance(effect, Mapping)
            and effect.get("id") == 1182
            and effect.get("playerIndex") == focal
            and context_card is None
            and owners == {1 - focal}
        ):
            family = Family.OPPONENT_TARGET
        if family is not None:
            candidate_indices = [
                index for index, option in enumerate(options)
                if isinstance(option, Mapping) and option.get("type") == 3
            ]
    elif (
        select_type == 1
        and context == 21
        and chosen.get("type") == 3
        and isinstance(effect, Mapping)
        and effect.get("id") == 1116
        and effect.get("playerIndex") == focal
        and isinstance(context_card, Mapping)
        and context_card.get("playerIndex") == focal
        and all(
            isinstance(option, Mapping)
            and option.get("type") == 3
            and option.get("playerIndex") == focal
            for option in options
        )
    ):
        family = Family.ATTACHED_RESOURCE_DESTINATION
        candidate_indices = list(range(len(options)))

    if family is None:
        return None, "HANDLER_SHAPE_NOT_WHITELISTED"
    if chosen_index not in candidate_indices:
        return None, "POLICY_SELECTION_MAPPING_AMBIGUOUS"
    unique = tuple((index,) for index in dict.fromkeys(candidate_indices))
    if len(unique) < 2:
        return None, "FORCED_OR_SINGLETON"
    return CandidateGroup(family, unique, (chosen_index,)), "ELIGIBLE_SHAPE"


__all__ = ["CandidateGroup", "Family", "policy_conditioned_group"]
