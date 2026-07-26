"""Encode actor-visible observations into named typed tokens."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .schema import (
    DeckToken,
    EntityToken,
    NumericFeature,
    OptionToken,
    Relation,
    RelationType,
    StateToken,
    TypedPolicyInput,
)

_FORBIDDEN = frozenset({"terminal_outcome", "reward", "rewards", "future_frame", "opponent_deck", "value_target", "q_target"})


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(value: object, *, cap: int, missing: bool = False) -> NumericFeature:
    if value is None:
        return NumericFeature.missing() if missing else NumericFeature.unknown()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return NumericFeature.missing()
    return NumericFeature.observed(float(value), cap=float(cap))


def encode_observation(raw: Mapping[str, Any], *, registered_deck: Sequence[int]) -> TypedPolicyInput:
    forbidden = set(raw) & _FORBIDDEN
    if forbidden:
        raise ValueError(f"forbidden policy input fields: {sorted(forbidden)}")
    current = _mapping(raw.get("current"), "current")
    select = _mapping(raw.get("select"), "select")
    players = current.get("players")
    if not isinstance(players, Sequence):
        raise ValueError("players must be a sequence")
    actor = current.get("yourIndex")
    if isinstance(actor, bool) or not isinstance(actor, int) or not 0 <= actor < len(players):
        raise ValueError("invalid actor index")
    state = StateToken(
        "state",
        {"actor": actor, "first_player": current.get("firstPlayer", -1), "select_type": select.get("type", -1), "select_context": select.get("context", -1), "select_effect": select.get("effect", -1)},
        {"turn": _integer(current.get("turn"), cap=1000), "turn_action_count": _integer(current.get("turnActionCount"), cap=1000), "min_count": _integer(select.get("minCount"), cap=64), "max_count": _integer(select.get("maxCount"), cap=64)},
    )
    entities: list[EntityToken] = []
    relations: list[Relation] = []
    for player_index, player_value in enumerate(players):
        player = _mapping(player_value, "player")
        player_id = f"player:{player_index}"
        entities.append(EntityToken(player_id, player_index, "player", None, None, {}, {"deck_count": _integer(player.get("deckCount"), cap=60), "hand_count": _integer(player.get("handCount"), cap=60), "bench_max": _integer(player.get("benchMax"), cap=8)}))
        for zone in ("active", "bench", "hand", "discard", "prize"):
            values = player.get(zone, [])
            if values is None:
                values = []
            if not isinstance(values, Sequence):
                raise ValueError(f"player.{zone} must be a sequence")
            for slot, item_value in enumerate(values):
                if not isinstance(item_value, Mapping):
                    continue
                card_id = item_value.get("id")
                card = card_id if isinstance(card_id, int) and not isinstance(card_id, bool) else None
                serial = item_value.get("serial")
                instance = serial if isinstance(serial, int) and not isinstance(serial, bool) else None
                token_id = f"entity:{player_index}:{zone}:{slot}:{instance if instance is not None else 'unknown'}"
                numeric = {"hp": _integer(item_value.get("hp"), cap=1000), "max_hp": _integer(item_value.get("maxHp"), cap=1000)}
                entities.append(EntityToken(token_id, player_index, zone, card, instance, {"slot": slot}, numeric))
                relations.append(Relation(RelationType.OWNED_BY, token_id, player_id))
                relations.append(Relation(RelationType.LOCATED_IN, token_id, player_id))
    deck_tokens = tuple(DeckToken(f"deck:{card_id}", card_id, count) for card_id, count in sorted(Counter(registered_deck).items()))
    option_values = select.get("option", [])
    if not isinstance(option_values, Sequence):
        raise ValueError("select.option must be a sequence")
    options: list[OptionToken] = []
    for index, option_value in enumerate(option_values):
        option = _mapping(option_value, "option")
        card_id = option.get("cardId")
        card = card_id if isinstance(card_id, int) and not isinstance(card_id, bool) else None
        categorical = {
            key: value
            for key, value in option.items()
            if key not in {"serial", "cardId"}
            and isinstance(value, (str, int))
            and not isinstance(value, bool)
        }
        numeric = {
            key: _integer(option.get(key), cap=128, missing=True)
            for key in ("count", "number", "energyIndex", "toolIndex")
        }
        options.append(OptionToken(
            f"option:{index}", option.get("type", "missing"), card, None, None,
            categorical, numeric,
        ))
    typed = TypedPolicyInput(state, tuple(entities), deck_tokens, (), (), tuple(options), tuple(relations))
    typed.validate()
    return typed


__all__ = ["encode_observation"]
