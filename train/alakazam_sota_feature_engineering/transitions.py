from __future__ import annotations

from typing import Any


TRANSITION_DIVISORS = (10.0, 10.0, 6.0, 10.0, 1.0, 5.0, 300.0, 5.0, 1.0, 5.0, 300.0)


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _player(current: dict[str, Any], index: int) -> dict[str, Any]:
    players = _list(current.get("players"))
    return players[index] if 0 <= index < len(players) and isinstance(players[index], dict) else {}


def _pokemon(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        value
        for value in [*_list(player.get("active")), *_list(player.get("bench"))]
        if isinstance(value, dict)
    ]


def state_vector(observation: dict[str, Any], actor: int) -> tuple[int, ...]:
    current = observation.get("current") or {}
    own = _player(current, actor)
    opp = _player(current, 1 - actor)
    own_pokemon = _pokemon(own)
    opp_pokemon = _pokemon(opp)
    return (
        _int(own.get("handCount"), len(_list(own.get("hand")))),
        _int(own.get("deckCount")),
        _int(own.get("prizeCount"), len(_list(own.get("prize")))),
        len(_list(own.get("discard"))),
        len(_list(own.get("active"))),
        len(_list(own.get("bench"))),
        sum(max(0, _int(card.get("maxHp"), _int(card.get("hp"))) - _int(card.get("hp"))) for card in own_pokemon),
        sum(len(_list(card.get("energyCards", card.get("energies")))) for card in own_pokemon),
        len(_list(opp.get("active"))),
        len(_list(opp.get("bench"))),
        sum(max(0, _int(card.get("maxHp"), _int(card.get("hp"))) - _int(card.get("hp"))) for card in opp_pokemon),
    )


def transition_label(
    before: dict[str, Any],
    after: dict[str, Any] | None,
    actor: int,
) -> dict[str, Any]:
    if after is None:
        return {
            "transition_delta": [0.0] * len(TRANSITION_DIVISORS),
            "transition_delta_mask": False,
            "transition_turn_changed": 0.0,
            "transition_next_mask": False,
        }
    before_current = before.get("current") or {}
    after_current = after.get("current") or {}
    raw_delta = [
        right - left
        for left, right in zip(state_vector(before, actor), state_vector(after, actor))
    ]
    normalized = [
        max(-2.0, min(2.0, value / divisor))
        for value, divisor in zip(raw_delta, TRANSITION_DIVISORS)
    ]
    turn_changed = _int(before_current.get("turn")) != _int(after_current.get("turn"))
    return {
        "transition_delta": normalized,
        "transition_delta_mask": not turn_changed,
        "transition_turn_changed": float(turn_changed),
        "transition_next_mask": True,
    }
