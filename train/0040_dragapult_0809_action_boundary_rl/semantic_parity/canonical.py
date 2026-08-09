"""Canonical public records used to localize the first semantic divergence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Iterable


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _zone(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [_plain(item) for item in value]


def _public_player(player: Mapping[str, Any], *, own: bool) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in player.items():
        if key in {"deck", "prize"}:
            public[f"{key}Count"] = len(value) if isinstance(value, Sequence) else value
        elif key == "hand" and not own:
            public["handCount"] = len(value) if isinstance(value, Sequence) else value
        elif key in {"active", "bench", "discard", "lostZone", "stadium"}:
            public[key] = _zone(value)
        elif key == "hand":
            public[key] = _zone(value)
        elif key not in {"private", "hidden", "deckOrder"}:
            public[key] = _plain(value)
    return public


def canonical_public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("observation.current is not a mapping")
    actor = current.get("yourIndex")
    if actor not in (0, 1):
        raise ValueError("observation has no valid public actor")
    players = current.get("players")
    if not isinstance(players, Sequence) or len(players) != 2:
        raise ValueError("observation has no two-player public state")
    canonical_current = {
        str(key): _plain(value)
        for key, value in current.items()
        if key != "players"
    }
    canonical_current["players"] = [
        _public_player(player, own=index == actor)
        if isinstance(player, Mapping) else {"invalid": repr(player)}
        for index, player in enumerate(players)
    ]
    select = observation.get("select")
    result = {
        "current": canonical_current,
        "select": _plain(select),
        "logs": _plain(observation.get("logs") or []),
    }
    return result


def canonical_option_signature(option: Any) -> str:
    return json.dumps(
        _plain(option), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def canonical_legal_set(select: Mapping[str, Any]) -> dict[str, Any]:
    options = select.get("option")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ValueError("select.option is not a sequence")
    ordered = [canonical_option_signature(option) for option in options]
    bounds = {
        "type": select.get("type"),
        "context": select.get("context"),
        "minCount": select.get("minCount"),
        "maxCount": select.get("maxCount"),
        "remainDamageCounter": select.get("remainDamageCounter"),
    }
    return {
        "bounds": bounds,
        "ordered_signatures": ordered,
        "canonical_signatures": sorted(ordered),
        "ordered_hash": stable_hash({"bounds": bounds, "options": ordered}),
        "set_hash": stable_hash({"bounds": bounds, "options": sorted(ordered)}),
    }


@dataclass(frozen=True, slots=True)
class FirstDivergence:
    stage: str
    left_hash: str
    right_hash: str


def first_divergence(
    stages: Iterable[tuple[str, str, str]],
) -> FirstDivergence | None:
    for stage, left, right in stages:
        if left != right:
            return FirstDivergence(str(stage), str(left), str(right))
    return None


__all__ = [
    "FirstDivergence",
    "canonical_legal_set",
    "canonical_option_signature",
    "canonical_public_observation",
    "first_divergence",
    "stable_hash",
]
