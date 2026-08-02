"""Strict actor-visible replay projection and ordered-action validation for 0019."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SUPPORTED_OPTION_FIELDS = frozenset(
    {
        "type",
        "playerIndex",
        "area",
        "index",
        "cardId",
        "inPlayPlayerIndex",
        "targetPlayerIndex",
        "inPlayArea",
        "inPlayIndex",
        "number",
        "attackId",
        "count",
        "energyIndex",
        "toolIndex",
        "specialConditionType",
        "serial",
    }
)
_TOP_FIELDS = frozenset(
    {"current", "logs", "remainingOverageTime", "search_begin_input", "select", "step"}
)
_CURRENT_FIELDS = frozenset(
    {
        "energyAttached",
        "firstPlayer",
        "looking",
        "players",
        "result",
        "retreated",
        "stadium",
        "stadiumPlayed",
        "supporterPlayed",
        "turn",
        "turnActionCount",
        "yourIndex",
    }
)
_PLAYER_FIELDS = frozenset(
    {
        "active",
        "asleep",
        "bench",
        "benchMax",
        "burned",
        "confused",
        "deckCount",
        "discard",
        "hand",
        "handCount",
        "paralyzed",
        "poisoned",
        "prize",
    }
)
_ENTITY_FIELDS = frozenset(
    {
        "appearThisTurn",
        "energies",
        "energyCards",
        "hp",
        "id",
        "maxHp",
        "playerIndex",
        "preEvolution",
        "serial",
        "tools",
    }
)
_SELECT_FIELDS = frozenset(
    {
        "context",
        "contextCard",
        "deck",
        "effect",
        "maxCount",
        "minCount",
        "option",
        "remainDamageCounter",
        "remainEnergyCost",
        "type",
    }
)
_LOG_FIELDS = frozenset(
    {
        "attackId",
        "cardId",
        "cardIdActive",
        "cardIdAfter",
        "cardIdBefore",
        "cardIdBench",
        "cardIdTarget",
        "count",
        "energyIndex",
        "fromArea",
        "hasBasicPokemon",
        "head",
        "inPlayArea",
        "inPlayIndex",
        "index",
        "isRecover",
        "number",
        "playerIndex",
        "putDamageCounter",
        "reason",
        "result",
        "serial",
        "serialActive",
        "serialAfter",
        "serialBefore",
        "serialBench",
        "serialTarget",
        "specialConditionType",
        "toArea",
        "toolIndex",
        "type",
        "value",
    }
)


@dataclass(frozen=True, slots=True)
class DeckManifest:
    counts: tuple[tuple[int, int], ...]
    sha256: str

    @classmethod
    def from_card_ids(cls, card_ids: Sequence[int]) -> DeckManifest:
        cards = tuple(card_ids)
        if len(cards) != 60 or not all(
            isinstance(card, int) and not isinstance(card, bool) and card >= 0 for card in cards
        ):
            raise ValueError("registered deck must contain exactly 60 integer card IDs")
        counts = tuple(sorted(Counter(cards).items()))
        encoded = json.dumps(
            [list(item) for item in counts], sort_keys=True, separators=(",", ":")
        ) + "\n"
        return cls(counts, hashlib.sha256(encoded.encode("utf-8")).hexdigest())

    def as_dict(self) -> dict[str, object]:
        return {
            "counts": [list(item) for item in self.counts],
            "card_count": 60,
            "sha256": self.sha256,
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _project_mapping(value: object, allowed: frozenset[str], label: str) -> dict[str, Any]:
    source = _mapping(value, label)
    unknown = set(source) - allowed
    if unknown:
        raise ValueError(f"{label} has unaudited fields: {sorted(unknown)}")
    return {key: _plain(item) for key, item in source.items()}


def _project_entity(value: object) -> dict[str, Any]:
    entity = _project_mapping(value, _ENTITY_FIELDS, "actor-visible entity")
    if "energies" in entity:
        energies = entity["energies"]
        if not isinstance(energies, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in energies
        ):
            raise ValueError("entity energies must contain integer card IDs")
    for field in ("energyCards", "preEvolution", "tools"):
        if field in entity:
            if not isinstance(entity[field], list):
                raise ValueError(f"entity {field} must be a sequence")
            entity[field] = [_project_entity(item) for item in entity[field]]
    return entity


def _project_entities(value: object, label: str) -> list[Any]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be a sequence")
    return [_project_entity(item) if isinstance(item, Mapping) else item for item in value]


def project_observation(value: object) -> dict[str, Any]:
    """Keep only fields available to the acting player and audited by V12."""
    observation = _project_mapping(value, _TOP_FIELDS, "actor observation")
    for field in ("remainingOverageTime", "search_begin_input", "step"):
        observation.pop(field, None)
    current = _project_mapping(observation["current"], _CURRENT_FIELDS, "actor current")
    current.pop("result", None)
    players = current.get("players", [])
    if not isinstance(players, list):
        raise ValueError("current.players must be a sequence")
    actor = current.get("yourIndex")
    if (
        isinstance(actor, bool)
        or not isinstance(actor, int)
        or actor < 0
        or actor >= len(players)
    ):
        raise ValueError("actor observation has invalid player perspective")
    projected_players: list[dict[str, Any]] = []
    for player_index, player in enumerate(players):
        projected = _project_mapping(player, _PLAYER_FIELDS, "actor-visible player")
        if player_index != actor:
            for private_zone in ("hand", "prize"):
                zone = projected.get(private_zone, [])
                opaque = isinstance(zone, list) and all(item is None for item in zone)
                if zone is not None and zone != [] and not opaque:
                    raise ValueError(f"opponent private zone {private_zone} exposes identities")
        for field in ("active", "bench", "discard", "hand", "prize"):
            if field not in projected:
                continue
            if player_index != actor and field in {"hand", "prize"} and projected[field] is None:
                projected[field] = []
            projected[field] = _project_entities(projected[field], f"player.{field}")
        projected_players.append(projected)
    current["players"] = projected_players
    for field in ("looking", "stadium"):
        if field in current:
            current[field] = _project_entities(current[field] or [], f"current.{field}")
    observation["current"] = current

    select = _project_mapping(observation["select"], _SELECT_FIELDS, "actor select")
    options = select.get("option")
    if not isinstance(options, list):
        raise ValueError("select.option must be a sequence")
    projected_options = []
    for option in options:
        raw_option = _mapping(option, "actor option")
        unknown = set(raw_option) - SUPPORTED_OPTION_FIELDS
        if unknown:
            raise ValueError(f"actor option has unaudited fields: {sorted(unknown)}")
        projected_options.append({key: _plain(item) for key, item in raw_option.items()})
    select["option"] = projected_options
    if "contextCard" in select and isinstance(select["contextCard"], Mapping):
        select["contextCard"] = _project_entity(select["contextCard"])
    if "deck" in select:
        select["deck"] = _project_entities(select["deck"] or [], "select.deck")
    observation["select"] = select
    logs = observation.get("logs", [])
    if not isinstance(logs, list):
        raise ValueError("observation.logs must be a sequence")
    observation["logs"] = [_project_mapping(item, _LOG_FIELDS, "causal log") for item in logs]
    return observation


def registration_decks(payload: Mapping[str, Any]) -> list[list[int]]:
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("replay steps are absent")
    first = steps[0][0]
    visualize = first.get("visualize") if isinstance(first, Mapping) else None
    decks = visualize[0].get("action") if isinstance(visualize, list) and visualize else None
    if decks is None:
        decks = [member.get("action") for member in steps[1]]
    if not isinstance(decks, list):
        raise ValueError("registration decks are absent")
    result: list[list[int]] = []
    for deck in decks:
        if not isinstance(deck, list):
            raise ValueError("registration deck must be a sequence")
        DeckManifest.from_card_ids(deck)
        result.append(deck)
    return result


def _duration(value: object) -> float:
    if value is None:
        return 0.0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("invalid duration")
    return float(value)


def _causal_compare(value: object) -> object:
    projected = project_observation(value)
    projected.setdefault("logs", [])
    projected["current"].setdefault("looking", [])
    projected["current"].setdefault("stadium", [])
    return projected


def decision_frames(
    payload: Mapping[str, Any], actor: int
) -> Iterator[tuple[int, Mapping[str, Any], tuple[int, ...], float]]:
    """Yield aligned pre-action observation and the actor's subsequent full action."""
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("replay steps are absent")
    first = steps[0][0] if steps and steps[0] else None
    visualize = first.get("visualize") if isinstance(first, Mapping) else None
    if visualize is not None:
        if (
            not isinstance(visualize, list)
            or not visualize
            or len(visualize) > len(steps) - 1
        ):
            raise ValueError("visual frames exceed the causally alignable replay prefix")
        for index, frame in enumerate(visualize):
            if index == 0:
                if frame.get("selected") is not None:
                    raise ValueError("visual registration selected must be null")
                continue
            selected = frame.get("selected")
            obs = _mapping(frame.get("obs"), "visual pre-action observation")
            if _mapping(obs.get("current"), "visual current").get("yourIndex") != actor:
                continue
            if not isinstance(selected, list):
                raise ValueError("actor decision visual frame has invalid selected action")
            target = tuple(selected)
            actions = frame.get("action")
            if not isinstance(actions, list) or tuple(actions[actor]) != target:
                raise ValueError("visual selected/action actor mismatch")
            raw_observation = steps[index][actor].get("observation")
            if _causal_compare(obs) != _causal_compare(raw_observation):
                raise ValueError("visual/raw causal observation alignment mismatch")
            if tuple(steps[index + 1][actor].get("action")) != target:
                raise ValueError("visual/raw shifted action alignment mismatch")
            yield index, obs, target, _duration(steps[index + 1][actor].get("duration"))
        return
    for index in range(len(steps) - 1):
        member = steps[index][actor]
        observation = member.get("observation")
        if member.get("status") != "ACTIVE" or not isinstance(observation, Mapping):
            continue
        if "select" not in observation:
            continue
        if _mapping(observation.get("current"), "raw current").get("yourIndex") != actor:
            raise ValueError("raw fallback observation has wrong actor perspective")
        action = steps[index + 1][actor].get("action")
        if action is not None:
            yield index, observation, tuple(action), _duration(
                steps[index + 1][actor].get("duration")
            )


def validate_ordered_action(
    indices: Sequence[int], *, option_count: int, min_count: int, max_count: int, capacity: int
) -> tuple[tuple[int, ...], str]:
    values = (option_count, min_count, max_count, capacity)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("action bounds must be exact integers")
    if option_count < 0 or min_count < 0 or max_count < min_count or capacity < 0:
        raise ValueError("invalid action count bounds")
    ordered = tuple(indices)
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in ordered):
        raise ValueError("action indices must be exact integers")
    if len(ordered) > capacity or not min_count <= len(ordered) <= max_count:
        raise ValueError("action length violates bounds")
    if len(set(ordered)) != len(ordered):
        raise ValueError("action indices must be distinct")
    if any(index < 0 or index >= option_count for index in ordered):
        raise ValueError("action index out of range")
    termination = "forced_max" if len(ordered) == max_count else "optional_stop"
    return ordered, termination


__all__ = [
    "DeckManifest",
    "SUPPORTED_OPTION_FIELDS",
    "decision_frames",
    "project_observation",
    "registration_decks",
    "validate_ordered_action",
]
