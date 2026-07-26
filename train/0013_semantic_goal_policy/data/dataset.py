"""Build causal BC records directly from canonical expert-winner episodes."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..features.action_contract import (
    build_occurrence_identities,
    serialize_occurrence_identities,
    unrecognized_option_fields,
    validate_ordered_action,
)
from .records import DecisionIdentity, SourceIdentity
from .source import (
    CanonicalEpisode,
    EpisodeSource,
    iter_canonical_episodes,
    iter_canonical_episodes_parallel,
    normalize_team_identity,
)
from ..protocol import PreRunProtocol
from .split import SplitGroup, SplitManifest, assign_groups

SCHEMA_VERSION = "decision_record_v3"
ACTION_CONTRACT_VERSION = "ordered_full_action_v1"
PROTOCOL_SHA256 = "da6482cf2d4cdc8d9e56fd4c03431e60dbf63b8327720c83185e907a039d44d3"
SOURCE_MANIFEST_SHA256 = "f0ac5c654b82f05ffa3be8fa4e50c6d71f9f132b685217aa40cd9f72cc49818b"
_HEX_DIGITS = frozenset("0123456789abcdef")
_TOP_FIELDS = frozenset(
    {"current", "logs", "remainingOverageTime", "search_begin_input", "select", "step"}
)
_CURRENT_FIELDS = frozenset(
    {
        "energyAttached", "firstPlayer", "looking", "players", "result", "retreated",
        "stadium", "stadiumPlayed", "supporterPlayed", "turn", "turnActionCount", "yourIndex",
    }
)
_PLAYER_FIELDS = frozenset(
    {
        "active", "asleep", "bench", "benchMax", "burned", "confused", "deckCount", "discard",
        "hand", "handCount", "paralyzed", "poisoned", "prize",
    }
)
_ENTITY_FIELDS = frozenset(
    {
        "appearThisTurn", "energies", "energyCards", "hp", "id", "maxHp", "playerIndex",
        "preEvolution", "serial", "tools",
    }
)
_SELECT_FIELDS = frozenset(
    {
        "context", "contextCard", "deck", "effect", "maxCount", "minCount", "option",
        "remainDamageCounter", "remainEnergyCost", "type",
    }
)
_LOG_FIELDS = frozenset(
    {
        "attackId", "cardId", "cardIdActive", "cardIdBench", "cardIdTarget", "fromArea",
        "hasBasicPokemon", "head", "playerIndex", "putDamageCounter", "serial", "serialActive",
        "serialBench", "serialTarget", "toArea", "type", "value",
    }
)
_NONMODEL_TOP_FIELDS = frozenset({"remainingOverageTime", "search_begin_input", "step"})
_DEFAULTS: Mapping[tuple[str, ...], object] = MappingProxyType({
    ("logs",): [],
    ("current", "looking"): [],
    ("current", "stadium"): [],
})


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
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
            raise ValueError("entity energies must be exact integer card IDs")
    for field in ("energyCards", "preEvolution", "tools"):
        if field in entity:
            if not isinstance(entity[field], list):
                raise ValueError(f"entity {field} must be a sequence")
            entity[field] = [_project_entity(item) for item in entity[field]]
    return entity


def _project_entity_sequence(value: object, label: str) -> list[Any]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be a sequence")
    return [_project_entity(item) if isinstance(item, Mapping) else item for item in value]


def _project_observation(value: object) -> dict[str, Any]:
    observation = _project_mapping(value, _TOP_FIELDS, "actor observation")
    for field in _NONMODEL_TOP_FIELDS:
        observation.pop(field, None)
    current = _project_mapping(observation["current"], _CURRENT_FIELDS, "actor current")
    current.pop("result", None)  # post-game signal is never a model input
    players = current.get("players", [])
    if not isinstance(players, list):
        raise ValueError("current.players must be a sequence")
    actor_index = current.get("yourIndex")
    if (
        isinstance(actor_index, bool)
        or not isinstance(actor_index, int)
        or actor_index < 0
        or actor_index >= len(players)
    ):
        raise ValueError("actor observation has invalid player perspective")
    projected_players: list[dict[str, Any]] = []
    for player_index, player in enumerate(players):
        projected = _project_mapping(player, _PLAYER_FIELDS, "actor-visible player")
        if player_index != actor_index:
            for private_zone in ("hand", "prize"):
                zone = projected.get(private_zone, [])
                if zone not in ([], None):
                    raise ValueError(f"opponent private zone {private_zone} exposes card identities")
        for field in ("active", "bench", "discard", "hand", "prize"):
            if field in projected:
                if player_index != actor_index and field in {"hand", "prize"} and projected[field] is None:
                    projected[field] = []
                projected[field] = _project_entity_sequence(projected[field], f"player.{field}")
        projected_players.append(projected)
    current["players"] = projected_players
    for field in ("looking", "stadium"):
        if field in current:
            if current[field] is None:
                current[field] = []
            current[field] = _project_entity_sequence(current[field], f"current.{field}")
    observation["current"] = current

    select = _project_mapping(observation["select"], _SELECT_FIELDS, "actor select")
    options = select.get("option")
    if not isinstance(options, list):
        raise ValueError("select.option must be a sequence")
    projected_options: list[dict[str, Any]] = []
    for option in options:
        raw_option = _mapping(option, "actor option")
        if not all(isinstance(key, str) for key in raw_option):
            raise ValueError("actor option keys must be strings")
        projected_options.append(
            {key: _plain(item) for key, item in raw_option.items() if key in _OPTION_FIELDS_WITH_SERIAL}
        )
    select["option"] = projected_options
    if "contextCard" in select and isinstance(select["contextCard"], Mapping):
        select["contextCard"] = _project_entity(select["contextCard"])
    if "deck" in select:
        select["deck"] = _project_entity_sequence(select["deck"], "select.deck")
    observation["select"] = select

    logs = observation.get("logs", [])
    if not isinstance(logs, list):
        raise ValueError("observation.logs must be a sequence")
    observation["logs"] = [_project_mapping(item, _LOG_FIELDS, "causal log") for item in logs]
    return observation


_OPTION_FIELDS_WITH_SERIAL = frozenset(
    {
        "area", "attackId", "cardId", "count", "energyIndex", "inPlayArea", "inPlayIndex",
        "index", "number", "playerIndex", "serial", "type", "inPlayPlayerIndex",
        "targetPlayerIndex", "toolIndex",
    }
)


@dataclass(frozen=True, slots=True)
class DeckManifest:
    counts: tuple[tuple[int, int], ...]
    sha256: str

    @classmethod
    def from_card_ids(cls, card_ids: Sequence[int]) -> DeckManifest:
        cards = tuple(card_ids)
        if len(cards) != 60:
            raise ValueError("deck manifest must contain exactly 60 cards")
        if not all(isinstance(card, int) and not isinstance(card, bool) and card >= 0 for card in cards):
            raise ValueError("invalid deck card ID")
        counts = tuple(sorted(Counter(cards).items()))
        return cls(counts, hashlib.sha256(_canonical_bytes([list(item) for item in counts])).hexdigest())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeckManifest:
        if set(value) != {"counts", "card_count", "sha256"}:
            raise ValueError("invalid deck manifest schema")
        cards: list[int] = []
        for card_id, count in value["counts"]:
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("invalid deck multiplicity")
            cards.extend([card_id] * count)
        result = cls.from_card_ids(cards)
        if value["card_count"] != 60 or value["sha256"] != result.sha256:
            raise ValueError("deck manifest hash mismatch")
        return result

    def as_dict(self) -> dict[str, object]:
        return {"counts": [list(item) for item in self.counts], "card_count": 60, "sha256": self.sha256}


class RawTerminalOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"
    ERROR = "error"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    identity: DecisionIdentity
    split: str
    deck_manifest: DeckManifest
    actor_observation: Mapping[str, Any]
    legal_options: tuple[Mapping[str, Any], ...]
    option_alignment_audit: tuple[Mapping[str, Any], ...]
    ordered_action: tuple[int, ...]
    action_termination: str
    event_cursor: Mapping[str, int]
    terminal_outcome: RawTerminalOutcome
    decision_duration_ms: float
    episode_duration_ms: float
    source_payload_sha256: str
    schema_version: str = SCHEMA_VERSION
    action_contract_version: str = ACTION_CONTRACT_VERSION

    def as_dict(self) -> dict[str, object]:
        source = self.identity.source
        return {
            "identity": {"date": source.date, "episode_id": source.episode_id,
                         "player_index": source.player_index, "submission_id_unavailable": True,
                         "episode_step": self.identity.episode_step},
            "split": self.split, "deck_manifest": self.deck_manifest.as_dict(),
            "actor_observation": _plain(self.actor_observation),
            "legal_options": _plain(self.legal_options),
            "option_alignment_audit": _plain(self.option_alignment_audit),
            "ordered_action": list(self.ordered_action), "action_termination": self.action_termination,
            "event_cursor": dict(self.event_cursor), "terminal_outcome": self.terminal_outcome.value,
            "decision_duration_ms": self.decision_duration_ms,
            "episode_duration_ms": self.episode_duration_ms,
            "source_payload_sha256": self.source_payload_sha256,
            "schema_version": self.schema_version,
            "action_contract_version": self.action_contract_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionRecord:
        required = {"identity", "split", "deck_manifest", "actor_observation", "legal_options",
                    "option_alignment_audit", "ordered_action", "action_termination", "event_cursor",
                    "terminal_outcome", "decision_duration_ms", "episode_duration_ms",
                    "source_payload_sha256", "schema_version", "action_contract_version"}
        if set(value) != required:
            raise ValueError("invalid decision record schema")
        if value["schema_version"] != SCHEMA_VERSION or value["action_contract_version"] != ACTION_CONTRACT_VERSION:
            raise ValueError("unsupported decision record version")
        source_hash = value["source_payload_sha256"]
        if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
            character not in "0123456789abcdef" for character in source_hash
        ):
            raise ValueError("invalid source payload SHA-256")
        if value["terminal_outcome"] != RawTerminalOutcome.WIN.value:
            raise ValueError("winner-only dataset record must have raw win outcome")
        identity = value["identity"]
        if not isinstance(identity, Mapping) or set(identity) != {"date", "episode_id", "player_index", "submission_id_unavailable", "episode_step"}:
            raise ValueError("invalid decision identity schema")
        source = SourceIdentity(identity["date"], identity["episode_id"], identity["player_index"],
                                identity["submission_id_unavailable"])
        projected = _project_observation(value["actor_observation"])
        raw_options = value["legal_options"]
        if not isinstance(raw_options, (tuple, list)):
            raise ValueError("legal_options must be a sequence")
        projected_options = tuple(
            {key: _plain(item) for key, item in _mapping(option, "actor option").items()
             if key in _OPTION_FIELDS_WITH_SERIAL}
            for option in raw_options
        )
        if projected_options != tuple(projected["select"]["option"]):
            raise ValueError("legal options disagree with actor observation")
        options = tuple(_freeze(item) for item in projected_options)
        select = projected["select"]
        checked = validate_ordered_action(value["ordered_action"], option_count=len(options),
                                          min_count=select["minCount"], max_count=select["maxCount"],
                                          decoder_capacity=max(len(options), len(value["ordered_action"])))
        if checked.termination.value != value["action_termination"]:
            raise ValueError("serialized action termination mismatch")
        expected = tuple(_freeze(item) for item in serialize_occurrence_identities(
            build_occurrence_identities(options)))
        audit = tuple(_freeze(item) for item in value["option_alignment_audit"])
        if expected != audit:
            raise ValueError("serialized occurrence audit mismatch")
        cursor = value["event_cursor"]
        if not isinstance(cursor, Mapping) or set(cursor) != {
            "visual_frame_index", "actor_decision_index", "incoming_log_count"
        } or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                     for item in cursor.values()):
            raise ValueError("invalid event cursor")
        durations = (value["decision_duration_ms"], value["episode_duration_ms"])
        if any(isinstance(item, bool) or not isinstance(item, (int, float))
               or not math.isfinite(item) or item < 0 for item in durations):
            raise ValueError("invalid duration")
        if value["split"] not in {"train", "validation"}:
            raise ValueError("invalid split")
        return cls(DecisionIdentity(source, identity["episode_step"]), value["split"],
                   DeckManifest.from_dict(value["deck_manifest"]), _freeze(projected), options, audit,
                   checked.indices, checked.termination.value, MappingProxyType(dict(cursor)),
                   RawTerminalOutcome(value["terminal_outcome"]), *durations,
                   value["source_payload_sha256"], value["schema_version"],
                   value["action_contract_version"])


def _causal_compare(value: object) -> object:
    projected = _project_observation(value)
    for field in _NONMODEL_TOP_FIELDS:
        projected.pop(field, None)
    for path, default in _DEFAULTS.items():
        target = projected
        for key in path[:-1]:
            child = target.get(key)
            if not isinstance(child, dict):
                child = {}
                target[key] = child
            target = child
        if target.get(path[-1]) is None:
            target[path[-1]] = _plain(default)
    return projected


def _duration(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid duration")
    return float(value)


def _assert_expert_winner(episode: CanonicalEpisode) -> None:
    rewards = episode.payload.get("rewards")
    statuses = episode.payload.get("statuses")
    if not isinstance(rewards, (tuple, list)) or not isinstance(statuses, (tuple, list)):
        raise ValueError("authoritative terminal fields are absent")
    info = episode.payload.get("info")
    teams = info.get("TeamNames") if isinstance(info, Mapping) else None
    if not isinstance(teams, (tuple, list)) or episode.source.player_index >= len(teams) or normalize_team_identity(teams[episode.source.player_index]) != normalize_team_identity("Yushin Ito"):
        raise ValueError("canonical source winner is not the configured expert")
    maximum = max(rewards)
    winners = [index for index, reward in enumerate(rewards) if reward == maximum]
    if winners != [episode.source.player_index] or any(status != "DONE" for status in statuses):
        raise ValueError("canonical source is not the unique completed expert winner")


def _visual_frames(episode: CanonicalEpisode) -> tuple[Mapping[str, Any], ...] | None:
    steps = episode.payload["steps"]
    first = steps[0][0]
    visualize = first.get("visualize") if isinstance(first, Mapping) else None
    if visualize is None:
        return None
    if not isinstance(visualize, (tuple, list)) or len(visualize) != len(steps) - 1:
        raise ValueError("visual frame count must equal steps minus registration")
    frames = tuple(_mapping(frame, "visual frame") for frame in visualize)
    if frames[0].get("selected", object()) is not None:
        raise ValueError("visual frame zero must be registration with selected null")
    return frames


def _registration_decks(episode: CanonicalEpisode, visual: tuple[Mapping[str, Any], ...] | None) -> Any:
    if visual is not None:
        action = visual[0].get("action")
    else:
        steps = episode.payload["steps"]
        if len(steps) < 2:
            raise ValueError("registration frame decks are absent")
        action = [steps[1][index].get("action") for index in range(len(steps[1]))]
    if not isinstance(action, (tuple, list)) or len(action) <= episode.source.player_index:
        raise ValueError("registration frame decks are absent")
    return action[episode.source.player_index]


def _normal(value: object) -> object:
    return json.loads(json.dumps(_plain(value), sort_keys=True, separators=(",", ":")))


def _decision_frames(episode: CanonicalEpisode) -> Iterator[tuple[int, Mapping[str, Any], tuple[int, ...], float]]:
    steps = episode.payload["steps"]
    actor = episode.source.player_index
    visual = _visual_frames(episode)
    if visual is not None:
        for index, frame in enumerate(visual):
            selected = frame.get("selected")
            if index == 0:
                continue
            obs = _mapping(frame.get("obs"), "visual pre-action observation")
            current = _mapping(obs.get("current"), "visual current")
            if current.get("yourIndex") != actor:
                continue
            if selected is None:
                raise ValueError("actor decision visual frame has selected null")
            target = tuple(selected)
            actions = frame.get("action")
            if not isinstance(actions, (tuple, list)) or tuple(actions[actor]) != target:
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
        if member.get("status") != "ACTIVE" or observation is None:
            continue
        if not isinstance(observation, Mapping) or "select" not in observation:
            continue  # registration/non-decision frame
        current = observation.get("current")
        if not isinstance(current, Mapping) or current.get("yourIndex") != actor:
            raise ValueError("raw fallback observation has wrong actor perspective")
        action = steps[index + 1][actor].get("action")
        if action is None:
            continue
        yield index, _mapping(member["observation"], "raw pre-action observation"), tuple(action), _duration(
            steps[index + 1][actor].get("duration"))


def _records(episode: CanonicalEpisode, manifest: SplitManifest, unknown: Counter[str],
             frozen: frozenset[str], capacity: int) -> Iterator[DecisionRecord]:
    _assert_expert_winner(episode)
    visual = _visual_frames(episode)
    deck = DeckManifest.from_card_ids(_registration_decks(episode, visual))
    assignment = manifest.assignment_for(SplitGroup(episode.source.date, deck.sha256,
                                                     episode.source.episode_id,
                                                     episode.source.player_index))
    episode_duration = _duration(episode.payload.get("duration"))
    decision_index = 0
    for frame_index, raw_observation, target, decision_duration in _decision_frames(episode):
        raw_select = _mapping(raw_observation.get("select"), "raw actor select")
        raw_options = raw_select.get("option")
        if not isinstance(raw_options, (tuple, list)):
            raise ValueError("raw select.option must be a sequence")
        counts = unrecognized_option_fields(raw_options)
        unknown.update(counts)
        if not set(counts).issubset(frozen):
            raise ValueError("unknown option field is outside frozen schema")
        observation = _project_observation(raw_observation)
        select = observation["select"]
        options = select["option"]
        ordered = validate_ordered_action(target, option_count=len(options),
                                          min_count=select["minCount"], max_count=select["maxCount"],
                                          decoder_capacity=capacity)
        occurrences = build_occurrence_identities(options)
        cursor = MappingProxyType({"visual_frame_index": frame_index,
                                   "actor_decision_index": decision_index,
                                   "incoming_log_count": len(observation.get("logs", []))})
        yield DecisionRecord(DecisionIdentity(episode.source, frame_index), assignment.split, deck,
                             _freeze(observation), tuple(_freeze(item) for item in options),
                             tuple(_freeze(item) for item in serialize_occurrence_identities(occurrences)),
                             ordered.indices, ordered.termination.value, cursor, RawTerminalOutcome.WIN,
                             decision_duration, episode_duration, episode.payload_sha256)
        decision_index += 1


def iter_records(episode: CanonicalEpisode, split_manifest: SplitManifest, *,
                 frozen_unknown_option_fields: frozenset[str], decoder_capacity: int = 64,
                 audit: Counter[str] | None = None) -> Iterator[DecisionRecord]:
    yield from _records(episode, split_manifest, audit if audit is not None else Counter(),
                        frozen_unknown_option_fields, decoder_capacity)


def _validate_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"invalid {label} SHA-256")
    return value


def _action_type(record: DecisionRecord) -> str:
    options = record.legal_options
    return "+".join(str(options[index].get("type", "missing")) for index in record.ordered_action)


def _comparable_state_key(record: DecisionRecord) -> str:
    value = record.as_dict()
    value.pop("identity")
    value.pop("split")
    value.pop("deck_manifest")
    value.pop("ordered_action")
    value.pop("action_termination")
    value.pop("terminal_outcome")
    value.pop("decision_duration_ms")
    value.pop("episode_duration_ms")
    value.pop("source_payload_sha256")
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class _DistributionAudit:
    def __init__(self) -> None:
        self.groups_by_date: Counter[str] = Counter()
        self.groups_by_deck: Counter[str] = Counter()
        self.decisions_by_date: Counter[str] = Counter()
        self.decisions_by_deck: Counter[str] = Counter()
        self.action_types: Counter[str] = Counter()
        self.select_types: Counter[str] = Counter()
        self.labels: dict[str, set[tuple[str, tuple[int, ...], str]]] = {}

    def begin_group(self, episode: CanonicalEpisode, deck: DeckManifest) -> None:
        self.groups_by_date[episode.source.date] += 1
        self.groups_by_deck[deck.sha256] += 1

    def add(self, record: DecisionRecord) -> None:
        date = record.identity.source.date
        deck_hash = record.deck_manifest.sha256
        self.decisions_by_date[date] += 1
        self.decisions_by_deck[deck_hash] += 1
        self.action_types[_action_type(record)] += 1
        select_type = record.actor_observation["select"].get("type", "missing")
        self.select_types[str(select_type)] += 1
        label = (_action_type(record), record.ordered_action, record.action_termination)
        self.labels.setdefault(_comparable_state_key(record), set()).add(label)

    def as_dict(self) -> dict[str, object]:
        comparable = sum(len(labels) > 1 for labels in self.labels.values())
        conflicting = sum(len(labels) > 1 for labels in self.labels.values())
        return {
            "group_counts_by_date": dict(sorted(self.groups_by_date.items())),
            "group_counts_by_deck_manifest_sha256": dict(sorted(self.groups_by_deck.items())),
            "decision_counts_by_date": dict(sorted(self.decisions_by_date.items())),
            "decision_counts_by_deck_manifest_sha256": dict(sorted(self.decisions_by_deck.items())),
            "decision_counts_by_action_type": dict(sorted(self.action_types.items())),
            "decision_counts_by_select_type": dict(sorted(self.select_types.items())),
            "cross_stratum_conflicts": {
                "comparable_state_groups": comparable,
                "conflicting_groups": conflicting,
                "conflict_rate": conflicting / comparable if comparable else 0.0,
            },
        }


def build_dataset(sources: Iterable[EpisodeSource], output_dir: Path | str, *,
                  split_manifest: SplitManifest, source_manifest_sha256: str,
                  protocol_sha256: str, frozen_unknown_option_fields: frozenset[str],
                  decoder_capacity: int = 64, shard_size: int = 10_000) -> dict[str, Any]:
    from .shards import AtomicShardWriter
    if _validate_sha256(source_manifest_sha256, "source manifest") != SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest SHA-256 does not match the tracked manifest")
    if _validate_sha256(protocol_sha256, "protocol") != PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 does not match the frozen protocol")
    if split_manifest.audit is None or split_manifest.audit.protocol_hash != protocol_sha256:
        raise ValueError("split manifest is not bound to the frozen protocol")
    writer = AtomicShardWriter(Path(output_dir), shard_size=shard_size)
    unknown: Counter[str] = Counter()
    distribution = _DistributionAudit()
    try:
        def sink(episodes: Iterable[CanonicalEpisode]) -> None:
            for episode in episodes:
                visual = _visual_frames(episode)
                deck = DeckManifest.from_card_ids(_registration_decks(episode, visual))
                distribution.begin_group(episode, deck)
                for record in _records(episode, split_manifest, unknown,
                                       frozen_unknown_option_fields, decoder_capacity):
                    distribution.add(record)
                    writer.write(record.split, record)
        source_audit = iter_canonical_episodes(sources, sink)
        metadata = {"source_manifest_sha256": source_manifest_sha256,
                    "source_audit": {"seen": source_audit.seen, "eligible": source_audit.eligible,
                                     "ineligible_noncomplete": source_audit.ineligible_noncomplete,
                                     "spool_bytes": source_audit.spool_bytes},
                    "distribution_audit": distribution.as_dict(),
                    "split_private_assignments_sha256": split_manifest.private_sha256(),
                    "split_audit_sha256": split_manifest.sha256(),
                    "protocol_sha256": protocol_sha256,
                    "record_schema_version": SCHEMA_VERSION,
                    "action_contract_version": ACTION_CONTRACT_VERSION}
        return writer.finalize(dict(unknown), metadata)
    except BaseException:
        writer.abort()
        raise


def build_dataset_with_derived_split(
    sources: Iterable[EpisodeSource],
    output_dir: Path | str,
    *,
    source_manifest_sha256: str,
    protocol: PreRunProtocol,
    frozen_unknown_option_fields: frozenset[str],
    decoder_capacity: int = 64,
    shard_size: int = 10_000,
    source_workers: int = 1,
) -> tuple[dict[str, Any], SplitManifest]:
    """Derive the complete group split and publish records during one validated source scan."""
    from .shards import AtomicShardWriter

    protocol.validate()
    protocol_sha256 = protocol.sha256()
    if _validate_sha256(source_manifest_sha256, "source manifest") != SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest SHA-256 does not match the tracked manifest")
    if protocol_sha256 != PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 does not match the frozen protocol")
    writer = AtomicShardWriter(Path(output_dir), shard_size=shard_size)
    unknown: Counter[str] = Counter()
    distribution = _DistributionAudit()
    private_manifest: SplitManifest | None = None
    try:
        def sink(episodes: Iterable[CanonicalEpisode]) -> None:
            nonlocal private_manifest
            groups: list[SplitGroup] = []
            for episode in episodes:
                deck = DeckManifest.from_card_ids(_registration_decks(episode, _visual_frames(episode)))
                groups.append(SplitGroup(episode.source.date, deck.sha256, episode.source.episode_id, episode.source.player_index))
            private_manifest = assign_groups(groups, protocol)
            for episode in episodes:
                deck = DeckManifest.from_card_ids(_registration_decks(episode, _visual_frames(episode)))
                distribution.begin_group(episode, deck)
                for record in _records(episode, private_manifest, unknown, frozen_unknown_option_fields, decoder_capacity):
                    distribution.add(record)
                    writer.write(record.split, record)
        reader = iter_canonical_episodes_parallel if source_workers > 1 else iter_canonical_episodes
        source_audit = reader(
            sources,
            sink,
            **({"workers": source_workers} if source_workers > 1 else {}),
        )
        if private_manifest is None:
            raise ValueError("source scan did not produce a split manifest")
        metadata = {
            "source_manifest_sha256": source_manifest_sha256,
            "source_audit": {"seen": source_audit.seen, "eligible": source_audit.eligible, "ineligible_noncomplete": source_audit.ineligible_noncomplete, "spool_bytes": source_audit.spool_bytes},
            "distribution_audit": distribution.as_dict(),
            "split_private_assignments_sha256": private_manifest.private_sha256(),
            "split_audit_sha256": private_manifest.sha256(),
            "protocol_sha256": protocol_sha256,
            "record_schema_version": SCHEMA_VERSION,
            "action_contract_version": ACTION_CONTRACT_VERSION,
        }
        return writer.finalize(dict(unknown), metadata), private_manifest
    except BaseException:
        writer.abort()
        raise


__all__ = ["DecisionRecord", "DeckManifest", "RawTerminalOutcome", "build_dataset", "build_dataset_with_derived_split", "iter_records"]
