"""Strict loader for the extended audited 0031-validation parity fixture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from ..data.replay_contract import DeckManifest


FIXTURE_SCHEMA_VERSION = "0035_extended_semantic_trajectories_v1"
SOURCE_DATASET_SCHEMA_VERSION = "0031_rule_faithful_raw_dataset_reference_v1"
SOURCE_ROW_SCHEMA_VERSION = "0019_universal_winner_decision_v1"
SOURCE_SHARD_SHA256 = "9ab50cf5be193b32c039085412fe11fb9ef3884227de3edb779dcb748c7c42bd"
SOURCE_DATASET_CONTENT_SHA256 = "d3f2e679f1d403b52f9f90276d171002430282a05ffd7654ccb3d6066de89dbd"
FIXTURE_FILE_SHA256 = "87aa39bb7dd8023db0bd05afa954416519bc4abf64552d4a4eff270288d96485"
FIXTURE_PATH = Path(__file__).with_name("fixtures") / "extended_semantic_trajectories.json.gz"
DECISION_KEYS = frozenset(
    {
        "identity",
        "split",
        "deck_manifest",
        "actor_observation",
        "ordered_action",
        "action_termination",
        "event_cursor",
        "schema_version",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _expand_deck(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    counts = manifest.get("counts")
    if not isinstance(counts, Sequence):
        raise ValueError("fixture deck manifest has no counts")
    deck: list[int] = []
    for index, item in enumerate(counts):
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"fixture deck count {index} is invalid")
        card_id = _integer(item[0], f"fixture deck card {index}")
        count = _integer(item[1], f"fixture deck count {index}")
        if card_id < 1 or count < 1:
            raise ValueError("fixture deck identities/counts must be positive")
        deck.extend([card_id] * count)
    if dict(manifest) != DeckManifest.from_card_ids(deck).as_dict():
        raise ValueError("fixture deck manifest commitment mismatch")
    return tuple(deck)


@dataclass(frozen=True, slots=True)
class ExtendedDecision:
    raw_row: Mapping[str, Any]

    @property
    def observation(self) -> Mapping[str, Any]:
        return _mapping(self.raw_row["actor_observation"], "actor observation")

    @property
    def event_cursor(self) -> Mapping[str, Any]:
        return _mapping(self.raw_row["event_cursor"], "event cursor")

    def row(self) -> dict[str, Any]:
        return dict(self.raw_row)


@dataclass(frozen=True, slots=True)
class ExtendedTrajectory:
    actor: int
    deck: tuple[int, ...]
    deck_manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    decisions: tuple[ExtendedDecision, ...]


def load_extended_trajectories(
    path: Path = FIXTURE_PATH,
) -> tuple[ExtendedTrajectory, ...]:
    """Load the fixture while rejecting provenance, content, or chronology drift."""
    if path == FIXTURE_PATH and _file_sha256(path) != FIXTURE_FILE_SHA256:
        raise ValueError("extended fixture file commitment mismatch")
    with gzip.open(path, "rt", encoding="ascii") as handle:
        payload = json.load(handle)
    root = _mapping(payload, "fixture root")
    if frozenset(root) != {
        "schema_version",
        "content_sha256",
        "source",
        "selection",
        "trajectories",
    }:
        raise ValueError("extended fixture root contract drift")
    if root.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError("unsupported extended fixture schema")
    unhashed = dict(root)
    claimed_content = unhashed.pop("content_sha256", None)
    actual_content = hashlib.sha256(_canonical_bytes(unhashed)).hexdigest()
    if claimed_content != actual_content:
        raise ValueError("extended fixture content commitment mismatch")

    source = _mapping(root.get("source"), "fixture source")
    expected_source = {
        "dataset_schema_version": SOURCE_DATASET_SCHEMA_VERSION,
        "dataset_content_sha256": SOURCE_DATASET_CONTENT_SHA256,
        "split": "validation",
        "shard": "validation-00000.jsonl.gz",
        "shard_decision_count": 39263,
        "shard_sha256": SOURCE_SHARD_SHA256,
        "row_schema_version": SOURCE_ROW_SCHEMA_VERSION,
    }
    if dict(source) != expected_source:
        raise ValueError("extended fixture audited source commitment mismatch")

    selection = _mapping(root.get("selection"), "fixture selection contract")
    if frozenset(selection) != {
        "trajectory_unit",
        "trajectory_count",
        "decision_count",
        "actor_counts",
        "exact_deck_count",
        "selected_episode_actor_pairs",
    }:
        raise ValueError("extended fixture selection contract drift")
    if selection.get("trajectory_unit") != "complete_actor_local":
        raise ValueError("fixture is not committed to complete actor-local trajectories")
    expected_trajectory_count = _integer(
        selection.get("trajectory_count"), "selection trajectory count"
    )
    expected_decision_count = _integer(
        selection.get("decision_count"), "selection decision count"
    )
    raw_trajectories = root.get("trajectories")
    if not isinstance(raw_trajectories, list):
        raise ValueError("fixture trajectories must be a list")
    if len(raw_trajectories) != expected_trajectory_count:
        raise ValueError("fixture trajectory count commitment mismatch")

    trajectories: list[ExtendedTrajectory] = []
    identities: set[tuple[int, int]] = set()
    actual_decision_count = 0
    for raw_trajectory in raw_trajectories:
        item = _mapping(raw_trajectory, "fixture trajectory")
        if frozenset(item) != {
            "actor",
            "decision_count",
            "deck_manifest",
            "provenance",
            "decisions",
        }:
            raise ValueError("extended fixture trajectory contract drift")
        actor = _integer(item.get("actor"), "trajectory actor")
        if actor not in (0, 1):
            raise ValueError("trajectory actor must be player 0 or 1")
        provenance = _mapping(item.get("provenance"), "trajectory provenance")
        if frozenset(provenance) != {
            "episode_id",
            "player_index",
            "source_id",
            "source_payload_sha256",
            "terminal_outcome",
        }:
            raise ValueError("trajectory provenance contract drift")
        episode_id = _integer(provenance.get("episode_id"), "trajectory episode id")
        player_index = _integer(provenance.get("player_index"), "trajectory player index")
        identity = (episode_id, player_index)
        if player_index != actor or identity in identities:
            raise ValueError("trajectory actor identity drift or duplication")
        identities.add(identity)
        source_id = _integer(provenance.get("source_id"), "trajectory source id")
        if source_id < 1:
            raise ValueError("trajectory source id must be positive")
        source_payload_sha256 = provenance.get("source_payload_sha256")
        if (
            not isinstance(source_payload_sha256, str)
            or len(source_payload_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_payload_sha256)
        ):
            raise ValueError("trajectory source payload commitment is absent")
        if provenance.get("terminal_outcome") != "win":
            raise ValueError("audited source trajectory is not a complete winner trajectory")

        manifest = _mapping(item.get("deck_manifest"), "trajectory deck manifest")
        deck = _expand_deck(manifest)
        raw_decisions = item.get("decisions")
        if not isinstance(raw_decisions, list) or not raw_decisions:
            raise ValueError("trajectory decisions must be a non-empty list")
        declared_count = _integer(item.get("decision_count"), "trajectory decision count")
        if declared_count != len(raw_decisions):
            raise ValueError("trajectory decision count commitment mismatch")

        decisions: list[ExtendedDecision] = []
        previous_step = -1
        for expected_index, raw_decision in enumerate(raw_decisions):
            decision = _mapping(raw_decision, "fixture decision")
            if frozenset(decision) != DECISION_KEYS:
                raise ValueError("fixture decision exceeds the audited compiler row contract")
            if decision.get("split") != "validation":
                raise ValueError("fixture decision is not from the validation split")
            if decision.get("schema_version") != SOURCE_ROW_SCHEMA_VERSION:
                raise ValueError("fixture decision row schema drift")
            if dict(
                _mapping(decision.get("deck_manifest"), "decision deck manifest")
            ) != dict(manifest):
                raise ValueError("fixture decision deck manifest drift")

            row_identity = _mapping(decision.get("identity"), "decision identity")
            if (
                _integer(row_identity.get("episode_id"), "decision episode id") != episode_id
                or _integer(row_identity.get("player_index"), "decision player index") != actor
            ):
                raise ValueError("fixture decision identity drift")
            step = _integer(row_identity.get("episode_step"), "decision episode step")
            if step <= previous_step:
                raise ValueError("fixture episode steps are not strictly chronological")
            previous_step = step

            cursor = _mapping(decision.get("event_cursor"), "decision event cursor")
            if cursor.get("actor_decision_index") != expected_index:
                raise ValueError("actor decision indexes must be exactly 0..N-1")
            observation = _mapping(decision.get("actor_observation"), "decision observation")
            if frozenset(observation) != {"current", "logs", "select"}:
                raise ValueError("actor observation contract drift")
            current = _mapping(observation.get("current"), "decision current state")
            select = _mapping(observation.get("select"), "decision selection")
            if current.get("yourIndex") != actor:
                raise ValueError("actor observation perspective drift")
            logs = observation.get("logs")
            options = select.get("option")
            if not isinstance(logs, list) or not isinstance(options, list):
                raise ValueError("observation logs/options must be lists")
            if cursor.get("incoming_log_count") != len(logs):
                raise ValueError("event cursor log count mismatch")
            action_value = decision.get("ordered_action")
            if not isinstance(action_value, list):
                raise ValueError("ordered action must be a list")
            action = tuple(_integer(value, "ordered action index") for value in action_value)
            if len(action) != len(set(action)) or any(not 0 <= value < len(options) for value in action):
                raise ValueError("ordered action is not a legal unique option sequence")
            minimum = _integer(select.get("minCount"), "selection minCount")
            maximum = _integer(select.get("maxCount"), "selection maxCount")
            if not minimum <= len(action) <= maximum:
                raise ValueError("ordered action violates min/max count")
            if (
                not isinstance(decision.get("action_termination"), str)
                or not decision["action_termination"]
            ):
                raise ValueError("action termination is absent")
            decisions.append(ExtendedDecision(raw_row=dict(decision)))

        actual_decision_count += len(decisions)
        trajectories.append(
            ExtendedTrajectory(
                actor=actor,
                deck=deck,
                deck_manifest=dict(manifest),
                provenance=dict(provenance),
                decisions=tuple(decisions),
            )
        )

    if actual_decision_count != expected_decision_count:
        raise ValueError("fixture total decision count commitment mismatch")
    if {trajectory.actor for trajectory in trajectories} != {0, 1}:
        raise ValueError("fixture must cover both actor seats")
    actor_counts = {
        str(actor): sum(trajectory.actor == actor for trajectory in trajectories)
        for actor in (0, 1)
    }
    if selection.get("actor_counts") != actor_counts:
        raise ValueError("fixture actor count commitment mismatch")
    exact_deck_count = len(
        {trajectory.deck_manifest["sha256"] for trajectory in trajectories}
    )
    if selection.get("exact_deck_count") != exact_deck_count:
        raise ValueError("fixture exact-deck count commitment mismatch")
    selected_pairs = [
        [trajectory.provenance["episode_id"], trajectory.actor]
        for trajectory in trajectories
    ]
    if selection.get("selected_episode_actor_pairs") != selected_pairs:
        raise ValueError("fixture selected trajectory identity commitment mismatch")
    if exact_deck_count < 5:
        raise ValueError("fixture must cover at least five exact decks")
    if expected_trajectory_count < 10 or expected_decision_count < 512:
        raise ValueError("fixture is below the extended semantic coverage floor")
    return tuple(trajectories)


__all__ = [
    "ExtendedDecision",
    "ExtendedTrajectory",
    "FIXTURE_FILE_SHA256",
    "FIXTURE_PATH",
    "load_extended_trajectories",
]
