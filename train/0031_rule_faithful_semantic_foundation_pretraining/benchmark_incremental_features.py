"""Chronological full/incremental canonical-feature microbenchmark.

The frozen fixture is deliberately actor-local: every timed pass creates fresh
causal knowledge, then consumes a complete trajectory in decision order.  This
keeps the headline number representative of changing official observations
instead of repeatedly compiling one unchanged row.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import torch

from .contracts.fields import SCHEMA_VERSION
from .data.replay_contract import DeckManifest
from .domain.prototypes import PrototypeIndex
from .features.collate import collate_canonical_records
from .features.compiler import compile_canonical_row
from .knowledge.state import CausalKnowledge


BENCHMARK_SCHEMA_VERSION = "0031_incremental_feature_benchmark_v1"
FIXTURE_SCHEMA_VERSION = "0031_incremental_feature_trajectory_v1"
ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).with_name("tests") / "fixtures/incremental_feature_trajectory.json.gz"
ASSETS = Path(__file__).with_name("assets")
PROTOTYPES = PrototypeIndex.load(
    ASSETS / "official_public_prototypes_v1.json",
    ASSETS / "official_full_engine_prototypes_v2.json",
)
REQUIRED_COVERAGE = frozenset(
    {
        "setup",
        "ordinary_main",
        "nested_selection",
        "deck_view",
        "evolution",
        "ability",
        "attack",
        "switch_or_ko_replacement",
        "terminal_near",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class ParityDecision:
    identity: Mapping[str, Any]
    observation: Mapping[str, Any]
    event_cursor: Mapping[str, Any]
    ordered_action: tuple[int, ...]
    action_termination: str
    deck_manifest: Mapping[str, Any]
    split: str

    def row(self) -> dict[str, Any]:
        """Return the raw-row subset consumed by the canonical compiler."""
        return {
            "identity": dict(self.identity),
            "split": self.split,
            "deck_manifest": dict(self.deck_manifest),
            "actor_observation": self.observation,
            "ordered_action": list(self.ordered_action),
            "action_termination": self.action_termination,
            "event_cursor": dict(self.event_cursor),
            "schema_version": "0019_universal_winner_decision_v1",
        }


@dataclass(frozen=True, slots=True)
class ParityTrajectory:
    actor: int
    deck: tuple[int, ...]
    deck_manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    decisions: tuple[ParityDecision, ...]


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
    if len(deck) != 60:
        raise ValueError("fixture deck must contain exactly 60 cards")
    expected = DeckManifest.from_card_ids(deck).as_dict()
    if dict(manifest) != expected:
        raise ValueError("fixture deck manifest commitment mismatch")
    return tuple(deck)


def load_parity_trajectories(path: Path = FIXTURE_PATH) -> tuple[ParityTrajectory, ...]:
    """Load and strictly audit frozen actor-local chronological trajectories."""
    with gzip.open(path, "rt", encoding="ascii") as handle:
        payload = json.load(handle)
    root = _mapping(payload, "fixture root")
    if root.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError("unsupported incremental feature fixture schema")
    claimed = root.get("content_sha256")
    unhashed = dict(root)
    unhashed.pop("content_sha256", None)
    actual = hashlib.sha256(_canonical_bytes(unhashed)).hexdigest()
    if claimed != actual:
        raise ValueError("incremental feature fixture content commitment mismatch")
    source = _mapping(root.get("source"), "fixture source")
    if source.get("split") != "validation" or not source.get("shard_sha256"):
        raise ValueError("fixture must retain audited validation-shard provenance")
    raw_trajectories = root.get("trajectories")
    if not isinstance(raw_trajectories, list) or not raw_trajectories:
        raise ValueError("fixture has no trajectories")

    trajectories: list[ParityTrajectory] = []
    seen: set[tuple[int, int]] = set()
    for raw_trajectory in raw_trajectories:
        item = _mapping(raw_trajectory, "fixture trajectory")
        actor = _integer(item.get("actor"), "fixture actor")
        if actor not in (0, 1):
            raise ValueError("fixture actor must be player 0 or 1")
        provenance = _mapping(item.get("provenance"), "trajectory provenance")
        episode_id = _integer(provenance.get("episode_id"), "fixture episode id")
        player_index = _integer(provenance.get("player_index"), "fixture player index")
        if player_index != actor or (episode_id, actor) in seen:
            raise ValueError("fixture trajectory identity/actor is inconsistent or duplicated")
        seen.add((episode_id, actor))
        manifest = _mapping(item.get("deck_manifest"), "fixture deck manifest")
        deck = _expand_deck(manifest)
        raw_decisions = item.get("decisions")
        if not isinstance(raw_decisions, list) or not raw_decisions:
            raise ValueError("fixture trajectory has no decisions")
        decisions: list[ParityDecision] = []
        previous_step = -1
        for expected_index, raw_decision in enumerate(raw_decisions):
            decision = _mapping(raw_decision, "fixture decision")
            identity = _mapping(decision.get("identity"), "fixture decision identity")
            if (
                _integer(identity.get("episode_id"), "decision episode id") != episode_id
                or _integer(identity.get("player_index"), "decision player index") != actor
            ):
                raise ValueError("fixture decision provenance identity drift")
            step = _integer(identity.get("episode_step"), "decision episode step")
            if step <= previous_step:
                raise ValueError("fixture episode steps are not strictly chronological")
            previous_step = step
            cursor = _mapping(decision.get("event_cursor"), "fixture event cursor")
            if cursor.get("actor_decision_index") != expected_index:
                raise ValueError("fixture actor_decision_index must be exactly 0..N-1")
            observation = _mapping(decision.get("actor_observation"), "fixture observation")
            current = _mapping(observation.get("current"), "fixture current state")
            select = _mapping(observation.get("select"), "fixture selection")
            if current.get("yourIndex") != actor:
                raise ValueError("fixture observation actor perspective drift")
            logs = observation.get("logs")
            options = select.get("option")
            if not isinstance(logs, list) or not isinstance(options, list):
                raise ValueError("fixture logs/options must be lists")
            if cursor.get("incoming_log_count") != len(logs):
                raise ValueError("fixture event cursor log count mismatch")
            action_value = decision.get("ordered_action")
            if not isinstance(action_value, list):
                raise ValueError("fixture ordered action must be a list")
            action = tuple(_integer(value, "fixture action index") for value in action_value)
            if len(action) != len(set(action)) or any(not 0 <= value < len(options) for value in action):
                raise ValueError("fixture ordered action is not a legal unique option sequence")
            minimum = _integer(select.get("minCount"), "fixture minCount")
            maximum = _integer(select.get("maxCount"), "fixture maxCount")
            if not minimum <= len(action) <= maximum:
                raise ValueError("fixture ordered action violates min/max count")
            termination = decision.get("action_termination")
            if not isinstance(termination, str) or not termination:
                raise ValueError("fixture action termination is absent")
            decisions.append(
                ParityDecision(
                    identity=dict(identity),
                    observation=dict(observation),
                    event_cursor=dict(cursor),
                    ordered_action=action,
                    action_termination=termination,
                    deck_manifest=dict(manifest),
                    split="validation",
                )
            )
        trajectories.append(
            ParityTrajectory(
                actor=actor,
                deck=deck,
                deck_manifest=dict(manifest),
                provenance=dict(provenance),
                decisions=tuple(decisions),
            )
        )
    return tuple(trajectories)


def trajectory_coverage(trajectory: ParityTrajectory) -> frozenset[str]:
    """Return coarse scenario evidence without interpreting policy quality."""
    coverage: set[str] = set()
    maximum_turn = max(
        int(decision.observation["current"].get("turn", 0))
        for decision in trajectory.decisions
    )
    for index, decision in enumerate(trajectory.decisions):
        observation = decision.observation
        current = observation["current"]
        select = observation["select"]
        logs = [item for item in observation.get("logs", ()) if isinstance(item, Mapping)]
        options = [item for item in select.get("option", ()) if isinstance(item, Mapping)]
        log_types = {item.get("type") for item in logs}
        option_types = {item.get("type") for item in options}
        turn = int(current.get("turn", 0))
        if turn == 0:
            coverage.add("setup")
        if select.get("type") == 0:
            coverage.add("ordinary_main")
        if select.get("type") != 0 and (
            select.get("effect") is not None
            or select.get("contextCard") is not None
            or select.get("deck")
            or current.get("looking")
        ):
            coverage.add("nested_selection")
        if select.get("deck") or current.get("looking"):
            coverage.add("deck_view")
        if 12 in log_types or 9 in option_types:
            coverage.add("evolution")
        if 10 in option_types:
            coverage.add("ability")
        if 15 in log_types or 13 in option_types:
            coverage.add("attack")
        # Projected engine context 4 is ToActive; log 8 is an observed switch.
        if select.get("context") == 4 or 8 in log_types:
            coverage.add("switch_or_ko_replacement")
        if index == len(trajectory.decisions) - 1 and turn == maximum_turn:
            coverage.add("terminal_near")
    return frozenset(coverage)


def full_rebuild_sequence(
    trajectory: ParityTrajectory,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Compile one complete sequence through the authoritative stateless path."""
    knowledge = CausalKnowledge(trajectory.actor, trajectory.deck)
    batches: list[dict[str, torch.Tensor]] = []
    for decision in trajectory.decisions:
        snapshot = knowledge.consume(decision.observation, decision.event_cursor)
        record = compile_canonical_row(decision.row(), snapshot, PROTOTYPES)
        batches.append(collate_canonical_records([record]))
    return tuple(batches)


def incremental_available() -> bool:
    try:
        module = importlib.import_module(f"{__package__}.features.incremental")
    except ModuleNotFoundError as error:
        if error.name == f"{__package__}.features.incremental":
            return False
        raise
    return hasattr(module, "IncrementalCanonicalCompiler")


def _new_incremental_compiler() -> object:
    module = importlib.import_module(f"{__package__}.features.incremental")
    compiler_type = getattr(module, "IncrementalCanonicalCompiler")
    parameters = inspect.signature(compiler_type).parameters
    if "prototypes" in parameters:
        return compiler_type(prototypes=PROTOTYPES)
    if "prototype_index" in parameters:
        return compiler_type(prototype_index=PROTOTYPES)
    required = [
        parameter
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return compiler_type(PROTOTYPES) if required else compiler_type()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _timing_summary(nanoseconds: Sequence[int]) -> dict[str, float]:
    milliseconds = [value / 1_000_000.0 for value in nanoseconds]
    return {
        "median": float(statistics.median(milliseconds)),
        "p95": _percentile(milliseconds, 0.95),
        "total": float(sum(milliseconds)),
    }


def _measure(
    trajectories: Sequence[ParityTrajectory],
    *,
    mode: str,
    decisions: int,
) -> dict[str, Any]:
    if mode not in {"full", "incremental"}:
        raise ValueError(f"unsupported measured mode: {mode}")
    knowledge_times: list[int] = []
    compile_times: list[int] = []
    collate_times: list[int] = []
    total_times: list[int] = []
    compiled = 0
    while compiled < decisions:
        for trajectory in trajectories:
            knowledge = CausalKnowledge(trajectory.actor, trajectory.deck)
            incremental = _new_incremental_compiler() if mode == "incremental" else None
            for decision in trajectory.decisions:
                if compiled >= decisions:
                    break
                total_start = time.perf_counter_ns()
                started = time.perf_counter_ns()
                snapshot = knowledge.consume(decision.observation, decision.event_cursor)
                knowledge_times.append(time.perf_counter_ns() - started)
                started = time.perf_counter_ns()
                if incremental is None:
                    record = compile_canonical_row(decision.row(), snapshot, PROTOTYPES)
                else:
                    record = incremental.compile(decision.row(), snapshot)  # type: ignore[attr-defined]
                compile_times.append(time.perf_counter_ns() - started)
                started = time.perf_counter_ns()
                collate_canonical_records([record])
                collate_times.append(time.perf_counter_ns() - started)
                total_times.append(time.perf_counter_ns() - total_start)
                compiled += 1
    elapsed_seconds = sum(total_times) / 1_000_000_000.0
    return {
        "compiled_decisions": compiled,
        "decisions_per_second": compiled / elapsed_seconds,
        "milliseconds_per_decision": {
            "knowledge": _timing_summary(knowledge_times),
            "compile": _timing_summary(compile_times),
            "collate": _timing_summary(collate_times),
            "total": _timing_summary(total_times),
        },
    }


def _warm_up(
    trajectories: Sequence[ParityTrajectory], *, mode: str, decisions: int
) -> None:
    if decisions:
        _measure(trajectories, mode=mode, decisions=decisions)


def _validate_incremental_parity(
    trajectories: Sequence[ParityTrajectory],
) -> int:
    checked = 0
    for trajectory in trajectories:
        knowledge = CausalKnowledge(trajectory.actor, trajectory.deck)
        incremental = _new_incremental_compiler()
        for decision in trajectory.decisions:
            snapshot = knowledge.consume(decision.observation, decision.event_cursor)
            full = compile_canonical_row(decision.row(), snapshot, PROTOTYPES)
            candidate = incremental.compile(decision.row(), snapshot)  # type: ignore[attr-defined]
            full_batch = collate_canonical_records([full])
            candidate_batch = collate_canonical_records([candidate])
            if set(full_batch) != set(candidate_batch) or any(
                not torch.equal(full_batch[key], candidate_batch[key]) for key in full_batch
            ):
                raise ValueError(
                    "incremental/full tensor parity failed at "
                    f"Episode {trajectory.provenance['episode_id']} decision {checked}"
                )
            checked += 1
    return checked


def _cpu_identity() -> str:
    try:
        with Path("/proc/cpuinfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run_benchmark(
    *,
    mode: str,
    decisions: int,
    warmup_decisions: int,
    fixture_path: Path = FIXTURE_PATH,
) -> dict[str, Any]:
    if mode not in {"full", "incremental", "compare"}:
        raise ValueError("mode must be full, incremental, or compare")
    if decisions < 1 or warmup_decisions < 0:
        raise ValueError("decisions must be positive and warmup decisions non-negative")
    trajectories = load_parity_trajectories(fixture_path)
    measured_modes = ("full", "incremental") if mode == "compare" else (mode,)
    if "incremental" in measured_modes and not incremental_available():
        raise RuntimeError("incremental compiler is not available")
    parity_decisions = (
        _validate_incremental_parity(trajectories) if "incremental" in measured_modes else 0
    )
    results: dict[str, Any] = {}
    for measured_mode in measured_modes:
        _warm_up(trajectories, mode=measured_mode, decisions=warmup_decisions)
        results[measured_mode] = _measure(
            trajectories, mode=measured_mode, decisions=decisions
        )
    coverage = sorted(set().union(*(trajectory_coverage(item) for item in trajectories)))
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "actor_schema_version": SCHEMA_VERSION,
        "mode": mode,
        "requested_decisions": decisions,
        "warmup_decisions": warmup_decisions,
        "compiled_decisions": decisions,
        "incremental_parity_decisions": parity_decisions,
        "fixture": {
            "path": os.path.relpath(fixture_path.resolve(), ROOT),
            "sha256": _sha256(fixture_path),
            "trajectory_count": len(trajectories),
            "decisions_per_cycle": sum(len(item.decisions) for item in trajectories),
            "coverage": coverage,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu": _cpu_identity(),
            "torch": torch.__version__,
        },
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "incremental", "compare"), default="full")
    parser.add_argument("--decisions", type=int, default=2000)
    parser.add_argument("--warmup-decisions", type=int, default=200)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_benchmark(
        mode=args.mode,
        decisions=args.decisions,
        warmup_decisions=args.warmup_decisions,
        fixture_path=args.fixture,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "FIXTURE_PATH",
    "FIXTURE_SCHEMA_VERSION",
    "PROTOTYPES",
    "ParityDecision",
    "ParityTrajectory",
    "REQUIRED_COVERAGE",
    "full_rebuild_sequence",
    "incremental_available",
    "load_parity_trajectories",
    "run_benchmark",
    "trajectory_coverage",
]
