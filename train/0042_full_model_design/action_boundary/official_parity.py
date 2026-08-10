"""Official-runtime sequential/macro parity for exhaustive canonical allocations."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from evaluation.runtime.seeded import load_seeded_library

from ..league import load_frozen_catalog
from ..rollout.pool_worker import _PointerBattle
from .dragapult import DragapultDamageAllocation, StableTargetIdentity, enumerate_allocations
from .macro_protocol import PendingMacroTransaction


@dataclass(frozen=True, slots=True)
class ParityScenario:
    battle_id: str
    seed: int
    search_seed: int
    focal_deck: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    primitive_action_prefix: tuple[tuple[int, ...], ...]
    targets: tuple[StableTargetIdentity, ...]


def scenario_from_sample(sample: dict[str, Any], focal_deck: tuple[int, ...]) -> ParityScenario:
    catalog = {item.deck_id: item.deck for item in load_frozen_catalog()}
    prefix = tuple(tuple(int(value) for value in row)
                   for row in sample.get("primitive_action_prefix", ()))
    if not prefix:
        raise ValueError("official parity sample has no reproducible primitive prefix")
    return ParityScenario(
        str(sample["battle_id"]), int(sample["seed"]),
        int(sample.get("search_seed") or (((int(sample["seed"]) + 900_000_007) & 0x7FFFFFFF) or 1)),
        focal_deck,
        catalog[str(sample["opponent_id"])], prefix,
        tuple(StableTargetIdentity(**item) for item in sample["target_ids"]),
    )


def _target_action(observation: dict[str, Any], target: StableTargetIdentity) -> list[int]:
    current, selection = observation["current"], observation["select"]
    bench = current["players"][target.player_index]["bench"]
    matches = []
    for index, option in enumerate(selection.get("option") or []):
        if option.get("playerIndex") != target.player_index:
            continue
        slot = option.get("index")
        if isinstance(slot, int) and 0 <= slot < len(bench):
            raw = bench[slot]
            if raw.get("serial") == target.serial and raw.get("id") == target.card_id:
                matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"target {target.serial} has {len(matches)} official legal matches")
    return [matches[0]]


def _run_sequence(scenario: ParityScenario, serial_order: tuple[int, ...], library, lock):
    battle = _PointerBattle(library, lock, [])
    try:
        observation = battle.start(
            scenario.focal_deck, scenario.opponent_deck, scenario.seed, scenario.search_seed
        )
        for action in scenario.primitive_action_prefix:
            observation = battle.select(list(action))
        targets = {item.serial: item for item in scenario.targets}
        primitive = []
        for serial in serial_order:
            action = _target_action(observation, targets[serial])
            primitive.append(tuple(action))
            observation = battle.select(action)
        return observation, tuple(primitive)
    finally:
        battle.finish()


def _macro_sequence(scenario: ParityScenario, allocation: DragapultDamageAllocation,
                    library, lock):
    transaction = PendingMacroTransaction(scenario.battle_id, 0, allocation, timeout_seconds=3600)
    battle = _PointerBattle(library, lock, [])
    try:
        observation = battle.start(
            scenario.focal_deck, scenario.opponent_deck, scenario.seed, scenario.search_seed
        )
        for action in scenario.primitive_action_prefix:
            observation = battle.select(list(action))
        primitive = []
        while not transaction.complete:
            action = transaction.next_primitive(observation, battle_id=scenario.battle_id)
            primitive.append(tuple(action))
            observation = battle.select(action)
        return observation, tuple(primitive)
    finally:
        battle.finish()


def _authoritative_snapshot(observation: dict[str, Any]) -> dict[str, Any]:
    """Return stable official semantics, excluding presentation and opaque serialization.

    The engine's compressed binary search input contains raw bytes from structs whose
    padding is not stable even across two identical seeded primitive replays.  It is
    audited separately below and therefore cannot serve as an exact parity oracle.
    """
    return {
        key: value for key, value in observation.items()
        if key not in {"logs", "search_begin_input"}
    }


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix + ":type"]
    if isinstance(left, dict):
        paths = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                paths.append(f"{prefix}/{key}:missing")
            else:
                paths.extend(_diff_paths(left[key], right[key], f"{prefix}/{key}"))
            if len(paths) >= 20:
                break
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix + ":length"]
        paths = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            paths.extend(_diff_paths(a, b, f"{prefix}/{index}"))
            if len(paths) >= 20:
                break
        return paths
    return [] if left == right else [prefix + ":value"]


def _serialized_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Describe, but never suppress, differences in the official serialized state."""
    def decode(payload: str) -> bytes:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        values = {character: index for index, character in enumerate(alphabet)}
        expanded: list[str] = []
        index = 0
        while index < len(payload):
            marker = payload[index]
            if marker == "A":
                index += 1
                count = values[payload[index]]
            elif marker == "-":
                count = values[payload[index + 1]] + 64 * values[payload[index + 2]]
                index += 2
            elif marker == "*":
                count = (values[payload[index + 1]]
                         + 64 * values[payload[index + 2]]
                         + 64 * 64 * values[payload[index + 3]])
                index += 3
            else:
                expanded.append(marker)
                index += 1
                continue
            expanded.extend("A" for _ in range(count))
            index += 1
        return base64.b64decode("".join(expanded))

    a = decode(left["search_begin_input"])
    b = decode(right["search_begin_input"])
    positions = [index for index, pair in enumerate(zip(a, b, strict=False))
                 if pair[0] != pair[1]]
    return {
        "left_bytes": len(a),
        "right_bytes": len(b),
        "different_bytes_in_overlap": len(positions),
        "first_different_byte": positions[0] if positions else None,
        "last_different_byte": positions[-1] if positions else None,
    }


def exhaustive_scenario_parity(scenario: ParityScenario, engine_library) -> dict[str, int]:
    """Compare macro expansion and a reversed sequential alias for every allocation."""
    import threading
    library = load_seeded_library(engine_library)
    lock = threading.Lock()
    macro_action_failures = macro_state_failures = alias_state_failures = 0
    macro_serialized_differences = alias_serialized_differences = 0
    repeat_serialized_differences = 0
    first_macro_diff: list[str] = []
    first_alias_diff: list[str] = []
    first_macro_serialized_diff: dict[str, Any] = {}
    first_alias_serialized_diff: dict[str, Any] = {}
    allocations = enumerate_allocations(scenario.targets)
    for allocation in allocations:
        canonical_order = allocation.primitive_target_serials()
        macro_state, macro_actions = _macro_sequence(scenario, allocation, library, lock)
        sequential_state, sequential_actions = _run_sequence(
            scenario, canonical_order, library, lock
        )
        repeat_state, repeat_actions = _run_sequence(
            scenario, canonical_order, library, lock
        )
        alias_state, _ = _run_sequence(scenario, tuple(reversed(canonical_order)), library, lock)
        macro_action_failures += int(macro_actions != sequential_actions)
        if repeat_actions != sequential_actions:
            raise AssertionError("identical official primitive sequence changed across replay")
        repeat_serialized_differences += int(
            repeat_state.get("search_begin_input") != sequential_state.get("search_begin_input")
        )
        macro_different = (
            _authoritative_snapshot(macro_state)
            != _authoritative_snapshot(sequential_state)
        )
        alias_different = _authoritative_snapshot(alias_state) != _authoritative_snapshot(sequential_state)
        macro_serialized_different = (
            macro_state.get("search_begin_input") != sequential_state.get("search_begin_input")
        )
        alias_serialized_different = (
            alias_state.get("search_begin_input") != sequential_state.get("search_begin_input")
        )
        macro_state_failures += int(macro_different)
        alias_state_failures += int(alias_different)
        macro_serialized_differences += int(macro_serialized_different)
        alias_serialized_differences += int(alias_serialized_different)
        if macro_different and not first_macro_diff:
            first_macro_diff = _diff_paths(
                _authoritative_snapshot(macro_state),
                _authoritative_snapshot(sequential_state),
            )
        if macro_serialized_different and not first_macro_serialized_diff:
            first_macro_serialized_diff = _serialized_diff(macro_state, sequential_state)
        if alias_different and not first_alias_diff:
            first_alias_diff = _diff_paths(
                _authoritative_snapshot(alias_state), _authoritative_snapshot(sequential_state)
            )
        if alias_serialized_different and not first_alias_serialized_diff:
            first_alias_serialized_diff = _serialized_diff(alias_state, sequential_state)
    failures = macro_action_failures + macro_state_failures + alias_state_failures
    return {"n": len(scenario.targets), "allocations": len(allocations),
            "macro_action_failures": macro_action_failures,
            "macro_state_failures": macro_state_failures,
            "alias_state_failures": alias_state_failures,
            "macro_serialized_differences": macro_serialized_differences,
            "alias_serialized_differences": alias_serialized_differences,
            "repeat_serialized_differences": repeat_serialized_differences,
            "first_macro_diff_paths": first_macro_diff,
            "first_alias_diff_paths": first_alias_diff,
            "first_macro_serialized_diff": first_macro_serialized_diff,
            "first_alias_serialized_diff": first_alias_serialized_diff,
            "parity_failures": failures}


__all__ = [
    "ParityScenario", "exhaustive_scenario_parity", "scenario_from_sample",
]
