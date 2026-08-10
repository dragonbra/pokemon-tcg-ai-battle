"""In-memory negative controls for parity-suite sensitivity."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .fixed_snapshot_parity import compare_fixed_snapshot
from .deterministic_transition import compare_primitive_trace


def perturb_option_effect(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(snapshot)
    relations = changed["option_effect_relations"]
    if not isinstance(relations, list) or not relations:
        raise ValueError("option_effect_relations requires a nonempty list")
    first = relations[0]
    if isinstance(first, bool):
        relations[0] = not first
    elif isinstance(first, int):
        relations[0] = first + 1
    else:
        raise ValueError("first option effect relation must be bool/int")
    return changed


def perturb_continuation(step: Mapping[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(step)
    continuation = changed.get("continuation")
    if not isinstance(continuation, list) or not continuation:
        raise ValueError("continuation requires a nonempty list")
    if not isinstance(continuation[0], Mapping):
        raise ValueError("continuation frame must be a mapping")
    continuation[0]["opcode"] = int(continuation[0].get("opcode", 0)) + 1
    return changed


def run_negative_controls(snapshot: Mapping[str, Any], transition: Mapping[str, Any]) -> dict[str, Any]:
    cuda_transition = copy.deepcopy(transition)
    cuda_transition["post_state"]["backend"] = "cuda"
    option_failure = compare_fixed_snapshot(snapshot, perturb_option_effect(snapshot))
    continuation_failure = compare_primitive_trace(
        [transition], [perturb_continuation(cuda_transition)]
    )
    restored_snapshot = compare_fixed_snapshot(snapshot, snapshot)
    restored_transition = compare_primitive_trace([transition], [cuda_transition])
    passed = (
        option_failure.status == "FAIL" and option_failure.first_difference is not None
        and option_failure.first_difference.stage == "option_effect_relations"
        and continuation_failure.status == "FAIL"
        and continuation_failure.first_divergence is not None
        and continuation_failure.first_divergence.stage == "continuation"
        and restored_snapshot.status == "PASS" and restored_transition.status == "PASS"
    )
    return {"status": "PASS" if passed else "FAIL", "option_effect_detected": option_failure.status == "FAIL", "continuation_detected": continuation_failure.status == "FAIL", "restored": restored_snapshot.status == restored_transition.status == "PASS"}


__all__ = ["perturb_continuation", "perturb_option_effect", "run_negative_controls"]
