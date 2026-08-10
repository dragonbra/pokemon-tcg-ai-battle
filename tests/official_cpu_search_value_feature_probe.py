#!/usr/bin/env python3
"""Phase 3 parity audit for official Search observations and Value inputs.

The script consumes the C++ probe JSONL, forks the mutable 0031 causal feature
context in research code, and compares live versus Search DecisionBatch tensors.
It does not alter or call production inference routing.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
CPU_PROBE = ROOT / "tests/official_cpu_search_same_perspective_probe.py"
PROBE_OUTPUT = ROOT / ".tmp/official_cpu_search_same_perspective/probe_output.jsonl"
SUMMARY_OUTPUT = ROOT / ".tmp/official_cpu_search_same_perspective/feature_summary.json"
VALUE_ARCHIVE = ROOT / "archive/pretrained/0031_friend_0809_gsb_v5_value_v9"
SEMANTIC_PACKAGE = "train.0040_dragapult_0809_action_boundary_rl.semantic_policy"

ELIGIBLE_CASES = (
    "BENCH_PLACEMENT",
    "ENERGY_ATTACH_TARGET",
    "RETREAT_ENERGY_PAYMENT",
    "RETREAT_SWITCH_TARGET",
    "BOSS_OPPONENT_TARGET",
    "TRAINER_EFFECT_TARGET",
)
HIDDEN_COUNTEREXAMPLE = "HIDDEN_DRAW_COUNTEREXAMPLE"


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("PHASE3_JSON\t"):
            continue
        _, case_name, kind, payload = line.split("\t", 3)
        rows.setdefault(case_name, {})[kind] = json.loads(payload)
    return rows


def _search_observation(wrapper: Mapping[str, Any]) -> Mapping[str, Any]:
    if wrapper.get("error") != 0:
        raise ValueError(f"Search wrapper reported error: {wrapper.get('error')}")
    return wrapper["state"]["observation"]


def _clone_knowledge(knowledge: Any) -> Any:
    """Clone every mutable CausalKnowledge field without copying prototypes."""

    cloned = copy.copy(knowledge)
    cloned.initial = Counter(knowledge.initial)
    for name in ("_exact_deck", "_exact_prize"):
        value = getattr(knowledge, name)
        setattr(cloned, name, None if value is None else Counter(value))
    order = knowledge._exact_deck_order
    cloned._exact_deck_order = None if order is None else list(order)
    for name in (
        "_known_opponent_hand",
        "_possible_opponent_hand",
        "_remembered_opponent_cards",
    ):
        setattr(cloned, name, dict(getattr(knowledge, name)))
    # TypedEvent and its mapping-proxy payload are immutable.
    cloned._events = list(knowledge._events)
    return cloned


def _clone_encoder(encoder: Any) -> Any:
    cloned = copy.copy(encoder)
    cloned.knowledge = _clone_knowledge(encoder.knowledge)
    return cloned


def _tensor_differences(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, Any]]:
    if set(left) != set(right):
        return {"__keys__": {"left": sorted(left), "right": sorted(right)}}
    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(left):
        a, b = left[key], right[key]
        if torch.equal(a, b):
            continue
        differences[key] = {
            "left_shape": list(a.shape),
            "right_shape": list(b.shape),
            "different_elements": (
                int(torch.count_nonzero(a != b)) if a.shape == b.shape else None
            ),
        }
    return differences


def _json_difference_paths(left: Any, right: Any, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_json_difference_paths(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{path}.length")
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(_json_difference_paths(a, b, f"{path}[{index}]"))
        return differences
    return [] if left == right else [path]


def _feature_components() -> tuple[type[Any], Any]:
    online = importlib.import_module(f"{SEMANTIC_PACKAGE}.deployment.online_runtime")
    config = importlib.import_module(f"{SEMANTIC_PACKAGE}.model.config")
    return online.OnlineCausalEncoder, config.ModelConfig()


def _value_network() -> Any:
    sys.path.insert(0, str(VALUE_ARCHIVE))
    try:
        loader = importlib.import_module("loader")
        return loader.load_value_network("cpu")
    finally:
        sys.path.pop(0)


def audit(*, with_value_forward: bool) -> dict[str, Any]:
    if not PROBE_OUTPUT.is_file():
        subprocess.run([sys.executable, str(CPU_PROBE)], cwd=ROOT, check=True)
    rows = _load_rows(PROBE_OUTPUT)
    missing = set(ELIGIBLE_CASES + (HIDDEN_COUNTEREXAMPLE,)) - set(rows)
    if missing:
        raise ValueError(f"CPU probe is missing cases: {sorted(missing)}")

    encoder_class, model_config = _feature_components()
    network = _value_network() if with_value_forward else None
    summary: dict[str, Any] = {"cases": {}, "value_forward_loaded": network is not None}

    for case_name in ELIGIBLE_CASES:
        case = rows[case_name]
        root = case["ROOT_OBSERVATION"]
        live = case["LIVE_OBSERVATION"]
        search = _search_observation(case["SEARCH_BRANCH_0"])
        perturbed = _search_observation(case["PERTURBED_SEARCH_BRANCH"])
        deck = case["REGISTERED_DECK"]
        actor = root["current"]["yourIndex"]

        base = encoder_class(actor, deck, model_config)
        base.encode(root)
        base_decision_index = base.knowledge._decision_index
        live_batch = _clone_encoder(base).encode(live)
        search_batch = _clone_encoder(base).encode(search)
        perturbed_batch = _clone_encoder(base).encode(perturbed)
        if base.knowledge._decision_index != base_decision_index:
            raise AssertionError("branch encoding polluted the root causal context")

        live_search_diff = _tensor_differences(live_batch, search_batch)
        hidden_diff = _tensor_differences(search_batch, perturbed_batch)
        value_equal: bool | None = None
        value_outputs: dict[str, list[float]] | None = None
        if network is not None:
            with torch.no_grad():
                live_value = network(live_batch).value
                search_value = network(search_batch).value
            value_equal = torch.equal(live_value, search_value)
            value_outputs = {
                "live": [float(value) for value in live_value.flatten()],
                "search": [float(value) for value in search_value.flatten()],
            }
        raw_paths = _json_difference_paths(live, search)
        summary["cases"][case_name] = {
            "raw_observation_equal": not raw_paths,
            "raw_difference_paths": raw_paths[:24],
            "feature_exact_match": not live_search_diff,
            "feature_differences": live_search_diff,
            "hidden_perturbation_feature_invariant": not hidden_diff,
            "hidden_feature_differences": hidden_diff,
            "value_exact_match": value_equal,
            "value_outputs": value_outputs,
            "root_context_unpolluted": True,
        }
        if live_search_diff or hidden_diff or value_equal is False:
            raise AssertionError(f"eligible feature parity failed: {case_name}")

    counter = rows[HIDDEN_COUNTEREXAMPLE]
    root = counter["ROOT_OBSERVATION"]
    deck = counter["REGISTERED_DECK"]
    actor = root["current"]["yourIndex"]
    base = encoder_class(actor, deck, model_config)
    base.encode(root)
    canonical = _search_observation(counter["SEARCH_BRANCH_0"])
    perturbed = _search_observation(counter["PERTURBED_SEARCH_BRANCH"])
    canonical_batch = _clone_encoder(base).encode(canonical)
    perturbed_batch = _clone_encoder(base).encode(perturbed)
    hidden_differences = _tensor_differences(canonical_batch, perturbed_batch)
    if not hidden_differences:
        raise AssertionError("hidden draw counterexample produced identical Value features")
    summary["cases"][HIDDEN_COUNTEREXAMPLE] = {
        "raw_observation_equal": canonical == perturbed,
        "feature_exact_match": False,
        "hidden_perturbation_feature_invariant": False,
        "hidden_feature_differences": hidden_differences,
        "root_context_unpolluted": True,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-value-forward",
        action="store_true",
        help="also load the exact paired 0036 Value V9 weights and compare outputs",
    )
    args = parser.parse_args()
    summary = audit(with_value_forward=args.with_value_forward)
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("RESULT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
