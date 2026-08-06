"""Fail-closed parity between the evaluated Large Model 0806 package and V3 runtime."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import torch

from evaluation.packages.loader import _clear_cg_modules
from evaluation.runner.worker import _load_game_api_with_runtime

from .policy.actor_critic import SemanticActorCritic
from .semantic_policy.contracts.batch import DecisionBatch
from .semantic_policy.contracts.fields import ACTOR_KEYS, MASK_KEYS, SCHEMA_VERSION, WIDTHS
from .semantic_policy.deployment.online_runtime import OnlineCausalEncoder


EXPECTED_WIDTHS = {
    "global_cat": 12,
    "global_num": 24,
    "card_cat": 9,
    "card_num": 7,
    "resource_cat": 4,
    "resource_num": 15,
    "event_cat": 31,
    "event_num": 4,
    "option_cat": 19,
    "option_num": 2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_full_schema() -> None:
    if SCHEMA_VERSION != "0031_rule_faithful_semantic_decision_v2":
        raise RuntimeError(f"wrong 0031 schema: {SCHEMA_VERSION}")
    actual = {name: getattr(WIDTHS, name) for name in EXPECTED_WIDTHS}
    if actual != EXPECTED_WIDTHS:
        raise RuntimeError(f"0031 width inventory mismatch: {actual}")
    expected_keys = ACTOR_KEYS | MASK_KEYS | {"targets"}
    if len(expected_keys) != 39:
        raise RuntimeError(f"0031 actor key inventory mismatch: {len(expected_keys)}")


def collect_official_observations(
    runtime_root: Path,
    deck0: Sequence[int],
    deck1: Sequence[int],
    *,
    focal_index: int,
    decisions: int = 16,
    seed: int = 330031,
) -> list[dict[str, Any]]:
    if focal_index not in (0, 1) or decisions < 2:
        raise ValueError("invalid parity collection request")
    random.seed(seed)
    _clear_cg_modules()
    game = _load_game_api_with_runtime(runtime_root)
    captured: list[dict[str, Any]] = []
    try:
        observation, _ = game.battle_start(list(deck0), list(deck1))
        for _ in range(2_000):
            current = observation.get("current") or {}
            if isinstance(current.get("result"), int) and current["result"] >= 0:
                break
            if current.get("yourIndex") == focal_index:
                captured.append(observation)
                if len(captured) >= decisions:
                    break
            select = observation.get("select") or {}
            minimum = int(select.get("minCount", 0))
            observation = game.battle_select(list(range(minimum)))
        if len(captured) < decisions:
            raise RuntimeError(
                f"official parity sequence ended after {len(captured)} focal decisions"
            )
        return captured
    finally:
        try:
            game.battle_finish()
        finally:
            _clear_cg_modules()


def _load_evaluated_runtime(candidate_root: Path):
    root = str(candidate_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return (
        importlib.import_module("strategy.deployment.online_runtime").OnlineCausalEncoder,
        importlib.import_module("strategy.deployment.inference").PortableSemanticPolicy,
    )


def assert_large_model_0806_runtime_parity(
    *,
    observations: Sequence[dict[str, Any]],
    actor_index: int,
    deck: tuple[int, ...],
    checkpoint: Path,
    candidate_root: Path,
    model: SemanticActorCritic,
    output: Path | None = None,
) -> dict[str, Any]:
    assert_full_schema()
    package_encoder_type, package_policy_type = _load_evaluated_runtime(candidate_root)
    local_encoder = OnlineCausalEncoder(actor_index, deck, model.actor.config)
    package_policy = package_policy_type.from_checkpoint(checkpoint, deck)
    package_encoder = package_encoder_type(actor_index, deck, package_policy.config)
    package_model = package_policy.model.to(model.device).eval()
    local_state = model.actor.state_dict()
    package_state = package_model.state_dict()
    if set(local_state) != set(package_state) or any(
        not torch.equal(local_state[name], package_state[name]) for name in local_state
    ):
        raise RuntimeError("update-0 Large Model 0806 actor tensors are not bitwise identical")
    compared_tensors = 0
    compared_values = 0
    maximum_logit_error = 0.0
    for decision_index, observation in enumerate(observations):
        local = local_encoder.encode(observation)
        package = package_encoder.encode(observation)
        if set(local) != set(package):
            raise RuntimeError(f"feature keys differ at decision {decision_index}")
        for name in sorted(local):
            if not torch.equal(local[name], package[name]):
                difference = (
                    float((local[name] - package[name]).abs().max())
                    if local[name].dtype.is_floating_point
                    else None
                )
                raise RuntimeError(
                    f"feature mismatch at decision {decision_index} field {name}: {difference}"
                )
            compared_tensors += 1
            compared_values += local[name].numel()
        local_batch = DecisionBatch.from_mapping(local).to(model.device)
        package_batch = package_model.validate_batch(package).to(model.device)
        with torch.inference_mode():
            local_logits = model.actor(local_batch)
            package_logits = package_model(package_batch)
            local_action = model.actor.greedy_action(local_batch)
            package_action = package_model.greedy_action(package_batch)
        logit_error = float((local_logits - package_logits).abs().max())
        maximum_logit_error = max(maximum_logit_error, logit_error)
        if not torch.allclose(local_logits, package_logits, rtol=0.0, atol=2.0e-6):
            raise RuntimeError(
                f"update-0 logits mismatch at decision {decision_index}: "
                f"{logit_error}"
            )
        if local_action != package_action:
            raise RuntimeError(
                f"update-0 greedy action mismatch at decision {decision_index}"
            )
    report = {
        "schema": "0034_large_model_0806_full_semantic_runtime_parity_v1",
        "passed": True,
        "actor_schema": SCHEMA_VERSION,
        "observations": len(observations),
        "tensors_compared": compared_tensors,
        "values_compared": compared_values,
        "checkpoint_sha256": _sha256(checkpoint),
        "candidate_model_sha256": _sha256(candidate_root / "strategy/model.bin"),
        "feature_widths": EXPECTED_WIDTHS,
        "full_actor_keys": sorted(ACTOR_KEYS | MASK_KEYS),
        "actor_tensors_bitwise_identical": True,
        "logits_atol": 2.0e-6,
        "maximum_logit_abs_error": maximum_logit_error,
        "greedy_actions_exact": True,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


__all__ = [
    "EXPECTED_WIDTHS",
    "assert_full_schema",
    "assert_large_model_0806_runtime_parity",
    "collect_official_observations",
]
