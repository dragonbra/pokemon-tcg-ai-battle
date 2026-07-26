"""Load, validate, hash, and immutably freeze the pre-run protocol."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_RUNTIME_FORMULA = "min(10.0, 0.8 * measured_m0_smoke_decisions_per_second)"
_MODELS = ("M0", "M1", "M2", "M3", "M4", "M5")
_TOP_LEVEL_KEYS = {
    "schema_version", "split", "source_precedence", "event_window", "ontology",
    "view_contract", "formal_evaluation", "thresholds", "numerical_health",
    "minimum_formal_allocation", "runtime_floor", "gpu_time", "wandb_snapshot_limits",
}


def _immutable(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_immutable(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _object(data: Mapping[str, Any], key: str, keys: set[str]) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{key} keys invalid; missing={sorted(keys - actual)}, "
            f"unexpected={sorted(actual - keys)}"
        )
    return value


def _exact(value: Any, expected: Any, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{path} must equal {expected!r}")


def _number(value: Any, path: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric, excluding bool")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{path} must be finite and in [{minimum}, {maximum}]")
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _zero_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise ValueError(f"{path} must be integer zero")


def _exact_sequence(value: Any, expected: Sequence[str], path: str) -> None:
    if not isinstance(value, (list, tuple)) or tuple(value) != tuple(expected):
        raise ValueError(f"{path} must equal {list(expected)!r}")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path} entries must be strings")


@dataclass(frozen=True)
class PreRunProtocol:
    """Validated immutable representation of the frozen protocol."""

    data: Mapping[str, Any]

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "data", _immutable(_plain(data)))

    @classmethod
    def from_json(cls, path: str | Path) -> PreRunProtocol:
        with Path(path).open(encoding="utf-8") as source:
            payload = json.load(source, parse_constant=_reject_json_constant)
        if not isinstance(payload, dict):
            raise ValueError("pre-run protocol must be a JSON object")
        protocol = cls(payload)
        protocol.validate()
        return protocol

    def validate(self) -> None:
        actual = set(self.data)
        if actual != _TOP_LEVEL_KEYS:
            raise ValueError(
                f"protocol keys invalid; missing={sorted(_TOP_LEVEL_KEYS - actual)}, "
                f"unexpected={sorted(actual - _TOP_LEVEL_KEYS)}"
            )
        _exact(self.data["schema_version"], "pre_run_protocol_v1", "schema_version")
        self._validate_split()
        self._validate_source_and_views()
        self._validate_evaluation_and_thresholds()
        self._validate_allocation_and_runtime()

    def _validate_split(self) -> None:
        split = _object(self.data, "split", {
            "train_fraction", "validation_fraction", "seed", "group_unit", "stratify_by",
            "assignment", "digest_algorithm", "digest_input",
        })
        train = _number(split["train_fraction"], "split.train_fraction", minimum=0, maximum=1)
        validation = _number(
            split["validation_fraction"], "split.validation_fraction", minimum=0, maximum=1
        )
        if train <= 0 or validation <= 0 or not math.isclose(train + validation, 1.0):
            raise ValueError("split fractions must be positive and sum to 1")
        _exact(split["train_fraction"], 0.9, "split.train_fraction")
        _exact(split["validation_fraction"], 0.1, "split.validation_fraction")
        _exact(split["seed"], 20260726, "split.seed")
        _exact(split["group_unit"], "complete_episode_player", "split.group_unit")
        _exact_sequence(split["stratify_by"], ("date", "deck_manifest_hash"), "split.stratify_by")
        _exact(split["assignment"], "deterministic_digest_rank", "split.assignment")
        _exact(split["digest_algorithm"], "sha256", "split.digest_algorithm")
        _exact(split["digest_input"], (
            "semantic_goal_policy_split_v1|20260726|<date>|<deck_hash>|"
            "<episode_id>|<player>"
        ), "split.digest_input")

    def _validate_source_and_views(self) -> None:
        source = _object(self.data, "source_precedence", {
            "identity", "rule", "ambiguity", "first_date", "last_date", "date_count",
        })
        _exact(source["identity"], "canonical_episode_id", "source_precedence.identity")
        _exact(source["rule"], "patched_episode_replaces_archive_original", "source_precedence.rule")
        _exact(source["ambiguity"], "fail_closed", "source_precedence.ambiguity")
        _exact(source["first_date"], "2026-07-18", "source_precedence.first_date")
        _exact(source["last_date"], "2026-07-25", "source_precedence.last_date")
        _exact(source["date_count"], 8, "source_precedence.date_count")
        event = _object(self.data, "event_window", {"max_events", "selection", "visibility"})
        _exact(_positive_int(event["max_events"], "event_window.max_events"), 64, "event_window.max_events")
        _exact(event["selection"], "newest", "event_window.selection")
        _exact(event["visibility"], "actor_visible", "event_window.visibility")
        ontology = _object(self.data, "ontology", {
            "version", "coverage_population", "card_identity_coverage_min",
            "effect_instance_coverage_min", "residual_effect",
        })
        _exact(ontology["version"], "card_effect_ontology_v1", "ontology.version")
        _exact_sequence(ontology["coverage_population"], (
            "registered_deck_identities", "observed_legal_option_identities",
        ), "ontology.coverage_population")
        for key, expected in (
            ("card_identity_coverage_min", 1.0), ("effect_instance_coverage_min", 0.99)
        ):
            value = _number(ontology[key], f"ontology.{key}", minimum=0, maximum=1)
            if value != expected:
                raise ValueError(f"ontology.{key} must equal {expected}")
        _exact(ontology["residual_effect"], "UNKNOWN_EFFECT", "ontology.residual_effect")
        view = _object(self.data, "view_contract", {"kinds", "exact_prize_inference_requires"})
        _exact_sequence(view["kinds"], (
            "none", "eligible_subset", "full_membership", "ordered_view",
        ), "view_contract.kinds")
        _exact(view["exact_prize_inference_requires"], "verified_full_membership", (
            "view_contract.exact_prize_inference_requires"
        ))

    def _validate_evaluation_and_thresholds(self) -> None:
        evaluation = _object(self.data, "formal_evaluation", {
            "metric_profile", "metric_revision", "games_per_enabled_opponent", "seed_formula",
            "seat_schedule", "games_per_seat",
        })
        exact_values = {
            "metric_profile": "auto_iteration_v8_setup_relay", "metric_revision": 7,
            "seed_formula": "20260726 + game_index", "seat_schedule": "alternating",
        }
        for key, expected in exact_values.items():
            _exact(evaluation[key], expected, f"formal_evaluation.{key}")
        for key, expected in (("games_per_enabled_opponent", 20), ("games_per_seat", 10)):
            _exact(_positive_int(evaluation[key], f"formal_evaluation.{key}"), expected, f"formal_evaluation.{key}")
        thresholds = _object(self.data, "thresholds", {
            "invariance_aligned_agreement_min", "invariance_mean_kl_max",
            "free_running_legality", "free_running_completion",
            "counterfactual_direction_rate_min", "unknown_vs_zero_rate_min",
            "irrelevant_goal_mean_abs_logit_change_max",
        })
        expected_thresholds = {
            "invariance_aligned_agreement_min": 0.999, "invariance_mean_kl_max": 0.0001,
            "free_running_legality": 1.0, "free_running_completion": 1.0,
            "counterfactual_direction_rate_min": 0.8, "unknown_vs_zero_rate_min": 0.8,
            "irrelevant_goal_mean_abs_logit_change_max": 0.05,
        }
        for key, expected in expected_thresholds.items():
            value = _number(thresholds[key], f"thresholds.{key}", minimum=0, maximum=1)
            if value != expected:
                raise ValueError(f"thresholds.{key} must equal {expected}")
        health = _object(self.data, "numerical_health", {
            "nan_inf_count_max", "corrupt_checkpoint_count_max",
            "unacknowledged_amp_overflow_count_max",
        })
        for key, value in health.items():
            _zero_int(value, f"numerical_health.{key}")

    def _validate_allocation_and_runtime(self) -> None:
        allocation = _object(self.data, "minimum_formal_allocation", {
            "models", "seeds", "complete_epochs", "complete_train_passes_min",
            "requires_smoke_pass",
        })
        _exact_sequence(allocation["models"], _MODELS, "minimum_formal_allocation.models")
        for key, expected in (("seeds", 1), ("complete_epochs", 3), ("complete_train_passes_min", 1)):
            _exact(_positive_int(allocation[key], f"minimum_formal_allocation.{key}"), expected, f"minimum_formal_allocation.{key}")
        _exact(allocation["requires_smoke_pass"], True, "minimum_formal_allocation.requires_smoke_pass")
        runtime = _object(self.data, "runtime_floor", {"formula", "units"})
        _exact(runtime["formula"], _RUNTIME_FORMULA, "runtime_floor.formula")
        _exact(runtime["units"], "decisions_per_second", "runtime_floor.units")
        gpu = _object(self.data, "gpu_time", {
            "telemetry_interval_seconds", "active_seconds_target", "merge_gap_seconds_max",
            "intervals", "report_separately",
        })
        for key, expected in (("telemetry_interval_seconds", 10), ("active_seconds_target", 36000), ("merge_gap_seconds_max", 60)):
            _exact(_positive_int(gpu[key], f"gpu_time.{key}"), expected, f"gpu_time.{key}")
        _exact(gpu["intervals"], "non_overlapping_healthy_gpu_work", "gpu_time.intervals")
        _exact_sequence(gpu["report_separately"], ("idle", "failure", "overlap"), "gpu_time.report_separately")
        limits = _object(self.data, "wandb_snapshot_limits", {
            "bytes_per_file_max", "bytes_per_generation_max",
        })
        for key, expected in (("bytes_per_file_max", 1048576), ("bytes_per_generation_max", 4194304)):
            _exact(_positive_int(limits[key], f"wandb_snapshot_limits.{key}"), expected, f"wandb_snapshot_limits.{key}")

    def canonical_bytes(self) -> bytes:
        encoded = json.dumps(
            _plain(self.data), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        return f"{encoded}\n".encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def resolve_runtime_floor(self, measured_m0_smoke_decisions_per_second: float) -> float:
        measured = float(measured_m0_smoke_decisions_per_second)
        if not math.isfinite(measured) or measured <= 0:
            raise ValueError("M0 smoke throughput must be finite and positive")
        return min(10.0, 0.8 * measured)


def freeze_protocol(protocol: PreRunProtocol, destination: str | Path) -> str:
    """Create a canonical protocol file atomically, refusing any existing path."""
    protocol.validate()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(protocol.canonical_bytes())
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination_path.unlink(missing_ok=True)
        raise
    return protocol.sha256()
