from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias


Manifest: TypeAlias = dict[str, object]
GameResult: TypeAlias = dict[str, object]
GameRecord: TypeAlias = dict[str, object]
GameMetric: TypeAlias = dict[str, object]
AggregateMetric: TypeAlias = dict[str, object]
CaseRecord: TypeAlias = dict[str, object]

MANIFEST_FIELDS = ("name", "package_hash", "deck_hash", "cg_manifest")
GAME_RESULT_FIELDS = ("winner", "reason")
GAME_RECORD_FIELDS = ("game_id", "candidate", "opponent", "result")
GAME_METRIC_FIELDS = ("game_id", "metric", "value")
AGGREGATE_METRIC_FIELDS = ("metric", "scope", "value", "sample_size")
CASE_RECORD_FIELDS = ("case_id", "name", "status", "details")


def _require_fields(payload: Mapping[str, object], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"{label} is missing required field: {missing[0]}")


def _require_mapping(payload: object, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return payload


def validate_manifest(payload: object) -> None:
    _require_fields(_require_mapping(payload, "manifest"), MANIFEST_FIELDS, "manifest")


def validate_game_record(payload: object) -> None:
    record = _require_mapping(payload, "game record")
    _require_fields(record, GAME_RECORD_FIELDS, "game record")
    result = _require_mapping(record["result"], "game result")
    _require_fields(result, GAME_RESULT_FIELDS, "game result")


def validate_metric_payload(payload: object) -> None:
    metric = _require_mapping(payload, "metric payload")
    fields = GAME_METRIC_FIELDS if "game_id" in metric else AGGREGATE_METRIC_FIELDS
    _require_fields(metric, fields, "metric payload")


def validate_case_record(payload: object) -> None:
    _require_fields(_require_mapping(payload, "case record"), CASE_RECORD_FIELDS, "case record")
