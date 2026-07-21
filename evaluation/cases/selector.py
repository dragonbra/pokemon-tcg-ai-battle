from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ERROR_STATUSES = frozenset(
    {
        "candidate_error",
        "engine_error",
        "illegal_action",
        "opponent_error",
        "visualization_error",
        "worker_crash",
        "worker_error",
    }
)
TARGET_FAILURE_CLASSES = frozenset(
    {
        "rare_candy_not_played",
        "evolution_not_completed",
        "alakazam_not_active",
        "no_legal_attack",
        "attack_not_declared",
    }
)
RESOURCE_FAILURE_CLASSES = frozenset(
    {
        "empty_bench_run_away_draw",
        "library_deck_out",
        "library_pressure",
        "post_ko_zero_ready",
    }
)
RESOURCE_METRIC_IDS = frozenset(
    {
        "library_pressure",
        "post_ko_relay",
        "run_away_draw",
    }
)
MAX_CASES = 3


@dataclass(frozen=True)
class CaseCandidate:
    game_id: str
    opponent: str
    status: str
    failure_class: str
    is_loss: bool
    metric_ids: tuple
    evidence_steps: tuple
    trace_path: Path


def select_cases(candidates: list[CaseCandidate], limit: int = 3) -> list[CaseCandidate]:
    """确定性地选择可解释的错误、目标失败和资源异常回放。"""
    if limit < 0:
        raise ValueError("limit must not be negative")
    if limit == 0:
        return []
    _validate_unique_game_ids(candidates)

    effective_limit = min(limit, MAX_CASES)

    eligible = [candidate for candidate in candidates if _priority(candidate) is not None]
    selected: list[CaseCandidate] = []
    selected_opponents: set[str] = set()
    selected_failures: set[str] = set()

    while eligible and len(selected) < effective_limit:
        selected_candidate = min(
            eligible,
            key=lambda candidate: _selection_key(
                candidate,
                selected_opponents,
                selected_failures,
            ),
        )
        selected.append(selected_candidate)
        selected_opponents.add(selected_candidate.opponent)
        selected_failures.add(_normalized_name(selected_candidate.failure_class))
        eligible.remove(selected_candidate)

    return selected


def case_record(candidate: CaseCandidate) -> dict[str, object]:
    """把候选 case 规整为可直接写入 ``cases.jsonl`` 的记录。"""
    evidence = [_evidence_summary(step) for step in candidate.evidence_steps]
    terminal_evidence = evidence[-1] if evidence else {}
    return {
        "game_id": candidate.game_id,
        "opponent": candidate.opponent,
        "failure_class": candidate.failure_class,
        "metric_ids": _json_value(candidate.metric_ids),
        "evidence": evidence,
        "state_summary": terminal_evidence.get("state_summary"),
        "actual_action": terminal_evidence.get("actual_action"),
        "expected_action": terminal_evidence.get("expected_action"),
        "expected_reason": terminal_evidence.get("expected_reason"),
        "trace_path": str(candidate.trace_path),
    }


def write_case_records(cases: Sequence[CaseCandidate], output_path: Path) -> None:
    """写入 selected cases；空选择生成空文件而不制造虚假 case。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for candidate in cases:
            output_file.write(
                json.dumps(case_record(candidate), ensure_ascii=False, sort_keys=True)
            )
            output_file.write("\n")


def _priority(candidate: CaseCandidate) -> tuple[int, int, int] | None:
    status = _normalized_name(candidate.status)
    failure_class = _normalized_name(candidate.failure_class)
    metric_ids = {str(metric_id).lower() for metric_id in candidate.metric_ids}

    if _is_error(status) or _is_error(failure_class):
        return (0, 0, 0)
    if not candidate.is_loss:
        return None
    if failure_class in TARGET_FAILURE_CLASSES:
        return (1, 0, 0)
    if failure_class in RESOURCE_FAILURE_CLASSES or metric_ids & RESOURCE_METRIC_IDS:
        return (1, 1, 0)
    return None


def _is_error(value: str) -> bool:
    return value in ERROR_STATUSES or value.endswith("_error") or value.endswith("_crash")


def _validate_unique_game_ids(candidates: Sequence[CaseCandidate]) -> None:
    seen_game_ids: set[str] = set()
    duplicate_game_ids: set[str] = set()
    for candidate in candidates:
        if candidate.game_id in seen_game_ids:
            duplicate_game_ids.add(candidate.game_id)
        seen_game_ids.add(candidate.game_id)

    if duplicate_game_ids:
        duplicate_ids = ", ".join(sorted(duplicate_game_ids))
        raise ValueError(f"duplicate game_id: {duplicate_ids}")


def _normalized_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _selection_key(
    candidate: CaseCandidate,
    selected_opponents: set[str],
    selected_failures: set[str],
) -> tuple[int, int, int, bool, bool, str]:
    error_rank, target_failure_rank, diagnostic_rank = _priority(candidate) or (1, 1, 1)
    return (
        error_rank,
        target_failure_rank,
        diagnostic_rank,
        candidate.opponent in selected_opponents,
        _normalized_name(candidate.failure_class) in selected_failures,
        candidate.game_id,
    )


def _evidence_summary(step: object) -> dict[str, object]:
    if not isinstance(step, Mapping):
        return {"detail": _json_value(step)}

    state_summary = step.get("state_summary", step.get("state"))
    actual_action = step.get("actual_action", step.get("action"))
    summary = {
        "step": step.get("step"),
        "turn": step.get("turn"),
        "role": step.get("role"),
        "state_summary": state_summary,
        "actual_action": actual_action,
        "expected_action": step.get("expected_action"),
        "expected_reason": step.get("expected_reason"),
    }
    return {key: _json_value(value) for key, value in summary.items() if value is not None}


def _json_value(value: Any) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
