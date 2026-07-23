from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.cases import CaseCandidate, case_record, select_cases, write_case_records
from evaluation.metrics import GameContext, GameMetric
from evaluation.metrics.profiles import get_metric_profile
from evaluation.metrics.registry import MetricRegistry, create_metric_registry
from evaluation.packages.loader import SubmissionPackage
from evaluation.reporting import ReportData, json_ready, write_report
from evaluation.runner.models import GameRequest, GameResult
from evaluation.traces.store import TraceStore


@dataclass(frozen=True)
class BatchConfig:
    candidate: SubmissionPackage
    opponents: tuple[SubmissionPackage, ...]
    games_per_opponent: int
    output_root: Path
    visualize: bool
    max_steps: int
    control: SubmissionPackage | None = None
    plugin_ids: tuple[str, ...] = ()
    metric_module_paths: tuple[str, ...] = ()
    metric_profile_id: str = "core"
    keep_temp: bool = False
    worker_timeout_seconds: float = 30.0
    workers: int = 1


@dataclass(frozen=True)
class BatchResult:
    run_id: str
    manifest: dict[str, object]
    game_records: tuple
    metric_results: dict[str, object]
    case_records: tuple
    report_data: ReportData


def run_batch(config: BatchConfig) -> BatchResult:
    if config.games_per_opponent < 1:
        raise ValueError("games_per_opponent must be at least one")
    if config.max_steps < 1:
        raise ValueError("max_steps must be at least one")
    if config.worker_timeout_seconds <= 0:
        raise ValueError("worker_timeout_seconds must be greater than zero")
    if config.workers < 1:
        raise ValueError("workers must be at least one")

    run_id = f"run-{uuid.uuid4().hex}"
    report_root = config.output_root.resolve() / run_id
    temp_root = Path(tempfile.gettempdir()) / "evaluation" / run_id
    store = TraceStore(temp_root, report_root)
    registry = _metric_registry(
        config.metric_module_paths,
        config.plugin_ids,
        config.metric_profile_id,
    )
    started_at = _timestamp()
    metric_values: dict[str, list[GameMetric]] = {
        plugin.metric_id: [] for plugin in registry.plugins
    }
    case_candidates: list[CaseCandidate] = []

    try:
        jobs = _game_jobs(config, run_id, store)
        for request, trace_path, result in _run_workers(config, jobs, store.temp_root):
            result, trace = _read_or_create_trace(request, result, trace_path)
            context = GameContext(
                game_id=result.game_id,
                candidate_name=config.candidate.name,
                opponent_name=request.opponent.name,
                candidate_physical_index=result.candidate_physical_index,
                candidate_first=result.candidate_first,
            )
            game_metrics = registry.analyze(_trace_for_metrics(trace, result), context)
            for metric_id, metric in game_metrics.items():
                metric_values[metric_id].append(metric)
            trace["metric_refs"] = _metric_refs(game_metrics)
            store.write_game_record(result, trace)
            case_candidates.append(_case_candidate(result, game_metrics))

        aggregate_metrics = registry.aggregate(metric_values)
        metric_results = {
            metric_id: json_ready(asdict(metric))
            for metric_id, metric in aggregate_metrics.items()
        }
        _add_metric_diagnostics(metric_results, metric_values)
        presentations = registry.present(aggregate_metrics, metric_values)
        presentation_errors = registry.presentation_errors
        selected_cases = select_cases(case_candidates, limit=store.retain_limit)
        retained = store.retain({candidate.game_id for candidate in selected_cases})
        retained_cases = tuple(
            replace(candidate, trace_path=retained[candidate.game_id])
            for candidate in selected_cases
        )
        write_case_records(retained_cases, report_root / "cases.jsonl")
        case_records = tuple(case_record(candidate) for candidate in retained_cases)
        summary = _summary(store.game_records)
        if config.control is not None:
            summary["control"] = _control_summary(config.candidate, config.control)
        finished_at = _timestamp()
        manifest = _manifest(
            config,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            retained=retained,
            metric_ids=tuple(plugin.metric_id for plugin in registry.plugins),
            presentation_errors=presentation_errors,
        )
        (report_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report_data = ReportData(
            manifest=manifest,
            summary=summary,
            games=store.game_records,
            metrics=metric_results,
            cases=case_records,
            metric_profile=manifest["metric_profile"],
            presentations=presentations,
            presentation_errors=presentation_errors,
        )
        (report_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (report_root / "metrics.json").write_text(
            json.dumps(metric_results, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_report(report_data, report_root)
        return BatchResult(
            run_id=run_id,
            manifest=manifest,
            game_records=store.game_records,
            metric_results=metric_results,
            case_records=case_records,
            report_data=report_data,
        )
    finally:
        if not config.keep_temp:
            store.cleanup()
            _cleanup_empty_run_temp_root(temp_root)


def _metric_registry(
    metric_module_paths: tuple[str, ...],
    plugin_ids: tuple[str, ...] = (),
    metric_profile_id: str = "core",
) -> MetricRegistry:
    if plugin_ids and metric_module_paths:
        raise ValueError("plugin_ids and metric_module_paths cannot be used together")
    if plugin_ids:
        raise ValueError(
            "plugin_ids is deprecated because batch always enables core metrics; "
            "use metric_module_paths for extra plugins"
        )
    return create_metric_registry(metric_module_paths, metric_profile_id)


def _game_jobs(
    config: BatchConfig,
    run_id: str,
    store: TraceStore,
) -> list[tuple[GameRequest, Path]]:
    jobs: list[tuple[GameRequest, Path]] = []
    for opponent in config.opponents:
        for game_number in range(1, config.games_per_opponent + 1):
            game_id = f"{opponent.name}-{game_number:03d}"
            request = GameRequest(
                run_id=run_id,
                game_id=game_id,
                candidate=config.candidate,
                opponent=opponent,
                candidate_first=game_number % 2 == 1,
                max_steps=config.max_steps,
                visualize=config.visualize,
            )
            jobs.append((request, store.temp_path(game_id)))
    return jobs


def _run_workers(
    config: BatchConfig,
    jobs: list[tuple[GameRequest, Path]],
    temp_root: Path,
):
    if config.workers == 1:
        for request, trace_path in jobs:
            yield request, trace_path, _run_worker(
                request,
                trace_path,
                temp_root,
                config.worker_timeout_seconds,
            )
        return

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures: list[Future[GameResult]] = [
            executor.submit(
                _run_worker,
                request,
                trace_path,
                temp_root,
                config.worker_timeout_seconds,
            )
            for request, trace_path in jobs
        ]
        for (request, trace_path), future in zip(jobs, futures, strict=True):
            yield request, trace_path, future.result()


def _run_worker(
    request: GameRequest,
    trace_path: Path,
    temp_root: Path,
    timeout_seconds: float,
) -> GameResult:
    request_path = temp_root / f"{request.game_id}.request.json"
    result_path = temp_root / f"{request.game_id}.result.json"
    request_path.write_text(
        json.dumps(_request_payload(request, trace_path), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_path = [str(repository_root)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.runner.worker",
                str(request_path),
                str(result_path),
            ],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return _worker_crash_result(
            request,
            trace_path,
            f"worker timed out after {timeout_seconds:g} seconds",
        )
    except OSError as exc:
        return _worker_crash_result(request, trace_path, f"could not start worker: {exc}")

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return _worker_crash_result(
            request,
            trace_path,
            detail or f"worker exited with code {completed.returncode}",
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return GameResult(
            game_id=str(payload["game_id"]),
            opponent=str(payload["opponent"]),
            candidate_first=bool(payload["candidate_first"]),
            candidate_physical_index=int(payload["candidate_physical_index"]),
            finished=bool(payload["finished"]),
            winner=payload["winner"],
            status=str(payload["status"]),
            error_kind=payload["error_kind"],
            error=payload["error"],
            steps=int(payload["steps"]),
            trace_path=trace_path,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _worker_crash_result(request, trace_path, f"invalid worker result: {exc}")


def _request_payload(request: GameRequest, trace_path: Path) -> dict[str, object]:
    return {
        "run_id": request.run_id,
        "game_id": request.game_id,
        "candidate": _package_payload(request.candidate),
        "opponent": _package_payload(request.opponent),
        "candidate_first": request.candidate_first,
        "max_steps": request.max_steps,
        "visualize": request.visualize,
        "trace_path": str(trace_path),
    }


def _package_payload(package: SubmissionPackage) -> dict[str, object]:
    return {
        "name": package.name,
        "root": str(package.root),
        "deck": package.deck,
        "entrypoint": str(package.entrypoint),
        "package_hash": package.package_hash,
        "deck_hash": package.deck_hash,
        "cg_manifest": package.cg_manifest,
    }


def _worker_crash_result(request: GameRequest, trace_path: Path, error: str) -> GameResult:
    result = GameResult(
        game_id=request.game_id,
        opponent=request.opponent.name,
        candidate_first=request.candidate_first,
        candidate_physical_index=0 if request.candidate_first else 1,
        finished=False,
        winner=None,
        status="worker_crash",
        error_kind="worker_crash",
        error=error,
        steps=0,
        trace_path=trace_path,
    )
    trace_path.write_text(
        json.dumps({"trace": [], "result": _result_payload(result)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _read_or_create_trace(
    request: GameRequest,
    result: GameResult,
    trace_path: Path,
) -> tuple[GameResult, dict[str, Any]]:
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _invalid_worker_trace(request, trace_path, "worker trace is missing")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _invalid_worker_trace(request, trace_path, f"worker trace is malformed: {exc}")
    if not isinstance(payload, dict):
        return _invalid_worker_trace(request, trace_path, "worker trace is not a JSON object")
    if not isinstance(payload.get("trace"), list) or not isinstance(payload.get("result"), dict):
        return _invalid_worker_trace(
            request,
            trace_path,
            "worker trace is malformed: expected trace list and result object",
        )
    missing_fields = _missing_game_result_fields(payload["result"])
    if missing_fields:
        return _invalid_worker_trace(
            request,
            trace_path,
            f"worker trace is malformed: result is missing required field: {missing_fields[0]}",
        )
    mismatch = _nested_result_mismatch(payload["result"], result)
    if mismatch is not None:
        return _invalid_worker_trace(
            request,
            trace_path,
            f"worker trace is malformed: result.{mismatch} disagrees with worker result",
        )
    payload["result"].pop("trace_path", None)
    return result, payload


def _invalid_worker_trace(
    request: GameRequest,
    trace_path: Path,
    error: str,
) -> tuple[GameResult, dict[str, Any]]:
    result = _worker_crash_result(request, trace_path, error)
    trace = {"trace": [], "result": _result_payload(result)}
    trace["result"].pop("trace_path", None)
    return result, trace


def _trace_for_metrics(trace: dict[str, Any], result: GameResult) -> dict[str, Any]:
    metric_trace = dict(trace)
    metric_trace.update(_result_payload(result))
    metric_trace["reason"] = result.error_kind or ""
    return metric_trace


def _metric_refs(game_metrics: dict[str, GameMetric]) -> dict[str, dict[str, object]]:
    return {
        metric_id: {
            "status": metric.status,
            "numerator": metric.numerator,
            "denominator": metric.denominator,
            "value": metric.value,
            "payload": _metric_ref_payload(metric.payload),
        }
        for metric_id, metric in game_metrics.items()
    }


def _metric_ref_payload(payload: object) -> object:
    """保留单局计数与审计摘要，把详细事件留在 retained trace/聚合结果。"""
    value = json_ready(payload)
    if not isinstance(value, dict):
        return value
    compact = dict(value)
    for key in ("attacks", "events"):
        details = compact.pop(key, None)
        if isinstance(details, list):
            compact[f"{key}_count"] = len(details)
    return compact


def _add_metric_diagnostics(
    metric_results: dict[str, dict[str, object]],
    metric_values: dict[str, list[GameMetric]],
) -> None:
    """保留报告层需要的 failure_class 分布，不改变核心聚合字段。"""
    correctness_results = metric_values.get("correctness", [])
    failure_classes: dict[str, int] = {}
    for result in correctness_results:
        failure_class = str(result.value)
        if failure_class != "ok":
            failure_classes[failure_class] = failure_classes.get(failure_class, 0) + 1
    if "correctness" in metric_results:
        metric_results["correctness"]["diagnostics"] = {
            "failure_classes": failure_classes,
        }


def _control_summary(
    candidate: SubmissionPackage,
    control: SubmissionPackage,
) -> dict[str, object]:
    return {
        "status": "unavailable",
        "package": _manifest_package(control),
        "differences": {
            "candidate_package": candidate.name,
            "control_package": control.name,
            "status": "unavailable",
            "reason": "control package was not run in this batch",
        },
    }


def _case_candidate(
    result: GameResult,
    game_metrics: dict[str, GameMetric],
) -> CaseCandidate:
    failure_class, metric_ids, evidence_steps = _case_details(result, game_metrics)
    return CaseCandidate(
        game_id=result.game_id,
        opponent=result.opponent,
        status=result.status,
        failure_class=failure_class,
        is_loss=result.winner == 1,
        metric_ids=metric_ids,
        evidence_steps=evidence_steps,
        trace_path=result.trace_path,
    )


def _case_details(
    result: GameResult,
    game_metrics: dict[str, GameMetric],
) -> tuple[str, tuple[str, ...], tuple]:
    metric_error = _metric_error(game_metrics)
    if metric_error is not None:
        return metric_error

    if result.error_kind:
        error_metrics = tuple(
            metric_id
            for metric_id, metric in game_metrics.items()
            if metric.status == "error"
        )
        evidence = _first_evidence(game_metrics, error_metrics)
        return result.error_kind, error_metrics, evidence

    # Strategy failures need an observed turn; an empty/zero-step fixture or an
    # early worker result is merely unavailable, not evidence of a bad action.
    metric_failure = _metric_failure(game_metrics) if result.steps > 0 else None
    if metric_failure is not None:
        return metric_failure
    return "ordinary_loss" if result.winner == 1 else "ordinary_result", (), ()


def _metric_error(
    game_metrics: dict[str, GameMetric],
) -> tuple[str, tuple[str, ...], tuple] | None:
    for metric in game_metrics.values():
        if metric.status != "error":
            continue
        failure_class = str(metric.value) if metric.value else metric.metric_id
        return failure_class, (metric.metric_id,), metric.evidence
    return None


def _metric_failure(
    game_metrics: dict[str, GameMetric],
) -> tuple[str, tuple[str, ...], tuple] | None:
    rare_candy = game_metrics.get("rare_candy")
    if (
        rare_candy is not None
        and rare_candy.status == "failure"
        and _has_observed_evidence(rare_candy)
    ):
        return str(rare_candy.value), (rare_candy.metric_id,), rare_candy.evidence

    post_ko_relay = game_metrics.get("post_ko_relay")
    if post_ko_relay is not None and post_ko_relay.numerator > 0:
        return "post_ko_zero_ready", (post_ko_relay.metric_id,), post_ko_relay.evidence

    run_away_draw = game_metrics.get("run_away_draw")
    if run_away_draw is not None and run_away_draw.numerator > 0:
        return "empty_bench_run_away_draw", (run_away_draw.metric_id,), run_away_draw.evidence

    library_pressure = game_metrics.get("library_pressure")
    if library_pressure is not None and _has_diagnostic(library_pressure, "deck_out"):
        return "library_deck_out", (library_pressure.metric_id,), library_pressure.evidence
    if library_pressure is not None and library_pressure.numerator > 0:
        return "library_pressure", (library_pressure.metric_id,), library_pressure.evidence
    return None


def _has_diagnostic(metric: GameMetric, key: str) -> bool:
    return any(isinstance(item, dict) and item.get(key) for item in metric.diagnostics)


def _has_observed_evidence(metric: GameMetric) -> bool:
    return any(
        isinstance(item, dict) and item.get("role") not in {None, "trace"}
        for item in metric.evidence
    )


def _first_evidence(game_metrics: dict[str, GameMetric], metric_ids: tuple[str, ...]) -> tuple:
    for metric_id in metric_ids:
        evidence = game_metrics[metric_id].evidence
        if evidence:
            return evidence
    return ()


def _missing_game_result_fields(payload: dict[str, Any]) -> tuple[str, ...]:
    required_fields = (
        "game_id",
        "opponent",
        "candidate_first",
        "candidate_physical_index",
        "finished",
        "winner",
        "status",
        "error_kind",
        "error",
        "steps",
        "trace_path",
    )
    return tuple(field for field in required_fields if field not in payload)


def _nested_result_mismatch(payload: dict[str, Any], result: GameResult) -> str | None:
    expected = {
        "game_id": result.game_id,
        "opponent": result.opponent,
        "candidate_first": result.candidate_first,
        "candidate_physical_index": result.candidate_physical_index,
        "finished": result.finished,
        "winner": result.winner,
        "status": result.status,
        "error_kind": result.error_kind,
        "error": result.error,
        "steps": result.steps,
    }
    return next(
        (field for field, value in expected.items() if payload[field] != value),
        None,
    )


def _cleanup_empty_run_temp_root(temp_root: Path) -> None:
    """清理本 batch 创建且已空的 run 临时父目录，不触碰共享上级目录。"""
    try:
        temp_root.rmdir()
    except (FileNotFoundError, OSError):
        pass


def _manifest(
    config: BatchConfig,
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    retained: dict[str, Path],
    metric_ids: tuple[str, ...],
    presentation_errors: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    profile = get_metric_profile(config.metric_profile_id)
    return {
        "run_id": run_id,
        "candidate": _manifest_package(config.candidate),
        "opponents": [_manifest_package(opponent) for opponent in config.opponents],
        "control": _manifest_package(config.control) if config.control else None,
        "games": len(config.opponents) * config.games_per_opponent,
        "workers": config.workers,
        "swap_policy": "alternate_candidate_first",
        "plugins": list(metric_ids),
        "metric_profile": profile.manifest(),
        "presentation_errors": list(presentation_errors),
        "python_version": sys.version,
        "engine_runtime": {
            "cg_tree_hash": config.candidate.cg_manifest.get("tree_hash"),
            "native_files": config.candidate.cg_manifest.get("native_files", []),
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "trace_policy": {
            "retain_limit": 3,
            "retained_game_ids": sorted(retained),
            "keep_temp": config.keep_temp,
        },
    }


def _manifest_package(package: SubmissionPackage) -> dict[str, object]:
    return {
        "name": package.name,
        "package_hash": package.package_hash,
        "deck_hash": package.deck_hash,
        "cg_hash": package.cg_manifest.get("tree_hash"),
    }


def _summary(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    total_games = len(records)
    wins = sum(record["winner"] == 0 for record in records)
    losses = sum(record["winner"] == 1 for record in records)
    draws = sum(
        record["status"] == "finished" and record["winner"] is None for record in records
    )
    errors = sum(
        record["error_kind"] is not None and record["status"] != "unfinished"
        for record in records
    )
    unfinished = sum(record["status"] == "unfinished" for record in records)
    by_opponent: dict[str, dict[str, int]] = {}
    for record in records:
        opponent = str(record["opponent"])
        values = by_opponent.setdefault(
            opponent,
            {
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "errors": 0,
                "unfinished": 0,
            },
        )
        values["games"] += 1
        values["wins"] += int(record["winner"] == 0)
        values["losses"] += int(record["winner"] == 1)
        values["draws"] += int(
            record["status"] == "finished" and record["winner"] is None
        )
        values["errors"] += int(
            record["error_kind"] is not None and record["status"] != "unfinished"
        )
        values["unfinished"] += int(record["status"] == "unfinished")
    for values in by_opponent.values():
        values["win_rate"] = values["wins"] / values["games"] if values["games"] else 0.0
    return {
        "total_games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "errors": errors,
        "unfinished": unfinished,
        "completed_games": total_games - errors - unfinished,
        "win_rate": wins / total_games if total_games else 0.0,
        "completion_rate": (
            (total_games - errors - unfinished) / total_games if total_games else 0.0
        ),
        "by_opponent": by_opponent,
    }


def _result_payload(result: GameResult) -> dict[str, object]:
    payload = asdict(result)
    payload["trace_path"] = str(result.trace_path)
    return payload


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
