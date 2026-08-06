from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.cases import CaseCandidate, case_record, select_cases
from evaluation.cards import card_image_url, load_card_catalog
from evaluation.metrics import GameContext, GameMetric
from evaluation.metrics.profiles import get_metric_profile
from evaluation.metrics.registry import MetricRegistry, create_metric_registry
from evaluation.packages.loader import SubmissionPackage
from evaluation.reporting import ReportData, json_ready, write_evaluation_index, write_report
from evaluation.reporting.index import write_report_file_atomic
from evaluation.runner.models import GameRequest, GameResult
from evaluation.traces.store import TraceStore
from rl_environment.runs import project_version_paths


_INITIALIZED_MANIFEST_FIELDS = (
    "schema_version",
    "project_id",
    "objective",
    "deck",
    "expert_source",
    "dataset_contract",
    "engine_revision",
    "opponent_pool_snapshot",
    "status",
    "created_at",
    "paths",
)
_INITIALIZED_VERSION_STATES = frozenset(
    {"allocated", "running", "complete", "completed", "failed", "interrupted"}
)


def default_worker_count() -> int:
    """Use the calibrated local preset without exceeding available CPU affinity."""
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1
    return max(1, min(8, available_cpus))


@dataclass(frozen=True)
class BatchConfig:
    candidate: SubmissionPackage
    opponents: tuple[SubmissionPackage, ...]
    games_per_opponent: int
    output_root: Path
    visualize: bool
    max_steps: int
    report_path: Path | None = None
    update_project_index: bool = False
    control: SubmissionPackage | None = None
    plugin_ids: tuple[str, ...] = ()
    metric_module_paths: tuple[str, ...] = ()
    metric_profile_id: str = "core"
    keep_temp: bool = False
    worker_timeout_seconds: float = 30.0
    workers: int = field(default_factory=default_worker_count)
    worker_cpu_threads: int | None = 1
    candidate_inference_device: str | None = None
    candidate_inference_batch_size: int = 32
    candidate_inference_batch_wait_ms: float = 2.0
    candidate_inference_dtype: str = "fp32"
    opponent_inference_root: Path | None = None
    opponent_inference_device: str | None = None
    opponent_inference_dtype: str = "fp32"
    opponent_pool_id: str = "legacy_opponents"
    opponent_catalog_sha256: str | None = None
    opponent_policy_hash: str | None = None
    opponent_policy_label: str | None = None
    share_policy_inference_server: bool = False
    seed: int = 22022
    worker_crash_retries: int = 0
    games_by_opponent: tuple[int, ...] | None = None
    opponent_schedule_id: str | None = None
    candidate_inference_socket: Path | None = None
    opponent_inference_socket: Path | None = None
    inference_ability_repeat_limit: int = 0
    engine_turn_draw_limit: int = 0


@dataclass(frozen=True)
class BatchResult:
    run_id: str
    manifest: dict[str, object]
    game_records: tuple
    metric_results: dict[str, object]
    case_records: tuple
    report_data: ReportData
    report_path: Path


def run_batch(config: BatchConfig) -> BatchResult:
    if config.games_per_opponent < 1:
        raise ValueError("games_per_opponent must be at least one")
    if config.max_steps < 1:
        raise ValueError("max_steps must be at least one")
    if config.worker_timeout_seconds <= 0:
        raise ValueError("worker_timeout_seconds must be greater than zero")
    if config.workers < 1:
        raise ValueError("workers must be at least one")
    if config.worker_crash_retries < 0:
        raise ValueError("worker_crash_retries cannot be negative")
    if config.games_by_opponent is not None and (
        len(config.games_by_opponent) != len(config.opponents)
        or any(type(count) is not int or count < 1 for count in config.games_by_opponent)
    ):
        raise ValueError(
            "games_by_opponent must contain one positive integer per opponent"
        )
    if config.opponent_schedule_id is not None and not config.opponent_schedule_id:
        raise ValueError("opponent_schedule_id cannot be empty")
    if (config.candidate_inference_socket is None) != (
        config.opponent_inference_socket is None
    ):
        raise ValueError("external inference sockets must be supplied together")
    if config.candidate_inference_socket is not None:
        missing = [
            path
            for path in (
                config.candidate_inference_socket,
                config.opponent_inference_socket,
            )
            if path is None or not path.exists()
        ]
        if missing:
            raise ValueError("external inference socket does not exist")
    if config.worker_cpu_threads is not None and config.worker_cpu_threads < 1:
        raise ValueError("worker_cpu_threads must be at least one")
    if config.candidate_inference_batch_size < 1:
        raise ValueError("candidate_inference_batch_size must be at least one")
    if config.candidate_inference_batch_wait_ms < 0:
        raise ValueError("candidate_inference_batch_wait_ms cannot be negative")
    if config.candidate_inference_dtype not in {"fp32", "fp16"}:
        raise ValueError("candidate_inference_dtype must be fp32 or fp16")
    if config.opponent_inference_dtype not in {"fp32", "fp16"}:
        raise ValueError("opponent_inference_dtype must be fp32 or fp16")
    if config.inference_ability_repeat_limit < 0:
        raise ValueError("inference_ability_repeat_limit cannot be negative")
    if config.engine_turn_draw_limit < 0:
        raise ValueError("engine_turn_draw_limit cannot be negative")
    if (config.opponent_inference_root is None) != (
        config.opponent_inference_device is None
    ):
        raise ValueError("opponent inference root and device must be supplied together")
    if config.opponent_pool_id.startswith("0019_foundation_") and "_exact_decks_" in config.opponent_pool_id and (
        config.candidate_inference_device is None
        or config.opponent_inference_root is None
        or config.opponent_inference_device is None
    ):
        raise ValueError(
            "Frozen Arena requires shared GPU inference for both candidate and opponents"
        )
    if config.share_policy_inference_server:
        if (
            config.candidate_inference_device is None
            or config.opponent_inference_root is None
            or config.opponent_inference_device is None
        ):
            raise ValueError("shared policy inference requires both GPU routes")
        if config.candidate_inference_device != config.opponent_inference_device:
            raise ValueError("shared policy inference requires one identical device")
        if config.candidate.root.resolve() != config.opponent_inference_root.resolve():
            raise ValueError("shared policy inference requires one identical policy root")
        if config.candidate.entrypoint.resolve() != (
            config.opponent_inference_root / "main.py"
        ).resolve():
            raise ValueError("shared policy inference candidate entrypoint mismatch")

    run_id = f"run-{uuid.uuid4().hex}"
    explicit_report_path = config.report_path.resolve() if config.report_path else None
    if explicit_report_path is not None:
        if explicit_report_path.suffix.lower() != ".html":
            raise ValueError("evaluation report path must end with .html")
        _validate_formal_destination(explicit_report_path)
        report_root = explicit_report_path.parent
        final_report_path = explicit_report_path
    else:
        report_root = config.output_root.resolve() / run_id
        final_report_path = report_root / "report.html"
    temp_root = Path(tempfile.gettempdir()) / "evaluation" / run_id
    store = TraceStore(temp_root, report_root)
    registry = _metric_registry(
        config.metric_module_paths,
        config.plugin_ids,
        config.metric_profile_id,
    )
    started_at = _timestamp()
    wall_started = time.perf_counter()
    metric_values: dict[str, list[GameMetric]] = {
        plugin.metric_id: [] for plugin in registry.plugins
    }
    case_candidates: list[CaseCandidate] = []

    try:
        jobs = _game_jobs(config, run_id, store)
        actual_workers = min(config.workers, len(jobs))
        with _policy_inference_servers(config) as (
            candidate_socket,
            opponent_socket,
        ):
            for request, trace_path, result in _run_workers(
                config, jobs, store.temp_root, candidate_socket, opponent_socket
            ):
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
        selected_cases = select_cases(case_candidates)
        case_records = tuple(_report_case_record(candidate) for candidate in selected_cases)
        summary = _summary(store.game_records)
        if config.control is not None:
            summary["control"] = _control_summary(config.candidate, config.control)
        finished_at = _timestamp()
        wall_time_seconds = time.perf_counter() - wall_started
        total_selections = sum(int(record["steps"]) for record in store.game_records)
        summary["performance"] = {
            "wall_time_seconds": wall_time_seconds,
            "games_per_second": len(store.game_records) / wall_time_seconds,
            "engine_selections": total_selections,
            "selections_per_second": total_selections / wall_time_seconds,
            "workers": actual_workers,
            "worker_cpu_threads": config.worker_cpu_threads,
        }
        manifest = _manifest(
            config,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            wall_time_seconds=wall_time_seconds,
            actual_workers=actual_workers,
            metric_ids=tuple(plugin.metric_id for plugin in registry.plugins),
            presentation_errors=presentation_errors,
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
        if explicit_report_path is None:
            write_report(report_data, report_root)
        else:
            _finalize_formal_report(
                report_data,
                explicit_report_path,
                run_id=run_id,
                update_project_index=config.update_project_index,
            )
        return BatchResult(
            run_id=run_id,
            manifest=manifest,
            game_records=store.game_records,
            metric_results=metric_results,
            case_records=case_records,
            report_data=report_data,
            report_path=final_report_path,
        )
    finally:
        if not config.keep_temp:
            store.cleanup()
            _cleanup_empty_run_temp_root(temp_root)


def _validate_formal_destination(report_path: Path) -> None:
    paths = _formal_version_paths(report_path)
    if report_path.exists():
        raise ValueError(f"evaluation report already exists: {report_path}")
    manifest_path = paths.project_archive / "manifest.json"
    backlink = paths.artifact / "evaluation.json"
    if backlink.exists():
        raise ValueError(f"evaluation provenance already exists: {backlink}")
    required = (manifest_path, paths.run_root, paths.artifact, paths.status)
    initialized = all(
        path.is_file() if path in (manifest_path, paths.status) else path.is_dir()
        for path in required
    )
    if not initialized:
        raise ValueError(
            "formal evaluation runtime is not initialized; expected project manifest, "
            "version root, artifact directory, and status.json"
        )
    manifest = _read_json_object(manifest_path, "manifest")
    invalid_manifest = (
        manifest.get("project_id") != paths.project_id
        or manifest.get("schema_version") != "ptcg_experiment_project_v1"
        or manifest.get("status") != "initialized"
        or any(
            field not in manifest
            or manifest[field] in (None, "")
            for field in _INITIALIZED_MANIFEST_FIELDS
        )
        or not isinstance(manifest.get("paths"), dict)
    )
    if invalid_manifest:
        raise ValueError(f"formal evaluation manifest is invalid: {manifest_path}")
    status = _read_json_object(paths.status, "status")
    if (
        status.get("version") != paths.version_name
        or status.get("state") not in _INITIALIZED_VERSION_STATES
    ):
        raise ValueError(f"formal evaluation status is invalid: {paths.status}")


def _read_json_object(path: Path, kind: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"formal evaluation {kind} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"formal evaluation {kind} must be a JSON object: {path}")
    return value


def _finalize_formal_report(
    report_data: ReportData,
    report_path: Path,
    *,
    run_id: str,
    update_project_index: bool,
) -> None:
    """Publish a completed formal report before its derived index and provenance backlink."""
    write_report_file_atomic(report_data, report_path)
    write_evaluation_index(report_path.parent)
    paths = _formal_version_paths(report_path)
    _write_evaluation_backlink_atomic(
        paths,
        run_id=run_id,
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
    )


def _write_evaluation_backlink_atomic(
    paths,
    *,
    run_id: str,
    report_sha256: str,
) -> None:
    backlink = paths.artifact / "evaluation.json"
    temporary = backlink.with_name(f".{backlink.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "report": str(paths.evaluation.relative_to(Path(__file__).resolve().parents[2])),
        "report_sha256": report_sha256,
        "run_id": run_id,
        "version": paths.version_name,
    }
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, backlink)
        except FileExistsError as exc:
            raise FileExistsError(
                f"evaluation provenance already exists: {backlink}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _formal_version_paths(report_path: Path):
    repository_root = Path(__file__).resolve().parents[2]
    experiments_root = (repository_root / "experiments").resolve()
    try:
        relative = report_path.resolve().relative_to(experiments_root)
    except ValueError as exc:
        raise ValueError(
            "explicit report_path must be a formal evaluation report path below experiments"
        ) from exc
    if len(relative.parts) != 3 or relative.parts[1] != "evaluation":
        raise ValueError(
            "formal evaluation report path must be "
            "experiments/<project_id>/evaluation/V<n>_<tag>.html"
        )
    project_id, _, report_name = relative.parts
    if Path(report_name).suffix.lower() != ".html":
        raise ValueError("evaluation report path must end with .html")
    try:
        paths = project_version_paths(project_id, Path(report_name).stem)
    except ValueError as exc:
        raise ValueError(
            f"invalid version or project in formal evaluation path: {report_path}"
        ) from exc
    if paths.evaluation.resolve() != report_path.resolve():
        raise ValueError(
            f"formal evaluation report path does not match runtime version: {report_path}"
        )
    return paths


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
    counts = config.games_by_opponent or (config.games_per_opponent,) * len(config.opponents)
    global_game_number = 0
    for opponent, game_count in zip(config.opponents, counts, strict=True):
        for game_number in range(1, game_count + 1):
            global_game_number += 1
            game_id = f"{opponent.name}-{game_number:03d}"
            request = GameRequest(
                run_id=run_id,
                game_id=game_id,
                candidate=config.candidate,
                opponent=opponent,
                candidate_first=(
                    global_game_number % 2 == 1
                    if config.games_by_opponent is not None
                    else game_number % 2 == 1
                ),
                max_steps=config.max_steps,
                engine_turn_draw_limit=config.engine_turn_draw_limit,
                visualize=config.visualize,
                seed=_stable_game_seed(
                    config.seed, config.candidate.name, opponent.name, game_number
                ),
            )
            jobs.append((request, store.temp_path(game_id)))
    return jobs


def _stable_game_seed(base_seed: int, candidate: str, opponent: str, game_number: int) -> int:
    """Derive a reproducible seed independent of UUID/run ordering."""
    digest = hashlib.sha256(
        f"{base_seed}:{candidate}:{opponent}:{game_number}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _candidate_socket_path() -> Path:
    """Return a unique AF_UNIX path that stays below Linux's 108-byte sockaddr limit."""
    return Path(tempfile.gettempdir()) / f"ptcg-eval-{uuid.uuid4().hex}.sock"


@contextmanager
def _policy_inference_servers(
    config: BatchConfig,
) -> Iterator[tuple[Path | None, Path | None]]:
    if config.candidate_inference_socket is not None:
        assert config.opponent_inference_socket is not None
        yield config.candidate_inference_socket, config.opponent_inference_socket
        return
    with _policy_inference_server(
        root=config.candidate.root,
        device=config.candidate_inference_device,
        batch_size=config.candidate_inference_batch_size,
        batch_wait_ms=config.candidate_inference_batch_wait_ms,
        label="candidate",
        ability_repeat_limit=config.inference_ability_repeat_limit,
        inference_dtype=config.candidate_inference_dtype,
    ) as candidate_socket:
        if config.share_policy_inference_server:
            yield candidate_socket, candidate_socket
            return
        with _policy_inference_server(
            root=config.opponent_inference_root,
            device=config.opponent_inference_device,
            batch_size=config.candidate_inference_batch_size,
            batch_wait_ms=config.candidate_inference_batch_wait_ms,
            label="opponent",
            ability_repeat_limit=config.inference_ability_repeat_limit,
            inference_dtype=config.opponent_inference_dtype,
        ) as opponent_socket:
            yield candidate_socket, opponent_socket


@contextmanager
def _policy_inference_server(
    *,
    root: Path | None,
    device: str | None,
    batch_size: int,
    batch_wait_ms: float,
    label: str,
    ability_repeat_limit: int = 0,
    inference_dtype: str = "fp32",
) -> Iterator[Path | None]:
    if root is None or device is None:
        yield None
        return

    socket_path = _candidate_socket_path()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "evaluation.runner.inference_server",
            "--candidate",
            str(root),
            "--socket",
            str(socket_path),
            "--device",
            str(device),
            "--batch-size",
            str(batch_size),
            "--batch-wait-ms",
            str(batch_wait_ms),
            "--ability-repeat-limit",
            str(ability_repeat_limit),
            "--inference-dtype",
            inference_dtype,
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 60.0
    try:
        while not socket_path.exists():
            if process.poll() is not None:
                stderr = process.stderr.read().strip() if process.stderr else ""
                raise RuntimeError(
                    f"{label} inference server failed to start"
                    + (f": {stderr}" if stderr else "")
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{label} inference server did not become ready in 60 seconds")
            time.sleep(0.05)
        yield socket_path
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        socket_path.unlink(missing_ok=True)


def _run_workers(
    config: BatchConfig,
    jobs: list[tuple[GameRequest, Path]],
    temp_root: Path,
    candidate_inference_socket: Path | None = None,
    opponent_inference_socket: Path | None = None,
) -> Iterator[tuple[GameRequest, Path, GameResult]]:
    if config.workers == 1:
        for request, trace_path in jobs:
            yield request, trace_path, _run_worker_with_retries(
                config.worker_crash_retries,
                request,
                trace_path,
                temp_root,
                config.worker_timeout_seconds,
                config.worker_cpu_threads,
                candidate_inference_socket,
                opponent_inference_socket,
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
                config.worker_cpu_threads,
                candidate_inference_socket,
                opponent_inference_socket,
            )
            for request, trace_path in jobs
        ]
        for (request, trace_path), future in zip(jobs, futures, strict=True):
            result = future.result()
            if result.error_kind == "worker_crash":
                result = _run_worker_with_retries(
                    config.worker_crash_retries,
                    request,
                    trace_path,
                    temp_root,
                    config.worker_timeout_seconds,
                    config.worker_cpu_threads,
                    candidate_inference_socket,
                    opponent_inference_socket,
                    initial_result=result,
                )
            yield request, trace_path, result


def _run_worker_with_retries(
    retries: int,
    request: GameRequest,
    trace_path: Path,
    temp_root: Path,
    timeout_seconds: float,
    cpu_threads: int | None,
    candidate_inference_socket: Path | None,
    opponent_inference_socket: Path | None,
    *,
    initial_result: GameResult | None = None,
) -> GameResult:
    result = initial_result
    for _attempt in range(retries + 1):
        if result is None or result.error_kind == "worker_crash":
            result = _run_worker(
                request,
                trace_path,
                temp_root,
                timeout_seconds,
                cpu_threads,
                candidate_inference_socket,
                opponent_inference_socket,
            )
        if result.error_kind != "worker_crash":
            break
    assert result is not None
    return result


def _run_worker(
    request: GameRequest,
    trace_path: Path,
    temp_root: Path,
    timeout_seconds: float,
    cpu_threads: int | None = None,
    candidate_inference_socket: Path | None = None,
    opponent_inference_socket: Path | None = None,
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
    if candidate_inference_socket is not None:
        environment["EVALUATION_CANDIDATE_INFERENCE_SOCKET"] = str(
            candidate_inference_socket
        )
    if opponent_inference_socket is not None:
        environment["EVALUATION_OPPONENT_INFERENCE_SOCKET"] = str(
            opponent_inference_socket
        )
    if cpu_threads is not None:
        thread_count = str(cpu_threads)
        environment.update(
            {
                "OMP_NUM_THREADS": thread_count,
                "MKL_NUM_THREADS": thread_count,
                "OPENBLAS_NUM_THREADS": thread_count,
                "NUMEXPR_NUM_THREADS": thread_count,
            }
        )
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
        "engine_turn_draw_limit": request.engine_turn_draw_limit,
        "visualize": request.visualize,
        "seed": request.seed,
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
    wall_time_seconds: float,
    actual_workers: int,
    metric_ids: tuple[str, ...],
    presentation_errors: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    profile = get_metric_profile(config.metric_profile_id)
    return {
        "run_id": run_id,
        "candidate": _manifest_package(config.candidate, include_deck=True),
        "opponents": [_manifest_package(opponent) for opponent in config.opponents],
        "control": _manifest_package(config.control) if config.control else None,
        "games": (
            sum(config.games_by_opponent)
            if config.games_by_opponent is not None
            else len(config.opponents) * config.games_per_opponent
        ),
        "games_per_opponent": (
            list(config.games_by_opponent)
            if config.games_by_opponent is not None
            else config.games_per_opponent
        ),
        "opponent_schedule_id": config.opponent_schedule_id,
        "seed": config.seed,
        "seed_policy": "sha256(base_seed:candidate:opponent:game_number)",
        "engine_rng_contract": (
            "python_numpy_torch_only; official engine internal RNG is not exposed by runtime"
        ),
        "workers": actual_workers,
        "requested_workers": config.workers,
        "worker_crash_retries": config.worker_crash_retries,
        "worker_cpu_threads": config.worker_cpu_threads,
        "candidate_inference": {
            "mode": (
                "external_shared_server"
                if config.candidate_inference_socket is not None
                else "shared_server"
                if config.candidate_inference_device
                else "in_worker"
            ),
            "device": config.candidate_inference_device or "cpu",
            "batch_size": (
                config.candidate_inference_batch_size
                if config.candidate_inference_device
                else 1
            ),
            "batch_wait_ms": (
                config.candidate_inference_batch_wait_ms
                if config.candidate_inference_device
                else 0.0
            ),
            "dtype": config.candidate_inference_dtype,
        },
        "opponent_pool": {
            "pool_id": config.opponent_pool_id,
            "catalog_sha256": config.opponent_catalog_sha256,
            "policy_hash": config.opponent_policy_hash,
            "policy_label": config.opponent_policy_label,
        },
        "opponent_inference": {
            "mode": (
                "external_shared_server"
                if config.opponent_inference_socket is not None
                else "shared_server"
                if config.opponent_inference_device
                else "in_worker"
            ),
            "device": config.opponent_inference_device or "cpu",
            "batch_size": (
                config.candidate_inference_batch_size
                if config.opponent_inference_device
                else 1
            ),
            "batch_wait_ms": (
                config.candidate_inference_batch_wait_ms
                if config.opponent_inference_device
                else 0.0
            ),
            "dtype": config.opponent_inference_dtype,
        },
        "shared_policy_inference_process": config.share_policy_inference_server,
        "inference_progress_guard": {
            "kind": "same_turn_identical_ability_repeat_then_end",
            "ability_repeat_limit": config.inference_ability_repeat_limit,
            "applies_to": "candidate_and_opponent",
        },
        "engine_turn_draw_limit": config.engine_turn_draw_limit,
        "full_round_draw_limit": (
            config.engine_turn_draw_limit // 2
            if config.engine_turn_draw_limit > 0
            else 0
        ),
        "swap_policy": (
            "alternate_candidate_first_globally"
            if config.games_by_opponent is not None
            else "alternate_candidate_first"
        ),
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
        "wall_time_seconds": wall_time_seconds,
        "artifact_policy": {
            "mode": "report_only",
            "retained_files": [config.report_path.name if config.report_path else "report.html"],
        },
        "trace_policy": {
            "retain_limit": 0,
            "retained_game_ids": [],
            "keep_temp": config.keep_temp,
        },
    }


def _manifest_package(
    package: SubmissionPackage, *, include_deck: bool = False
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": package.name,
        "display_name": package.display_name or package.name,
        "representative_cards": list(package.representative_cards),
        "package_hash": package.package_hash,
        "deck_hash": package.deck_hash,
        "cg_hash": package.cg_manifest.get("tree_hash"),
    }
    if include_deck:
        card_catalog = load_card_catalog(
            Path(__file__).resolve().parents[2]
            / "data"
            / "official"
            / "EN_Card_Data.csv"
        )
        deck_cards = []
        category_counts = {"pokemon": 0, "trainer": 0, "energy": 0}
        for card_id, count in sorted(
            Counter(package.deck).items(),
            key=lambda item: (
                _deck_card_category(card_catalog.get(item[0], {})),
                str(card_catalog.get(item[0], {}).get("name", "")),
                item[0],
            ),
        ):
            metadata = card_catalog.get(card_id, {})
            category = _deck_card_category(metadata)
            category_counts[category] += count
            deck_cards.append(
                {
                    "card_id": card_id,
                    "count": count,
                    "name": metadata.get("name") or f"Card {card_id}",
                    "expansion": metadata.get("expansion", ""),
                    "collection_number": metadata.get("collection_number", ""),
                    "stage_or_type": metadata.get("stage_or_type", ""),
                    "category": category,
                    "image_url": card_image_url(
                        str(metadata.get("expansion", "")),
                        str(metadata.get("collection_number", "")),
                    ),
                }
            )
        payload.update(
            {
                "deck": list(package.deck),
                "deck_cards": deck_cards,
                "deck_category_counts": category_counts,
                "deck_total": len(package.deck),
            }
        )
    if package.package_manifest is not None:
        payload["package_manifest"] = package.package_manifest
    return payload


def _deck_card_category(metadata: dict[str, str]) -> str:
    stage_or_type = str(metadata.get("stage_or_type", ""))
    if stage_or_type.endswith("Pokémon"):
        return "pokemon"
    if "Energy" in stage_or_type:
        return "energy"
    return "trainer"


def _summary(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    total_games = len(records)
    wins = sum(record["winner"] == 0 for record in records)
    losses = sum(record["winner"] == 1 for record in records)
    draws = sum(
        record["status"] == "finished" and record["winner"] is None for record in records
    )
    errors = sum(
        record["error_kind"] is not None
        and record["status"] != "unfinished"
        and record["status"] != "finished"
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
            record["error_kind"] is not None
            and record["status"] != "unfinished"
            and record["status"] != "finished"
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


def _report_case_record(candidate: CaseCandidate) -> dict[str, object]:
    record = case_record(candidate)
    record["trace_path"] = None
    return record


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
