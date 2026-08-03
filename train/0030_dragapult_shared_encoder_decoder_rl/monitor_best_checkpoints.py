"""Audit strict 0019 frozen-evaluation improvements without stopping training."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from rl_environment.runs import initialize_version, write_version_status

from . import PROJECT_ID


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_VERSION = "V3_lightweight_engine_workers"
DEFAULT_INITIAL_UPDATE = 5
DEFAULT_INITIAL_WIN_RATE = 62 / 102
VERSION_PATTERN = re.compile(r"^V(?P<number>[1-9]\d*)_[a-z0-9][a-z0-9_]*$")
CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass(frozen=True)
class EvaluationPoint:
    update: int
    win_rate: float


@dataclass(frozen=True)
class AuditResult:
    update: int
    status: str
    reason: str | None = None
    version: str | None = None
    report: str | None = None


@dataclass(frozen=True)
class AuditState:
    best_update: int
    best_win_rate: float
    attempts: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def initial(cls, update: int, win_rate: float) -> "AuditState":
        return cls(best_update=int(update), best_win_rate=float(win_rate))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AuditState":
        attempts = payload.get("attempts", {})
        if not isinstance(attempts, dict):
            raise ValueError("audit attempts must be an object")
        return cls(
            best_update=int(payload["best_update"]),
            best_win_rate=float(payload["best_win_rate"]),
            attempts={str(key): dict(value) for key, value in attempts.items()},
        )

    @classmethod
    def read(cls, path: Path) -> "AuditState":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("audit state must be an object")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0030_best_checkpoint_audit_state_v1",
            "best_update": self.best_update,
            "best_win_rate": self.best_win_rate,
            "attempts": self.attempts,
        }

    def write(self, path: Path) -> None:
        _atomic_json(path, self.to_dict())

    def mark_attempted(self, point: EvaluationPoint) -> "AuditState":
        attempts = {key: dict(value) for key, value in self.attempts.items()}
        attempts[str(point.update)] = {
            "status": "attempted",
            "win_rate": point.win_rate,
            "attempted_at": time.time(),
        }
        return AuditState(
            best_update=point.update,
            best_win_rate=point.win_rate,
            attempts=attempts,
        )

    def mark_finished(
        self,
        update: int,
        status: str,
        *,
        reason: str | None = None,
        version: str | None = None,
        report: str | None = None,
    ) -> "AuditState":
        key = str(update)
        if key not in self.attempts:
            raise ValueError(f"update {update} was not marked attempted")
        attempts = {name: dict(value) for name, value in self.attempts.items()}
        attempts[key].update({"status": status, "finished_at": time.time()})
        for name, value in (("reason", reason), ("version", version), ("report", report)):
            if value is not None:
                attempts[key][name] = value
        return AuditState(self.best_update, self.best_win_rate, attempts)


@dataclass(frozen=True)
class AuditConfig:
    repository_root: Path = REPOSITORY_ROOT
    training_version: str = DEFAULT_TRAINING_VERSION
    state_path: Path = Path(".tmp/training_monitor") / PROJECT_ID / "best_checkpoint_audit/state.json"
    workers: int = 8
    worker_cpu_threads: int = 1
    device: str = "cuda:0"

    def resolved_state_path(self) -> Path:
        return self.state_path if self.state_path.is_absolute() else self.repository_root / self.state_path

    def training_root(self) -> Path:
        return self.repository_root / "rl_runs" / PROJECT_ID / "versions" / self.training_version

    def checkpoint(self, update: int) -> Path:
        return self.training_root() / "checkpoint" / f"update-{update:04d}.pt"

    def metrics_path(self) -> Path:
        return self.training_root() / "artifact" / "training_metrics.jsonl"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _emit(event: str, payload: dict[str, Any]) -> None:
    print(f"{event} {json.dumps(payload, ensure_ascii=True, sort_keys=True)}", flush=True)


def evaluation_record(record: dict[str, Any]) -> EvaluationPoint | None:
    key = "eval/foundation_0019/win_rate"
    if key not in record:
        return None
    update_value = record.get("eval/checkpoint_update", record.get("trainer/update"))
    if update_value is None:
        return None
    return EvaluationPoint(update=int(update_value), win_rate=float(record[key]))


def read_metric_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def select_new_bests(
    records: Iterable[dict[str, Any]], state: AuditState
) -> list[EvaluationPoint]:
    best = state.best_win_rate
    selected: list[EvaluationPoint] = []
    for record in records:
        point = evaluation_record(record)
        if point is None or str(point.update) in state.attempts:
            continue
        if point.win_rate > best:
            selected.append(point)
            best = point.win_rate
    return selected


def next_audit_version(existing_names: Iterable[str], update: int) -> str:
    numbers = [
        int(match.group("number"))
        for name in existing_names
        if (match := VERSION_PATTERN.fullmatch(name)) is not None
    ]
    return f"V{max(numbers, default=0) + 1}_update{update}_frozen0019_audit"


def _run(command_runner: CommandRunner, command: list[str], root: Path) -> int:
    completed = command_runner(command, cwd=root, check=False)
    return int(completed.returncode)


def audit_checkpoint(
    point: EvaluationPoint,
    config: AuditConfig,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> AuditResult:
    source_checkpoint = config.checkpoint(point.update)
    if not source_checkpoint.is_file():
        return AuditResult(point.update, "failed", reason="checkpoint_missing")

    versions_root = config.repository_root / "rl_runs" / PROJECT_ID / "versions"
    names = [path.name for path in versions_root.iterdir() if path.is_dir()]
    version = next_audit_version(names, point.update)
    try:
        paths = initialize_version(PROJECT_ID, version)
    except Exception as error:
        return AuditResult(
            point.update,
            "failed",
            reason=f"version_allocation_failed:{type(error).__name__}:{error}",
            version=version,
        )

    preserved_checkpoint = paths.checkpoints / f"source-update-{point.update:04d}.pt"
    candidate = (
        config.repository_root
        / "evaluation"
        / "arena"
        / "candidates"
        / f"0030_dragapult_ex_001_v3_update{point.update}_{version.lower()}"
    )
    audit_config = {
        "schema_version": "0030_frozen0019_checkpoint_audit_v1",
        "project_id": PROJECT_ID,
        "version": version,
        "source_training_version": config.training_version,
        "source_update": point.update,
        "trigger_eval_foundation_0019_win_rate": point.win_rate,
        "games_per_opponent": 10,
        "opponent_count": 51,
        "workers": config.workers,
        "worker_cpu_threads": config.worker_cpu_threads,
        "candidate_device": config.device,
        "opponent_device": config.device,
    }
    try:
        temporary = preserved_checkpoint.with_suffix(".pt.tmp")
        shutil.copy2(source_checkpoint, temporary)
        temporary.replace(preserved_checkpoint)
        _atomic_json(paths.config, audit_config)
        write_version_status(paths, {
            "state": "running",
            "kind": "frozen0019_checkpoint_audit",
            "source_training_version": config.training_version,
            "source_update": point.update,
        })

        export_command = [
            sys.executable,
            "-m",
            f"train.{PROJECT_ID}.export_candidate",
            "--checkpoint",
            str(preserved_checkpoint),
            "--output",
            str(candidate),
        ]
        if _run(command_runner, export_command, config.repository_root) != 0:
            raise RuntimeError("candidate_export_failed")

        evaluation_command = [
            sys.executable,
            "-m",
            "evaluation",
            "--pool",
            "frozen",
            "run",
            "--candidate",
            str(candidate),
            "--opponents",
            "all",
            "--games",
            "10",
            "--output",
            str(paths.evaluation),
            "--workers",
            str(config.workers),
            "--worker-cpu-threads",
            str(config.worker_cpu_threads),
            "--candidate-device",
            config.device,
            "--opponent-device",
            config.device,
        ]
        if _run(command_runner, evaluation_command, config.repository_root) != 0:
            raise RuntimeError("frozen0019_evaluation_failed")
        if not paths.evaluation.is_file():
            raise RuntimeError("formal_report_missing")
        write_version_status(paths, {"state": "completed", "report": str(paths.evaluation)})
        return AuditResult(
            point.update,
            "completed",
            version=version,
            report=str(paths.evaluation.relative_to(config.repository_root)),
        )
    except Exception as error:
        write_version_status(paths, {
            "state": "failed",
            "failure": f"{type(error).__name__}:{error}",
        })
        return AuditResult(
            point.update,
            "failed",
            reason=f"{type(error).__name__}:{error}",
            version=version,
        )


def _load_state(config: AuditConfig, initial_update: int, initial_win_rate: float) -> AuditState:
    path = config.resolved_state_path()
    return AuditState.read(path) if path.is_file() else AuditState.initial(initial_update, initial_win_rate)


def _record_result(state: AuditState, result: AuditResult, path: Path) -> AuditState:
    updated = state.mark_finished(
        result.update,
        result.status,
        reason=result.reason,
        version=result.version,
        report=result.report,
    )
    updated.write(path)
    return updated


def monitor(args: argparse.Namespace) -> int:
    config = AuditConfig(
        training_version=args.training_version,
        state_path=args.state_path,
        workers=args.workers,
        worker_cpu_threads=args.worker_cpu_threads,
        device=args.device,
    )
    state_path = config.resolved_state_path()
    state = _load_state(config, args.initial_update, args.initial_win_rate)

    if args.audit_update is not None:
        point = EvaluationPoint(args.audit_update, args.audit_win_rate)
        if str(point.update) not in state.attempts:
            state = state.mark_attempted(point)
            state.write(state_path)
            _emit("BEST_CHECKPOINT_AUDIT_ATTEMPT", {"update": point.update, "win_rate": point.win_rate})
            result = audit_checkpoint(point, config)
            state = _record_result(state, result, state_path)
            _emit(
                "BEST_CHECKPOINT_AUDIT_COMPLETE" if result.status == "completed" else "BEST_CHECKPOINT_AUDIT_ALERT",
                result.__dict__,
            )
        if args.once:
            return 0

    while True:
        records = read_metric_records(config.metrics_path())
        for point in select_new_bests(records, state):
            state = state.mark_attempted(point)
            state.write(state_path)
            _emit("BEST_CHECKPOINT_AUDIT_ATTEMPT", {"update": point.update, "win_rate": point.win_rate})
            result = audit_checkpoint(point, config)
            state = _record_result(state, result, state_path)
            _emit(
                "BEST_CHECKPOINT_AUDIT_COMPLETE" if result.status == "completed" else "BEST_CHECKPOINT_AUDIT_ALERT",
                result.__dict__,
            )
        heartbeat = {
            "checked_at": time.time(),
            "training_version": config.training_version,
            "metric_records": len(records),
            "best_update": state.best_update,
            "best_win_rate": state.best_win_rate,
            "attempt_count": len(state.attempts),
        }
        _atomic_json(state_path.parent / "heartbeat.json", heartbeat)
        _emit("BEST_CHECKPOINT_MONITOR_HEARTBEAT", heartbeat)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-version", default=DEFAULT_TRAINING_VERSION)
    parser.add_argument("--state-path", type=Path, default=AuditConfig.state_path)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--initial-update", type=int, default=DEFAULT_INITIAL_UPDATE)
    parser.add_argument("--initial-win-rate", type=float, default=DEFAULT_INITIAL_WIN_RATE)
    parser.add_argument("--audit-update", type=int)
    parser.add_argument("--audit-win-rate", type=float, default=DEFAULT_INITIAL_WIN_RATE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-cpu-threads", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    if arguments.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if arguments.workers <= 0 or arguments.worker_cpu_threads <= 0:
        parser.error("worker counts must be positive")
    if arguments.audit_update is None and "--audit-win-rate" in sys.argv:
        parser.error("--audit-win-rate requires --audit-update")
    return monitor(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
