"""Foreground process watchdog for the formal 0034 full-semantic CPU PPO runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT = "0042_full_model_design"
FATAL_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "official CPU engine errors",
    "behavior log-prob parity failed",
    "FloatingPointError",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, json.JSONDecodeError) as error:
        return {"read_error": str(error)}


def _tail(path: Path, limit: int = 131_072, *, start_offset: int = 0) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(start_offset, size - limit))
        return handle.read().decode(errors="replace")


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _gpu() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False}
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {"available": True, "returncode": result.returncode, "rows": result.stdout.strip()}


def _host(run_root: Path) -> dict[str, Any]:
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            memory[name] = int(value.strip().split()[0])
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(run_root if run_root.exists() else ROOT)
    return {
        "load_average": os.getloadavg(),
        "memory_available_kib": memory.get("MemAvailable"),
        "swap_free_kib": memory.get("SwapFree"),
        "disk_free_gib": disk.free / (1024**3),
    }


def _emit(name: str, payload: dict[str, Any]) -> None:
    print(f"{name} {json.dumps(payload, sort_keys=True)}", flush=True)


def _is_current_failed_status(
    state: object, modified_at: float | None, monitor_started_at: float
) -> bool:
    return (
        str(state).startswith("failed")
        and modified_at is not None
        and modified_at >= monitor_started_at
    )


def monitor(args: argparse.Namespace) -> int:
    if not args.command:
        raise ValueError("training command is required after --")
    run_root = ROOT / "rl_runs" / PROJECT / "versions" / args.version
    artifact = run_root / "artifact"
    status_path = artifact / "status.json"
    metrics_path = artifact / "training_metrics.jsonl"
    monitor_root = ROOT / ".tmp" / "training_monitor" / PROJECT / args.version
    log_path = monitor_root / "training.log"
    heartbeat_path = monitor_root / "heartbeat.json"
    alert_path = monitor_root / "alert.json"
    monitor_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_start_offset = log_path.stat().st_size if log_path.is_file() else 0
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(args.command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        last_lines = 0
        last_change = started
        while True:
            now = time.time()
            lines = _line_count(metrics_path)
            if lines != last_lines:
                last_lines = lines
                last_change = now
            status = _read_json(status_path)
            status_modified_at = (
                status_path.stat().st_mtime if status_path.is_file() else None
            )
            # Retries append to the audit log. Historical tracebacks must not
            # terminate a fresh process, so fatal scanning is attempt-local.
            tail = _tail(log_path, start_offset=log_start_offset)
            fatal = [pattern for pattern in FATAL_PATTERNS if pattern in tail]
            returncode = process.poll()
            heartbeat = {
                "checked_at": now,
                "elapsed_seconds": now - started,
                "pid": process.pid,
                "process_returncode": returncode,
                "status_state": status.get("state"),
                "status_modified_at": status_modified_at,
                "checkpoint_update": status.get("checkpoint_update"),
                "metrics_lines": lines,
                "seconds_since_metrics_change": now - last_change,
                "fatal_log_patterns": fatal,
                "gpu": _gpu(),
                "host": _host(run_root),
                "log_path": str(log_path.relative_to(ROOT)),
            }
            _atomic_json(heartbeat_path, heartbeat)
            _emit("TRAINING_MONITOR_HEARTBEAT", heartbeat)
            if returncode is not None:
                if returncode == 0 and status.get("state") in {"complete", "completed"}:
                    _emit("TRAINING_MONITOR_COMPLETE", heartbeat)
                    return 0
                heartbeat["alert_reason"] = "training_process_failed_or_incomplete"
                _atomic_json(alert_path, heartbeat)
                _emit("TRAINING_MONITOR_ALERT", heartbeat)
                return 2
            alert_reason = None
            if fatal:
                alert_reason = "fatal_pattern_in_log"
            elif _is_current_failed_status(
                status.get("state"), status_modified_at, started
            ):
                alert_reason = "status_failed"
            elif lines == 0 and now - started > args.first_metrics_grace_seconds:
                alert_reason = "first_metrics_missing"
            elif lines and now - last_change > args.metrics_stale_seconds:
                alert_reason = "metrics_stale"
            elif heartbeat["host"]["disk_free_gib"] < args.minimum_free_gib:
                alert_reason = "disk_low"
            if alert_reason:
                heartbeat["alert_reason"] = alert_reason
                _atomic_json(alert_path, heartbeat)
                _emit("TRAINING_MONITOR_ALERT", heartbeat)
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                return 2
            time.sleep(args.interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--first-metrics-grace-seconds", type=float, default=300.0)
    parser.add_argument("--metrics-stale-seconds", type=float, default=600.0)
    parser.add_argument("--minimum-free-gib", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
