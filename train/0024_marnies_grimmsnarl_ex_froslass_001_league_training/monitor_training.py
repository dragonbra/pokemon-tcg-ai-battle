"""Local watchdog for long 0024 League PPO runs.

The monitor is intentionally process-based instead of agent-based. It writes a
small heartbeat JSON on every poll and an alert JSON before exiting non-zero
when training looks unhealthy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


PROJECT = "0024_marnies_grimmsnarl_ex_froslass_001_league_training"
RUN_ROOT = Path("rl_runs") / PROJECT / "versions"
MONITOR_ROOT = Path(".tmp") / "training_monitor" / PROJECT
FATAL_LOG_PATTERNS = (
    "Traceback (most recent call last)",
    "RuntimeError:",
    "CUDA out of memory",
    "worker EOF",
    "FileNotFoundError:",
    "official-engine errors",
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"json_error": str(error)}


def _tail(path: Path, max_bytes: int = 131_072) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _nvidia_smi() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False}
    query = (
        "timestamp,index,name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _emit_event(event: str, payload: dict[str, Any]) -> None:
    """Write the terminal notification protocol consumed by the foreground agent."""
    print(f"{event} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def _latest_checkpoint_update(live_root: Path) -> int | None:
    latest: int | None = None
    if not live_root.exists():
        return None
    for checkpoint in live_root.glob("*/update-*.pt"):
        try:
            update = int(checkpoint.stem.split("-")[-1])
        except ValueError:
            continue
        latest = update if latest is None else max(latest, update)
    return latest


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def monitor(args: argparse.Namespace) -> int:
    run_dir = RUN_ROOT / args.version
    artifact = run_dir / "artifact"
    status_path = artifact / "status.json"
    metrics_path = artifact / "training_metrics.jsonl"
    live_root = run_dir / "checkpoint" / "live"
    monitor_dir = MONITOR_ROOT / args.version
    heartbeat_path = monitor_dir / "heartbeat.json"
    alert_path = monitor_dir / "alert.json"
    started = time.time()
    last_metrics_lines = _line_count(metrics_path)
    last_metrics_change = time.time()
    last_checkpoint_update = _latest_checkpoint_update(live_root)
    last_checkpoint_change = time.time()

    while True:
        now = time.time()
        elapsed_hours = (now - started) / 3600.0
        status = _read_json(status_path)
        metrics_lines = _line_count(metrics_path)
        checkpoint_update = _latest_checkpoint_update(live_root)
        if metrics_lines != last_metrics_lines:
            last_metrics_lines = metrics_lines
            last_metrics_change = now
        if checkpoint_update != last_checkpoint_update:
            last_checkpoint_update = checkpoint_update
            last_checkpoint_change = now

        log_tail = _tail(args.log)
        fatal_patterns = [pattern for pattern in FATAL_LOG_PATTERNS if pattern in log_tail]
        heartbeat = {
            "checked_at": now,
            "elapsed_hours": elapsed_hours,
            "pid": args.pid,
            "pid_alive": _pid_alive(args.pid),
            "status_state": status.get("state"),
            "metrics_lines": metrics_lines,
            "seconds_since_metrics_change": now - last_metrics_change,
            "latest_checkpoint_update": checkpoint_update,
            "seconds_since_checkpoint_change": now - last_checkpoint_change,
            "gpu": _nvidia_smi(),
            "fatal_log_patterns": fatal_patterns,
            "target_min_hours": args.min_hours,
        }
        _write_json(heartbeat_path, heartbeat)
        _emit_event("TRAINING_MONITOR_HEARTBEAT", heartbeat)

        alert_reason = None
        if not heartbeat["pid_alive"]:
            alert_reason = "process_exited"
        elif str(status.get("state", "")).startswith("failed"):
            alert_reason = "status_failed"
        elif fatal_patterns:
            alert_reason = "fatal_pattern_in_log"
        elif (
            metrics_lines == 0
            and elapsed_hours * 60 > args.first_metrics_grace_minutes
        ):
            alert_reason = "first_metrics_missing"
        elif (
            metrics_lines > 0
            and now - last_metrics_change > args.metrics_stale_minutes * 60
        ):
            alert_reason = "metrics_stale"
        elif (
            checkpoint_update is not None
            and now - last_checkpoint_change > args.checkpoint_stale_minutes * 60
        ):
            alert_reason = "checkpoint_stale"

        if alert_reason:
            heartbeat["alert_reason"] = alert_reason
            _write_json(alert_path, heartbeat)
            _emit_event("TRAINING_MONITOR_ALERT", heartbeat)
            return 2
        if args.min_hours is not None and elapsed_hours >= args.min_hours:
            heartbeat["state"] = "completed_minimum_watch"
            _write_json(heartbeat_path, heartbeat)
            _emit_event("TRAINING_MONITOR_COMPLETE", heartbeat)
            return 0
        time.sleep(args.interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor a long 0024 League PPO run")
    parser.add_argument("--version", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--min-hours", type=float,
        help="Optional finite monitoring duration; omit for continuous supervision.",
    )
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--metrics-stale-minutes", type=float, default=45.0)
    parser.add_argument("--first-metrics-grace-minutes", type=float, default=45.0)
    parser.add_argument("--checkpoint-stale-minutes", type=float, default=90.0)
    return parser


def main() -> int:
    return monitor(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
