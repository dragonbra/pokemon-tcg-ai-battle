"""Foreground watchdog for long 0030 Dragapult decoder-only PPO runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from . import PROJECT_ID


RUN_ROOT = Path("rl_runs") / PROJECT_ID / "versions"
MONITOR_ROOT = Path(".tmp") / "training_monitor" / PROJECT_ID
FATAL_PATTERNS = (
    "Traceback (most recent call last)", "CUDA out of memory", "OutOfMemoryError",
    "worker EOF", "official-engine errors", "TimeoutError", "Killed",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _tail(path: Path, size: int = 131_072) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - size))
        return stream.read().decode(errors="replace")


def _memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        values[name] = int(raw.strip().split()[0]) * 1024
    return {
        "ram_available_bytes": values.get("MemAvailable", 0),
        "ram_total_bytes": values.get("MemTotal", 0),
        "swap_free_bytes": values.get("SwapFree", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
    }


def _gpu() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    result = subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"available": True, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def _emit(name: str, payload: dict[str, Any]) -> None:
    print(f"{name} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def monitor(args: argparse.Namespace) -> int:
    root = RUN_ROOT / args.version
    status_path = root / "artifact" / "status.json"
    metrics_path = root / "artifact" / "training_metrics.jsonl"
    monitor_root = MONITOR_ROOT / args.version
    started = time.time()
    last_lines = _line_count(metrics_path)
    last_change = started
    while True:
        now = time.time()
        status = _read_json(status_path)
        lines = _line_count(metrics_path)
        if lines != last_lines:
            last_lines, last_change = lines, now
        disk = shutil.disk_usage(Path.cwd())
        log_tail = _tail(args.log)
        fatal = [pattern for pattern in FATAL_PATTERNS if pattern in log_tail]
        wandb = status.get("wandb") if isinstance(status.get("wandb"), dict) else {}
        heartbeat = {
            "checked_at": now, "elapsed_hours": (now - started) / 3600,
            "pid": args.pid, "pid_alive": _pid_alive(args.pid),
            "status_state": status.get("state"), "completed_update": status.get("completed_update"),
            "metrics_lines": lines, "seconds_since_metrics_change": now - last_change,
            "checkpoint_count": len(list((root / "checkpoint").glob("update-*.pt"))),
            "wandb_state": wandb.get("sync_state", wandb.get("state")),
            "disk_free_bytes": disk.free,
            "cpu_count": os.cpu_count(),
            "cpu_load_1m": os.getloadavg()[0],
            "cpu_load_5m": os.getloadavg()[1],
            **_memory(), "gpu": _gpu(),
            "fatal_log_patterns": fatal,
        }
        _atomic_json(monitor_root / "heartbeat.json", heartbeat)
        _emit("TRAINING_MONITOR_HEARTBEAT", heartbeat)
        reason = None
        if status.get("state") == "complete":
            _emit("TRAINING_MONITOR_COMPLETE", heartbeat)
            return 0
        if not heartbeat["pid_alive"]:
            reason = "process_exited"
        elif status.get("state") == "failed":
            reason = "status_failed"
        elif fatal:
            reason = "fatal_pattern_in_log"
        elif disk.free < args.min_disk_gib * 1024**3:
            reason = "disk_low"
        elif heartbeat["ram_available_bytes"] < args.min_ram_gib * 1024**3:
            reason = "ram_low"
        elif lines == 0 and now - started > args.first_metrics_minutes * 60:
            reason = "first_metrics_missing"
        elif lines and now - last_change > args.metrics_stale_minutes * 60:
            reason = "metrics_stale"
        elif wandb.get("sync_state") == "failed":
            reason = "wandb_failed"
        if reason:
            heartbeat["alert_reason"] = reason
            _atomic_json(monitor_root / "alert.json", heartbeat)
            _emit("TRAINING_MONITOR_ALERT", heartbeat)
            return 2
        time.sleep(args.interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=float, default=60)
    parser.add_argument("--first-metrics-minutes", type=float, default=30)
    parser.add_argument("--metrics-stale-minutes", type=float, default=75)
    parser.add_argument("--min-disk-gib", type=float, default=10)
    parser.add_argument("--min-ram-gib", type=float, default=1)
    return monitor(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
