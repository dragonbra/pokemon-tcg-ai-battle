from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from .constants import REPOSITORY_ROOT


DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "rl_runs/0018_alakazam_terminal_rl/gpu_runtime_audit.jsonl"
)


def _compute_processes(match: str) -> list[dict[str, object]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    command_lines = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    commands: dict[int, str] = {}
    for line in command_lines:
        fields = line.strip().split(maxsplit=1)
        if fields:
            commands[int(fields[0])] = fields[1] if len(fields) > 1 else ""
    processes: list[dict[str, object]] = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        pid = int(fields[0])
        command = commands.get(pid, "")
        if match not in command:
            continue
        used_memory = None if fields[2].startswith("[") else float(fields[2])
        processes.append(
            {
                "pid": pid,
                "process_name": fields[1],
                "used_memory_mib": used_memory,
                "command": command,
            }
        )
    return processes


def _gpu_metrics() -> dict[str, float]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in query.stdout.splitlines()[0].split(",")]
    return {
        "gpu_utilization_percent": float(fields[0]),
        "gpu_memory_used_mib": float(fields[1]),
        "gpu_power_watts": float(fields[2]),
        "gpu_temperature_c": float(fields[3]),
    }


def _previous_active_seconds(path: Path) -> float:
    if not path.is_file() or not path.stat().st_size:
        return 0.0
    last = path.read_text(encoding="utf-8").splitlines()[-1]
    return float(json.loads(last).get("cumulative_active_seconds", 0.0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit cumulative 0018 CUDA process time")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--target-active-hours", type=float, default=12.0)
    parser.add_argument("--match", default="train.0018_alakazam_terminal_rl.run")
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0 or args.target_active_hours <= 0:
        raise ValueError("interval and target active hours must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cumulative = _previous_active_seconds(args.output)
    previous_sample = time.monotonic()
    while cumulative < args.target_active_hours * 3_600.0:
        sampled_at = time.monotonic()
        processes = _compute_processes(args.match)
        active = bool(processes)
        if active:
            cumulative += max(0.0, sampled_at - previous_sample)
        previous_sample = sampled_at
        record = {
            "schema_version": "0018_gpu_runtime_audit_v1",
            "timestamp": time.time(),
            "timestamp_iso": datetime.now().astimezone().isoformat(),
            "active": active,
            "cumulative_active_seconds": cumulative,
            "target_active_seconds": args.target_active_hours * 3_600.0,
            "processes": processes,
            **_gpu_metrics(),
        }
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
        if cumulative >= args.target_active_hours * 3_600.0:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
