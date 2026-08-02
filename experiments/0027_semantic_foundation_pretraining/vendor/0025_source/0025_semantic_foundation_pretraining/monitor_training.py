"""Foreground watchdog for a formal 0025 BC training process."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ALERT_TERMS = (
    "out of memory",
    "cuda oom",
    "worker eof",
    "broken pipe",
    "timeout",
    "traceback (most recent call last)",
)


def _memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        token = raw.strip().split()[0]
        if key in {"MemAvailable", "SwapFree"}:
            values[key] = int(token) * 1024
    return values


def _gpu() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip() or f"exit {result.returncode}"}
    values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    return {
        "memory_used_mib": int(values[0]),
        "memory_free_mib": int(values[1]),
        "utilization_percent": int(values[2]),
        "temperature_c": int(values[3]),
    }


def _json_or_none(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _snapshot(run_root: Path, child: subprocess.Popen[str]) -> dict[str, Any]:
    artifact = run_root / "artifact"
    memory = _memory()
    disk = shutil.disk_usage(run_root.parent if run_root.parent.exists() else Path.cwd())
    status = _json_or_none(artifact / "status.json")
    metrics = artifact / "training_metrics.jsonl"
    return {
        "child_pid": child.pid,
        "child_returncode": child.poll(),
        "status": status,
        "metrics_bytes": metrics.stat().st_size if metrics.is_file() else 0,
        "checkpoint_files": len(list((run_root / "checkpoint").rglob("*.pt")))
        if (run_root / "checkpoint").is_dir()
        else 0,
        "wandb": status.get("wandb") if isinstance(status, dict) else None,
        "gpu": _gpu(),
        "memory_available_bytes": memory.get("MemAvailable", 0),
        "swap_free_bytes": memory.get("SwapFree", 0),
        "disk_free_bytes": disk.free,
        "load_average": list(os.getloadavg()),
    }


def _emit(event: str, payload: dict[str, Any], monitor_path: Path | None) -> None:
    record = {"event": event, "timestamp": time.time(), **payload}
    line = json.dumps(record, ensure_ascii=True, sort_keys=True)
    print(line, flush=True)
    if monitor_path is None:
        return
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    with monitor_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entrypoint",
        choices=("ablation", "canonical"),
        default="ablation",
    )
    parser.add_argument("--version", default="V2_james_cox_raging_bolt_ablation")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    args = parser.parse_args()
    if not 5 <= args.heartbeat_seconds <= 60:
        raise ValueError("heartbeat interval must be between 5 and 60 seconds")
    run_root = (
        Path("rl_runs/0025_semantic_foundation_pretraining/versions") / args.version
    )
    monitor_path = run_root / "artifact/training_monitor.jsonl"
    module = (
        "train.0025_semantic_foundation_pretraining.run_canonical_bc"
        if args.entrypoint == "canonical"
        else "train.0025_semantic_foundation_pretraining.run_bc_ablation"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--version",
        args.version,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--validation-batch-size",
        str(args.validation_batch_size),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
    ]
    environment = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    output: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert child.stdout is not None
        for line in child.stdout:
            output.put(line)
        output.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    _emit(
        "TRAINING_MONITOR_HEARTBEAT",
        {"state": "started", "command": command, "child_pid": child.pid},
        None,
    )
    next_heartbeat = time.monotonic()
    output_closed = False
    alerts: set[str] = set()
    try:
        while child.poll() is None or not output_closed:
            timeout = max(0.1, min(1.0, next_heartbeat - time.monotonic()))
            try:
                line = output.get(timeout=timeout)
            except queue.Empty:
                line = ""
            if line is None:
                output_closed = True
            elif line:
                print(line, end="", flush=True)
                lowered = line.casefold()
                for term in ALERT_TERMS:
                    if term in lowered and term not in alerts:
                        alerts.add(term)
                        _emit(
                            "TRAINING_MONITOR_ALERT",
                            {"reason": f"training output matched {term!r}"},
                            monitor_path if run_root.exists() else None,
                        )
            if time.monotonic() >= next_heartbeat:
                snapshot = _snapshot(run_root, child)
                _emit(
                    "TRAINING_MONITOR_HEARTBEAT",
                    snapshot,
                    monitor_path if run_root.exists() else None,
                )
                next_heartbeat = time.monotonic() + args.heartbeat_seconds
        returncode = child.wait()
        snapshot = _snapshot(run_root, child)
        if returncode != 0:
            _emit(
                "TRAINING_MONITOR_ALERT",
                {**snapshot, "reason": f"training child exited {returncode}"},
                monitor_path if run_root.exists() else None,
            )
            raise SystemExit(returncode)
        status = snapshot.get("status")
        if not isinstance(status, dict) or status.get("state") != "completed":
            _emit(
                "TRAINING_MONITOR_ALERT",
                {**snapshot, "reason": "child exited zero without completed status"},
                monitor_path if run_root.exists() else None,
            )
            raise SystemExit(2)
        _emit(
            "TRAINING_MONITOR_COMPLETE",
            snapshot,
            monitor_path,
        )
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise


if __name__ == "__main__":
    main()
