"""Fail-closed live supervisor for the V9-U50 to V10/G3 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import torch

from ..assets import sha256_file
from .run_v1 import ROOT, _atomic_json
from . import run_v10


V9_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / run_v10.PARENT_VERSION
CHECKPOINT = V9_ROOT / "checkpoint/update-000050.pt"
REPORT = V9_ROOT / "artifact/periodic_evaluation/update-000050/report.json"
METRICS = V9_ROOT / "artifact/training_metrics.jsonl"
MANIFEST = run_v10.HANDOFF_MANIFEST
SUPERVISOR_STATUS = V9_ROOT / "artifact/g3_handoff_supervisor_status.json"
V10_LOG = V9_ROOT / "artifact/v10_g3_training.log"
EXPECTED_V9_MODULE = "pokemon_tcg_ai.training.run_v9"


def _metrics_has_u50_evaluation(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("eval/checkpoint_update") == 50
            and row.get("eval/benchmark_v2") == 1.0
            and row.get("eval/games") == 2048
        ):
            return True
    return False


def build_handoff_manifest(
    *, checkpoint: Path = CHECKPOINT, report_path: Path = REPORT,
    metrics_path: Path = METRICS,
) -> dict[str, Any] | None:
    """Return exact immutable U50 evidence, or None while the gate is incomplete."""
    if not checkpoint.is_file() or not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        parent = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, ValueError, KeyError, RuntimeError):
        return None
    if (
        report.get("status") != "PASS"
        or report.get("benchmark_id") != "Benchmark-V2"
        or report.get("focal_checkpoint_update") != 50
        or report.get("summary", {}).get("games") != 2048
        or parent.get("schema_version") != "0044_focal_v1_model_only_v1"
        or parent.get("update") != 50
        or parent.get("metadata", {}).get("version") != run_v10.PARENT_VERSION
        or not _metrics_has_u50_evaluation(metrics_path)
    ):
        return None
    return {
        "schema_version": "0044_v9_u50_g3_handoff_v1",
        "status": "PASS",
        "parent_version": run_v10.PARENT_VERSION,
        "parent_checkpoint_update": 50,
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": sha256_file(checkpoint),
        "benchmark_v2": {
            "status": "PASS",
            "checkpoint_update": 50,
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "games": 2048,
            "win_rate": float(report["summary"]["win_rate"]),
        },
        "optimizer_boundary": "fresh_v10",
        "pfsp_state": None,
    }


def validated_v9_cmdline(pid: int) -> tuple[str, ...]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    argv = tuple(part.decode() for part in raw.split(b"\0") if part)
    if EXPECTED_V9_MODULE not in argv or "--launch-formal" not in argv:
        raise RuntimeError(f"PID {pid} is not the exact formal V9 process: {argv}")
    return argv


def v10_command() -> tuple[str, ...]:
    return (
        sys.executable, "-m",
        "pokemon_tcg_ai.training.run_v10",
        "--launch-formal", "--wandb-mode", "online",
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def supervise(*, v9_pid: int, poll_seconds: float = 2.0) -> None:
    validated_v9_cmdline(v9_pid)
    _atomic_json(SUPERVISOR_STATUS, {
        "state": "waiting_for_v9_u50", "v9_pid": v9_pid,
        "target_update": 50, "v10_version": run_v10.VERSION,
    })
    while True:
        if not _pid_alive(v9_pid):
            raise RuntimeError("V9 exited before the complete U50 handoff gate")
        validated_v9_cmdline(v9_pid)
        if (V9_ROOT / "checkpoint/update-000051.pt").exists():
            raise RuntimeError("V9 advanced to U51 before the exact-U50 handoff")
        manifest = build_handoff_manifest()
        if manifest is not None:
            break
        time.sleep(poll_seconds)

    _atomic_json(MANIFEST, manifest)
    _atomic_json(SUPERVISOR_STATUS, {
        "state": "stopping_v9_after_complete_u50_gate", "v9_pid": v9_pid,
        "parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
        "benchmark_v2": manifest["benchmark_v2"],
    })
    os.kill(v9_pid, signal.SIGINT)
    deadline = time.monotonic() + 120.0
    while _pid_alive(v9_pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if _pid_alive(v9_pid):
        raise RuntimeError("V9 did not stop gracefully within 120 seconds")
    if (V9_ROOT / "checkpoint/update-000051.pt").exists():
        raise RuntimeError("refusing V10 launch because V9 U51 is durable")
    if sha256_file(CHECKPOINT) != manifest["parent_checkpoint_sha256"]:
        raise RuntimeError("V9 U50 checkpoint changed after it was frozen")

    # Readiness must pass before the child can create any V10 paths.
    run_v10.readiness()
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    V10_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = V10_LOG.open("a", encoding="utf-8")
    process = subprocess.Popen(
        v10_command(), cwd=ROOT, env=env,
        stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    _atomic_json(SUPERVISOR_STATUS, {
        "state": "v10_launched", "v9_stopped_at_update": 50,
        "v10_pid": process.pid, "v10_version": run_v10.VERSION,
        "v10_command": list(v10_command()),
        "parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
    })

    v10_status = run_v10.VERSION_ROOT / "artifact/status.json"
    deadline = time.monotonic() + 1800.0
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"V10 exited during launch verification: {return_code}")
        if v10_status.is_file():
            payload = json.loads(v10_status.read_text(encoding="utf-8"))
            if payload.get("checkpoint_update", -1) >= 1:
                _atomic_json(SUPERVISOR_STATUS, {
                    "state": "v10_running_verified", "v9_stopped_at_update": 50,
                    "v10_pid": process.pid, "v10_version": run_v10.VERSION,
                    "v10_checkpoint_update": payload["checkpoint_update"],
                    "v10_wandb_run_id": run_v10.WANDB_RUN_ID,
                    "parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
                })
                return
        time.sleep(poll_seconds)
    raise RuntimeError("V10 did not reach its first durable update within 30 minutes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    try:
        supervise(v9_pid=args.v9_pid, poll_seconds=args.poll_seconds)
    except Exception as exc:
        _atomic_json(SUPERVISOR_STATUS, {
            "state": "failed_closed", "v9_pid": args.v9_pid,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
