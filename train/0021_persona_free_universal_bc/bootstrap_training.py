"""Finish corrected data, calibrate CUDA batches, and guard formal 0021 training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import torch

from .config import MODEL_READY_DATASET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "experiments/0021_persona_free_universal_bc/manifest.json"
CALIBRATION = (
    PROJECT_ROOT / "experiments/0021_persona_free_universal_bc/batch_calibration.json"
)
LEGACY_MODEL_READY = PROJECT_ROOT / "rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _run(command: list[str]) -> int:
    print(json.dumps({"event": "0021_bootstrap_command", "command": command}), flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def _smoke(batch_size: int) -> dict[str, Any]:
    tag = f"V0_batch{batch_size}_throughput_smoke"
    root = PROJECT_ROOT / f".tmp/0021_persona_free_universal_bc/{tag}"
    if root.exists():
        raise FileExistsError(f"smoke output already exists: {root}")
    return_code = _run(
        [
            sys.executable,
            "-m",
            "train.0021_persona_free_universal_bc.run_r15",
            "--smoke",
            "--smoke-tag",
            tag,
            "--batch-size",
            str(batch_size),
            "--validation-batch-size",
            "512",
        ]
    )
    if return_code != 0:
        return {"batch_size": batch_size, "completed": False, "return_code": return_code}
    summary = json.loads(
        (root / "artifact/training_summary.json").read_text(encoding="utf-8")
    )
    metrics = summary["history"][-1]
    return {
        "batch_size": batch_size,
        "completed": True,
        "return_code": 0,
        "train_decisions_per_second": metrics["system/train_decisions_per_second"],
        "train_iterations_per_second": metrics["system/train_iterations_per_second"],
        "max_memory_allocated_bytes": metrics["system/gpu/max_memory_allocated_bytes"],
        "memory_reserved_bytes": metrics["system/gpu/memory_reserved_bytes"],
        "optimization_loss": metrics["bc/optimization/loss"],
    }


def _choose_batch(results: list[dict[str, Any]]) -> int:
    successful = [result for result in results if result["completed"]]
    if not successful:
        raise RuntimeError("all 0021 CUDA batch calibration smokes failed")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    safe = [
        result
        for result in successful
        if result["max_memory_allocated_bytes"] <= 0.9 * total_memory
    ]
    candidates = safe or successful
    return int(max(candidates, key=lambda item: item["train_decisions_per_second"])["batch_size"])


def _delete_verified_legacy_features() -> None:
    expected = (
        PROJECT_ROOT / "rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner"
    ).resolve()
    if LEGACY_MODEL_READY.resolve() != expected:
        raise RuntimeError("legacy feature deletion target changed")
    if LEGACY_MODEL_READY.is_dir():
        shutil.rmtree(LEGACY_MODEL_READY)
        print(
            json.dumps(
                {
                    "event": "0021_legacy_features_deleted",
                    "path": str(LEGACY_MODEL_READY),
                    "recoverable": False,
                    "reason": "corrected dataset validated and smoke-tested",
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materializer-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    arguments = parser.parse_args()
    if arguments.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    while not MODEL_READY_DATASET.is_dir():
        if not _alive(arguments.materializer_pid):
            raise RuntimeError("materializer exited without atomically publishing the dataset")
        time.sleep(arguments.poll_seconds)
    if _run(
        [sys.executable, "-m", "train.0021_persona_free_universal_bc.finalize_data"]
    ):
        raise RuntimeError("corrected dataset validation/publication failed")
    calibration = [_smoke(256), _smoke(512)]
    chosen_batch = _choose_batch(calibration)
    calibration_payload = {
        "schema_version": "0021_batch_calibration_v1",
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "results": calibration,
        "chosen_batch_size": chosen_batch,
        "validation_batch_size": 512,
        "selection_rule": "highest decisions/s among successful runs using <=90% allocated VRAM",
        "created_at": time.time(),
    }
    _atomic_json(CALIBRATION, calibration_payload)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["training"]["batch_calibration"] = str(CALIBRATION.relative_to(PROJECT_ROOT))
    manifest["training"]["chosen_batch_size"] = chosen_batch
    manifest["training"]["validation_batch_size"] = 512
    _atomic_json(MANIFEST, manifest)
    _delete_verified_legacy_features()
    result = _run(
        [
            sys.executable,
            "-m",
            "train.0021_persona_free_universal_bc.watch_training",
            "--version",
            "V1_corrected_persona_free_r15",
            "--minimum-gpu-training-hours",
            "12",
            "--batch-size",
            str(chosen_batch),
            "--validation-batch-size",
            "512",
            "--max-restarts",
            "3",
        ]
    )
    if result:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
