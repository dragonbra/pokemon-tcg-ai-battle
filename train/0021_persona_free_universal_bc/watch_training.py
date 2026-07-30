"""Guard one formal 0021 run and resume only from its latest committed epoch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from rl_environment.runs import project_version_paths

from . import PROJECT_ID


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


def latest_checkpoint(checkpoint_root: Path) -> Path:
    criterion = checkpoint_root / "criteria/latest.json"
    try:
        manifest = json.loads(criterion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FileNotFoundError("no readable latest recovery checkpoint") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("path"), str):
        raise ValueError("latest checkpoint criterion is invalid")
    checkpoint = checkpoint_root / manifest["path"]
    if checkpoint.parent != checkpoint_root or not checkpoint.is_file():
        raise ValueError("latest checkpoint escapes or is absent from version root")
    return checkpoint


def _command(
    version: str,
    minimum_hours: float,
    batch_size: int,
    validation_batch_size: int,
    checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "train.0021_persona_free_universal_bc.run_r15",
        "--version",
        version,
        "--minimum-gpu-training-hours",
        str(minimum_hours),
        "--batch-size",
        str(batch_size),
        "--validation-batch-size",
        str(validation_batch_size),
    ]
    if checkpoint is not None:
        command.extend(("--resume-checkpoint", str(checkpoint)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="V1_corrected_persona_free_r15")
    parser.add_argument("--minimum-gpu-training-hours", type=float, default=12.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--restart-delay-seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    arguments = parser.parse_args()
    if arguments.minimum_gpu_training_hours < 12.0:
        raise ValueError("formal watchdog requires at least 12 GPU training hours")
    if (
        arguments.max_restarts < 0
        or arguments.restart_delay_seconds < 0
        or arguments.batch_size < 1
        or arguments.validation_batch_size < 1
    ):
        raise ValueError("watchdog restart limits must be nonnegative")
    paths = project_version_paths(PROJECT_ID, arguments.version)
    restart_count = 0
    while True:
        checkpoint = (
            latest_checkpoint(paths.checkpoints) if paths.run_root.exists() else None
        )
        if paths.status.is_file():
            status = json.loads(paths.status.read_text(encoding="utf-8"))
            if status.get("state") == "completed":
                return
        if checkpoint is not None and restart_count >= arguments.max_restarts:
            raise RuntimeError("0021 watchdog exhausted automatic restarts")
        attempt = restart_count + 1
        if paths.artifact.is_dir():
            _atomic_json(
                paths.artifact / "watchdog_status.json",
                {
                    "state": "launching",
                    "attempt": attempt,
                    "automatic_restarts_used": restart_count,
                    "maximum_restarts": arguments.max_restarts,
                    "minimum_gpu_training_hours": arguments.minimum_gpu_training_hours,
                    "batch_size": arguments.batch_size,
                    "validation_batch_size": arguments.validation_batch_size,
                    "resume_checkpoint": str(checkpoint) if checkpoint else None,
                    "updated_at": time.time(),
                },
            )
        result = subprocess.run(
            _command(
                arguments.version,
                arguments.minimum_gpu_training_hours,
                arguments.batch_size,
                arguments.validation_batch_size,
                checkpoint,
            ),
            check=False,
        )
        if result.returncode == 0:
            if paths.artifact.is_dir():
                _atomic_json(
                    paths.artifact / "watchdog_status.json",
                    {
                        "state": "completed",
                        "attempt": attempt,
                        "automatic_restarts_used": restart_count,
                        "maximum_restarts": arguments.max_restarts,
                        "minimum_gpu_training_hours": arguments.minimum_gpu_training_hours,
                        "batch_size": arguments.batch_size,
                        "validation_batch_size": arguments.validation_batch_size,
                        "updated_at": time.time(),
                    },
                )
            return
        restart_count += 1
        if restart_count > arguments.max_restarts:
            raise RuntimeError(
                f"0021 training failed with exit code {result.returncode}; "
                "watchdog exhausted automatic restarts"
            )
        latest_checkpoint(paths.checkpoints)
        time.sleep(arguments.restart_delay_seconds)


if __name__ == "__main__":
    main()


__all__ = ["latest_checkpoint", "main"]
