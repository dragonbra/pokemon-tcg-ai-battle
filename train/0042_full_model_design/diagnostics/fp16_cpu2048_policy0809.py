"""Eight independent official CPU-256 units for an FP16-deployed 0042 checkpoint.

The focal package follows kaggle_fp16_storage_fp32_runtime_v1. Both focal and
immutable Policy-0809 opponent inference may run on a separately materialized
GPU model while every official game engine remains CPU. Results are diagnostic
and never promote a checkpoint automatically.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import json
from pathlib import Path
import time
from typing import Any

import torch

from ..candidate_deployment import (
    CONTRACT_ID,
    audit_candidate_deployment,
)
from ..evaluation.frozen_jobs import build_frozen_jobs
from ..export_full_semantic_candidate import export_candidate
from ..rollout.collector import FullSemanticRolloutCollector
from ..training import run_full_semantic as runner
from .fp32_cpu2048_policy0809 import (
    _assert_cpu_chunk_health,
    _episode_rows,
    _load_package,
    _sha256,
)


SCHEMA = "0042_fp16_storage_fp32_runtime_cpu2048_policy0809_diagnostic_v1"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_update(checkpoint: Path) -> int:
    digest = _sha256(checkpoint)
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != digest:
        raise RuntimeError("checkpoint SHA-256 sidecar mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    update = payload.get("update")
    if not isinstance(update, int) or update < 0:
        raise RuntimeError("checkpoint update identity is missing")
    return update


def _materialize_cpu_candidate(
    checkpoint: Path, package: Path,
) -> tuple[Any, dict[str, Any], Any]:
    manifest = export_candidate(
        source=runner.CANDIDATE_ROOT,
        checkpoint=checkpoint,
        output=package,
        require_frozen_selection=False,
    )
    portable = package / "strategy/model.bin"
    payload = torch.load(portable, map_location="cpu", weights_only=True)
    model = _load_package(package, runner.focal_deck())
    audit = audit_candidate_deployment(
        manifest=manifest,
        payload=payload,
        runtime_model=model,
        source_checkpoint_sha256=_sha256(checkpoint),
        portable_checkpoint_sha256=_sha256(portable),
    )
    if audit.status != "PASS" or audit.contract_id != CONTRACT_ID:
        raise RuntimeError("FP16-storage/FP32-runtime candidate audit failed")
    model._candidate_deployment_audit = audit
    return model, manifest, audit


def run(
    *, checkpoint: Path, output_root: Path, torch_threads: int = 2,
    chunk_games: int = 32, resume: bool = False,
    inference_device: str = "cuda:0",
) -> dict[str, Any]:
    device = torch.device(inference_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but unavailable")
    if output_root.exists() and not resume:
        raise FileExistsError(output_root)
    if chunk_games < 1 or 256 % chunk_games:
        raise ValueError("chunk_games must be a positive divisor of 256")
    update = _checkpoint_update(checkpoint)
    output_root.mkdir(parents=True, exist_ok=resume)
    package = output_root / f"u{update:06d}_fp16_storage_fp32_runtime_package"
    if resume:
        manifest_path = package / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("resume requested without persistent candidate package")
        manifest = json.loads(manifest_path.read_text())
        model = _load_package(package, runner.focal_deck())
        payload = torch.load(
            package / "strategy/model.bin", map_location="cpu", weights_only=True
        )
        audit = audit_candidate_deployment(
            manifest=manifest,
            payload=payload,
            runtime_model=model,
            source_checkpoint_sha256=_sha256(checkpoint),
            portable_checkpoint_sha256=_sha256(package / "strategy/model.bin"),
        )
    else:
        model, manifest, audit = _materialize_cpu_candidate(checkpoint, package)
    if (
        manifest.get("checkpoint_update") != update
        or manifest.get("storage_dtype") != "fp16"
        or manifest.get("runtime_dtype") != "fp32"
        or audit.status != "PASS"
    ):
        raise RuntimeError("persistent candidate identity does not match checkpoint")

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    opponent = runner.load_frozen_opponent(device)
    opponent_audit = opponent._policy_identity_audit
    jobs, schedule_sha256 = build_frozen_jobs(
        focal_deck_id=runner.FOCAL_DECK_ID,
        focal_deck=runner.focal_deck(),
        runtime_root=runner.runtime_root(),
        source_policy_update=update,
        focal_deployment_identity=audit.effective_candidate_sha256,
        opponent_effective_policy_sha256=opponent_audit.effective_policy_sha256,
        evaluation_units=8,
    )

    rows: list[dict[str, Any]] = []
    completed_units = 0
    if resume:
        for number in range(1, 9):
            path = output_root / f"unit-{number:02d}.json"
            if not path.is_file():
                break
            entries = json.loads(path.read_text()).get("entries")
            if not isinstance(entries, list) or len(entries) != 256:
                raise RuntimeError(f"malformed completed unit: {path}")
            rows.extend(entries)
            completed_units += 1

    started = time.perf_counter()
    for unit in range(completed_units, 8):
        unit_started = time.perf_counter()
        unit_rows: list[dict[str, Any]] = []
        unit_metrics: list[dict[str, float]] = []
        unit_jobs = jobs[unit * 256:(unit + 1) * 256]
        for chunk_start in range(0, 256, chunk_games):
            chunk = unit_jobs[chunk_start:chunk_start + chunk_games]
            collector = FullSemanticRolloutCollector(
                model,
                opponent,
                device=device,
                worker_processes=min(8, len(chunk)),
                engines_per_worker=4,
                inference_channels_per_role=4,
                mode="greedy",
                coalesce_ms=5.0,
                timeout_seconds=300.0,
                record_trajectory=False,
            )
            episodes = collector.collect(chunk)
            _assert_cpu_chunk_health(episodes, expected_games=len(chunk))
            chunk_rows = _episode_rows(
                episodes, opponent_audit.effective_policy_sha256
            )
            unit_rows.extend(chunk_rows)
            rows.extend(chunk_rows)
            unit_metrics.append(collector.metrics())
            progress = {
                "schema_version": SCHEMA,
                "status": "RUNNING",
                "completed_units": unit,
                "total_units": 8,
                "active_unit": unit + 1,
                "active_unit_completed_games": len(unit_rows),
                "completed_games": len(rows),
                "total_games": 2048,
                "wins": sum(row["outcome"] == 1 for row in rows),
                "losses": sum(row["outcome"] == -1 for row in rows),
                "draws": sum(row["outcome"] == 0 for row in rows),
                "win_rate": sum(row["outcome"] == 1 for row in rows) / len(rows),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _atomic_json(output_root / "progress.json", progress)
            _atomic_json(output_root / f"unit-{unit + 1:02d}-partial.json", {
                **progress, "entries": unit_rows,
            })
            print(json.dumps(progress, sort_keys=True), flush=True)
            del episodes, collector
            gc.collect()

        unit_report = {
            "unit": unit + 1,
            "games": 256,
            "wins": sum(row["outcome"] == 1 for row in unit_rows),
            "losses": sum(row["outcome"] == -1 for row in unit_rows),
            "draws": sum(row["outcome"] == 0 for row in unit_rows),
            "win_rate": sum(row["outcome"] == 1 for row in unit_rows) / 256,
            "elapsed_seconds": time.perf_counter() - unit_started,
            "collector_metrics_by_chunk": unit_metrics,
            "entries": unit_rows,
        }
        _atomic_json(output_root / f"unit-{unit + 1:02d}.json", unit_report)

    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "created_at": datetime.now(UTC).isoformat(),
        "contract_id": CONTRACT_ID,
        "benchmark_kind": "eight_independent_official_cpu_256_units",
        "official_engine_device": "cpu",
        "neural_inference_device": str(device),
        "checkpoint_update": update,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "package": str(package),
        "package_manifest": manifest,
        "candidate_deployment_identity_audit": audit.to_manifest(),
        "opponent_policy_identity_audit": opponent_audit.to_manifest(),
        "schedule_sha256": schedule_sha256,
        "completed_games": len(rows),
        "wins": sum(row["outcome"] == 1 for row in rows),
        "losses": sum(row["outcome"] == -1 for row in rows),
        "draws": sum(row["outcome"] == 0 for row in rows),
        "win_rate": sum(row["outcome"] == 1 for row in rows) / len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "promotion_status": "NOT_PROMOTED_DIAGNOSTIC_ONLY",
        "entries": rows,
    }
    _atomic_json(output_root / "report.json", report)
    _atomic_json(output_root / "progress.json", {
        key: report[key] for key in (
            "schema_version", "status", "completed_games", "wins", "losses",
            "draws", "win_rate", "elapsed_seconds",
        )
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--chunk-games", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inference-device", default="cuda:0")
    args = parser.parse_args()
    report = run(
        checkpoint=args.checkpoint.resolve(),
        output_root=args.output_root.resolve(),
        torch_threads=args.torch_threads,
        chunk_games=args.chunk_games,
        resume=args.resume,
        inference_device=args.inference_device,
    )
    print(json.dumps({key: report[key] for key in (
        "status", "completed_games", "wins", "losses", "draws", "win_rate"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
