"""Read-only frozen official-engine evaluation for an 0042 model-only checkpoint."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from .policy import AdaptationConfig, load_actor_critic
from .training.run_full_semantic import (
    PROJECT,
    SOURCE_CHECKPOINT,
    FullSemanticRolloutCollector,
    _episode_metrics,
    build_jobs,
    focal_deck,
    load_frozen_opponent,
)
from .training.storage_full_semantic import load_adapted_model_only
from .checkpoint import CHECKPOINT_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_FIELDS = (
    "opponent_id",
    "focal_first",
    "engine_seed",
    "search_seed",
    "policy_seed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _environment_rows(jobs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "opponent_id": job.opponent_id,
            "focal_first": job.focal_first,
            "engine_seed": job.seed,
            "search_seed": job.search_seed,
            "policy_seed": job.policy_seed,
        }
        for job in jobs
    ]


def assert_fixed_environment(jobs: list[Any], baseline_schedule: Path) -> str:
    baseline = json.loads(baseline_schedule.read_text())
    expected = [
        {field: row[field] for field in ENVIRONMENT_FIELDS}
        for row in baseline["jobs"]
    ]
    actual = _environment_rows(jobs)
    if actual != expected:
        mismatch = next(
            (index for index, pair in enumerate(zip(actual, expected, strict=False)) if pair[0] != pair[1]),
            min(len(actual), len(expected)),
        )
        raise RuntimeError(f"frozen evaluation environment changed at job {mismatch}")
    encoded = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_adaptation(checkpoint: Path, expected_update: int) -> AdaptationConfig:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("update") != expected_update:
        raise ValueError(
            f"checkpoint update mismatch: {payload.get('update')} != {expected_update}"
        )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint is not an 0042 model-only checkpoint")
    return AdaptationConfig()


def load_checkpoint(model: Any, checkpoint: Path, expected_update: int) -> dict[str, Any]:
    digest = _sha256(checkpoint)
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != digest:
        raise ValueError("checkpoint SHA-256 sidecar mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint is not an 0042 model-only checkpoint")
    if payload.get("update") != expected_update:
        raise ValueError(
            f"checkpoint update mismatch: {payload.get('update')} != {expected_update}"
        )
    representation = model.representation_sha256()
    load_adapted_model_only(model, checkpoint)
    if model.representation_sha256() != representation:
        raise RuntimeError("checkpoint load changed frozen actor representation")
    model.eval()
    return {
        "checkpoint_sha256": digest,
        "checkpoint_schema": payload["schema_version"],
        "checkpoint_update": payload["update"],
        "metadata": payload.get("metadata") or {},
        "representation_sha256": representation,
        "decoder_sha256": model.decoder_sha256(),
    }


def wilson_interval(wins: int, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    spread = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return centre - spread, centre + spread


def run(
    *,
    checkpoint: Path,
    update: int,
    baseline_schedule: Path,
    output: Path,
    device_name: str = "cuda:0",
    worker_processes: int = 16,
    engines_per_worker: int = 8,
    inference_channels_per_role: int = 8,
    coalesce_ms: float = 5.0,
) -> dict[str, Any]:
    device = torch.device(device_name)
    adaptation = checkpoint_adaptation(checkpoint, update)
    model, source_identity = load_actor_critic(
        SOURCE_CHECKPOINT,
        focal_deck(),
        device,
        adaptation=adaptation,
    )
    checkpoint_identity = load_checkpoint(model, checkpoint, update)
    opponent = load_frozen_opponent(device)
    jobs = build_jobs(
        source_policy_update=update,
        seed=330031001 + 70_000_000,
        count=512,
        greedy=True,
    )
    environment_sha256 = assert_fixed_environment(jobs, baseline_schedule)
    collector = FullSemanticRolloutCollector(
        model,
        opponent,
        device=device,
        worker_processes=worker_processes,
        engines_per_worker=engines_per_worker,
        inference_channels_per_role=inference_channels_per_role,
        mode="greedy",
        coalesce_ms=coalesce_ms,
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    wall_seconds = time.perf_counter() - started
    metrics = _episode_metrics(episodes, "eval")
    metrics.update(collector.metrics())
    wins = int(metrics["eval/wins"])
    low, high = wilson_interval(wins, len(episodes))
    result = {
        "schema": "0042_frozen_checkpoint_evaluation_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "project": PROJECT,
        "mode": "greedy",
        "official_engine": True,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_identity": checkpoint_identity,
        "source_identity": {
            key: getattr(source_identity, key)
            for key in source_identity.__dataclass_fields__
        },
        "opponent_policy_id": opponent._policy_id,
        "policy_identity_audit": opponent._policy_identity_audit.to_manifest(),
        "schedule": {
            "baseline_schedule": str(baseline_schedule.relative_to(ROOT)),
            "environment_sha256": environment_sha256,
            "environment_matches_baseline": True,
            "episodes": len(jobs),
            "seed_pairs": len(jobs) // 2,
        },
        "topology": {
            "worker_processes": worker_processes,
            "engines_per_worker": engines_per_worker,
            "inference_channels_per_role": inference_channels_per_role,
            "coalesce_ms": coalesce_ms,
        },
        "wall_seconds": wall_seconds,
        "wilson_95": [low, high],
        "metrics": metrics,
        "games": [
            {
                "game_id": episode.job.game_id,
                "opponent_id": episode.job.opponent_id,
                "focal_first": episode.job.focal_first,
                "engine_seed": episode.job.seed,
                "search_seed": episode.job.search_seed,
                "policy_seed": episode.job.policy_seed,
                "reward": episode.reward,
                "turns": episode.turns,
                "decisions": len(episode.decisions),
                "valid": episode.valid,
                "error": episode.error,
            }
            for episode in episodes
        ],
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    version_root = ROOT / "rl_runs" / PROJECT / "versions/V2_engine_pool_prototype_cache_seeded512"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", type=int, default=10)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=version_root / "checkpoint/update-000010.pt",
    )
    parser.add_argument(
        "--baseline-schedule",
        type=Path,
        default=version_root / "artifact/schedules/eval_fixed_seeded512.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".tmp/evaluation/0037_update10_frozen/result.json",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(
        checkpoint=args.checkpoint.resolve(),
        update=args.update,
        baseline_schedule=args.baseline_schedule.resolve(),
        output=args.output.resolve(),
        device_name=args.device,
    )
    print(json.dumps({
        "output": str(args.output),
        "wall_seconds": result["wall_seconds"],
        "wilson_95": result["wilson_95"],
        "metrics": result["metrics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
