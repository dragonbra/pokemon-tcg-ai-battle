"""Reproducible official-engine and cached-PPO performance gate."""

from __future__ import annotations

import json
import os
import time
import resource
import sys
from pathlib import Path
from unittest import mock

import torch

from .checkpoint import load_model_checkpoint
from .legacy_foundation import LegacyFoundationService
from .policy import load_actor_critic
from .rollout import HeterogeneousRolloutCollector
from .smoke import smoke_jobs
from .training.batch import prepare_episodes
from .training.ppo import PPOConfig, PPOTrainer
from .training.run import RunConfig, _episode_metrics, _evaluate_dual, build_jobs
from .worker_diagnostics import WorkerProcessSampler


def run(
    *,
    games: int,
    workers: int,
    device_name: str,
    output: Path,
    initialization_checkpoint: Path | None = None,
    opponent_foundation: str = "mixed",
) -> dict[str, object]:
    device = torch.device(device_name)
    torch.manual_seed(30030)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    model, _, identity = load_actor_critic(device)
    initialization_identity = (
        load_model_checkpoint(initialization_checkpoint, model)
        if initialization_checkpoint is not None
        else None
    )
    if opponent_foundation not in {"0019", "0028", "mixed"}:
        raise ValueError("unknown opponent foundation")
    legacy_service = (
        LegacyFoundationService(device)
        if opponent_foundation in {"0019", "mixed"}
        else None
    )
    collector = HeterogeneousRolloutCollector(
        model,
        device=device,
        workers=workers,
        mode="sample",
        coalesce_ms=0.5,
        opponent_foundation=opponent_foundation,
        legacy_service=legacy_service,
    )
    exact_training_schedule = (
        opponent_foundation == "mixed" and games >= 204 and games % 4 == 0
    )
    if exact_training_schedule:
        jobs = build_jobs(
            count=games,
            source_policy_update=5,
            seed=30030,
        )
    else:
        jobs = smoke_jobs(
            games,
            seed=30030,
            opponent_foundation=opponent_foundation,
        )
    sampler = WorkerProcessSampler()
    started = time.perf_counter()
    sampler.start()
    try:
        episodes = collector.collect(jobs)
    finally:
        sampler.stop()
    rollout_seconds = time.perf_counter() - started
    errors = [episode.error for episode in episodes if not episode.valid]
    if errors or len(episodes) != games:
        raise RuntimeError(f"official-engine benchmark errors: {errors}")
    batch = prepare_episodes(episodes)
    trainer = PPOTrainer(model, device=device, config=PPOConfig(epochs=2, batch_size=1024))
    representation = model.representation_sha256()
    opponent = model.opponent_decoder_sha256()
    ppo_started = time.perf_counter()
    with mock.patch.object(model.actor, "encode", side_effect=AssertionError("encoder called during PPO")):
        metrics = trainer.update(batch)
    ppo_seconds = time.perf_counter() - ppo_started
    decisions = batch.decisions
    result: dict[str, object] = {
        "schema_version": "0030_shared_encoder_performance_gate_v3_dual_foundation",
        "foundation_sha256": identity.checkpoint_sha256,
        "initialization_model_checkpoint": initialization_identity,
        "games": games,
        "workers": workers,
        "opponent_foundation": opponent_foundation,
        "schedule_contract": (
            "exact_dual_foundation_training"
            if exact_training_schedule
            else "smoke_subset"
        ),
        "valid": len(episodes),
        "errors": errors,
        "rollout_wall_seconds": rollout_seconds,
        "episodes_per_second": games / rollout_seconds,
        "focal_decisions": decisions,
        "focal_decisions_per_second": decisions / rollout_seconds,
        "ppo_wall_seconds": ppo_seconds,
        "ppo_cached_decisions_per_second": decisions * 2 / ppo_seconds,
        "ppo_encoder_calls": 0,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "representation_unchanged": model.representation_sha256() == representation,
        "opponent_decoder_unchanged": model.opponent_decoder_sha256() == opponent,
        **sampler.metrics(),
        **collector.metrics(),
        "ppo/minibatches_completed": metrics["ppo/minibatches_completed"],
    }
    if opponent_foundation == "mixed":
        for foundation_name in ("0019", "0028"):
            foundation_episodes = [
                episode
                for episode in episodes
                if episode.job.opponent_foundation == foundation_name
            ]
            result.update(_episode_metrics(
                foundation_episodes,
                f"rollout/foundation_{foundation_name}",
            ))
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    if not result["representation_unchanged"] or not result["opponent_decoder_unchanged"]:
        raise RuntimeError("frozen parameter contract changed during benchmark")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def run_dual_evaluation(
    *,
    initialization_checkpoint: Path,
    workers: int,
    device_name: str,
    seed: int,
    output: Path,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    model, _, foundation = load_actor_critic(device)
    checkpoint = load_model_checkpoint(initialization_checkpoint, model)
    legacy_service = LegacyFoundationService(device)
    config = RunConfig(
        version="V0_dual_foundation_diagnostic",
        workers=workers,
        device=device_name,
        seed=seed,
    )
    started = time.perf_counter()
    metrics, selection_score = _evaluate_dual(model, config, 0, legacy_service)
    result: dict[str, object] = {
        "schema_version": "0030_dual_foundation_evaluation_v1",
        "candidate_checkpoint": checkpoint,
        "candidate_foundation_sha256": foundation.checkpoint_sha256,
        "checkpoint_selection_foundation": "0019",
        "checkpoint_selection_score": selection_score,
        "games_per_foundation": config.eval_games,
        "seed": seed,
        "total_wall_seconds": time.perf_counter() - started,
        "workers": workers,
        **metrics,
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return result


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "train.0030_dragapult_shared_encoder_decoder_rl",
        "benchmark",
        *sys.argv[1:],
    ]
    os.execv(sys.executable, command)
    raise AssertionError("os.execv returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
