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
from .policy import load_actor_critic
from .rollout import HeterogeneousRolloutCollector
from .smoke import smoke_jobs
from .training.batch import prepare_episodes
from .training.ppo import PPOConfig, PPOTrainer
from .worker_diagnostics import WorkerProcessSampler


def run(
    *,
    games: int,
    workers: int,
    device_name: str,
    output: Path,
    initialization_checkpoint: Path | None = None,
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
    collector = HeterogeneousRolloutCollector(model, device=device, workers=workers, mode="sample", coalesce_ms=0.5)
    sampler = WorkerProcessSampler()
    started = time.perf_counter()
    sampler.start()
    try:
        episodes = collector.collect(smoke_jobs(games, seed=30030))
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
        "schema_version": "0030_shared_encoder_performance_gate_v2",
        "foundation_sha256": identity.checkpoint_sha256,
        "initialization_model_checkpoint": initialization_identity,
        "games": games,
        "workers": workers,
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
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    if not result["representation_unchanged"] or not result["opponent_decoder_unchanged"]:
        raise RuntimeError("frozen parameter contract changed during benchmark")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
