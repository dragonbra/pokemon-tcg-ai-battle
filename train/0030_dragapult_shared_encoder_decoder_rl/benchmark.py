"""Reproducible official-engine and cached-PPO performance gate."""

from __future__ import annotations

import argparse
import json
import time
import resource
from pathlib import Path
from unittest import mock

import torch

from .policy import load_actor_critic
from .rollout import HeterogeneousRolloutCollector
from .smoke import smoke_jobs
from .training.batch import prepare_episodes
from .training.ppo import PPOConfig, PPOTrainer


def run(*, games: int, workers: int, device_name: str, output: Path) -> dict[str, object]:
    device = torch.device(device_name)
    torch.manual_seed(30030)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    model, _, identity = load_actor_critic(device)
    collector = HeterogeneousRolloutCollector(model, device=device, workers=workers, mode="sample", coalesce_ms=0.5)
    started = time.perf_counter()
    episodes = collector.collect(smoke_jobs(games, seed=30030))
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
        "schema_version": "0030_shared_encoder_performance_gate_v1",
        "foundation_sha256": identity.checkpoint_sha256,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path(".tmp/evaluation/0030_performance/performance_gate.json"))
    args = parser.parse_args()
    print(json.dumps(run(games=args.games, workers=args.workers, device_name=args.device, output=args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
