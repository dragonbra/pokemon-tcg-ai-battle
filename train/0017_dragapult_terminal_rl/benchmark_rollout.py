from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from .opponents import balanced_jobs, load_frozen_pool, select_slice
from .policy.actor_critic import load_source_actor_critic
from .rollout.collector import RolloutCollector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0017 official-engine rollout benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args(argv)
    if args.games < 1 or args.workers < 1:
        raise ValueError("games and workers must be positive")

    torch.set_num_threads(4 if args.device == "cuda" else 1)
    device = torch.device(args.device)
    model, metadata = load_source_actor_critic()
    model.to(device).eval()
    packages, snapshot = load_frozen_pool()
    train_pool = select_slice(packages, snapshot, "train")
    jobs = balanced_jobs(
        train_pool,
        count=args.games,
        seed=args.seed,
        prefix=f"benchmark-w{args.workers}",
    )
    collector = RolloutCollector(model, device=device, workers=args.workers, mode="sample")
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    wall_seconds = time.perf_counter() - started
    valid = [episode for episode in episodes if episode.valid]
    inference_ms = [
        decision.inference_ms for episode in valid for decision in episode.decisions
    ]
    result = {
        "source_version": metadata["version"],
        "device": str(device),
        "workers": args.workers,
        "games": args.games,
        "valid_episodes": len(valid),
        "errors": len(episodes) - len(valid),
        "decisions": sum(len(episode.decisions) for episode in valid),
        "wall_seconds": wall_seconds,
        "episodes_per_second": len(valid) / wall_seconds,
        "decisions_per_second": (
            sum(len(episode.decisions) for episode in valid) / wall_seconds
        ),
        "inference_p50_ms": statistics.median(inference_ms) if inference_ms else 0.0,
        "cuda_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if len(valid) == args.games else 1


if __name__ == "__main__":
    raise SystemExit(main())
