from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .opponents import balanced_jobs, load_frozen_pool, select_slice
from .opponent_inference import ResidentOpponentPool
from .opponent_inference.catalog import OpponentInferenceKind, classify_opponent
from .policy.actor_critic import load_source_actor_critic
from .rollout.collector import RolloutCollector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0018 official-engine rollout benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--resident-opponents", action="store_true")
    parser.add_argument("--pool", choices=("all", "resident"), default="all")
    parser.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--coalesce-ms", type=float, default=0.0)
    parser.add_argument("--opponent")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.games < 1 or args.workers < 1 or args.coalesce_ms < 0.0:
        raise ValueError("games/workers must be positive and coalesce-ms non-negative")

    torch.set_num_threads(4 if args.device == "cuda" else 1)
    device = torch.device(args.device)
    model, metadata = load_source_actor_critic()
    model.to(device).eval()
    packages, snapshot = load_frozen_pool()
    resident_started = time.perf_counter()
    resident_pool = None
    if args.resident_opponents:
        resident_pool = ResidentOpponentPool(packages, device=device)
    resident_load_seconds = time.perf_counter() - resident_started
    train_pool = select_slice(packages, snapshot, "train")
    if args.pool == "resident":
        train_pool = [
            package
            for package in train_pool
            if classify_opponent(package) != OpponentInferenceKind.CPU
            and (resident_pool is None or resident_pool.supports(package.name))
        ]
    if args.opponent:
        train_pool = [package for package in train_pool if package.name == args.opponent]
        if not train_pool:
            raise ValueError(f"opponent is absent from selected pool: {args.opponent}")
    jobs = balanced_jobs(
        train_pool,
        count=args.games,
        seed=args.seed,
        prefix=f"benchmark-w{args.workers}",
    )
    collector = RolloutCollector(
        model,
        device=device,
        workers=args.workers,
        mode=args.mode,
        resident_opponents=resident_pool,
        coalesce_seconds=args.coalesce_ms / 1_000.0,
    )
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
        "pool": args.pool,
        "mode": args.mode,
        "coalesce_ms": args.coalesce_ms,
        "valid_episodes": len(valid),
        "errors": len(episodes) - len(valid),
        "decisions": sum(len(episode.decisions) for episode in valid),
        "wall_seconds": wall_seconds,
        "episodes_per_second": len(valid) / wall_seconds,
        "decisions_per_second": (
            sum(len(episode.decisions) for episode in valid) / wall_seconds
        ),
        "inference_p50_ms": statistics.median(inference_ms) if inference_ms else 0.0,
        "candidate_inference": {
            "batch_count": collector.inference_batch_count,
            "request_count": collector.inference_request_count,
            "max_batch_size": collector.inference_batch_size_max,
            "mean_batch_size": (
                collector.inference_request_count / collector.inference_batch_count
                if collector.inference_batch_count
                else 0.0
            ),
            "inference_seconds": collector.inference_seconds,
        },
        "cuda_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "resident_opponents": resident_pool.audit() if resident_pool else None,
        "resident_load_seconds": resident_load_seconds,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if len(valid) == args.games else 1


if __name__ == "__main__":
    raise SystemExit(main())
