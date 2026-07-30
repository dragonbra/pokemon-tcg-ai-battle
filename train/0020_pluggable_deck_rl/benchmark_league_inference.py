from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Literal


Arm = Literal[
    "a_cpu_packages",
    "b_shared_cpu",
    "c_shared_gpu",
    "d_unified_shared_gpu",
]
ARMS: tuple[Arm, ...] = (
    "a_cpu_packages",
    "b_shared_cpu",
    "c_shared_gpu",
    "d_unified_shared_gpu",
)


class _GpuSampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        utilization = [sample[0] for sample in self.samples]
        memory_mib = [sample[1] for sample in self.samples]
        return {
            "sample_count": len(self.samples),
            "utilization_mean_percent": statistics.fmean(utilization) if utilization else 0.0,
            "utilization_max_percent": max(utilization, default=0.0),
            "memory_used_mean_mib": statistics.fmean(memory_mib) if memory_mib else 0.0,
            "memory_used_max_mib": max(memory_mib, default=0.0),
        }

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                        "--id=0",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                utilization, memory_mib = completed.stdout.strip().split(",", maxsplit=1)
                self.samples.append((float(utilization), float(memory_mib)))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_seconds)


def _cuda_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {"allocated": 0, "reserved": 0, "peak_allocated": 0, "peak_reserved": 0}
    torch.cuda.synchronize(device)
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
    }


def _run_arm(
    arm: Arm,
    *,
    candidate: torch.nn.Module,
    candidate_device: torch.device,
    packages: list[Any],
    games: int,
    workers: int,
    seed: int,
    coalesce_ms: float,
    mode: Literal["greedy", "sample"],
) -> dict[str, Any]:
    opponent_pool = None
    opponent_load_seconds = 0.0
    unified = arm == "d_unified_shared_gpu"
    candidate.to(torch.device("cpu") if unified else candidate_device).eval()
    if arm != "a_cpu_packages":
        opponent_device = torch.device("cpu" if arm == "b_shared_cpu" else "cuda")
        load_started = time.perf_counter()
        opponent_pool = SharedFoundationOpponentPool(
            packages,
            device=opponent_device,
            include_candidate=unified,
        )
        opponent_load_seconds = time.perf_counter() - load_started

    if candidate_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(candidate_device)
    memory_before = _cuda_memory(candidate_device)
    jobs = balanced_jobs(
        packages,
        count=games,
        seed=seed,
        prefix=f"0022-{arm}",
    )
    collector = RolloutCollector(
        candidate,
        device=candidate_device,
        workers=workers,
        mode=mode,
        resident_opponents=opponent_pool,
        coalesce_seconds=coalesce_ms / 1_000.0,
        unified_league_inference=unified,
    )
    gpu_sampler = _GpuSampler()
    gpu_sampler.start()
    started = time.perf_counter()
    try:
        episodes = collector.collect(jobs)
    finally:
        wall_seconds = time.perf_counter() - started
        gpu_samples = gpu_sampler.stop()
    memory_after = _cuda_memory(candidate_device)
    valid = [episode for episode in episodes if episode.valid]
    candidate_decisions = sum(len(episode.decisions) for episode in valid)
    engine_selections = sum(episode.engine_selections for episode in valid)
    opponent_decisions = (
        engine_selections - candidate_decisions
        if opponent_pool is None or unified
        else opponent_pool.request_count
    )
    inference_ms = [
        decision.inference_ms for episode in valid for decision in episode.decisions
    ]
    trajectory_decisions = [
        decision for episode in valid for decision in episode.decisions
    ]
    return {
        "arm": arm,
        "contract": {
            "official_engine": True,
            "candidate_device": str(candidate_device),
            "opponent_mode": (
                "isolated_cpu_package"
                if opponent_pool is None
                else (
                    "unified_candidate_and_opponent_shared_foundation"
                    if unified
                    else "shared_foundation_routed_decoders"
                )
            ),
            "opponent_device": None if opponent_pool is None else str(opponent_pool.device),
            "games": games,
            "workers": workers,
            "seed": seed,
            "coalesce_ms": coalesce_ms,
            "mode": mode,
            "deck_schedule": [job.opponent.name for job in jobs],
            "candidate_first": [job.candidate_first for job in jobs],
        },
        "completed": len(valid),
        "errors": len(episodes) - len(valid),
        "statuses": {
            status: sum(episode.status == status for episode in episodes)
            for status in sorted({episode.status for episode in episodes})
        },
        "error_messages": [episode.error for episode in episodes if episode.error],
        "wall_seconds": wall_seconds,
        "episodes_per_second": len(valid) / wall_seconds,
        "engine_selections": engine_selections,
        "engine_selections_per_second": engine_selections / wall_seconds,
        "candidate_decisions": candidate_decisions,
        "candidate_decisions_per_second": candidate_decisions / wall_seconds,
        "opponent_decisions": opponent_decisions,
        "opponent_decisions_per_second": opponent_decisions / wall_seconds,
        "complete_rounds_mean": statistics.fmean(
            episode.complete_rounds for episode in valid if episode.complete_rounds is not None
        ) if valid else 0.0,
        "candidate_inference": {
            "scope": "all_league_requests" if unified else "candidate_only",
            "batch_count": collector.inference_batch_count,
            "request_count": collector.inference_request_count,
            "max_batch_size": collector.inference_batch_size_max,
            "mean_batch_size": (
                collector.inference_request_count / collector.inference_batch_count
                if collector.inference_batch_count
                else 0.0
            ),
            "seconds": collector.inference_seconds,
            "request_p50_ms": statistics.median(inference_ms) if inference_ms else 0.0,
            "log_prob_mean": (
                statistics.fmean(decision.old_log_prob for decision in trajectory_decisions)
                if trajectory_decisions
                else 0.0
            ),
            "entropy_mean": (
                statistics.fmean(decision.entropy for decision in trajectory_decisions)
                if trajectory_decisions
                else 0.0
            ),
            "value_mean": (
                statistics.fmean(decision.old_value for decision in trajectory_decisions)
                if trajectory_decisions
                else 0.0
            ),
            "nonfinite_trajectory_scalars": sum(
                not math.isfinite(value)
                for decision in trajectory_decisions
                for value in (
                    decision.old_log_prob,
                    decision.entropy,
                    decision.old_value,
                )
            ),
        },
        "opponent_inference": opponent_pool.audit() if opponent_pool is not None else None,
        "opponent_load_seconds": opponent_load_seconds,
        "cuda_memory_before": memory_before,
        "cuda_memory_after": memory_after,
        "nvidia_smi": gpu_samples,
    }


def _ratios(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {result["arm"]: result for result in results}
    ratios: dict[str, Any] = {}
    if "a_cpu_packages" in by_arm and "c_shared_gpu" in by_arm:
        baseline = by_arm["a_cpu_packages"]
        shared = by_arm["c_shared_gpu"]
        ratios["c_vs_a"] = {
            "episodes_per_second": (
                shared["episodes_per_second"] / baseline["episodes_per_second"]
            ),
            "engine_selections_per_second": (
                shared["engine_selections_per_second"]
                / baseline["engine_selections_per_second"]
            ),
            "evidence_boundary": (
                "Same engine, candidate, deck/seat schedule, workers and seeds; opponent policy "
                "implementation differs, so selection-normalized throughput is also reported."
            ),
        }
    if "b_shared_cpu" in by_arm and "c_shared_gpu" in by_arm:
        cpu = by_arm["b_shared_cpu"]
        gpu = by_arm["c_shared_gpu"]
        ratios["c_vs_b"] = {
            "episodes_per_second": gpu["episodes_per_second"] / cpu["episodes_per_second"],
            "engine_selections_per_second": (
                gpu["engine_selections_per_second"] / cpu["engine_selections_per_second"]
            ),
            "opponent_decisions_per_second": (
                gpu["opponent_decisions_per_second"] / cpu["opponent_decisions_per_second"]
            ),
            "evidence_boundary": "Same shared policy weights, decks and schedule; device differs.",
        }
    if "a_cpu_packages" in by_arm and "d_unified_shared_gpu" in by_arm:
        baseline = by_arm["a_cpu_packages"]
        unified = by_arm["d_unified_shared_gpu"]
        ratios["d_vs_a"] = {
            "episodes_per_second": (
                unified["episodes_per_second"] / baseline["episodes_per_second"]
            ),
            "engine_selections_per_second": (
                unified["engine_selections_per_second"]
                / baseline["engine_selections_per_second"]
            ),
            "evidence_boundary": (
                "Same engine, deck/seat schedule, workers and seeds; all League policies use the "
                "shared Foundation in D, while A uses the current heterogeneous CPU opponents."
            ),
        }
    if "c_shared_gpu" in by_arm and "d_unified_shared_gpu" in by_arm:
        separate = by_arm["c_shared_gpu"]
        unified = by_arm["d_unified_shared_gpu"]
        ratios["d_vs_c"] = {
            "episodes_per_second": (
                unified["episodes_per_second"] / separate["episodes_per_second"]
            ),
            "engine_selections_per_second": (
                unified["engine_selections_per_second"]
                / separate["engine_selections_per_second"]
            ),
            "evidence_boundary": (
                "Same Foundation family and GPU; D combines candidate and opponent requests into "
                "one encoder batch, while C performs separate forwards."
            ),
        }
    return ratios


def main(argv: list[str] | None = None) -> int:
    global SharedFoundationOpponentPool
    global RolloutCollector
    global balanced_jobs
    global load_frozen_pool
    global load_source_actor_critic
    global torch

    import torch

    from .rl.opponent_inference import SharedFoundationOpponentPool
    from .rl.opponents import balanced_jobs, load_frozen_pool
    from .rl.policy.actor_critic import load_source_actor_critic
    from .rl.rollout.collector import RolloutCollector

    parser = argparse.ArgumentParser(description="0022 official-engine League throughput benchmark")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--deck-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--coalesce-ms", type=float, default=1.0)
    parser.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.games < 1 or args.workers < 1 or args.deck_count < 1 or args.coalesce_ms < 0:
        raise ValueError("games/workers/deck-count must be positive; coalesce-ms non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA for the candidate and C arm")

    torch.set_num_threads(1)
    candidate_device = torch.device("cuda")
    candidate, metadata = load_source_actor_critic()
    torch.manual_seed(args.seed)
    candidate.to(candidate_device).eval()
    packages, snapshot = load_frozen_pool()
    if len(packages) < args.deck_count:
        raise ValueError(
            f"requested {args.deck_count} decks, but catalog only has {len(packages)}"
        )
    packages = packages[: args.deck_count]
    results = []
    for arm in args.arms:
        results.append(
            _run_arm(
                arm,
                candidate=candidate,
                candidate_device=candidate_device,
                packages=packages,
                games=args.games,
                workers=args.workers,
                seed=args.seed,
                coalesce_ms=args.coalesce_ms,
                mode=args.mode,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "schema_version": "0022_league_throughput_benchmark_v1",
        "source_version": metadata["version"],
        "catalog_sha256": snapshot["catalog_sha256"],
        "deck_count": len(packages),
        "decks": [package.name for package in packages],
        "results": results,
        "ratios": _ratios(results),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all(result["completed"] == args.games for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
