"""Paired CUDA benchmark for model-static prototype embedding persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

import torch

from .benchmark_incremental_features import PROTOTYPES, load_parity_trajectories
from .contracts.batch import DecisionBatch
from .deployment.inference import PortableSemanticPolicy
from .features.collate import collate_canonical_records
from .features.compiler import compile_canonical_row
from .knowledge.state import CausalKnowledge


SCHEMA_VERSION = "0035_prototype_embedding_cache_benchmark_v1"


def align_model_dtype(
    values: Any, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Match model floating dtype while preserving categorical/mask contracts."""
    return {
        name: value.to(dtype=dtype) if value.dtype.is_floating_point else value
        for name, value in values.items()
    }


def paired_orders(repetitions: int) -> list[tuple[str, str]]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    return [
        ("uncached", "cached") if index % 2 == 0 else ("cached", "uncached")
        for index in range(repetitions)
    ]


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("timing summary requires values")
    ordered = sorted(float(value) for value in values)
    median = float(statistics.median(ordered))
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": median,
        "mad_ms": float(statistics.median(abs(value - median) for value in ordered)),
        "p95_ms": ordered[p95_index],
    }


def _batch(batch_size: int, device: torch.device) -> DecisionBatch:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    records: list[dict[str, Any]] = []
    trajectories = load_parity_trajectories()
    while len(records) < batch_size:
        for trajectory in trajectories:
            knowledge = CausalKnowledge(trajectory.actor, trajectory.deck)
            for decision in trajectory.decisions:
                snapshot = knowledge.consume(
                    decision.observation, decision.event_cursor
                )
                records.append(
                    compile_canonical_row(decision.row(), snapshot, PROTOTYPES)
                )
                if len(records) == batch_size:
                    break
            if len(records) == batch_size:
                break
    return DecisionBatch.from_mapping(
        collate_canonical_records(records)
    ).to(device)


def _digest(result: Any) -> str:
    digest = hashlib.sha256()
    for name in ("sequences", "lengths", "legal"):
        value = getattr(result, name).detach().contiguous().cpu()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _measure_arm(
    model: Any,
    batch: DecisionBatch,
    *,
    arm: str,
    iterations: int,
) -> dict[str, Any]:
    if arm not in {"uncached", "cached"}:
        raise ValueError(f"unsupported arm: {arm}")
    model.clear_prototype_cache(f"benchmark_{arm}_start")
    cache_build_ms = 0.0
    if arm == "cached":
        build_start = torch.cuda.Event(enable_timing=True)
        build_end = torch.cuda.Event(enable_timing=True)
        build_start.record()
        model.prepare_prototype_cache()
        build_end.record()
        build_end.synchronize()
        cache_build_ms = float(build_start.elapsed_time(build_end))

    pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    result = None
    with torch.inference_mode():
        for _ in range(iterations):
            if arm == "uncached":
                model.clear_prototype_cache("benchmark_uncached_iteration")
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            result = model.deterministic_action_tensors(batch)
            finished.record()
            pairs.append((started, finished))
    assert result is not None
    pairs[-1][1].synchronize()
    timings = [float(start.elapsed_time(end)) for start, end in pairs]
    summary = timing_summary(timings)
    summary.update(
        {
            "iterations": iterations,
            "batch_size": batch.batch_size,
            "batches_per_second": 1000.0 / summary["median_ms"],
            "decisions_per_second": (
                1000.0 * batch.batch_size / summary["median_ms"]
            ),
            "cache_build_ms": cache_build_ms,
            "output_sha256": _digest(result),
            "cache_stats": model.prototype_cache_stats(),
        }
    )
    return summary


def run_benchmark(
    *,
    checkpoint: Path,
    deck: Sequence[int],
    batch_size: int,
    iterations: int,
    warmup_iterations: int,
    repetitions: int,
    dtype: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if iterations < 1 or warmup_iterations < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    model_dtype = torch.float16 if dtype == "fp16" else torch.float32
    policy = PortableSemanticPolicy.from_checkpoint(checkpoint, deck)
    model = policy.model.to(device=device, dtype=model_dtype).eval()
    batch = DecisionBatch.from_mapping(
        align_model_dtype(_batch(batch_size, device), model_dtype)
    )

    for arm in ("uncached", "cached"):
        if warmup_iterations:
            _measure_arm(
                model, batch, arm=arm, iterations=warmup_iterations
            )
    measurements: dict[str, list[dict[str, Any]]] = {
        "uncached": [], "cached": []
    }
    paired: list[dict[str, Any]] = []
    for index, order in enumerate(paired_orders(repetitions)):
        item: dict[str, Any] = {"index": index, "order": list(order), "results": {}}
        for arm in order:
            result = _measure_arm(model, batch, arm=arm, iterations=iterations)
            measurements[arm].append(result)
            item["results"][arm] = result
        paired.append(item)

    commitments = {
        result["output_sha256"]
        for arm_results in measurements.values()
        for result in arm_results
    }
    if len(commitments) != 1:
        raise RuntimeError("cached and uncached deterministic outputs differ")

    def aggregate(arm: str) -> dict[str, Any]:
        results = measurements[arm]
        return {
            "median_ms": float(statistics.median(x["median_ms"] for x in results)),
            "p95_ms": float(statistics.median(x["p95_ms"] for x in results)),
            "decisions_per_second": float(
                statistics.median(x["decisions_per_second"] for x in results)
            ),
            "cache_build_ms": float(
                statistics.median(x["cache_build_ms"] for x in results)
            ),
            "repetitions": results,
        }

    cache = model.prepare_prototype_cache()
    cache_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (cache.cards, cache.attacks, cache.skills, cache.effects)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "device": torch.cuda.get_device_name(device),
        "dtype": dtype,
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "repetitions": repetitions,
        "prototype_cache_bytes": cache_bytes,
        "output_sha256": next(iter(commitments)),
        "results": {arm: aggregate(arm) for arm in ("uncached", "cached")},
        "paired_repetitions": paired,
    }


def _deck(path: Path) -> list[int]:
    deck = [int(line) for line in path.read_text().splitlines() if line.strip()]
    if len(deck) != 60 or any(identity <= 0 for identity in deck):
        raise ValueError("deck must contain 60 positive card IDs")
    return deck


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_benchmark(
        checkpoint=args.checkpoint,
        deck=_deck(args.deck),
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        repetitions=args.repetitions,
        dtype=args.dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "align_model_dtype", "paired_orders", "run_benchmark", "timing_summary"
]
