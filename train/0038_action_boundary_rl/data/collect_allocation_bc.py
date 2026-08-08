"""Collect Zero-Shot sequential Phantom Dive labels; never performs PPO."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import time

import torch

from ..initialization import build_update0_model, initialization_manifest
from ..rollout import FullSemanticRolloutCollector
from ..training.run_full_semantic import build_jobs, focal_deck
from .allocation_dataset import AllocationBCSample, split_for_battle

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / "rl_runs/0038_action_boundary_rl/datasets/allocation_zero_shot_v1"


def _plain(sample: AllocationBCSample) -> dict[str, object]:
    return {
        **{field.name: getattr(sample, field.name) for field in fields(sample)
           if field.name != "pre_action_features"},
        "pre_action_features": {key: value.cpu() for key, value in sample.pre_action_features.items()},
        "split": split_for_battle(sample.battle_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=800)
    parser.add_argument("--minimum-samples", type=int, default=3000)
    parser.add_argument("--chunk-games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--engines-per-worker", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.chunk_games < 2 or args.chunk_games % 2:
        raise ValueError("game counts must be positive and seat-balanced")
    if (args.output_root / "manifest.json").exists():
        raise FileExistsError(f"completed allocation dataset already exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    chunk_root = args.output_root / "chunks"
    chunk_root.mkdir(exist_ok=True)
    device = torch.device(args.device)
    model, identity = build_update0_model(focal_deck(), device=device)
    samples: list[dict[str, object]] = []
    valid_games = errors = invalid_chains = 0
    invalid_reasons = Counter()
    started = time.perf_counter()
    for offset in range(0, args.games, args.chunk_games):
        count = min(args.chunk_games, args.games - offset)
        chunk_path = chunk_root / f"chunk-{offset:06d}.pt"
        if chunk_path.exists():
            existing = torch.load(chunk_path, map_location="cpu", weights_only=True)
            samples.extend(existing["samples"])
            valid_games += int(existing["valid_games"])
            errors += int(existing["errors"])
            invalid_chains += int(existing.get("invalid_chains", 0))
            invalid_reasons.update(existing.get("invalid_reasons", {}))
            if len(samples) >= args.minimum_samples:
                break
            continue
        jobs = build_jobs(source_policy_update=0, seed=380_100_000 + offset, count=count)
        jobs = [replace(
            job, game_id=f"allocation-bc-{offset + index:06d}",
            action_boundary_mode="shadow", trace_policy="compact"
        ) for index, job in enumerate(jobs)]
        collector = FullSemanticRolloutCollector(
            model, model.actor, device=device,
            worker_processes=min(args.workers, count),
            engines_per_worker=min(args.engines_per_worker, count),
            inference_channels_per_role=min(args.engines_per_worker, count),
            mode="sample", coalesce_ms=5.0, timeout_seconds=240.0,
        )
        episodes = collector.collect(jobs)
        chunk_samples = []
        chunk_valid = chunk_errors = chunk_invalid = 0
        chunk_invalid_reasons = Counter()
        for episode in episodes:
            if episode.valid:
                chunk_valid += 1
                chunk_samples.extend(_plain(item) for item in episode.diagnostics["allocation_bc_samples"])
                chunk_invalid += int(episode.diagnostics.get("allocation_bc_invalid", 0))
                chunk_invalid_reasons.update(
                    episode.diagnostics.get("allocation_bc_invalid_reasons", {})
                )
            else:
                chunk_errors += 1
        torch.save({"samples": chunk_samples, "valid_games": chunk_valid,
                    "errors": chunk_errors, "invalid_chains": chunk_invalid,
                    "invalid_reasons": dict(chunk_invalid_reasons)}, chunk_path)
        samples.extend(chunk_samples)
        valid_games += chunk_valid
        errors += chunk_errors
        invalid_chains += chunk_invalid
        invalid_reasons.update(chunk_invalid_reasons)
        print(json.dumps({"completed_games": offset + count, "samples": len(samples),
                          "invalid_chains": invalid_chains, "errors": errors}), flush=True)
        if len(samples) >= args.minimum_samples:
            break
    if len(samples) < args.minimum_samples:
        raise RuntimeError(f"only collected {len(samples)} complete labels; need {args.minimum_samples}")
    dataset_path = args.output_root / "allocation_samples.pt"
    torch.save({"schema_version": "0038_allocation_bc_dataset_v1", "samples": samples}, dataset_path)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    counts = Counter(len(item["counters"]) for item in samples)
    allocations = Counter(tuple(item["counters"]) for item in samples)
    turns = Counter("early" if item["turn"] <= 4 else "mid" if item["turn"] <= 10 else "late"
                    for item in samples)
    opponents = Counter(str(item["opponent_id"]) for item in samples)
    manifest = {
        "schema_version": "0038_allocation_bc_manifest_v1",
        "teacher": initialization_manifest(),
        "explicitly_excluded": ["V5", "RL-updated checkpoints", "PPO old_logprob/ratio"],
        "requested_games": args.games, "valid_games": valid_games, "errors": errors,
        "invalid_or_stale_chains": invalid_chains,
        "invalid_chain_reasons": invalid_reasons,
        "unique_battles": len({item["battle_id"] for item in samples}),
        "unique_engine_seeds": len({item["seed"] for item in samples}),
        "macro_samples": len(samples),
        "split": Counter(str(item["split"]) for item in samples),
        "target_count_n": counts, "allocation_distribution": {
            ",".join(map(str, key)): value for key, value in allocations.items()
        },
        "focal_first": sum(bool(item["focal_first"]) for item in samples),
        "focal_second": sum(not bool(item["focal_first"]) for item in samples),
        "opponent_deck": opponents, "game_stage": turns,
        "immediate_ko_samples": sum(int(item["immediate_ko_targets"]) > 0 for item in samples),
        "immediate_prize_samples": sum(int(item["immediate_prizes"]) > 0 for item in samples),
        "dataset_sha256": dataset_hash,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=dict) + "\n")
    print(json.dumps({"dataset": str(dataset_path), "sha256": dataset_hash,
                      "samples": len(samples), "valid_games": valid_games,
                      "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
