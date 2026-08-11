"""Run an auditable 0038 CUDA rollout benchmark and write compact JSON."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from .initialization import (
    ALLOCATION_HEAD_CHECKPOINT,
    ALLOCATION_HEAD_SHA256,
    build_preset_from_common_update0,
)
from .integrated.presets import preset
from .rollout.cuda_collector import CudaFullSemanticRolloutCollector
from .training.run_full_semantic import (
    CUDA_EXTENSION, CUDA_RULES, build_jobs, focal_deck, load_frozen_opponent,
)


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--lanes", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    model, identity = build_preset_from_common_update0(
        focal_deck(), preset("FULL_MODEL"), device=device
    )
    opponent = load_frozen_opponent(device)
    jobs = build_jobs(source_policy_update=0, seed=330_038_401, count=args.games)
    collector = CudaFullSemanticRolloutCollector(
        model, opponent, device=device, rules_path=CUDA_RULES,
        extension_dir=CUDA_EXTENSION, lane_count=args.lanes, mode=args.mode,
        check_interval=8, record_trajectory=args.mode == "sample",
        opponent_policy_id=opponent._policy_id,
        opponent_identity_audit=opponent._policy_identity_audit,
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    wall = time.perf_counter() - started
    errors = sum(not episode.valid for episode in episodes)
    fallback = sum(int(episode.diagnostics.get("macro_fallback", 0)) for episode in episodes)
    fallback_reasons = Counter(
        str(episode.diagnostics.get("macro_fallback_reason"))
        for episode in episodes if episode.diagnostics.get("macro_fallback")
    )
    metrics = collector.metrics()
    throughput = len(episodes) / wall
    payload = {
        "schema": "0038_cuda_action_boundary_benchmark_v1",
        "games": len(episodes),
        "mode": args.mode,
        "lanes": args.lanes,
        "wall_seconds": wall,
        "games_per_second": throughput,
        "errors": errors,
        "fallbacks": fallback,
        "fallback_reasons": dict(fallback_reasons),
        "policy_transitions": sum(len(episode.policy_transitions) for episode in episodes),
        "primitive_selects": sum(int(episode.diagnostics["engine_selections"]) for episode in episodes),
        "metrics": metrics,
        "initialization": {
            "allocation_head_checkpoint": str(ALLOCATION_HEAD_CHECKPOINT.relative_to(ROOT)),
            "allocation_head_checkpoint_sha256": ALLOCATION_HEAD_SHA256,
            "source": asdict(identity),
            "opponent_policy_identity": (
                opponent._policy_identity_audit.to_manifest()
            ),
        },
        "cuda": {
            "extension": str((CUDA_EXTENSION / "_ptcg_cuda.so").relative_to(ROOT)),
            "extension_sha256": sha256(CUDA_EXTENSION / "_ptcg_cuda.so"),
            "rules_sha256": sha256(CUDA_RULES),
            "gpu": torch.cuda.get_device_name(device),
        },
        "historical_comparison": {
            "0038_cpu_greedy_games_per_second": 1.7821430006093517,
            "0037_cuda_stochastic_games_per_second": 9.60,
            "speedup_vs_0038_cpu_greedy": throughput / 1.7821430006093517,
            "ratio_vs_0037_cuda_stochastic": throughput / 9.60,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if errors:
        raise RuntimeError(
            f"CUDA benchmark failed closed: errors={errors}, fallback={fallback}, "
            f"reasons={dict(fallback_reasons)}"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
