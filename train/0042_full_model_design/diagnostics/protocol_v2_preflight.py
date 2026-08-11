"""Real-rollout preflight for the 0042 PPO Protocol V2 baseline."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import torch

from ..initialization import build_preset_from_common_update0
from ..integrated.presets import preset
from ..training.batch_full_semantic import prepare_episodes
from ..training.ppo_full_semantic import (
    PPOConfig,
    PPOTrainer,
    assert_frozen_targets,
    complete_epoch_orders,
    epoch_minibatches,
    snapshot_frozen_targets,
)
from ..training.run_full_semantic import (
    RunConfig,
    TRAINING_OPPONENT_POLICY_ID,
    _replace_chance_boundary_episodes,
    build_collector,
    build_jobs,
    focal_deck,
    load_frozen_catalog,
    load_frozen_opponent,
)


ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_ROOT = ROOT / ".tmp/0042_protocol_v2_preflight"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _frequency_fingerprint(jobs: list[Any]) -> str:
    payload = [
        {
            "slot": index,
            "opponent_id": job.opponent_id,
            "focal_first": job.focal_first,
            "seed": job.seed,
            "policy_seed": job.policy_seed,
        }
        for index, job in enumerate(jobs)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _instrumented_epoch_contract(
    decisions: int, batch_size: int, forward_microbatch_size: int = 1024
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(420_042_002)
    usage = torch.zeros(decisions, dtype=torch.int16)
    rows = []
    for epoch, order in enumerate(
        complete_epoch_orders(decisions, 3, generator=generator), start=1
    ):
        batches = epoch_minibatches(order, batch_size)
        physical_batches = [
            micro
            for logical in batches
            for micro in epoch_minibatches(logical, forward_microbatch_size)
        ]
        usage[order] += 1
        expected = torch.full_like(usage, epoch)
        exact = torch.equal(usage, expected)
        rows.append({
            "epoch": epoch,
            "optimizer_steps": len(batches),
            "sample_slots": sum(int(batch.numel()) for batch in batches),
            "last_minibatch_size": int(batches[-1].numel()),
            "logical_minibatch_size": batch_size,
            "physical_microbatch_size": forward_microbatch_size,
            "physical_forward_backward_passes": len(physical_batches),
            "unique_decisions": int(torch.unique(order).numel()),
            "coverage_ratio": float(usage.gt(0).float().mean()),
            "cumulative_usage_min": int(usage.min()),
            "cumulative_usage_max": int(usage.max()),
            "exact_usage_contract": exact,
        })
        if not exact:
            raise RuntimeError(f"instrumented data epoch {epoch} violated exact traversal")
    return {"passed": True, "epochs": rows}


def run(output: Path, *, seed: int = 420_042_002) -> dict[str, Any]:
    output = output.resolve()
    if PREFLIGHT_ROOT.resolve() not in output.parents or output.exists():
        raise ValueError("preflight output must be a fresh child of .tmp/0042_protocol_v2_preflight")
    output.mkdir(parents=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    flags = preset("FULL_MODEL")
    ppo_config = PPOConfig(meta_anchor_coef=flags.meta_anchor_coef)
    run_config = RunConfig(
        version="V999_protocol_v2_preflight",
        updates=1,
        eval_every=999,
        wandb_mode="offline",
        ppo=ppo_config,
    )
    run_config.validate()
    model, identity = build_preset_from_common_update0(
        focal_deck(), flags, device="cuda:0"
    )
    trainer = PPOTrainer(model, device=torch.device("cuda:0"), config=ppo_config)
    opponent = load_frozen_opponent(torch.device("cuda:0"))
    jobs = build_jobs(source_policy_update=0, seed=seed, count=256)
    catalog = load_frozen_catalog()
    expected_frequency = {item.deck_id: item.games for item in catalog}
    actual_frequency = dict(Counter(job.opponent_id for job in jobs))
    if actual_frequency != expected_frequency:
        raise RuntimeError("256-game preflight lost the canonical opponent frequency unit")
    collector = build_collector(model, opponent, run_config, mode="sample")
    episodes = collector.collect(jobs)

    replacement_metrics: list[dict[str, float]] = []

    def collect_replacements(replacement_jobs):
        replacement = build_collector(model, opponent, run_config, mode="sample")
        values = replacement.collect(replacement_jobs)
        replacement_metrics.append(replacement.metrics())
        return values

    episodes, exclusions = _replace_chance_boundary_episodes(
        episodes, collect_replacements
    )
    if len(episodes) != 256 or any(not episode.valid for episode in episodes):
        raise RuntimeError("preflight did not complete exactly 256 valid games")
    batch = prepare_episodes(
        episodes,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        credit_clock=ppo_config.credit_clock,
        loss_weighting=ppo_config.loss_weighting,
        prize_mode=flags.prize_aux_mode,
        prize_scale=flags.prize_aux_scale,
        require_policy_identity=True,
    )
    completed_games = len(episodes)
    del episodes
    torch.cuda.empty_cache()
    targets = snapshot_frozen_targets(batch)
    assert_frozen_targets(batch, targets)
    epoch_contract = _instrumented_epoch_contract(
        batch.decisions,
        ppo_config.batch_size,
        ppo_config.forward_microbatch_size,
    )
    collector_metrics = collector.metrics()
    performance_gates = {
        "features_remained_device_resident": (
            collector_metrics.get("rollout/cuda_features_device_resident") == 1.0
        ),
        "no_bulk_feature_device_to_host_copy": (
            collector_metrics.get("rollout/cuda_feature_d2h_bytes") == 0.0
        ),
        "per_lane_exact_deck_routing_passed": (
            collector_metrics.get("rollout/lane_routing_audit_pass") == 1.0
            and collector_metrics.get("rollout/lane_routing_audit_failures") == 0.0
        ),
        "full_frequency_pool_observed": (
            collector_metrics.get("rollout/unique_opponent_exact_decks") == 55.0
        ),
        "heterogeneous_roles_are_compacted_before_model_forward": (
            collector_metrics.get("rollout/role_compacted_routing") == 1.0
        ),
    }
    if not all(performance_gates.values()):
        raise RuntimeError(f"CUDA residency/routing preflight failed: {performance_gates}")
    report = {
        "schema_version": "0042_ppo_protocol_v2_preflight_v1",
        "passed": True,
        "seed": seed,
        "runtime_config": asdict(run_config),
        "source_identity": asdict(identity),
        "rollout": {
            "games_requested": 256,
            "games_completed": completed_games,
            "frequency_unit_games": 256,
            "frequency_units": 1,
            "frequency_semantics_correct": actual_frequency == expected_frequency,
            "frequency_fingerprint_sha256": _frequency_fingerprint(jobs),
            "opponent_policy_id": TRAINING_OPPONENT_POLICY_ID,
            "valid_decisions": batch.decisions,
            "mean_decisions_per_game": batch.decisions / completed_games,
            "chance_boundary_replacements": len(exclusions),
            "collector_metrics": collector_metrics,
            "replacement_collector_metrics": replacement_metrics,
        },
        "epoch_semantics": epoch_contract,
        "frozen_targets": {
            "names": sorted(targets),
            "unchanged": True,
            "computed_once_before_optimizer": True,
        },
        "optimizer_groups": trainer.optimizer_group_manifest(),
        "parameter_partitions": trainer.parameter_partition_manifest(),
        "precision": "fp32",
        "performance_gates": performance_gates,
    }
    _atomic_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=420_042_002)
    args = parser.parse_args()
    report = run(args.output, seed=args.seed)
    print(json.dumps({
        "output": str(args.output),
        "passed": report["passed"],
        "games": report["rollout"]["games_completed"],
        "valid_decisions": report["rollout"]["valid_decisions"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
