"""One-rollout, fresh-optimizer LR comparison for the sealed 0042 U250 policy.

This is diagnostic-only: it never writes a formal version or candidate package.
Every arm consumes the same frozen PreparedBatch and starts from identical U250
model weights while retaining Policy-0809/U0 as the reference-KL policy.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from ..integrated.presets import preset
from ..initialization import build_preset_from_common_update0
from ..training.batch_full_semantic import prepare_episodes
from ..training.ppo_full_semantic import PPOConfig, PPOTrainer
from ..training.run_full_semantic import (
    RunConfig,
    _assert_acceptance_episode_health,
    _episode_metrics,
    build_collector,
    build_jobs,
    focal_deck,
    load_frozen_opponent,
)
from ..training.storage_full_semantic import load_adapted_model_only


ARMS = {
    "current_1x": (5.0e-6, 5.0e-6, 5.0e-6),
    "balanced_2x_adapter_4x": (1.0e-5, 2.0e-5, 1.0e-5),
    "balanced_4x_adapter_8x": (2.0e-5, 4.0e-5, 2.0e-5),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_branch(checkpoint: Path, device: torch.device, config: PPOConfig):
    flags = preset("FULL_MODEL")
    model, _ = build_preset_from_common_update0(focal_deck(), flags, device=device)
    trainer = PPOTrainer(model, device=device, config=config)
    payload = load_adapted_model_only(model, checkpoint)
    if model.representation_sha256() != trainer.initial_representation_sha256:
        raise RuntimeError("U250 changed frozen Policy-0809 representation")
    return model, trainer, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=420042251)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_config = PPOConfig()
    behavior_model, _, payload = build_branch(checkpoint, device, base_config)
    opponent = load_frozen_opponent(device)
    rollout_config = RunConfig(
        version="V999_diagnostic_u250_lr_probe",
        updates=1,
        wandb_mode="offline",
        initial_model_checkpoint=str(checkpoint),
    )
    collector = build_collector(
        behavior_model, opponent, rollout_config, mode="sample"
    )
    jobs = build_jobs(
        source_policy_update=int(payload["update"]),
        seed=args.seed,
        count=256,
    )
    rollout_started = time.perf_counter()
    episodes = collector.collect(jobs)
    rollout_seconds = time.perf_counter() - rollout_started
    collector_metrics = collector.metrics()
    health = _assert_acceptance_episode_health(
        episodes, collector_metrics, scope="diagnostic/u250_lr_probe"
    )
    batch = prepare_episodes(
        episodes,
        gamma=base_config.gamma,
        gae_lambda=base_config.gae_lambda,
        credit_clock=base_config.credit_clock,
        loss_weighting=base_config.loss_weighting,
        prize_mode=preset("FULL_MODEL").prize_aux_mode,
        prize_scale=preset("FULL_MODEL").prize_aux_scale,
        require_policy_identity=True,
    )
    del behavior_model, collector
    torch.cuda.empty_cache()

    rng_state = torch.random.get_rng_state()
    arm_reports = {}
    for name, (decoder_lr, adapter_lr, allocation_lr) in ARMS.items():
        torch.random.set_rng_state(rng_state)
        config = replace(
            base_config,
            decoder_learning_rate=decoder_lr,
            policy_adapter_learning_rate=adapter_lr,
            allocation_learning_rate=allocation_lr,
        )
        model, trainer, _ = build_branch(checkpoint, device, config)
        reference_before = {
            key: value.detach().cpu().clone()
            for key, value in trainer.reference.state_dict().items()
        }
        started = time.perf_counter()
        metrics = trainer.update(batch, update=251)
        elapsed = time.perf_counter() - started
        reference_unchanged = all(
            torch.equal(reference_before[key], value.detach().cpu())
            for key, value in trainer.reference.state_dict().items()
        )
        arm_reports[name] = {
            "learning_rates": config.actor_group_learning_rates(),
            "optimizer_seconds": elapsed,
            "reference_remains_policy0809_u0": reference_unchanged,
            "metrics": metrics,
        }
        del model, trainer
        torch.cuda.empty_cache()

    report = {
        "schema": "0042_u250_identical_rollout_lr_probe_v1",
        "diagnostic_only": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_update": int(payload["update"]),
        "seed": args.seed,
        "games": len(episodes),
        "valid_decisions": batch.decisions,
        "rollout_seconds": rollout_seconds,
        "rollout_metrics": {**_episode_metrics(episodes, "rollout"), **collector_metrics},
        "acceptance_health": health,
        "arms": arm_reports,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "games": len(episodes),
        "valid_decisions": batch.decisions,
        "arms": {
            name: {
                "behavior_kl": row["metrics"]["ppo/epoch_end_behavior_kl"],
                "epochs": row["metrics"]["ppo/epochs_completed"],
                "clip_fraction": row["metrics"]["ppo/clip_fraction"],
                "actor_relative_l2": row["metrics"]["ppo/actor_relative_l2_vs_reference"],
            }
            for name, row in arm_reports.items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
