"""Profile CUDA-resident PPO feature widths without changing training state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..initialization import build_preset_from_common_update0
from ..integrated.presets import preset
from ..training.batch_full_semantic import prepare_episodes
from ..training.run_full_semantic import (
    RunConfig,
    build_collector,
    build_jobs,
    focal_deck,
    load_frozen_opponent,
)
from ..training.storage_full_semantic import load_adapted_model_only


def run(checkpoint: Path, output: Path, *, seed: int, source_update: int) -> dict:
    device = torch.device("cuda:0")
    flags = preset("FULL_MODEL")
    model, _ = build_preset_from_common_update0(focal_deck(), flags, device=device)
    load_adapted_model_only(model, checkpoint)
    opponent = load_frozen_opponent(device)
    config = RunConfig(
        version="V999_feature_width_profile",
        updates=1,
        games_per_update=256,
        trajectory_games_per_update=256,
        rollout_batch_size=256,
        cuda_lane_count=256,
        eval_every=999,
        preset_name="FULL_MODEL",
        wandb_mode="offline",
        launch_formal=False,
    )
    collector = build_collector(model, opponent, config, mode="sample")
    episodes = collector.collect(build_jobs(
        source_policy_update=source_update, seed=seed, count=256
    ))
    batch = prepare_episodes(
        episodes,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        credit_clock=config.ppo.credit_clock,
        loss_weighting=config.ppo.loss_weighting,
        prize_mode=flags.prize_aux_mode,
        prize_scale=flags.prize_aux_scale,
        require_policy_identity=True,
    )
    widths = {}
    for name, values in sorted(batch.feature_widths.items()):
        values = values.float()
        widths[name] = {
            "mean": float(values.mean()),
            "p50": float(values.quantile(0.50)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "max": float(values.max()),
        }
    report = {
        "checkpoint": str(checkpoint),
        "source_update": source_update,
        "decisions": batch.decisions,
        "feature_widths": widths,
        "feature_store_bytes": sum(
            value.numel() * value.element_size() for value in batch.features.values()
        ),
        "collector": collector.metrics(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=420_042_911)
    parser.add_argument("--source-update", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(
        args.checkpoint, args.output, seed=args.seed,
        source_update=args.source_update,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
