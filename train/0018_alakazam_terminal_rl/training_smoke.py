from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from evaluation.cli import _load_official_card_ids
from evaluation.packages.loader import load_submission_package

from .checkpoint import checkpoint_metadata, save_model_only
from .constants import REPOSITORY_ROOT
from .policy.actor_critic import load_source_actor_critic
from .rollout.collector import RolloutCollector
from .rollout.protocol import RolloutJob
from .training.batch import prepare_episodes
from .training.ppo import PPOConfig, PPOTrainer, frozen_reference
from .training.value import ValueConfig, calibrate_value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0018 end-to-end CUDA training smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--coalesce-ms", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / ".tmp/0018_alakazam_terminal_rl/training_smoke",
    )
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested without a CUDA device")
    torch.manual_seed(20260728)
    torch.set_num_threads(2)

    evaluation_root = REPOSITORY_ROOT / "evaluation"
    opponent = load_submission_package(
        evaluation_root / "arena/opponents/dragapult_ex_01",
        _load_official_card_ids(evaluation_root),
        name="dragapult_ex_01",
    )
    model, metadata = load_source_actor_critic()
    model.to(device).eval()
    collector = RolloutCollector(
        model,
        device=device,
        workers=args.workers,
        mode="sample",
        coalesce_seconds=args.coalesce_ms / 1_000.0,
    )
    episodes = collector.collect(
        [
            RolloutJob(
                episode_id=f"cuda-smoke-{index:04d}",
                opponent=opponent,
                candidate_first=index % 2 == 0,
                seed=20260728 + index,
            )
            for index in range(args.games)
        ]
    )
    if not episodes or not all(episode.valid for episode in episodes):
        raise RuntimeError("CUDA smoke rollout did not produce only valid terminal episodes")
    batch = prepare_episodes(episodes)
    value_history = calibrate_value(
        model,
        batch,
        device=device,
        config=ValueConfig(epochs=1, batch_size=16, learning_rate=1e-4),
    )
    reference = frozen_reference(model, device)
    ppo = PPOTrainer(
        model,
        reference,
        device=device,
        config=PPOConfig(
            gae_lambda=0.97,
            epochs=1,
            batch_size=16,
            actor_learning_rate=3e-6,
            value_learning_rate=1e-4,
        ),
    )
    ppo_metrics = ppo.update(batch)

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    checkpoint = save_model_only(
        args.output / "update-000001.pt",
        model,
        update=1,
        metadata={
            "smoke": True,
            "source_version": metadata["version"],
            "optimizer_state_saved": False,
        },
    )
    contract = checkpoint_metadata(checkpoint)
    result = {
        "device": str(device),
        "coalesce_ms": args.coalesce_ms,
        "candidate_batch_count": collector.inference_batch_count,
        "candidate_batch_size_mean": (
            collector.inference_request_count / collector.inference_batch_count
            if collector.inference_batch_count
            else 0.0
        ),
        "episodes": len(episodes),
        "wins": sum(episode.reward == 1 for episode in episodes),
        "losses": sum(episode.reward == -1 for episode in episodes),
        "decisions": batch.decisions,
        "value": value_history[-1],
        "ppo": ppo_metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_contract": contract,
        "cuda_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
