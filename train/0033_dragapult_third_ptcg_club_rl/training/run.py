"""Formal resident-CUDA PPO runner for 0033."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from rl_environment.logging import TrainingLogger

from .ppo import PPOConfig, PPOTrainer
from .rollout import ResidentRolloutCollector
from .runtime import FOCAL_DECK_ID, FOCAL_DECK_PATH, RuntimeBundle, build_runtime
from .storage import save_trainable_heads
from ..transfer import write_transfer_report


ROOT = Path(__file__).resolve().parents[3]
PROJECT = "0033_dragapult_third_ptcg_club_rl"
FORMAL_VERSION = "V2_third_ptcg_club_cuda_ppo_200u"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _merge_json(path: Path, payload: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict):
            existing = loaded
    _atomic_json(path, {**existing, **payload})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fresh(paths: tuple[Path, ...]) -> None:
    occupied = [str(path) for path in paths if path.exists() and any(path.iterdir())]
    if occupied:
        raise FileExistsError("formal version paths are already occupied: " + ", ".join(occupied))


def _require_wandb_online_auth(mode: str) -> None:
    if mode != "online":
        return
    import wandb

    if not getattr(wandb.Api(), "api_key", None):
        raise RuntimeError("W&B online mode requires an authenticated SDK credential")


def _opponent_snapshot(runtime: RuntimeBundle, args: argparse.Namespace) -> dict[str, Any]:
    support = json.loads(args.support_report.read_text(encoding="utf-8"))
    support_rows = support.get("decks", support.get("opponents", []))
    hashes = {
        row["deck_id"]: row["deck_sha256"]
        for row in support_rows
        if row.get("status", "supported") == "supported"
    }
    return {
        "schema": "0033_frozen_opponent_snapshot_v1",
        "focal_deck_id": FOCAL_DECK_ID,
        "focal_deck_sha256": _sha256(FOCAL_DECK_PATH),
        "inference": "one_shared_0019_epoch13_foundation_head",
        "foundation_checkpoint": str(args.foundation_checkpoint),
        "foundation_checkpoint_sha256": _sha256(args.foundation_checkpoint),
        "decks_per_update": len(runtime.full_opponent_deck_ids),
        "copies_per_deck": args.lanes // len(runtime.full_opponent_deck_ids),
        "opponents": [
            {
                "deck_id": deck_id,
                "deck_sha256": hashes[deck_id],
            }
            for deck_id in runtime.full_opponent_deck_ids
        ],
    }


def _build(args: argparse.Namespace, learner=None) -> RuntimeBundle:
    supported_decks = 38
    return build_runtime(
        rules=args.rules,
        extension_dir=args.extension_dir,
        source_checkpoint=args.source_checkpoint,
        support_report=args.support_report,
        frozen_root=args.frozen_root,
        checkpoint_root=args.checkpoint_root,
        shared_foundation_checkpoint=args.foundation_checkpoint,
        copies_per_opponent=args.lanes // supported_decks,
        opponents_per_cohort=supported_decks,
        cohort_index=0,
        seed_start=args.seed,
        learner=learner,
    )


def run(args: argparse.Namespace) -> int:
    run_root = ROOT / "rl_runs" / PROJECT / "versions" / args.version
    artifact = run_root / "artifact"
    checkpoint_root = run_root / "checkpoint"
    tensorboard = run_root / "tensorboard"
    wandb_dir = run_root / "wandb"
    _require_fresh((artifact, checkpoint_root, tensorboard, wandb_dir))
    for path in (artifact, checkpoint_root, tensorboard, wandb_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "WANDB_MODE": args.wandb_mode,
            "WANDB_ENTITY": "dragon_bra",
            "WANDB_PROJECT": "pokemon-tcg-policy-learning",
            "WANDB_JOB_TYPE": "ppo",
            "WANDB_RUN_ID": f"0033-{args.version.lower().replace('_', '-')}",
            "WANDB_NAME": f"0033 · dragapult_third_ptcg_club_rl · {args.version}",
            "WANDB_RUN_GROUP": PROJECT,
            "WANDB_TAGS": "0033,cuda,ppo,dragapult_third_ptcg_club,third_ptcg_club",
            "WANDB_DIR": str(wandb_dir),
        }
    )
    wandb_url = (
        "https://wandb.ai/dragon_bra/pokemon-tcg-policy-learning/runs/"
        + os.environ["WANDB_RUN_ID"]
    )
    _require_wandb_online_auth(args.wandb_mode)
    config = {
        "schema": "0033_cuda_ppo_training_config_v1",
        "version": args.version,
        "focal_deck_id": FOCAL_DECK_ID,
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "source_checkpoint_role": "pt0805_epoch13_initialization_only",
        "rules": str(args.rules),
        "rules_sha256": _sha256(args.rules),
        "extension": str(args.extension_dir / "_ptcg_cuda.so"),
        "extension_sha256": _sha256(args.extension_dir / "_ptcg_cuda.so"),
        "support_report": str(args.support_report),
        "support_report_sha256": _sha256(args.support_report),
        "frozen_root": str(args.frozen_root),
        "updates": args.updates,
        "rollout_steps": args.steps,
        "lanes": args.lanes,
        "opponents_per_update": 38,
        "copies_per_opponent": args.lanes // 38,
        "frozen_policy": "0019_epoch13_shared_foundation",
        "foundation_checkpoint": str(args.foundation_checkpoint),
        "foundation_checkpoint_sha256": _sha256(args.foundation_checkpoint),
        "actor_execution": "cuda_graph",
        "checkpoint_retention": "all",
        "policy_trainable": "action_decoder_only",
        "critic_trainable": "value_head_only_not_deployed_for_action_logits",
        "wandb_mode": args.wandb_mode,
        "wandb_offline_reason": (
            "WANDB_API_KEY unavailable on launch host" if args.wandb_mode == "offline" else None
        ),
        "ppo": asdict(PPOConfig()),
    }
    _atomic_json(artifact / "training_config.json", config)
    _atomic_json(
        artifact / "status.json",
        {
            "project_id": PROJECT,
            "version": args.version,
            "state": "initializing",
            "training_started": False,
            "wandb_mode": args.wandb_mode,
            "wandb_run_id": os.environ["WANDB_RUN_ID"],
            "wandb_url": wandb_url,
            "wandb_sync_status": "initializing",
        },
    )

    started = time.time()
    update = 0
    episodes_total = 0
    decisions_total = 0
    try:
        runtime = _build(args)
        write_transfer_report(runtime.transfer_report, artifact / "weight_transfer.json")
        _atomic_json(artifact / "opponent_snapshot.json", _opponent_snapshot(runtime, args))
        trainer = PPOTrainer(runtime.learner, device=torch.device("cuda"))
        trainable_parameters = sum(
            parameter.numel() for parameter in runtime.learner.parameters()
            if parameter.requires_grad
        )
        config["trainable_parameter_count"] = trainable_parameters
        config["total_parameter_count"] = sum(
            parameter.numel() for parameter in runtime.learner.parameters()
        )
        _atomic_json(artifact / "training_config.json", config)
        with TrainingLogger(artifact / "training_metrics.jsonl", tensorboard) as logger:
            _merge_json(
                artifact / "status.json",
                {
                    "state": "running",
                    "training_started": True,
                    "started_at_unix": started,
                    "checkpoint_update": 0,
                    "trainable_parameters": trainable_parameters,
                    "wandb_mode": args.wandb_mode,
                    "wandb_url": wandb_url,
                    "wandb_sync_status": (
                        "online_active" if args.wandb_mode == "online" else "offline_staging"
                    ),
                },
            )
            collector = ResidentRolloutCollector(
                engine=runtime.engine,
                policy=runtime.policy,
                decks=runtime.decks,
                seeds=runtime.seeds,
                opponent_head_indices=runtime.opponent_head_indices,
                opponent_deck_ids=runtime.opponent_deck_ids,
                steps=args.steps,
                actor_execution="cuda_graph",
            )
            for update_index in range(args.updates):
                iteration_started = time.perf_counter()
                rollout = collector.collect(source_policy_update=update)
                ppo_started = time.perf_counter()
                ppo_metrics = trainer.update(rollout.batch)
                torch.cuda.synchronize()
                ppo_seconds = time.perf_counter() - ppo_started
                update += 1
                episodes_total += int(rollout.metrics["rollout/completed_episodes"])
                decisions_total += int(rollout.metrics["rollout/engine_decisions"])
                checkpoint = checkpoint_root / f"update-{update:06d}.pt"
                digest = save_trainable_heads(
                    runtime.learner,
                    checkpoint,
                    update=update,
                    metadata={
                        "project": PROJECT,
                        "version": args.version,
                        "focal_deck_id": FOCAL_DECK_ID,
                        "source_policy_update": update - 1,
                        "opponent_pool": "38_cuda_supported_0019_shared_foundation",
                        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
                    },
                )
                iteration_seconds = time.perf_counter() - iteration_started
                metrics = {
                    "trainer/update": update,
                    "rollout/source_policy_update": update - 1,
                    "checkpoint/update": update,
                    "env/episodes": episodes_total,
                    "env/decisions": decisions_total,
                    "ppo/wall_seconds": ppo_seconds,
                    "system/end_to_end/wall_seconds": iteration_seconds,
                    "system/end_to_end/decisions_per_second": (
                        rollout.metrics["rollout/engine_decisions"] / iteration_seconds
                    ),
                    "system/end_to_end/episodes_per_second": (
                        rollout.metrics["rollout/completed_episodes"] / iteration_seconds
                    ),
                    "checkpoint/sha256_present": float(bool(digest)),
                    "system/gpu/allocated_bytes": torch.cuda.memory_allocated(),
                    "system/gpu/reserved_bytes": torch.cuda.memory_reserved(),
                    "system/disk/free_gib": shutil.disk_usage(run_root).free / (1024**3),
                    **rollout.metrics,
                    **ppo_metrics,
                }
                logger.log(update, metrics)
                _merge_json(
                    artifact / "status.json",
                    {
                        "state": "running",
                        "training_started": True,
                        "started_at_unix": started,
                        "checkpoint_update": update,
                        "rollout_source_policy_update": update - 1,
                        "episodes": episodes_total,
                        "decisions": decisions_total,
                        "opponent_pool": "38_cuda_supported_0019_shared_foundation",
                        "wandb_mode": args.wandb_mode,
                        "wandb_url": wandb_url,
                        "wandb_sync_status": (
                            "online_active" if args.wandb_mode == "online" else "offline_staging"
                        ),
                    },
                )
        summary = {
            "state": "completed_updates",
            "updates": update,
            "episodes": episodes_total,
            "decisions": decisions_total,
            "elapsed_seconds": time.time() - started,
            "checkpoint_retention": "all",
            "wandb_url": wandb_url,
            "wandb_sync_status": (
                "online_finished" if args.wandb_mode == "online" else "offline_staged"
            ),
        }
        _atomic_json(artifact / "training_summary.json", summary)
        _merge_json(artifact / "status.json", summary)
        return 0
    except BaseException as error:
        _merge_json(
            artifact / "status.json",
            {
                "state": "failed",
                "training_started": update > 0,
                "checkpoint_update": update,
                "error": f"{type(error).__name__}: {error}",
                "wandb_mode": args.wandb_mode,
                "wandb_url": wandb_url,
                "wandb_sync_status": "failed_or_interrupted",
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=FORMAL_VERSION)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--lanes", type=int, default=152)
    parser.add_argument("--seed", type=int, default=320320001)
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default="online")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.updates < 1 or args.steps < 1 or args.lanes < 1:
        raise ValueError("updates, steps, and lanes must be positive")
    if args.lanes % 38:
        raise ValueError("lanes must be divisible by the 38 supported decks")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
