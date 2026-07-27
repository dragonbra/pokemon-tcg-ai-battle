from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from rl_environment.logging import TrainingLogger
from rl_environment.runs import initialize_version, write_version_status

from .checkpoint import prune_model_checkpoints, save_model_only
from .constants import PROJECT_ID, SOURCE_CHECKPOINT_SHA256
from .observability.metrics import OutcomeTracker, episode_metrics
from .opponents import balanced_jobs, load_frozen_pool, select_slice, write_snapshot
from .policy.actor_critic import DragapultActorCritic, load_source_actor_critic
from .rollout.collector import RolloutCollector
from .storage import storage_guard
from .training.batch import prepare_episodes
from .training.ppo import PPOConfig, PPOTrainer, frozen_reference
from .training.value import ValueConfig, calibrate_value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _device(value: str, allow_gpu: bool) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda":
        if not allow_gpu:
            raise ValueError("CUDA requires the explicit --allow-gpu gate")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
    return requested


def _configure_wandb(args: argparse.Namespace) -> None:
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_ENTITY", "dragon_bra")
    os.environ.setdefault("WANDB_PROJECT", "pokemon-tcg-policy-learning")
    job_types = {"probe": "eval", "value": "value_calibration", "ppo": "ppo_train"}
    os.environ.setdefault("WANDB_JOB_TYPE", job_types[args.command])
    os.environ.setdefault("WANDB_TAGS", "0017,dragapult,terminal_reward,official_engine")


def _load_model(device: torch.device) -> tuple[DragapultActorCritic, dict[str, Any]]:
    model, metadata = load_source_actor_critic()
    return model.to(device), metadata


def _runtime_metrics(
    model: DragapultActorCritic, device: torch.device
) -> dict[str, float]:
    metrics = {
        "system/parameter_count": float(sum(item.numel() for item in model.parameters())),
        "system/trainable_parameter_count": float(
            sum(item.numel() for item in model.parameters() if item.requires_grad)
        ),
    }
    if device.type == "cuda":
        metrics.update(
            {
                "system/gpu/memory_allocated_bytes": float(
                    torch.cuda.memory_allocated(device)
                ),
                "system/gpu/memory_reserved_bytes": float(
                    torch.cuda.memory_reserved(device)
                ),
                "system/gpu/max_memory_allocated_bytes": float(
                    torch.cuda.max_memory_allocated(device)
                ),
            }
        )
        utilization = getattr(torch.cuda, "utilization", None)
        if callable(utilization):
            try:
                metrics["system/gpu/utilization_percent"] = float(
                    utilization(device)
                )
            except (OSError, RuntimeError):
                pass
    return metrics


def _base_config(
    args: argparse.Namespace,
    *,
    snapshot: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": "0017_terminal_rl_training_v1",
        "project_id": PROJECT_ID,
        "version": args.version,
        "command": args.command,
        "device": str(device),
        "workers": args.workers,
        "seed": args.seed,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "reward": {"win": 1.0, "loss": -1.0, "draw": 0.0, "gamma": 1.0},
        "opponent_snapshot_sha256": snapshot["catalog_sha256"],
        "storage": {"warning_free_gib": 80, "hard_stop_free_gib": 50, "version_cap_gib": 20},
        "wandb": {
            "mode": args.wandb_mode,
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
        },
    }


def _allocate(args: argparse.Namespace, device: torch.device):
    packages, snapshot = load_frozen_pool()
    paths = initialize_version(PROJECT_ID, args.version)
    write_snapshot(paths.artifact / "opponent_snapshot.json", snapshot)
    config = _base_config(args, snapshot=snapshot, device=device)
    _write_json(paths.config, config)
    write_version_status(
        paths,
        {
            "state": "running",
            "stage": args.command,
            "device": str(device),
            "optimizer_state_saved": False,
            "raw_rollout_saved": False,
        },
    )
    return paths, packages, snapshot, config


def _log_episodes(
    logger: TrainingLogger,
    episodes,
    tracker: OutcomeTracker,
    *,
    episode_offset: int,
    decision_offset: int,
    extra: dict[str, float] | None = None,
) -> tuple[int, int]:
    episodes_seen = episode_offset
    decisions_seen = decision_offset
    for episode in episodes:
        tracker.add(episode)
        if episode.valid:
            episodes_seen += 1
            decisions_seen += len(episode.decisions)
        metrics = {
            "env/episodes": episodes_seen,
            "env/decisions": decisions_seen,
            **tracker.metrics(),
            **(extra or {}),
        }
        logger.log(decisions_seen, metrics)
    return episodes_seen, decisions_seen


def run_probe(args: argparse.Namespace) -> int:
    device = _device(args.device, args.allow_gpu)
    paths, packages, snapshot, config = _allocate(args, device)
    try:
        model, metadata = _load_model(device)
        model.eval()
        jobs = balanced_jobs(
            packages,
            count=len(packages) * args.games_per_opponent,
            seed=args.seed,
            prefix="probe",
        )
        collector = RolloutCollector(
            model, device=device, workers=args.workers, mode="greedy"
        )
        tracker = OutcomeTracker()
        started = time.perf_counter()
        episodes = collector.collect(jobs)
        wall = time.perf_counter() - started
        with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
            episode_count, decision_count = _log_episodes(
                logger, episodes, tracker, episode_offset=0, decision_offset=0
            )
            summary_metrics = {
                "trainer/update": 0,
                "env/episodes": episode_count,
                "env/decisions": decision_count,
                **tracker.metrics(),
                **episode_metrics(episodes),
                "system/rollout/wall_seconds": wall,
                "system/rollout/episodes_per_second": len(episodes) / wall,
                **_runtime_metrics(model, device),
                **storage_guard(paths.run_root).metrics(),
            }
            logger.log(decision_count, summary_metrics)
        summary = {
            "state": "completed",
            "source_version": metadata["version"],
            "episodes": len(episodes),
            "valid_episodes": sum(item.valid for item in episodes),
            "wins": sum(item.reward == 1 for item in episodes),
            "losses": sum(item.reward == -1 for item in episodes),
            "draws": sum(item.reward == 0 for item in episodes),
            "errors": sum(not item.valid for item in episodes),
            "wall_seconds": wall,
            "catalog_count": snapshot["enabled_count"],
            "config": config,
        }
        _write_json(paths.summary, summary)
        write_version_status(paths, summary)
        return 0
    except BaseException as error:
        write_version_status(
            paths,
            {"state": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def run_value(args: argparse.Namespace) -> int:
    device = _device(args.device, args.allow_gpu)
    paths, packages, snapshot, config = _allocate(args, device)
    value_config = ValueConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    config["value"] = asdict(value_config)
    config["episodes"] = args.episodes
    _write_json(paths.config, config)
    try:
        model, metadata = _load_model(device)
        train_pool = select_slice(packages, snapshot, "train")
        jobs = balanced_jobs(
            train_pool, count=args.episodes, seed=args.seed, prefix="value"
        )
        collector = RolloutCollector(
            model.eval(), device=device, workers=args.workers, mode="sample"
        )
        tracker = OutcomeTracker()
        with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
            progress = {"episodes": 0, "decisions": 0}

            def on_episode(episode) -> None:
                tracker.add(episode)
                if episode.valid:
                    progress["episodes"] += 1
                    progress["decisions"] += len(episode.decisions)
                if progress["episodes"] % 10 == 0 or not episode.valid:
                    logger.log(
                        progress["decisions"],
                        {
                            "trainer/epoch": 0,
                            "env/episodes": progress["episodes"],
                            "env/decisions": progress["decisions"],
                            **tracker.metrics(),
                        },
                    )

            started = time.perf_counter()
            episodes = collector.collect(jobs, on_episode=on_episode)
            rollout_seconds = time.perf_counter() - started
            batch = prepare_episodes(episodes)
            history = calibrate_value(model, batch, device=device, config=value_config)
            for metrics in history:
                epoch = int(metrics["trainer/epoch"])
                logger.log(
                    epoch,
                    {
                        "env/episodes": progress["episodes"],
                        "env/decisions": progress["decisions"],
                        "trainer/update": epoch,
                        **metrics,
                        **tracker.metrics(),
                        **episode_metrics(episodes),
                        "system/rollout/wall_seconds": rollout_seconds,
                        **_runtime_metrics(model, device),
                        **storage_guard(paths.run_root).metrics(),
                    },
                )
        checkpoint = save_model_only(
            paths.checkpoints / "update-000000-value.pt",
            model,
            update=0,
            metadata={
                "project_id": PROJECT_ID,
                "version": args.version,
                "stage": "value_calibration",
                "source_version": metadata["version"],
                "optimizer_state_saved": False,
            },
        )
        summary = {
            "state": "completed",
            "episodes": len(episodes),
            "valid_episodes": sum(item.valid for item in episodes),
            "decisions": batch.decisions,
            "checkpoint": str(checkpoint),
            "final_metrics": history[-1],
            "optimizer_state_saved": False,
        }
        _write_json(paths.summary, summary)
        write_version_status(paths, summary)
        return 0
    except BaseException as error:
        write_version_status(
            paths,
            {"state": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def _load_model_only(path: Path, device: torch.device) -> DragapultActorCritic:
    model, _ = _load_model(torch.device("cpu"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if any(key in payload for key in ("optimizer", "scheduler", "scaler", "rng_state")):
        raise ValueError("warm start violates model-only checkpoint contract")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device)


def run_ppo(args: argparse.Namespace) -> int:
    device = _device(args.device, args.allow_gpu)
    paths, packages, snapshot, config = _allocate(args, device)
    ppo_config = PPOConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        actor_learning_rate=args.actor_learning_rate,
        value_learning_rate=args.value_learning_rate,
    )
    config["ppo"] = asdict(ppo_config)
    config["episodes_per_update"] = args.episodes_per_update
    config["updates"] = args.updates
    config["warm_start"] = str(args.warm_start)
    _write_json(paths.config, config)
    try:
        model = _load_model_only(args.warm_start, device)
        reference, _ = _load_model(device)
        reference = frozen_reference(reference, device)
        trainer = PPOTrainer(model, reference, device=device, config=ppo_config)
        train_pool = select_slice(packages, snapshot, "train")
        collector = RolloutCollector(
            model, device=device, workers=args.workers, mode="sample"
        )
        tracker = OutcomeTracker()
        episodes_seen = 0
        decisions_seen = 0
        with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
            for update in range(1, args.updates + 1):
                storage = storage_guard(paths.run_root)
                jobs = balanced_jobs(
                    train_pool,
                    count=args.episodes_per_update,
                    seed=args.seed + update * args.episodes_per_update,
                    prefix=f"ppo-u{update:05d}",
                )
                progress_at_start = (episodes_seen, decisions_seen)

                def on_episode(episode) -> None:
                    nonlocal episodes_seen, decisions_seen
                    tracker.add(episode)
                    if episode.valid:
                        episodes_seen += 1
                        decisions_seen += len(episode.decisions)
                    collected = episodes_seen - progress_at_start[0]
                    if collected % 10 == 0 or not episode.valid:
                        logger.log(
                            decisions_seen,
                            {
                                "trainer/update": update - 1,
                                "env/episodes": episodes_seen,
                                "env/decisions": decisions_seen,
                                **tracker.metrics(),
                            },
                        )

                started = time.perf_counter()
                episodes = collector.collect(jobs, on_episode=on_episode)
                rollout_seconds = time.perf_counter() - started
                batch = prepare_episodes(episodes)
                metrics = trainer.update(batch)
                logger.log(
                    update,
                    {
                        "trainer/update": update,
                        "env/episodes": episodes_seen,
                        "env/decisions": decisions_seen,
                        **metrics,
                        **tracker.metrics(),
                        **episode_metrics(episodes),
                        "system/rollout/wall_seconds": rollout_seconds,
                        "system/rollout/episodes_per_second": len(episodes) / rollout_seconds,
                        **_runtime_metrics(model, device),
                        **storage.metrics(),
                    },
                )
                if update == 1 or update % args.checkpoint_every == 0 or update == args.updates:
                    save_model_only(
                        paths.checkpoints / f"update-{update:06d}.pt",
                        model,
                        update=update,
                        metadata={
                            "project_id": PROJECT_ID,
                            "version": args.version,
                            "stage": "ppo",
                            "optimizer_state_saved": False,
                            "env_episodes": episodes_seen,
                            "env_decisions": decisions_seen,
                        },
                    )
                    prune_model_checkpoints(paths.checkpoints, keep=8)
        summary = {
            "state": "completed",
            "updates": args.updates,
            "episodes": episodes_seen,
            "decisions": decisions_seen,
            "optimizer_state_saved": False,
        }
        _write_json(paths.summary, summary)
        write_version_status(paths, summary)
        return 0
    except BaseException as error:
        write_version_status(
            paths,
            {"state": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="0017 Dragapult terminal-reward RL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(name: str) -> argparse.ArgumentParser:
        item = subparsers.add_parser(name)
        item.add_argument("--version", required=True)
        item.add_argument("--device", default="cpu")
        item.add_argument("--allow-gpu", action="store_true")
        item.add_argument("--workers", type=int, default=8)
        item.add_argument("--seed", type=int, default=20260728)
        item.add_argument(
            "--wandb-mode",
            choices=("online", "offline", "disabled"),
            default="online",
        )
        return item

    probe = common("probe")
    probe.add_argument("--games-per-opponent", type=int, default=10)
    probe.set_defaults(handler=run_probe)

    value = common("value")
    value.add_argument("--episodes", type=int, default=2_000)
    value.add_argument("--epochs", type=int, default=4)
    value.add_argument("--batch-size", type=int, default=1024)
    value.add_argument("--learning-rate", type=float, default=1e-4)
    value.set_defaults(handler=run_value)

    ppo = common("ppo")
    ppo.add_argument("--warm-start", type=Path, required=True)
    ppo.add_argument("--updates", type=int, default=100)
    ppo.add_argument("--episodes-per-update", type=int, default=256)
    ppo.add_argument("--epochs", type=int, default=4)
    ppo.add_argument("--batch-size", type=int, default=1024)
    ppo.add_argument("--actor-learning-rate", type=float, default=1e-5)
    ppo.add_argument("--value-learning-rate", type=float, default=1e-4)
    ppo.add_argument("--checkpoint-every", type=int, default=10)
    ppo.set_defaults(handler=run_ppo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    _configure_wandb(args)
    torch.set_num_threads(1 if args.device == "cpu" else max(1, min(4, os.cpu_count() or 1)))
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
