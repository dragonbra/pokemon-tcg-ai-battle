"""Guarded focal-only PPO orchestration for the first 0022 formal version."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rl_environment.logging import TrainingLogger
from rl_environment.runs import project_version_paths, write_version_status

from .. import PROJECT_ID
from ..decoder import save_decoder_checkpoint
from ..decks import DeckPlugin, load_deck_plugins
from ..foundation import verify_foundation
from ..league import DEFAULT_DECK_ROOT, initialize_league_version
from ..policy import load_league_actor_critic
from ..rollout import LeaguePolicyView, LeagueRolloutCollector, RolloutJob
from ..storage import preflight_storage, prune_latest, runtime_storage
from .batch import prepare_episodes
from .ppo import PPOConfig, PPOTrainer, frozen_reference


@dataclass(frozen=True)
class LeagueTrainingConfig:
    version: str
    device: str = "cuda:0"
    workers: int = 8
    games_per_update: int = 512
    duration_hours: float = 20.0
    frozen_eval_interval: int = 5
    seed: int = 22022
    coalesce_ms: float = 0.5
    checkpoint_keep_latest: int = 2
    launch_minimum_free_gib: float = 100.0
    runtime_stop_free_gib: float = 80.0
    version_cap_gib: float = 10.0


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_root() -> Path:
    candidates = sorted(Path("evaluation/arena/opponents").glob("*/cg/game.py"))
    if not candidates: raise FileNotFoundError("official engine runtime package is unavailable")
    return candidates[0].parents[1].resolve()


def compose_training_config(initialization: dict[str, object], config: LeagueTrainingConfig, *,
                            identity: dict[str, object], catalog_count: int, runtime_root: Path) -> dict[str, object]:
    if "checkpoints" not in initialization or "catalog" not in initialization:
        raise ValueError("League initialization config is missing asset provenance")
    return {
        **initialization, "schema_version": "0022_focal_league_ppo_v1",
        "run": asdict(config), "foundation": identity,
        "focal_deck_id": "dragapult_ex_001", "catalog_count": catalog_count,
        "opponent_views": ["frozen", "live"],
        "live_opponent_updates_enabled": False, "runtime_root": str(runtime_root),
        "ppo": asdict(PPOConfig()),
    }


def schedule_jobs(plugins: tuple[DeckPlugin, ...], *, count: int, update: int, seed: int, runtime_root: Path, evaluation: bool = False) -> list[RolloutJob]:
    focal = next(item for item in plugins if item.focal)
    views = [LeaguePolicyView.FROZEN] if evaluation else [LeaguePolicyView.FROZEN, LeaguePolicyView.LIVE]
    matchups = [(item, view) for view in views for item in plugins]
    if evaluation:
        count = len(matchups) * 2
    rng = random.Random(seed + update * 1_000_003 + int(evaluation) * 97)
    offset = rng.randrange(len(matchups))
    jobs = []
    for index in range(count):
        opponent, view = matchups[(offset + index // 2) % len(matchups)]
        focal_first = index % 2 == 0
        jobs.append(RolloutJob(
            game_id=f"{'eval' if evaluation else 'rollout'}-u{update:06d}-{index:04d}",
            focal_deck_id=focal.deck_id, opponent_deck_id=opponent.deck_id, opponent_view=view,
            focal_first=focal_first, seed=seed + update * 100_000 + index,
            source_policy_update=update, focal_deck=focal.deck, opponent_deck=opponent.deck,
            runtime_root=runtime_root,
        ))
    return jobs


def _episode_metrics(episodes, prefix: str) -> dict[str, float]:
    valid = [item for item in episodes if item.valid]
    wins = sum(item.reward == 1 for item in valid); losses = sum(item.reward == -1 for item in valid)
    draws = sum(item.reward == 0 for item in valid)
    return {
        f"{prefix}/episodes": float(len(episodes)), f"{prefix}/finished": float(len(valid)),
        f"{prefix}/errors": float(len(episodes) - len(valid)), f"{prefix}/wins": float(wins),
        f"{prefix}/losses": float(losses), f"{prefix}/draws": float(draws),
        f"{prefix}/win_rate": wins / len(valid) if valid else 0.0,
    }


def _frozen_eval(model, plugins, config, update, runtime_root):
    collector = LeagueRolloutCollector(model, device=torch.device(config.device), workers=config.workers, mode="greedy", coalesce_ms=config.coalesce_ms)
    started = time.perf_counter()
    episodes = collector.collect(schedule_jobs(plugins, count=0, update=update, seed=config.seed, runtime_root=runtime_root, evaluation=True))
    wall = time.perf_counter() - started
    metrics = {"eval/checkpoint_update": float(update), "eval/wall_seconds": wall, **_episode_metrics(episodes, "eval/frozen"), **collector.metrics()}
    if metrics["eval/frozen/errors"] != 0 or metrics["eval/frozen/finished"] != len(plugins) * 2:
        raise RuntimeError(f"Frozen greedy evaluation gate failed: {metrics}")
    return metrics


def run_training(config: LeagueTrainingConfig, *, deck_root: Path = DEFAULT_DECK_ROOT) -> int:
    if config.games_per_update < 1 or config.duration_hours <= 0 or config.frozen_eval_interval < 1:
        raise ValueError("training counts, duration, and evaluation interval must be positive")
    prospective = project_version_paths(PROJECT_ID, config.version)
    preflight_storage(prospective.run_root.parent, minimum_free_gib=config.launch_minimum_free_gib)
    identity = verify_foundation(); plugins = load_deck_plugins(deck_root)
    paths = initialize_league_version(config.version, deck_root=deck_root)
    runtime_root = _runtime_root(); device = torch.device(config.device)
    if device.type == "cuda": torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
    initialization_config = json.loads(paths.config.read_text(encoding="utf-8"))
    training_config = compose_training_config(
        initialization_config, config, identity=identity.as_dict(),
        catalog_count=len(plugins), runtime_root=runtime_root,
    )
    _atomic_json(paths.config, training_config)
    os.environ.update({
        "WANDB_MODE": "online", "WANDB_ENTITY": "dragon_bra",
        "WANDB_PROJECT": "pokemon-tcg-policy-learning", "WANDB_JOB_TYPE": "ppo_train",
        "WANDB_TAGS": "0022,league,ppo,dragapult,focal_only",
    })
    focal = next(item for item in plugins if item.focal)
    focal_record = training_config["checkpoints"][focal.deck_id]
    if not isinstance(focal_record, dict) or not focal_record.get("materialized"):
        raise RuntimeError("focal update-0 decoder asset was not materialized")
    focal_initial_checkpoint = Path(str(focal_record["decoder_ref"]))
    if not focal_initial_checkpoint.is_absolute():
        focal_initial_checkpoint = Path.cwd() / focal_initial_checkpoint
    model, _ = load_league_actor_critic(
        config.device, decoder_checkpoint=focal_initial_checkpoint,
        deck_id=focal.deck_id, deck_sha256=focal.deck_sha256,
    )
    model.eval()
    reference = frozen_reference(model, device)
    trainer = PPOTrainer(model, reference, device=device, config=PPOConfig())
    checkpoint_dir = paths.checkpoints / "focal"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.time(); deadline = started + config.duration_hours * 3600; update = 0
    episodes_total = 0; decisions_total = 0; best_checkpoints: list[tuple[float, Path]] = []
    write_version_status(paths, {"state": "preflight_frozen_eval", "training_started": False})
    try:
        with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
            eval_metrics = _frozen_eval(model, plugins, config, 0, runtime_root)
            logger.log(0, {"trainer/update": 0, "env/episodes": 0, "env/decisions": 0, **eval_metrics})
            write_version_status(paths, {"state": "running", "training_started": True, "started_at_unix": started})
            while time.time() < deadline:
                disk = runtime_storage(paths.run_root, stop_free_gib=config.runtime_stop_free_gib, version_cap_gib=config.version_cap_gib)
                if disk.stop_requested: break
                collector = LeagueRolloutCollector(model, device=device, workers=config.workers, mode="sample", coalesce_ms=config.coalesce_ms)
                jobs = schedule_jobs(plugins, count=config.games_per_update, update=update, seed=config.seed, runtime_root=runtime_root)
                rollout_started = time.perf_counter(); episodes = collector.collect(jobs); rollout_wall = time.perf_counter() - rollout_started
                if sum(item.valid for item in episodes) != len(jobs):
                    raise RuntimeError(f"rollout update {update} has worker errors: {[item.error for item in episodes if not item.valid][:5]}")
                batch = prepare_episodes(episodes, gamma=1.0, gae_lambda=0.95)
                metrics = trainer.update(batch); update += 1
                episodes_total += len(episodes); decisions_total += batch.decisions
                checkpoint = checkpoint_dir / f"update-{update:06d}.pt"
                digest = save_decoder_checkpoint(checkpoint, model.actor, model.value_head, foundation_sha256=identity.weights_sha256, deck_id=focal.deck_id, deck_sha256=focal.deck_sha256, policy_role="live", policy_version=config.version, update=update)
                eval_metrics = {}
                if update % config.frozen_eval_interval == 0:
                    eval_metrics = _frozen_eval(model, plugins, config, update, runtime_root)
                    score = float(eval_metrics["eval/frozen/win_rate"])
                    best_checkpoints.append((score, checkpoint))
                    best_checkpoints = sorted(best_checkpoints, key=lambda item: (-item[0], str(item[1])))[:2]
                prune_latest(checkpoint_dir, keep=config.checkpoint_keep_latest, protected={path for _, path in best_checkpoints})
                disk = runtime_storage(paths.run_root, stop_free_gib=config.runtime_stop_free_gib, version_cap_gib=config.version_cap_gib)
                log = {
                    "trainer/update": update, "rollout/source_policy_update": update - 1,
                    "checkpoint/update": update, "env/episodes": episodes_total, "env/decisions": decisions_total,
                    "rollout/wall_seconds": rollout_wall, "rollout/episodes_per_second": len(episodes) / rollout_wall,
                    "checkpoint/sha256_present": float(bool(digest)), **_episode_metrics(episodes, "rollout"),
                    **metrics, **collector.metrics(), **eval_metrics, **disk.metrics(),
                }
                if device.type == "cuda":
                    log.update({"system/gpu/allocated_bytes": torch.cuda.memory_allocated(device), "system/gpu/reserved_bytes": torch.cuda.memory_reserved(device), "system/gpu/peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
                logger.log(update, log)
                write_version_status(paths, {"state": "running", "checkpoint_update": update, "rollout_source_policy_update": update - 1, "episodes": episodes_total, "decisions": decisions_total, "disk_free_gib": disk.free_bytes / (1024**3)})
        final_state = "completed_time_budget" if time.time() >= deadline else "completed_disk_low_water"
        summary = {"state": final_state, "updates": update, "episodes": episodes_total, "decisions": decisions_total, "best_frozen_eval_win_rate": best_checkpoints[0][0] if best_checkpoints else None, "elapsed_hours": (time.time() - started) / 3600}
        _atomic_json(paths.summary, summary); write_version_status(paths, summary); return 0
    except BaseException as error:
        write_version_status(paths, {"state": "failed", "error": f"{type(error).__name__}: {error}", "checkpoint_update": update})
        raise


__all__ = ["LeagueTrainingConfig", "compose_training_config", "run_training", "schedule_jobs"]
