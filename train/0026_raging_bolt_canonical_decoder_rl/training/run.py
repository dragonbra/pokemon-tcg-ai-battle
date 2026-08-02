"""Versioned decoder-only PPO training against the Frozen 0019 league."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rl_environment.logging import TrainingLogger

from .. import FOCAL_DECK_ID, PROJECT_ID
from ..checkpoint import save_model_checkpoint
from ..league import load_frozen_catalog
from ..policy import load_actor_critic
from ..rollout import HeterogeneousRolloutCollector, RolloutJob
from ..smoke import runtime_root, smoke_jobs
from .batch import prepare_episodes
from .ppo import PPOConfig, PPOTrainer


RUN_ROOT = Path("rl_runs") / PROJECT_ID / "versions"


@dataclass(frozen=True)
class RunConfig:
    version: str
    updates: int = 20
    games_per_update: int = 512
    workers: int = 8
    device: str = "cuda:0"
    seed: int = 20260802
    eval_every: int = 5
    eval_games: int = 102
    checkpoint_retention: int = 6
    wandb_mode: str = "online"
    ppo: PPOConfig = PPOConfig()

    def validate(self) -> None:
        if not self.version.startswith("V") or "/" in self.version:
            raise ValueError("version must be a V<n>_<tag> directory name")
        if min(self.updates, self.games_per_update, self.workers) < 1:
            raise ValueError("updates, games_per_update, and workers must be positive")
        if self.eval_every < 1 or self.eval_games != 102:
            raise ValueError("frozen evaluation contract is exactly 102 games")
        if self.checkpoint_retention < 2:
            raise ValueError("checkpoint_retention must preserve at least last and best")
        if self.wandb_mode not in {"disabled", "offline", "online"}:
            raise ValueError("invalid W&B mode")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _status(path: Path, payload: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            existing = value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            existing = {}
    if "wandb" in existing and "wandb" not in payload:
        payload = {**payload, "wandb": existing["wandb"]}
    _atomic_json(path, payload)


def _paths(version: str) -> dict[str, Path]:
    root = RUN_ROOT / version
    return {
        "root": root,
        "artifact": root / "artifact",
        "metrics": root / "artifact" / "training_metrics.jsonl",
        "status": root / "artifact" / "status.json",
        "summary": root / "artifact" / "training_summary.json",
        "config": root / "artifact" / "training_config.json",
        "checkpoint": root / "checkpoint",
        "tensorboard": root / "tensorboard",
        "wandb": root / "wandb",
    }


def assert_fresh_version(version: str) -> dict[str, Path]:
    paths = _paths(version)
    if any(path.exists() for key, path in paths.items() if key != "root"):
        raise FileExistsError(f"0026 version has already been used: {version}")
    return paths


def build_jobs(
    *, count: int, source_policy_update: int, seed: int, greedy_eval: bool = False
) -> list[RolloutJob]:
    catalog = load_frozen_catalog()
    focal = next(item for item in catalog if item.deck_id == FOCAL_DECK_ID)
    if count < 2 * len(catalog):
        raise ValueError("a league schedule must cover every opponent in both seats")
    if greedy_eval and count != 2 * len(catalog):
        raise ValueError("greedy evaluation must be one game per opponent and seat")
    root = runtime_root()
    jobs: list[RolloutJob] = []
    cycle = 0
    while len(jobs) < count:
        rotated = list(catalog[cycle % len(catalog) :]) + list(catalog[: cycle % len(catalog)])
        for opponent in rotated:
            for focal_first in (True, False):
                if len(jobs) >= count:
                    break
                index = len(jobs)
                jobs.append(RolloutJob(
                    game_id=("eval" if greedy_eval else "rollout")
                    + f"-u{source_policy_update:04d}-{index:04d}",
                    opponent_id=opponent.deck_id,
                    focal_first=focal_first,
                    seed=seed + source_policy_update * 1_000_003 + index,
                    source_policy_update=source_policy_update,
                    focal_deck=focal.deck,
                    opponent_deck=opponent.deck,
                    runtime_root=root,
                ))
            if len(jobs) >= count:
                break
        cycle += 1
    if len({job.opponent_id for job in jobs}) != len(catalog):
        raise RuntimeError("league schedule omitted a Frozen opponent")
    if abs(sum(job.focal_first for job in jobs) - count / 2) > 1:
        raise RuntimeError("league schedule is not seat-balanced")
    return jobs


def _episode_metrics(episodes: list[Any], prefix: str) -> dict[str, float]:
    valid = [episode for episode in episodes if episode.valid]
    errors = [episode for episode in episodes if not episode.valid]
    if errors or len(valid) != len(episodes):
        detail = [episode.error for episode in errors[:5]]
        raise RuntimeError(f"official-engine errors ({len(errors)}): {detail}")
    games = len(valid)
    wins = sum(episode.reward == 1 for episode in valid)
    draws = sum(episode.reward == 0 for episode in valid)
    focal_first = [episode for episode in valid if episode.job.focal_first]
    focal_second = [episode for episode in valid if not episode.job.focal_first]
    return {
        f"{prefix}/episodes": float(games),
        f"{prefix}/wins": float(wins),
        f"{prefix}/losses": float(sum(episode.reward == -1 for episode in valid)),
        f"{prefix}/draws": float(draws),
        f"{prefix}/win_rate": wins / games,
        f"{prefix}/score_rate": (wins + 0.5 * draws) / games,
        f"{prefix}/first_win_rate": sum(ep.reward == 1 for ep in focal_first) / len(focal_first),
        f"{prefix}/second_win_rate": sum(ep.reward == 1 for ep in focal_second) / len(focal_second),
        f"{prefix}/mean_engine_turn": sum(ep.turns for ep in valid) / games,
        f"{prefix}/decisions": float(sum(len(ep.decisions) for ep in valid)),
    }


def _system_metrics(device: torch.device) -> dict[str, float]:
    disk = shutil.disk_usage(Path.cwd())
    metrics = {
        "system/disk/free_bytes": float(disk.free),
        "system/disk/used_bytes": float(disk.used),
    }
    if device.type == "cuda":
        metrics.update({
            "system/gpu/max_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
            "system/gpu/max_reserved_bytes": float(torch.cuda.max_memory_reserved(device)),
        })
    return metrics


def _retain_checkpoints(directory: Path, limit: int, protected: set[Path]) -> None:
    checkpoints = sorted(directory.glob("update-*.pt"))
    removable = [path for path in checkpoints if path not in protected]
    while len(checkpoints) > limit and removable:
        victim = removable.pop(0)
        victim.unlink()
        checkpoints.remove(victim)


def _evaluate(model: Any, config: RunConfig, update: int) -> tuple[dict[str, float], float]:
    collector = HeterogeneousRolloutCollector(
        model, device=torch.device(config.device), workers=config.workers, mode="greedy"
    )
    started = time.perf_counter()
    episodes = collector.collect(build_jobs(
        count=config.eval_games,
        source_policy_update=update,
        seed=config.seed + 70_000_000,
        greedy_eval=True,
    ))
    metrics = _episode_metrics(episodes, "eval")
    metrics.update(collector.metrics())
    metrics["eval/checkpoint_update"] = float(update)
    metrics["eval/wall_seconds"] = time.perf_counter() - started
    return metrics, metrics["eval/win_rate"]


def run(config: RunConfig) -> dict[str, Any]:
    config.validate()
    paths = assert_fresh_version(config.version)
    for key in ("artifact", "checkpoint", "tensorboard", "wandb"):
        paths[key].mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.update({
        "WANDB_MODE": config.wandb_mode,
        "WANDB_ENTITY": "dragon_bra",
        "WANDB_PROJECT": "pokemon-tcg-policy-learning",
        "WANDB_JOB_TYPE": "ppo_decoder_only",
        "WANDB_TAGS": "0026,raging_bolt,decoder_only,official_engine,frozen_0019",
        "WANDB_DIR": str(paths["wandb"].resolve()),
    })
    config_payload = {
        "schema_version": "0026_decoder_only_ppo_run_v1",
        "project_id": PROJECT_ID,
        **asdict(config),
        "opponent_count": 51,
        "trainable_contract": ["actor.action_decoder.*", "value_head.*"],
        "reward": "official_engine_terminal_minus_one_zero_plus_one",
        "checkpoint_payload": "model_only_action_decoder_and_value_head",
    }
    _atomic_json(paths["config"], config_payload)
    _status(paths["status"], {"state": "initializing", "version": config.version})
    model, identity = load_actor_critic(device)
    trainer = PPOTrainer(model, device=device, config=config.ppo)
    representation_sha = model.representation_sha256()
    initial_decoder_sha = model.decoder_sha256()
    initial_checkpoint_sha = save_model_checkpoint(
        paths["checkpoint"] / "update-0000.pt", model,
        policy_version=config.version, update=0,
    )
    best_eval = -math.inf
    best_update = 0
    best_path = paths["checkpoint"] / "update-0000.pt"
    cumulative_episodes = 0
    cumulative_decisions = 0
    protected = {best_path}
    started = time.time()
    _status(paths["status"], {
        "state": "running", "version": config.version, "pid": os.getpid(),
        "representation_sha256": representation_sha,
        "initial_decoder_sha256": initial_decoder_sha,
        "initial_checkpoint_sha256": identity.checkpoint_sha256,
        "model_checkpoint_update_0_sha256": initial_checkpoint_sha,
        "wandb": {"state": config.wandb_mode},
    })
    try:
        with TrainingLogger(paths["metrics"], paths["tensorboard"]) as logger:
            eval_metrics, best_eval = _evaluate(model, config, 0)
            eval_decisions = int(eval_metrics["eval/decisions"])
            logger.log(0, {
                "trainer/update": 0,
                "env/episodes": 0,
                "env/decisions": 0,
                "checkpoint/update": 0,
                "rollout/source_policy_update": 0,
                **eval_metrics,
                **_system_metrics(device),
            })
            for update in range(1, config.updates + 1):
                source_update = update - 1
                collector = HeterogeneousRolloutCollector(
                    model, device=device, workers=config.workers, mode="sample"
                )
                rollout_started = time.perf_counter()
                episodes = collector.collect(build_jobs(
                    count=config.games_per_update,
                    source_policy_update=source_update,
                    seed=config.seed,
                ))
                rollout_metrics = _episode_metrics(episodes, "rollout")
                rollout_metrics.update(collector.metrics())
                rollout_metrics["rollout/wall_seconds"] = time.perf_counter() - rollout_started
                cumulative_episodes += len(episodes)
                cumulative_decisions += int(rollout_metrics["rollout/decisions"])
                batch = prepare_episodes(
                    episodes, gamma=1.0, gae_lambda=config.ppo.gae_lambda
                )
                ppo_metrics = trainer.update(batch)
                checkpoint_path = paths["checkpoint"] / f"update-{update:04d}.pt"
                checkpoint_sha = save_model_checkpoint(
                    checkpoint_path, model, policy_version=config.version, update=update
                )
                metrics: dict[str, Any] = {
                    "trainer/update": update,
                    "env/episodes": cumulative_episodes,
                    "env/decisions": cumulative_decisions,
                    "rollout/source_policy_update": source_update,
                    "checkpoint/update": update,
                    "checkpoint/sha256": checkpoint_sha,
                    "representation/sha256_unchanged": float(
                        model.representation_sha256() == representation_sha
                    ),
                    **rollout_metrics,
                    **ppo_metrics,
                    **_system_metrics(device),
                }
                if update % config.eval_every == 0 or update == config.updates:
                    eval_metrics, eval_score = _evaluate(model, config, update)
                    eval_decisions += int(eval_metrics["eval/decisions"])
                    metrics.update(eval_metrics)
                    if eval_score > best_eval:
                        best_eval = eval_score
                        best_update = update
                        best_path = checkpoint_path
                protected = {best_path, checkpoint_path}
                _retain_checkpoints(paths["checkpoint"], config.checkpoint_retention, protected)
                logger.log(update, metrics)
                _status(paths["status"], {
                    "state": "running", "version": config.version, "pid": os.getpid(),
                    "completed_update": update, "source_policy_update": source_update,
                    "checkpoint_update": update, "best_eval_update": best_update,
                    "best_eval_win_rate": best_eval,
                    "representation_sha256": representation_sha,
                    "decoder_sha256": model.decoder_sha256(),
                    "elapsed_seconds": time.time() - started,
                })
        summary = {
            "state": "complete", "version": config.version,
            "updates": config.updates, "episodes": cumulative_episodes,
            "rollout_decisions": cumulative_decisions,
            "eval_decisions": eval_decisions,
            "best_eval_update": best_update, "best_eval_win_rate": best_eval,
            "best_checkpoint": str(best_path),
            "representation_sha256": representation_sha,
            "representation_unchanged": model.representation_sha256() == representation_sha,
            "initial_decoder_sha256": initial_decoder_sha,
            "final_decoder_sha256": model.decoder_sha256(),
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(paths["summary"], summary)
        status = dict(summary)
        status["wandb"] = json.loads(paths["status"].read_text()).get("wandb", {})
        _status(paths["status"], status)
        return summary
    except BaseException as error:
        _status(paths["status"], {
            "state": "failed", "version": config.version, "pid": os.getpid(),
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.time() - started,
            "representation_sha256": representation_sha,
        })
        raise


def run_smoke_gate(
    *, version: str, games: int = 4, workers: int = 2, device_name: str = "cuda:0"
) -> dict[str, Any]:
    """Run one small official-engine PPO update without claiming formal coverage."""
    if games < 2 or games % 2:
        raise ValueError("smoke games must be an even number of at least two")
    paths = assert_fresh_version(version)
    for key in ("artifact", "checkpoint", "tensorboard", "wandb"):
        paths[key].mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ["WANDB_MODE"] = "disabled"
    smoke_config = {
        "schema_version": "0026_decoder_only_ppo_smoke_v1",
        "project_id": PROJECT_ID,
        "version": version,
        "games": games,
        "workers": workers,
        "device": device_name,
        "coverage": "smoke_only_not_all_51_opponents",
        "ppo": asdict(PPOConfig()),
        "wandb": {"mode": "disabled", "reason": "local smoke diagnostic"},
    }
    _atomic_json(paths["config"], smoke_config)
    _status(paths["status"], {"state": "initializing", "version": version})
    model, identity = load_actor_critic(device)
    trainer = PPOTrainer(model, device=device)
    representation_sha = model.representation_sha256()
    decoder_before = model.decoder_sha256()
    save_model_checkpoint(
        paths["checkpoint"] / "update-0000.pt", model,
        policy_version=version, update=0,
    )
    started = time.perf_counter()
    try:
        collector = HeterogeneousRolloutCollector(
            model, device=device, workers=workers, mode="sample"
        )
        episodes = collector.collect(smoke_jobs(games))
        rollout_metrics = _episode_metrics(episodes, "rollout")
        rollout_metrics.update(collector.metrics())
        batch = prepare_episodes(episodes)
        ppo_metrics = trainer.update(batch)
        final_sha = save_model_checkpoint(
            paths["checkpoint"] / "update-0001.pt", model,
            policy_version=version, update=1,
        )
        metrics = {
            "trainer/update": 1,
            "env/episodes": games,
            "env/decisions": int(rollout_metrics["rollout/decisions"]),
            "rollout/source_policy_update": 0,
            "checkpoint/update": 1,
            "checkpoint/sha256": final_sha,
            **rollout_metrics,
            **ppo_metrics,
            **_system_metrics(device),
        }
        with TrainingLogger(paths["metrics"], paths["tensorboard"]) as logger:
            logger.log(1, metrics, mirror_wandb=False)
        summary = {
            "state": "complete", "version": version,
            "scope": "smoke_only_not_formal_strength_evidence",
            "games": games, "engine_errors": 0,
            "representation_sha256": representation_sha,
            "representation_unchanged": model.representation_sha256() == representation_sha,
            "initial_checkpoint_sha256": identity.checkpoint_sha256,
            "decoder_sha256_before": decoder_before,
            "decoder_sha256_after": model.decoder_sha256(),
            "decoder_changed": model.decoder_sha256() != decoder_before,
            "final_checkpoint_sha256": final_sha,
            "elapsed_seconds": time.perf_counter() - started,
            "wandb": {"state": "disabled", "reason": "local smoke diagnostic"},
        }
        _atomic_json(paths["summary"], summary)
        _status(paths["status"], summary)
        return summary
    except BaseException as error:
        _status(paths["status"], {
            "state": "failed", "version": version,
            "error": f"{type(error).__name__}: {error}",
            "representation_sha256": representation_sha,
        })
        raise


__all__ = [
    "RunConfig", "assert_fresh_version", "build_jobs", "run", "run_smoke_gate",
]
