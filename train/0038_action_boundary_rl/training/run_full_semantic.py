"""Formal full-0031 decoder PPO over paired Frozen-0806 seed scenarios."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

import torch

from rl_environment.logging import TrainingLogger
from evaluation.runtime.seeded import build_seeded_runtime

from ..league import load_frozen_catalog
from ..parity import (
    assert_full_schema,
    assert_large_model_0806_runtime_parity,
    collect_official_observations,
)
from ..policy import AdaptationConfig, load_actor_critic
from ..rollout import (
    DEFAULT_FULL_ROUND_DRAW_LIMIT,
    ChunkedCudaRolloutCollector,
    FullSemanticRolloutCollector,
    RolloutJob,
)
from .batch_full_semantic import prepare_episodes
from .ppo_full_semantic import PPOConfig, PPOTrainer
from .storage_full_semantic import save_model_only
from .metric_frequency import is_sparse_diagnostic_update
from .accelerated_transfer import (
    AcceleratedTransferConfig,
    AcceleratedTransferController,
)
from ..semantic_policy.deployment.inference import PortableSemanticPolicy
from ..evaluation.frozen_jobs import (
    CANONICAL_CONTRACT_ID,
    EXPECTED_007_SCHEDULE_SHA256,
    build_frozen_jobs,
)
from ..evaluation.frozen_panel import paired_summary, wilson_interval
from ..initialization import (
    COMMON_UPDATE0_CHECKPOINT,
    COMMON_UPDATE0_SHA256,
    build_preset_from_common_update0,
)
from ..integrated.presets import PRESETS, preset
from ..export_full_semantic_candidate import export_candidate
from ..action_boundary.contracts import (
    ACTION_BOUNDARY_SCHEMA_VERSION,
    CANONICALIZER_VERSION,
    DECISION_GATE_VERSION,
    FEATURE_PREPROCESSING_VERSION,
    OFFICIAL_PROTOCOL_ADAPTER_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT = "0038_action_boundary_rl"
WANDB_DISPLAY_PREFIX = "0038 · action_boundary"
FORMAL_VERSION = "V13_accelerated_transfer_acceptance"
IMMUTABLE_GATE_C_TRACE_SHA256 = (
    "18c1684a3dc8158494fd9820278a85e60e351b2b274b520bb9fb413b1fa056ca"
)
SOURCE_CHECKPOINT = ROOT / "rl_runs/0037_dragapult_value_initialized_rl/source/friend_0806_epoch11/model.pt"
CANDIDATE_ROOT = ROOT / "evaluation/arena/candidates/0034_dragapult_third_large_model_zero_shot"
FOCAL_DECK_ID = "dragapult_ex_07bedfffbfad"
FOCAL_EXACT_DECK_SHA256 = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
FOCAL_DECK_PATH = ROOT / "train" / PROJECT / "league/decks" / FOCAL_DECK_ID / "deck.csv"
CUDA_RULES = ROOT / ".tmp/cuda_0032_rules/official_rules.bin"
CUDA_EXTENSION = ROOT / ".tmp/engine_cuda_benchmark/build_sm120_staged"
CUDA_RESIDENT_SOURCE = (
    ROOT / "engine_cuda/python/ptcg_cuda_engine/semantic0031_resident.py"
)
CUDA_PROGRESS_GUARD_SOURCE = (
    ROOT / "engine_cuda/python/ptcg_cuda_engine/progress_guard.py"
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    version: str = FORMAL_VERSION
    updates: int | None = None
    worker_processes: int = 16
    engines_per_worker: int = 8
    inference_channels_per_role: int = 8
    coalesce_ms: float = 5.0
    device: str = "cuda:0"
    seed: int = 330031001
    games_per_update: int = 512
    engine_backend: str = "accelerated:cuda_resident"
    cuda_lane_count: int = 256
    rollout_batch_size: int = 512
    trajectory_games_per_update: int = 512
    inference_batch_size: int = 64
    max_inflight_requests: int = 256
    rollout_queue_depth: int = 512
    seed_shard_count: int = 8
    optimization_mode: str = "fixed_optimizer_budget"
    optimizer_steps_per_update: int = 32
    ppo_gradient_accumulation: int = 1
    ppo_minibatch_size: int = 1024
    ppo_epochs: int = 4
    eval_every: int = 5
    adaptation_arm: str = "lora"
    preset_name: str = "INTEGRATED"
    wandb_mode: str = "online"
    launch_formal: bool = False
    resume_update0: bool = False
    accelerated_transfer_acceptance: bool = False
    accelerated_transfer: AcceleratedTransferConfig = AcceleratedTransferConfig()
    ppo: PPOConfig = PPOConfig()

    def validate(self) -> None:
        if re.fullmatch(r"V[1-9]\d*_[a-z0-9]+(?:_[a-z0-9]+)*", self.version) is None:
            raise ValueError("version must match V<n>_<ascii_snake_case>")
        if min(
            self.worker_processes,
            self.engines_per_worker,
            self.inference_channels_per_role,
            self.eval_every,
        ) < 1 or self.coalesce_ms < 0:
            raise ValueError("rollout topology and eval_every must be positive")
        if self.updates is not None and self.updates < 1:
            raise ValueError("updates must be positive when a finite limit is configured")
        if self.resume_update0 and not self.launch_formal:
            raise ValueError("resume_update0 requires the formal launch token")
        if self.inference_channels_per_role > self.engines_per_worker:
            raise ValueError(
                "inference_channels_per_role cannot exceed engines_per_worker"
            )
        if self.games_per_update < 256 or self.games_per_update % 256:
            raise ValueError(
                "games_per_update must contain complete 256-slot frequency units"
            )
        if self.engine_backend != "official" and not self.engine_backend.startswith("accelerated:"):
            raise ValueError("invalid engine backend")
        if self.optimization_mode not in {"fixed_epochs", "fixed_optimizer_budget"}:
            raise ValueError("invalid optimization mode")
        if min(self.inference_batch_size, self.max_inflight_requests,
               self.rollout_queue_depth, self.seed_shard_count,
               self.optimizer_steps_per_update, self.ppo_gradient_accumulation,
               self.cuda_lane_count, self.rollout_batch_size) < 1:
            raise ValueError("capacity settings must be positive")
        if not 1 <= self.trajectory_games_per_update <= self.games_per_update:
            raise ValueError(
                "trajectory_games_per_update must fit inside the rollout"
            )
        if (
            self.optimization_mode == "fixed_epochs"
            and self.trajectory_games_per_update != self.games_per_update
        ):
            raise ValueError("fixed_epochs must retain every rollout trajectory")
        if (self.ppo.batch_size != self.ppo_minibatch_size
                or self.ppo.epochs != self.ppo_epochs
                or self.ppo.gradient_accumulation != self.ppo_gradient_accumulation
                or self.ppo.optimizer_steps_per_update != self.optimizer_steps_per_update
                or self.ppo.optimization_mode != self.optimization_mode):
            raise ValueError("RunConfig and PPOConfig scaling contracts disagree")
        if self.wandb_mode not in {"online", "offline"}:
            raise ValueError("invalid W&B mode")
        if self.adaptation_arm not in {"lora", "lora_layernorm"}:
            raise ValueError("invalid adaptation arm")
        if self.preset_name not in PRESETS:
            raise ValueError("invalid integrated preset")
        if self.version == FORMAL_VERSION and not self.accelerated_transfer_acceptance:
            raise ValueError("V13 requires the accelerated transfer acceptance contract")
        if self.accelerated_transfer_acceptance:
            if self.updates is not None:
                raise ValueError("accelerated transfer acceptance must run until manual stop")
            if self.engine_backend != "accelerated:cuda_resident":
                raise ValueError("accelerated transfer acceptance requires CUDA resident")
            if self.preset_name != "PRIZE":
                raise ValueError("accelerated transfer acceptance disables Opponent Meta")
            if (
                self.games_per_update != 512
                or self.trajectory_games_per_update != 512
                or self.optimizer_steps_per_update != 32
                or self.eval_every != 5
                or self.optimization_mode != "fixed_optimizer_budget"
            ):
                raise ValueError("accelerated transfer scaling contract changed")
            self.accelerated_transfer.validate()
        self.ppo.validate()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _record_training_schedule(path: Path, payload: dict[str, Any]) -> None:
    """Keep all update schedules in one atomic, auditable version-level JSON."""
    rows = payload.get("jobs") or []
    updates = {int(row["source_policy_update"]) for row in rows}
    if len(updates) != 1:
        raise ValueError("training schedule must contain exactly one source-policy update")
    source_update = updates.pop()
    if path.is_file():
        aggregate = json.loads(path.read_text())
        if aggregate.get("schema") != "0038_dynamic_training_schedules_v2":
            raise ValueError("unexpected aggregate training schedule schema")
    else:
        aggregate = {
            "schema": "0038_dynamic_training_schedules_v2",
            "schedules": [],
        }
    schedules = aggregate.get("schedules")
    if not isinstance(schedules, list):
        raise ValueError("aggregate training schedules must be a list")
    existing = {
        int(item["source_policy_update"])
        for item in schedules
        if isinstance(item, dict) and "source_policy_update" in item
    }
    if source_update in existing:
        raise FileExistsError(f"training schedule update {source_update} already exists")
    schedules.append({"source_policy_update": source_update, **payload})
    schedules.sort(key=lambda item: int(item["source_policy_update"]))
    _atomic_json(path, aggregate)


def _merge_status(path: Path, payload: dict[str, Any]) -> None:
    existing = json.loads(path.read_text()) if path.is_file() else {}
    _atomic_json(path, {**existing, **payload})


def _paths(version: str) -> dict[str, Path]:
    root = ROOT / "rl_runs" / PROJECT / "versions" / version
    return {
        "root": root,
        "artifact": root / "artifact",
        "checkpoint": root / "checkpoint",
        "tensorboard": root / "tensorboard",
        "wandb": root / "wandb",
    }


def assert_fresh_version(version: str) -> dict[str, Path]:
    paths = _paths(version)
    occupied = [str(paths[name]) for name in ("artifact", "checkpoint", "tensorboard", "wandb") if paths[name].exists()]
    if occupied:
        raise FileExistsError("full-semantic version already used: " + ", ".join(occupied))
    target_number = int(version.split("_", 1)[0][1:])
    existing_numbers = [
        int(path.name.split("_", 1)[0][1:])
        for path in paths["root"].parent.glob("V[1-9]*_*")
        if re.fullmatch(r"V[1-9]\d*_[a-z0-9]+(?:_[a-z0-9]+)*", path.name)
    ]
    if existing_numbers and target_number <= max(existing_numbers):
        raise ValueError(
            f"version number must be greater than existing V{max(existing_numbers)}"
        )
    return paths


def assert_update0_resume(version: str) -> dict[str, Path]:
    paths = _paths(version)
    status_path = paths["artifact"] / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError("update-0 resume requires an existing status.json")
    status = json.loads(status_path.read_text())
    checkpoints = sorted(paths["checkpoint"].glob("update-*.pt"))
    metrics_path = paths["artifact"] / "training_metrics.jsonl"
    if (
        status.get("state") != "failed"
        or int(status.get("checkpoint_update", -1)) != 0
        or [path.name for path in checkpoints] != ["update-000000.pt"]
        or (metrics_path.is_file() and metrics_path.stat().st_size != 0)
    ):
        raise RuntimeError("resume is allowed only for an interrupted pre-PPO update-0 run")
    return paths


def focal_deck() -> tuple[int, ...]:
    cards = tuple(int(line) for line in FOCAL_DECK_PATH.read_text().splitlines())
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError("0034 focal deck is not exact 60")
    manifest = json.loads(FOCAL_DECK_PATH.with_name("manifest.json").read_text())
    if manifest.get("deck_id") != FOCAL_DECK_ID:
        raise ValueError("0034 focal deck ID does not match Frozen 007")
    if manifest.get("exact_deck_sha256") != FOCAL_EXACT_DECK_SHA256:
        raise ValueError("0034 focal deck hash does not match Frozen 007")
    return cards


def runtime_root() -> Path:
    roots = sorted((ROOT / "evaluation/arena/opponents").glob("*/cg/game.py"))
    if not roots:
        raise FileNotFoundError("official CPU engine runtime is unavailable")
    return roots[0].parents[1].resolve()


def build_collector(model, opponent, config: RunConfig, *, mode: str,
                    record_trajectory: bool | None = None):
    if record_trajectory is None:
        record_trajectory = mode == "sample"
    if config.engine_backend == "accelerated:cuda_resident":
        return ChunkedCudaRolloutCollector(
            model,
            opponent,
            rollout_batch_size=config.rollout_batch_size,
            trajectory_games_per_update=(
                config.trajectory_games_per_update if record_trajectory else None
            ),
            device=torch.device(config.device),
            rules_path=CUDA_RULES,
            extension_dir=CUDA_EXTENSION,
            lane_count=config.cuda_lane_count,
            mode=mode,
            check_interval=8,
            record_trajectory=record_trajectory,
        )
    return FullSemanticRolloutCollector(
        model,
        opponent,
        device=torch.device(config.device),
        worker_processes=config.worker_processes,
        engines_per_worker=config.engines_per_worker,
        inference_channels_per_role=config.inference_channels_per_role,
        mode=mode,
        coalesce_ms=config.coalesce_ms,
    )


def build_jobs(
    *,
    source_policy_update: int,
    seed: int,
    count: int = 512,
    greedy: bool = False,
) -> list[RolloutJob]:
    catalog = load_frozen_catalog()
    if count < 256 or count % 256:
        raise ValueError("job count must contain complete 256-slot frequency units")
    fixed_slots = [opponent for opponent in catalog for _ in range(opponent.games)]
    update_offset = 0 if greedy else source_policy_update * 1_000_003
    scenario_rng = random.Random(seed + update_offset)
    selected: list[Any] = []
    seats: list[bool] = []
    for _unit in range(count // 256):
        unit = list(fixed_slots)
        scenario_rng.shuffle(unit)
        selected.extend(unit)
        unit_seats = [True] * 128 + [False] * 128
        scenario_rng.shuffle(unit_seats)
        seats.extend(unit_seats)
    engine_seeds = scenario_rng.sample(range(1, 0x80000000), count)
    jobs: list[RolloutJob] = []
    deck = focal_deck()
    root = runtime_root()
    runtime = build_seeded_runtime()
    for game_index, (opponent, focal_first, engine_seed) in enumerate(
        zip(selected, seats, engine_seeds, strict=True)
    ):
        search_seed = (
            (engine_seed + 900_000_007 + update_offset) & 0x7FFFFFFF
        ) or 1
        jobs.append(
            RolloutJob(
                game_id=("eval" if greedy else "rollout")
                + f"-u{source_policy_update:04d}-{game_index:04d}",
                opponent_id=opponent.deck_id,
                focal_first=focal_first,
                seed=engine_seed,
                source_policy_update=source_policy_update,
                focal_deck=deck,
                opponent_deck=opponent.deck,
                runtime_root=root,
                policy_seed=(
                    (seed + 1_700_000_009 + update_offset + game_index)
                    & 0x7FFFFFFF
                )
                or 1,
                search_seed=search_seed,
                engine_library=runtime.library_path,
                full_round_draw_limit=DEFAULT_FULL_ROUND_DRAW_LIMIT,
                action_boundary_mode="enabled",
            )
        )
    for start in range(0, len(jobs), 256):
        unit_counts = {
            item.deck_id: sum(job.opponent_id == item.deck_id for job in jobs[start:start + 256])
            for item in catalog
        }
        if any(unit_counts[item.deck_id] != item.games for item in catalog):
            raise RuntimeError("formal rollout lost the 256-slot environment frequency")
    if len(jobs) != count:
        raise RuntimeError(f"schedule produced {len(jobs)} jobs instead of {count}")
    return jobs


def _episode_metrics(episodes: list[Any], prefix: str) -> dict[str, float]:
    invalid = [episode for episode in episodes if not episode.valid]
    if invalid:
        raise RuntimeError(
            f"official CPU engine errors ({len(invalid)}): "
            + repr([episode.error for episode in invalid[:5]])
        )
    games = len(episodes)
    wins = sum(episode.reward == 1.0 for episode in episodes)
    losses = sum(episode.reward == -1.0 for episode in episodes)
    draws = sum(episode.reward == 0.0 for episode in episodes)
    first = [episode for episode in episodes if episode.job.focal_first]
    second = [episode for episode in episodes if not episode.job.focal_first]
    pairs: dict[tuple[str, int], list[Any]] = {}
    for episode in episodes:
        pairs.setdefault((episode.job.opponent_id, episode.job.seed), []).append(episode)
    complete_pairs = [pair for pair in pairs.values() if len(pair) == 2]
    unique_seed_panel = all(len(pair) == 1 for pair in pairs.values())
    if not unique_seed_panel and (
        len(complete_pairs) * 2 != games or any(
            {episode.job.focal_first for episode in pair} != {True, False}
            for pair in complete_pairs
        )
    ):
        raise RuntimeError("0038 Episode results are neither unique-seed nor opposite-seat paired")
    pair_wins = [sum(episode.reward == 1.0 for episode in pair) for pair in complete_pairs]
    return {
        f"{prefix}/episodes": float(games),
        f"{prefix}/wins": float(wins),
        f"{prefix}/losses": float(losses),
        f"{prefix}/draws": float(draws),
        f"{prefix}/win_rate": wins / games,
        f"{prefix}/first_win_rate": sum(ep.reward == 1.0 for ep in first) / len(first),
        f"{prefix}/second_win_rate": sum(ep.reward == 1.0 for ep in second) / len(second),
        f"{prefix}/decisions": float(sum(
            int(ep.diagnostics.get("strategic_decisions", len(ep.decisions)))
            for ep in episodes
        )),
        f"{prefix}/policy_transitions": float(
            sum(len(ep.policy_transitions) for ep in episodes)
        ),
        f"{prefix}/trajectory_episodes": float(sum(
            bool(ep.policy_transitions) for ep in episodes
        )),
        f"{prefix}/engine_selections": float(
            sum(int(ep.diagnostics.get("engine_selections", 0)) for ep in episodes)
        ),
        f"{prefix}/mean_engine_turn": sum(ep.turns for ep in episodes) / games,
        f"{prefix}/error_games": 0.0,
        f"{prefix}/seed_pairs": float(len(complete_pairs)),
        f"{prefix}/pair_ww": float(sum(value == 2 for value in pair_wins)),
        f"{prefix}/pair_split": float(sum(value == 1 for value in pair_wins)),
        f"{prefix}/pair_ll": float(sum(value == 0 for value in pair_wins)),
    }


def _persist_frozen_results(path: Path, episodes: list[Any], *, checkpoint_update: int,
                            panel_version: str) -> dict[int, int]:
    rows = []
    outcomes = {}
    for episode in episodes:
        outcome = 1 if episode.reward == 1.0 else -1 if episode.reward == -1.0 else 0
        outcomes[episode.job.seed] = outcome
        rows.append({
            "game_id": episode.job.game_id, "seed": episode.job.seed,
            "opponent": episode.job.opponent_id, "focal_first": episode.job.focal_first,
            "outcome": outcome, "turns": episode.turns, "valid": episode.valid,
            "error": episode.error,
            "fallback": int(episode.diagnostics.get("macro_fallback", 0)),
        })
    if len(rows) != 2048 or len(outcomes) != len(rows):
        raise RuntimeError("Frozen result persistence requires 2,048 unique games")
    _atomic_json(path, {
        "schema_version": "0038_frozen_per_game_results_v1",
        "frozen_panel_version": panel_version,
        "checkpoint_update": checkpoint_update, "entries": rows,
    })
    return outcomes


def _paired_frozen_metrics(baseline: dict[int, int], checkpoint: dict[int, int],
                           prefix: str = "eval") -> dict[str, float]:
    summary = paired_summary(baseline, checkpoint)
    return {
        f"{prefix}/win_rate": float(summary["win_rate"]),
        f"{prefix}/wilson_low": float(summary["wilson_95"][0]),
        f"{prefix}/wilson_high": float(summary["wilson_95"][1]),
        f"{prefix}/baseline_loss_to_win": float(summary["baseline_loss_to_checkpoint_win"]),
        f"{prefix}/baseline_win_to_loss": float(summary["baseline_win_to_checkpoint_loss"]),
        f"{prefix}/mcnemar_chi2": float(summary["mcnemar_continuity_corrected_chi2"]),
    }


def _schedule_payload(jobs: list[RolloutJob], *, source_checkpoint_sha256: str) -> dict[str, Any]:
    rows = [
        {
            "game_id": job.game_id,
            "opponent_id": job.opponent_id,
            "focal_first": job.focal_first,
            "engine_seed": job.seed,
            "search_seed": job.search_seed,
            "policy_seed": job.policy_seed,
            "source_policy_update": job.source_policy_update,
            "full_round_draw_limit": job.full_round_draw_limit,
        }
        for job in jobs
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
    environment_sha256 = _environment_schedule_sha256(jobs)
    return {
        "schema": "0038_dynamic_rollout_schedule_v2",
        "episodes": len(rows),
        "seed_pairs": len({(row["opponent_id"], row["engine_seed"]) for row in rows}),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "schedule_sha256": hashlib.sha256(encoded).hexdigest(),
        "environment_sha256": environment_sha256,
        "jobs": rows,
    }


def _environment_schedule_sha256(jobs: list[RolloutJob]) -> str:
    rows = [{
        "opponent_id": job.opponent_id,
        "focal_first": job.focal_first,
        "engine_seed": job.seed,
        "search_seed": job.search_seed,
        "policy_seed": job.policy_seed,
        "full_round_draw_limit": job.full_round_draw_limit,
    } for job in jobs]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _opponent_snapshot() -> dict[str, Any]:
    catalog = load_frozen_catalog()
    return {
        "schema": "0038_frozen0806_pool_snapshot_v2",
        "opponent_count": len(catalog),
        "total_games": sum(item.games for item in catalog),
        "seat_contract": (
            "rollout_balanced_inside_each_randomized_256-slot unit; "
            "evaluation fixed by frozen_0806_seeded_2048_v2"
        ),
        "pool_id": "0806_kaggle_top100_plus_v1",
        "schedule_sha256": "16dbd18ce417405571c88997c9e97f9b2ec2adf96db544d1a9af988bb3c3cc3c",
        "policy": "0806_large_model_pretrained_immutable",
        "policy_sha256": _sha256(SOURCE_CHECKPOINT),
        "opponents": [
            {
                "deck_id": item.deck_id,
                "deck_sha256": item.deck_sha256,
                "games": item.games,
                "segment": item.segment,
            }
            for item in catalog
        ],
    }


def _trainable_manifest(model, trainer) -> dict[str, Any]:
    rows = [
        {"name": name, "shape": list(parameter.shape), "parameters": parameter.numel()}
        for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    groups = []
    for group in trainer.optimizer.param_groups:
        groups.append({
            "name": str(group.get("name", "unnamed")),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "tensor_count": len(group["params"]),
        })
    return {
        "trainable_parameters": sum(row["parameters"] for row in rows),
        "trainable_tensors": len(rows),
        "parameter_table": rows,
        "optimizer_groups": groups,
    }


def _run_parity(model, output: Path) -> dict[str, Any]:
    catalog = load_frozen_catalog()
    observations = collect_official_observations(
        runtime_root(),
        focal_deck(),
        catalog[0].deck,
        focal_index=0,
        decisions=8,
    )
    return assert_large_model_0806_runtime_parity(
        observations=observations,
        actor_index=0,
        deck=focal_deck(),
        checkpoint=SOURCE_CHECKPOINT,
        candidate_root=CANDIDATE_ROOT,
        model=model,
        output=output,
    )


def _git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return {"git_commit": commit, "git_dirty": dirty}


def _run_static_update0_contract(model, output: Path) -> dict[str, Any]:
    """Validate strict U0 identity without scheduling a new official CPU game."""
    assert_full_schema()
    model.assert_trainable_contract()
    report = {
        "schema": "0038_accelerated_u0_static_contract_v1",
        "passed": True,
        "official_cpu_engine_scheduled": False,
        "source_policy_checkpoint_sha256": _sha256(SOURCE_CHECKPOINT),
        "source_value_checkpoint": "0036_v2_epoch5_pre_rl",
        "common_update0_sha256": COMMON_UPDATE0_SHA256,
        "representation_sha256": model.representation_sha256(),
        "actor_schema": "0031_rule_faithful_semantic_decision_v2",
        "followup_gate": "immutable_283_decision_cuda_training_package_parity",
    }
    _atomic_json(output, report)
    return report


def _assert_acceptance_episode_health(
    episodes: list[Any], collector_metrics: dict[str, float], *, scope: str
) -> dict[str, float]:
    fallbacks = sum(
        int(episode.diagnostics.get("macro_fallback", 0)) for episode in episodes
    )
    invalid = int(collector_metrics.get("rollout/invalid_macros", 0.0))
    unsupported = sum(
        int(episode.diagnostics.get("unsupported_effect", 0)) for episode in episodes
    )
    pending_resets = sum(
        int(episode.diagnostics.get("pending_macro_reset", 0)) for episode in episodes
    )
    if any((fallbacks, invalid, unsupported, pending_resets)):
        raise RuntimeError(
            f"{scope} action-contract health failed: fallback={fallbacks}, "
            f"invalid={invalid}, unsupported={unsupported}, pending_reset={pending_resets}"
        )
    return {
        f"{scope}/fallback": 0.0,
        f"{scope}/unsupported": 0.0,
        f"{scope}/pending_macro_reset": 0.0,
    }


CHANCE_BOUNDARY_FALLBACK = "chance_boundary_before_allocation"


def _chance_boundary_replacement_job(job: RolloutJob, retry: int) -> RolloutJob:
    if retry < 1:
        raise ValueError("chance-boundary retry must be positive")
    digest = hashlib.sha256(
        f"{job.game_id}:{job.seed}:{job.policy_seed}:{job.search_seed}:"
        f"chance-boundary-retry:{retry}".encode("ascii")
    ).digest()

    def seeded(offset: int) -> int:
        return int.from_bytes(digest[offset : offset + 8], "big") % 0x7FFFFFFF or 1

    return replace(
        job,
        game_id=f"{job.game_id}-chance-retry-{retry:02d}",
        seed=seeded(0),
        policy_seed=seeded(8),
        search_seed=seeded(16),
    )


def _replace_chance_boundary_episodes(
    episodes: list[Any],
    collect_replacements: Any,
    *,
    max_retries: int = 8,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Return the same environment slots with chance-boundary episodes resampled.

    Confusion reveals a real random result before Phantom Dive allocation.  The
    legacy fallback trace is retained for diagnostics but cannot enter compound
    PPO because it contains six primitive policy callbacks.  Only this explicit
    chance boundary is replaceable; every protocol drift remains fail closed.
    """

    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    accepted = list(episodes)
    originals = [episode.job for episode in episodes]
    retry_count = [0] * len(episodes)
    evidence: list[dict[str, Any]] = []
    while True:
        positions: list[int] = []
        replacement_jobs: list[RolloutJob] = []
        for position, episode in enumerate(accepted):
            if not episode.diagnostics.get("macro_fallback"):
                continue
            reason = str(episode.diagnostics.get("macro_fallback_reason") or "")
            if reason != CHANCE_BOUNDARY_FALLBACK:
                raise RuntimeError(
                    f"non-resampleable macro failure in {episode.job.game_id}: {reason}"
                )
            retry_count[position] += 1
            if retry_count[position] > max_retries:
                raise RuntimeError(
                    f"chance-boundary replacement exhausted for "
                    f"{originals[position].game_id}"
                )
            replacement = _chance_boundary_replacement_job(
                originals[position], retry_count[position]
            )
            evidence.append({
                "source_policy_update": originals[position].source_policy_update,
                "slot_game_id": originals[position].game_id,
                "opponent_id": originals[position].opponent_id,
                "focal_first": originals[position].focal_first,
                "excluded_engine_seed": episode.job.seed,
                "excluded_policy_seed": episode.job.policy_seed,
                "excluded_search_seed": episode.job.search_seed,
                "replacement_game_id": replacement.game_id,
                "replacement_engine_seed": replacement.seed,
                "replacement_policy_seed": replacement.policy_seed,
                "replacement_search_seed": replacement.search_seed,
                "retry": retry_count[position],
                "reason": reason,
            })
            positions.append(position)
            replacement_jobs.append(replacement)
        if not positions:
            return accepted, evidence
        replacements = list(collect_replacements(replacement_jobs))
        if len(replacements) != len(positions):
            raise RuntimeError("chance-boundary replacement collector changed cardinality")
        for position, replacement in zip(positions, replacements, strict=True):
            accepted[position] = replacement


def _attested_package_parity_passed(report: dict[str, Any]) -> bool:
    deployment = report.get("training_to_package") or {}
    root = deployment.get("root") or {}
    return bool(
        report.get("full_cpu_causalknowledge_parity")
        and deployment.get("passed")
        and int(root.get("greedy_action_divergences", -1)) == 0
    )


def _run_attested_update0_package_parity(
    *, version: str, checkpoint: Path, artifact: Path
) -> dict[str, Any]:
    temporary_root = (
        ROOT / ".tmp/evaluation/0038_accelerated_transfer" / version
    )
    package = temporary_root / "u0_package"
    report_path = artifact / "update0_cuda_package_fixed_snapshot_parity.json"
    if package.exists():
        raise FileExistsError(f"attested U0 package path already exists: {package}")
    manifest = export_candidate(
        source=CANDIDATE_ROOT, checkpoint=checkpoint, output=package
    )
    trace = (
        ROOT
        / ".tmp/evaluation/0038_semantic_parity_audit/gate_c_final/"
        "fixed_283_primitive_trace.jsonl"
    )
    trace_manifest = (
        ROOT
        / ".tmp/evaluation/0038_semantic_parity_audit/gate_c/semantic_trace_manifest.json"
    )
    if not trace.is_file() or not trace_manifest.is_file():
        raise FileNotFoundError("immutable repaired-schema Gate C trace is unavailable")
    if (
        _sha256(trace) != IMMUTABLE_GATE_C_TRACE_SHA256
        or sum(bool(line.strip()) for line in trace.read_text(encoding="utf-8").splitlines()) != 283
    ):
        raise RuntimeError("immutable 283-decision Gate C trace identity mismatch")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "engine_cuda/tools/run_official_semantic0031_v2_parity_scaffold.py"),
            "--manifest", str(trace_manifest),
            "--package", str(package),
            "--training-checkpoint", str(checkpoint),
            "--extension-dir", str(CUDA_EXTENSION),
            "--reuse-trace", "--trace", str(trace),
            "--compare-decisions", "283",
            "--require-history-wrap", "--strict",
            "--output", str(report_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not _attested_package_parity_passed(report):
        raise RuntimeError("attested U0 CUDA/package fixed-snapshot parity failed")
    _atomic_json(
        artifact / "update0_package_manifest.json",
        {
            **manifest,
            "diagnostic_only": True,
            "official_cpu_engine_scheduled": False,
            "fixed_snapshot_trace_sha256": _sha256(trace),
        },
    )
    return report


def load_frozen_opponent(device: torch.device):
    policy = PortableSemanticPolicy.from_checkpoint(SOURCE_CHECKPOINT, focal_deck())
    actor = policy.model.to(device).eval()
    actor.requires_grad_(False)
    return actor


def run_gate(
    *,
    output: Path,
    games: int,
    worker_processes: int,
    engines_per_worker: int = 1,
    inference_channels_per_role: int = 1,
    run_ppo: bool,
    device_name: str = "cuda:0",
    mode: str = "sample",
    coalesce_ms: float = 5.0,
    gae_lambda: float = 0.95,
    credit_clock: str = "turn",
    loss_weighting: str = "episode_equal_decisions",
    adaptation_arm: str = "lora",
    preset_name: str = "BASE",
) -> dict[str, Any]:
    device = torch.device(device_name)
    adaptation = AdaptationConfig(
        lora=True, layernorm_tuning=adaptation_arm == "lora_layernorm"
    )
    flags = preset(preset_name)
    model, identity = build_preset_from_common_update0(focal_deck(), flags, device=device)
    opponent = load_frozen_opponent(device)
    parity = _run_parity(model, output.parent / "large_model_0806_runtime_parity.json")
    representation = model.representation_sha256()
    decoder_before = model.decoder_sha256()
    collector = FullSemanticRolloutCollector(
        model,
        opponent,
        device=device,
        worker_processes=worker_processes,
        engines_per_worker=engines_per_worker,
        inference_channels_per_role=inference_channels_per_role,
        mode=mode,
        coalesce_ms=coalesce_ms,
    )
    started = time.perf_counter()
    episodes = collector.collect(
        build_jobs(source_policy_update=0, seed=330033001, count=games)
    )
    rollout_seconds = time.perf_counter() - started
    metrics = _episode_metrics(episodes, "rollout")
    metrics.update(collector.metrics())
    ppo_metrics: dict[str, float] = {}
    ppo_seconds = 0.0
    if run_ppo:
        ppo_config = PPOConfig(
            gae_lambda=gae_lambda,
            credit_clock=credit_clock,
            loss_weighting=loss_weighting,
            optimization_mode="fixed_epochs",
            epochs=1,
        )
        batch = prepare_episodes(
            episodes,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            credit_clock=ppo_config.credit_clock,
            loss_weighting=ppo_config.loss_weighting,
            prize_mode=flags.prize_aux_mode,
            prize_scale=flags.prize_aux_scale,
        )
        trainer = PPOTrainer(model, device=device, config=ppo_config)
        ppo_metrics.update(trainer.sparse_gradient_diagnostics(batch))
        ppo_started = time.perf_counter()
        ppo_metrics.update(trainer.update(batch))
        ppo_seconds = time.perf_counter() - ppo_started
    report = {
        "schema": "0038_action_boundary_official_cpu_gate_v1",
        "passed": True,
        "games": games,
        "paired_seed_scenarios": games // 2,
        "paired_seats": games % 2 == 0,
        "ppo_update": run_ppo,
        "credit_clock": credit_clock,
        "gae_lambda": gae_lambda,
        "loss_weighting": loss_weighting,
        "adaptation": asdict(adaptation),
        "integrated_flags": flags.metadata(),
        "adaptation_inventory": model.adaptation_inventory,
        "rollout_seconds": rollout_seconds,
        "rollout_topology": {
            "worker_processes": worker_processes,
            "engines_per_worker": engines_per_worker,
            "inference_channels_per_role": inference_channels_per_role,
        },
        "ppo_seconds": ppo_seconds,
        "source_identity": asdict(identity),
        "parity": parity,
        "representation_sha256": representation,
        "representation_unchanged": model.representation_sha256() == representation,
        "decoder_sha256_before": decoder_before,
        "decoder_sha256_after": model.decoder_sha256(),
        "decoder_changed": model.decoder_sha256() != decoder_before,
        "metrics": {**metrics, **ppo_metrics},
        "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    if run_ppo and not report["decoder_changed"]:
        raise RuntimeError("PPO gate did not change the original Large Model 0806 decoder")
    if not report["representation_unchanged"]:
        raise RuntimeError("PPO gate changed the original Large Model 0806 representation")
    _atomic_json(output, report)
    return report


def run(config: RunConfig) -> dict[str, Any]:
    if not config.launch_formal:
        raise RuntimeError(
            "0038 formal PPO requires explicit user approval via the "
            "--launch-formal token; "
            "tests and imports must remain side-effect free"
        )
    config.validate()
    flags = replace(
        preset(config.preset_name),
        enable_local_output_layernorm=config.adaptation_arm == "lora_layernorm",
    )
    flags.validate()
    adaptation = AdaptationConfig(
        lora=True,
        layernorm_tuning=config.adaptation_arm == "lora_layernorm",
    )
    resuming = config.resume_update0
    paths = assert_update0_resume(config.version) if resuming else assert_fresh_version(config.version)
    if not resuming:
        for name in ("artifact", "checkpoint", "tensorboard", "wandb"):
            paths[name].mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference/training requested but unavailable")
    if config.engine_backend == "accelerated:cuda_resident":
        if not CUDA_RULES.is_file() or not (CUDA_EXTENSION / "_ptcg_cuda.so").is_file():
            raise RuntimeError("validated CUDA rules/extension artifacts are unavailable")
    source_identity = _git_identity()
    if config.accelerated_transfer_acceptance and source_identity["git_dirty"]:
        raise RuntimeError("accelerated transfer acceptance requires a clean source commit")
    run_id = "0038-" + config.version.lower().replace("_", "-")
    wandb_url = f"https://wandb.ai/dragon_bra/pokemon-tcg-policy-learning/runs/{run_id}"
    os.environ.update(
        {
            "WANDB_MODE": config.wandb_mode,
            "WANDB_ENTITY": "dragon_bra",
            "WANDB_PROJECT": "pokemon-tcg-policy-learning",
            "WANDB_JOB_TYPE": "ppo_decoder_value_encoder_adapter",
            "WANDB_RUN_ID": run_id,
            "WANDB_NAME": f"{WANDB_DISPLAY_PREFIX} · {config.version}",
            "WANDB_RUN_GROUP": PROJECT,
            "WANDB_TAGS": (
                "0038,action_boundary,cuda_resident,full_stack,ppo,"
                "frozen_0806_seeded_2048_v2,"
                + ("accelerated_transfer_acceptance," if config.accelerated_transfer_acceptance else "")
                + config.ppo.credit_clock
                + "_clock"
            ),
            "WANDB_DIR": str(paths["wandb"]),
        }
    )
    if config.wandb_mode == "online":
        import wandb
        if not getattr(wandb.Api(), "api_key", None):
            raise RuntimeError("W&B online authentication is unavailable")
    config_payload = {
        "schema": "0038_action_boundary_scalable_ppo_config_v2",
        "project_id": PROJECT,
        **asdict(config),
        "initialization_checkpoint": str(COMMON_UPDATE0_CHECKPOINT.relative_to(ROOT)),
        "initialization_checkpoint_sha256": COMMON_UPDATE0_SHA256,
        "base_source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "base_source_checkpoint_sha256": _sha256(SOURCE_CHECKPOINT),
        "integrated_flags": flags.metadata(),
        **source_identity,
        "cuda_rules_sha256": _sha256(CUDA_RULES),
        "cuda_extension_sha256": _sha256(CUDA_EXTENSION / "_ptcg_cuda.so"),
        "feature_preprocessing_version": FEATURE_PREPROCESSING_VERSION,
        "action_schema_version": ACTION_BOUNDARY_SCHEMA_VERSION,
        "decision_gate_version": DECISION_GATE_VERSION,
        "canonicalizer_version": CANONICALIZER_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "official_protocol_adapter_version": OFFICIAL_PROTOCOL_ADAPTER_VERSION,
        "cpu_evaluation_contract": (
            "not_scheduled_wait_for_explicit_user_checkpoint_selection"
            if config.accelerated_transfer_acceptance else "unchanged"
        ),
        "actor_schema": "0031_rule_faithful_semantic_decision_v2",
        "actor": "exact_0031_semantic_policy_no_reduction",
        "focal_deck_id": FOCAL_DECK_ID,
        "focal_exact_deck_sha256": FOCAL_EXACT_DECK_SHA256,
        "ppo_setting_source": {
            "project": "0023_mega_lopunny_ex_mega_froslass_ex_002_league_training",
            "version": "V2_mega_lopunny_ex_mega_froslass_ex_002_continuous_league",
            "config": "rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/V2_mega_lopunny_ex_mega_froslass_ex_002_continuous_league/artifact/training_config.json",
            "intentional_difference": (
                f"rollout size is configured as {config.games_per_update} episodes; "
                "optimization mode is explicit and does not scale silently"
            ),
        },
        "opponent_count": 55,
        "opponent_games_per_batch": config.games_per_update,
        "opponent_policy": "0806_large_model_pretrained_immutable",
        "opponent_policy_sha256": _sha256(SOURCE_CHECKPOINT),
        "opponent_schedule_sha256": "16dbd18ce417405571c88997c9e97f9b2ec2adf96db544d1a9af988bb3c3cc3c",
        "official_engine": "seeded_official_engine_abi_v1_runtime_0002",
        "engine_backend": {
            "name": config.engine_backend,
            "cuda_rules": str(CUDA_RULES.relative_to(ROOT)),
            "cuda_rules_sha256": _sha256(CUDA_RULES)
            if config.engine_backend == "accelerated:cuda_resident" else None,
            "cuda_extension": str((CUDA_EXTENSION / "_ptcg_cuda.so").relative_to(ROOT)),
            "cuda_extension_sha256": _sha256(CUDA_EXTENSION / "_ptcg_cuda.so")
            if config.engine_backend == "accelerated:cuda_resident" else None,
            "cuda_resident_source": str(CUDA_RESIDENT_SOURCE.relative_to(ROOT)),
            "cuda_resident_source_sha256": _sha256(CUDA_RESIDENT_SOURCE)
            if config.engine_backend == "accelerated:cuda_resident" else None,
            "cuda_progress_guard_source": str(
                CUDA_PROGRESS_GUARD_SOURCE.relative_to(ROOT)
            ),
            "cuda_progress_guard_source_sha256": _sha256(
                CUDA_PROGRESS_GUARD_SOURCE
            ) if config.engine_backend == "accelerated:cuda_resident" else None,
            "lane_count": config.cuda_lane_count,
            "trajectory_chunk_games": config.rollout_batch_size,
        },
        "rollout_seed_contract": {
            "frequency_unit_games": 256,
            "frequency_units_per_update": config.games_per_update // 256,
            "unique_engine_seeds": config.games_per_update,
            "exact_environment_slot_counts_per_unit": True,
            "seat_balanced_per_unit": True,
            "trajectory_games_per_update": config.trajectory_games_per_update,
            "trajectory_sampling": (
                "all configured environment slots; a real pre-allocation chance "
                "boundary is retained as diagnostic evidence and replaced with a "
                "new auditable seed in the same opponent/seat slot"
                if config.trajectory_games_per_update == config.games_per_update
                else "uniform deterministic episodes; stratified across every 256-slot unit"
            ),
            "chance_boundary_replacement": {
                "reason_allowlist": [CHANCE_BOUNDARY_FALLBACK],
                "same_opponent_and_seat": True,
                "max_retries_per_slot": 8,
                "excluded_legacy_callbacks_enter_ppo": False,
                "provenance": "artifact/schedules/chance_boundary_replacements/",
            },
        },
        "frozen_evaluation_contract": {
            "contract_id": CANONICAL_CONTRACT_ID,
            "evaluation_seed": 341512806,
            "games": 2048,
            "schedule_sha256": EXPECTED_007_SCHEDULE_SHA256,
            "frequency_unit_games": 256,
            "replicas": 8,
            "reference_report": (
                "evaluation/arena/combat_mat/policy_0806/"
                "0806_kaggle_top100_plus_v1_cuda_seeded_2048_resident_v2/"
                "reports/007_dragapult_ex.html"
            ),
        },
        "rollout_topology": {
            "worker_processes": config.worker_processes,
            "engines_per_worker": config.engines_per_worker,
            "inference_channels_per_role": config.inference_channels_per_role,
            "worker_local_compiler": True,
            "worker_compiler_backend": "policy_stateless",
            "prototype_gpu_cache": True,
        },
        "trainable_contract": [
            "actor.action_decoder.*",
            "actor.option_encoder.cross_attention_transformer.layers.1.{self_attn,multihead_attn}.in_proj Q/V LoRA",
            "actor.option_encoder.cross_attention_transformer.norm.{weight,bias} (setting 2 only)",
            "value_head.queries",
            "value_head.blocks.*",
            "value_head.final_norm.*",
            "value_head.heads.value.*",
        ],
        "critic_initialization": {
            "project": "0036_dedicated_action_value_network",
            "version": "V2_latent_value_archetype_diff",
            "epoch": 5,
            "checkpoint_sha256": "e88b2f18089911c4ebd810fa3abfba800a103189184b6265b9d38c03a9e36360",
            "output": "2*sigmoid(value_logit)-1",
            "on_policy_recalibration": True,
            "auxiliary_heads_in_ppo": False,
        },
        "checkpoint_retention": "all",
        "run_duration": {
            "mode": "manual_stop" if config.updates is None else "finite_updates",
            "max_updates": config.updates,
            "stop_sentinel": "artifact/STOP_REQUESTED",
            "stop_check_boundary": "after_each_complete_update",
        },
        "adaptation": asdict(adaptation),
        "value_diagnostics": {
            "source": "on_policy_rollout_old_value",
            "absolute_turn_bins": ["00_03", "04_07", "08_11", "12_plus"],
            "remaining_turn_bins": ["00_01", "02_03", "04_plus"],
            "outcomes": ["focal_win", "focal_loss", "draw"],
            "targets": ["gae_return", "terminal_outcome"],
        },
        "reward": {
            "v_win_target": "terminal_only_minus_one_zero_plus_one",
            "prize_mode": flags.prize_aux_mode,
            "net_prize_delta": "own_prizes_taken-opponent_prizes_taken",
            "prize_scale": flags.prize_aux_scale,
            "prize_actor_weight": flags.prize_aux_actor_weight,
            "double_scaling": False,
        },
        "temporal_credit": {
            "gamma": config.ppo.gamma,
            "gae_lambda": config.ppo.gae_lambda,
            "clock": config.ppo.credit_clock,
            "same_turn_lambda": 1.0 if config.ppo.credit_clock == "turn" else config.ppo.gae_lambda,
            "boundary_lambda": config.ppo.gae_lambda,
        },
        "episode_weighting": config.ppo.loss_weighting,
        "termination_contract": {
            "ability_repeat_limit": 20,
            "ability_repeat_action": "opponent_win",
            "ability_repeat_key_version": "turn_scoped_stable_semantic_identity_v3",
            "ability_repeat_key_fields": [
                "actor", "selection_kind", "option_type", "source_card",
                "context_card", "effect_card", "source_serial",
            ],
            "ability_repeat_key_excludes": ["option_index", "target_order"],
            "ability_repeat_counter_reset": "official_turn_change",
            "full_round_draw_limit": DEFAULT_FULL_ROUND_DRAW_LIMIT,
            "engine_turn_limit": 2 * DEFAULT_FULL_ROUND_DRAW_LIMIT - 1,
            "turn_limit_result": "terminal_draw_reward_zero",
            "ppo_terminal_semantics": "done_true_next_value_zero",
            "official_rule": False,
            "owner": "rollout_scheduler_safety_contract",
        },
    }
    if not resuming:
        _atomic_json(paths["artifact"] / "training_config.json", config_payload)
        _atomic_json(paths["artifact"] / "opponent_snapshot.json", _opponent_snapshot())
        _atomic_json(
            paths["artifact"] / "status.json",
            {
                "state": "initializing",
                "version": config.version,
                "wandb_run_id": run_id,
                "wandb_url": wandb_url,
                "wandb_sync_status": "initializing",
            },
        )
    else:
        previous_config_path = paths["artifact"] / "training_config.json"
        interrupted_config_path = paths["artifact"] / "interrupted_update0_config.json"
        if not interrupted_config_path.exists():
            _atomic_json(interrupted_config_path, json.loads(previous_config_path.read_text()))
        _atomic_json(previous_config_path, config_payload)
        _merge_status(paths["artifact"] / "status.json", {
            "state": "resuming_update0",
            "resume_reason": "interrupted test-triggered baseline before PPO update 1",
            "resume_time": time.time(),
            "run_duration_mode": "manual_stop" if config.updates is None else "finite_updates",
            "max_updates": config.updates,
            "interrupted_config_preserved": "artifact/interrupted_update0_config.json",
        })
    started = time.time()
    update = 0
    completed_update = 0
    cumulative_episodes = 0
    cumulative_decisions = 0
    try:
        model, identity = build_preset_from_common_update0(
            focal_deck(), flags, device=device
        )
        opponent = load_frozen_opponent(device)
        parity = (
            _run_static_update0_contract(
                model, paths["artifact"] / "large_model_0806_runtime_parity.json"
            )
            if config.accelerated_transfer_acceptance
            else _run_parity(
                model, paths["artifact"] / "large_model_0806_runtime_parity.json"
            )
        )
        if not parity["passed"]:
            raise RuntimeError("Large Model 0806 runtime parity did not pass")
        trainer = PPOTrainer(model, device=device, config=config.ppo)
        transfer_controller = (
            AcceleratedTransferController(config.accelerated_transfer)
            if config.accelerated_transfer_acceptance else None
        )
        trainable_manifest = _trainable_manifest(model, trainer)
        if transfer_controller is not None:
            base_rates = trainer.base_learning_rates
            trainable_manifest.update({
                "accelerated_transfer": transfer_controller.metadata(),
                "optimizer_group_contract": trainer.optimizer_group_manifest(),
                "planned_update1_learning_rates": transfer_controller.learning_rates(
                    base_rates, update=1
                ),
                "planned_update3_learning_rates": transfer_controller.learning_rates(
                    base_rates, update=3
                ),
                "planned_update6_learning_rates": transfer_controller.learning_rates(
                    base_rates, update=6
                ),
            })
        _atomic_json(
            paths["artifact"] / "trainable_parameters.json",
            trainable_manifest,
        )
        representation = model.representation_sha256()
        initial_decoder = model.decoder_sha256()
        if not resuming:
            save_model_only(
                model,
                paths["checkpoint"] / "update-000000.pt",
                update=0,
                metadata={
                    "project": PROJECT,
                    "version": config.version,
                    "source_checkpoint_sha256": identity.checkpoint_sha256,
                    "parent_update0_sha256": COMMON_UPDATE0_SHA256,
                    "integrated_flags": flags.metadata(),
                    "representation_sha256": representation,
                    **source_identity,
                    "feature_preprocessing_version": FEATURE_PREPROCESSING_VERSION,
                    "action_schema_version": ACTION_BOUNDARY_SCHEMA_VERSION,
                    "decision_gate_version": DECISION_GATE_VERSION,
                    "canonicalizer_version": CANONICALIZER_VERSION,
                    "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                    "official_protocol_adapter_version": OFFICIAL_PROTOCOL_ADAPTER_VERSION,
                    "initialization_contract": "zero_shot_pretrain_common_update0_no_rl_state",
                    "lora_initialization": "zero_delta",
                },
            )
        _merge_status(
            paths["artifact"] / "status.json",
            {
                "state": "running",
                "pid": os.getpid(),
                "checkpoint_update": 0,
                "rollout_source_policy_update": 0,
                "representation_sha256": representation,
                "initial_decoder_sha256": initial_decoder,
                "full_schema_parity": True,
                "wandb_sync_status": "online_active" if config.wandb_mode == "online" else "offline_staging",
            },
        )
        with TrainingLogger(
            paths["artifact"] / "training_metrics.jsonl", paths["tensorboard"]
        ) as logger:
            logger.initialize_wandb(
                {
                    "trainer/update": 0,
                    "checkpoint/update": 0,
                    "rollout/source_policy_update": 0,
                }
            )
            baseline_evaluator = build_collector(model, opponent, config, mode="greedy")
            baseline_started = time.perf_counter()
            baseline_jobs, baseline_schedule_sha = build_frozen_jobs(
                focal_deck_id=FOCAL_DECK_ID,
                focal_deck=focal_deck(), runtime_root=runtime_root(),
                source_policy_update=0,
            )
            if baseline_schedule_sha != EXPECTED_007_SCHEDULE_SHA256:
                raise RuntimeError("update-0 Frozen schedule is not canonical 007")
            _atomic_json(
                paths["artifact"] / "schedules/eval_frozen_2048.json",
                {
                    **_schedule_payload(
                        baseline_jobs,
                        source_checkpoint_sha256=identity.checkpoint_sha256,
                    ),
                    "contract_id": CANONICAL_CONTRACT_ID,
                    "canonical_schedule_sha256": baseline_schedule_sha,
                },
            )
            baseline = baseline_evaluator.collect(baseline_jobs)
            baseline_collector_metrics = baseline_evaluator.metrics()
            baseline_health_metrics = (
                _assert_acceptance_episode_health(
                    baseline, baseline_collector_metrics, scope="eval/core"
                )
                if transfer_controller is not None else {}
            )
            baseline_core_outcomes = _persist_frozen_results(
                paths["artifact"] / "frozen_results/core-update-000000.json",
                baseline, checkpoint_update=0, panel_version=CANONICAL_CONTRACT_ID,
            )
            update0_package_parity = None
            if transfer_controller is not None:
                update0_package_parity = _run_attested_update0_package_parity(
                    version=config.version,
                    checkpoint=paths["checkpoint"] / "update-000000.pt",
                    artifact=paths["artifact"],
                )
            diagnostic_evaluator = build_collector(
                model, opponent, config, mode="greedy", record_trajectory=True
            )
            diagnostic_episodes = diagnostic_evaluator.collect(baseline_jobs[:2])
            diagnostic_batch = prepare_episodes(
                diagnostic_episodes,
                gamma=config.ppo.gamma,
                gae_lambda=config.ppo.gae_lambda,
                credit_clock=config.ppo.credit_clock,
                loss_weighting=config.ppo.loss_weighting,
                prize_mode=flags.prize_aux_mode,
                prize_scale=flags.prize_aux_scale,
            )
            logger.log(
                0,
                {
                    "trainer/update": 0,
                    "checkpoint/update": 0,
                    "eval/checkpoint_update": 0,
                    "eval/wall_seconds": time.perf_counter() - baseline_started,
                    "representation/sha256_unchanged": 1.0,
                    **trainer.sparse_gradient_diagnostics(diagnostic_batch),
                    **_episode_metrics(baseline, "eval/core"),
                    **baseline_collector_metrics,
                    **baseline_health_metrics,
                    "parity/update0_cuda_package_passed": float(
                        update0_package_parity is None
                        or update0_package_parity["full_cpu_causalknowledge_parity"]
                    ),
                },
            )
            baseline_win_rate = sum(
                value == 1 for value in baseline_core_outcomes.values()
            ) / len(baseline_core_outcomes)
            best_frozen_update = 0
            best_frozen_win_rate = baseline_win_rate
            frozen_below_best_streak = 0
            _atomic_json(paths["artifact"] / "champion.json", {
                "schema": "0038_cuda_frozen_champion_v1",
                "checkpoint_update": 0,
                "win_rate": baseline_win_rate,
                "checkpoint": "checkpoint/update-000000.pt",
                "selection_runtime": "accelerated:cuda_resident",
                "cpu_evaluation_run": False,
            })
            update_iterator = (
                itertools.count(1)
                if config.updates is None
                else range(1, config.updates + 1)
            )
            stop_requested = False
            for update in update_iterator:
                source_update = update - 1
                collector = build_collector(model, opponent, config, mode="sample")
                rollout_started = time.perf_counter()
                rollout_jobs = build_jobs(
                    source_policy_update=source_update,
                    seed=config.seed,
                    count=config.games_per_update,
                )
                _record_training_schedule(
                    paths["artifact"] / "schedules/train_schedules.json",
                    _schedule_payload(
                        rollout_jobs,
                        source_checkpoint_sha256=identity.checkpoint_sha256,
                    ),
                )
                episodes = collector.collect(rollout_jobs)
                collector_metrics = collector.metrics()
                replacement_metric_rows: list[dict[str, float]] = []

                def collect_chance_replacements(
                    replacement_jobs: list[RolloutJob],
                ) -> list[Any]:
                    replacement_collector = build_collector(
                        model, opponent, config, mode="sample"
                    )
                    replacement_episodes = replacement_collector.collect(
                        replacement_jobs
                    )
                    replacement_metric_rows.append(replacement_collector.metrics())
                    return replacement_episodes

                episodes, chance_exclusions = _replace_chance_boundary_episodes(
                    episodes, collect_chance_replacements
                )
                rollout_seconds = time.perf_counter() - rollout_started
                additive_metrics = {
                    "rollout/inference_batches", "rollout/inference_requests",
                    "rollout/focal_requests", "rollout/inference_seconds",
                    "rollout/cuda_hot_loop_seconds", "rollout/cuda_materialize_seconds",
                    "rollout/cuda_refill_events", "rollout/cuda_repeat_forfeits",
                    "rollout/cuda_turn_limit_draws", "rollout/cuda_staged_trajectory_bytes",
                    "rollout/cuda_chunks",
                }
                maximum_metrics = {
                    "rollout/max_batch_size", "rollout/cuda_lane_count",
                    "rollout/cuda_peak_allocated_bytes", "rollout/cuda_peak_reserved_bytes",
                }
                for row in replacement_metric_rows:
                    for name in additive_metrics:
                        collector_metrics[name] = (
                            collector_metrics.get(name, 0.0) + row.get(name, 0.0)
                        )
                    for name in maximum_metrics:
                        collector_metrics[name] = max(
                            collector_metrics.get(name, 0.0), row.get(name, 0.0)
                        )
                collector_metrics.update({
                    "rollout/focal_requests": float(sum(
                        int(episode.diagnostics.get("strategic_decisions", 0))
                        for episode in episodes
                    )),
                    "rollout/forced_shortcuts": float(sum(
                        int(episode.diagnostics.get("forced_shortcuts", 0))
                        for episode in episodes
                    )),
                    "rollout/macro_actions": float(sum(
                        int(episode.diagnostics.get("macro_actions", 0))
                        for episode in episodes
                    )),
                    "rollout/macro_callbacks": float(sum(
                        int(episode.diagnostics.get("macro_callbacks", 0))
                        for episode in episodes
                    )),
                    "rollout/invalid_macros": 0.0,
                    "rollout/chance_boundary_exclusions": float(len(chance_exclusions)),
                    "rollout/attempted_games": float(
                        len(episodes) + len(chance_exclusions)
                    ),
                    "rollout/cuda_trajectory_games": float(sum(
                        bool(episode.policy_transitions) for episode in episodes
                    )),
                    "rollout/cuda_stored_policy_transitions": float(sum(
                        len(episode.policy_transitions) for episode in episodes
                    )),
                    "rollout/cuda_games_per_second": len(episodes)
                    / max(rollout_seconds, 1e-9),
                    "rollout/cuda_attempted_games_per_second": (
                        len(episodes) + len(chance_exclusions)
                    ) / max(rollout_seconds, 1e-9),
                    "rollout/strategic_decisions_per_second": float(sum(
                        int(episode.diagnostics.get("strategic_decisions", 0))
                        for episode in episodes
                    )) / max(rollout_seconds, 1e-9),
                })
                if chance_exclusions:
                    _atomic_json(
                        paths["artifact"]
                        / "schedules/chance_boundary_replacements"
                        / f"source-update-{source_update:06d}.json",
                        {
                            "schema": "0038_chance_boundary_replacements_v1",
                            "source_policy_update": source_update,
                            "accepted_episodes": len(episodes),
                            "attempted_episodes": len(episodes) + len(chance_exclusions),
                            "entries": chance_exclusions,
                        },
                    )
                rollout_metrics = _episode_metrics(episodes, "rollout")
                rollout_metrics.update(collector_metrics)
                if transfer_controller is not None:
                    rollout_metrics.update(_assert_acceptance_episode_health(
                        episodes, collector_metrics, scope="rollout"
                    ))
                rollout_metrics["rollout/wall_seconds"] = rollout_seconds
                cumulative_episodes += len(episodes)
                cumulative_decisions += int(rollout_metrics["rollout/decisions"])
                batch = prepare_episodes(
                    episodes,
                    gamma=config.ppo.gamma,
                    gae_lambda=config.ppo.gae_lambda,
                    credit_clock=config.ppo.credit_clock,
                    loss_weighting=config.ppo.loss_weighting,
                    prize_mode=flags.prize_aux_mode,
                    prize_scale=flags.prize_aux_scale,
                )
                ppo_started = time.perf_counter()
                ppo_metrics = {}
                if is_sparse_diagnostic_update(update):
                    ppo_metrics.update(trainer.sparse_gradient_diagnostics(diagnostic_batch))
                rollback_retries = 0
                applied_actor_multiplier = 1.0
                if transfer_controller is None:
                    ppo_metrics.update(trainer.update(batch, update=update))
                else:
                    trainable_before = trainer.snapshot_trainable_state()
                    cpu_rng_before = torch.get_rng_state()
                    cuda_rng_before = torch.cuda.get_rng_state_all()
                    while True:
                        applied_actor_multiplier = transfer_controller.actor_multiplier(update)
                        trainer.set_group_learning_rates(
                            transfer_controller.learning_rates(
                                trainer.base_learning_rates, update=update
                            )
                        )
                        attempt_metrics = trainer.update(batch, update=update)
                        health = transfer_controller.health(attempt_metrics)
                        if health.rollback:
                            trainer.restore_trainable_state(
                                trainable_before, reset_optimizer=True
                            )
                            torch.set_rng_state(cpu_rng_before)
                            torch.cuda.set_rng_state_all(cuda_rng_before)
                            rollback_retries += 1
                            transfer_controller.reduce_actor_cap(
                                current_multiplier=applied_actor_multiplier
                            )
                            if rollback_retries > 2:
                                raise RuntimeError(
                                    "accelerated PPO exceeded rollback retry budget"
                                )
                            continue
                        ppo_metrics.update(attempt_metrics)
                        if (
                            health.reduce_actor_lr
                            and applied_actor_multiplier > 3.0
                        ):
                            transfer_controller.reduce_actor_cap(
                                current_multiplier=applied_actor_multiplier
                            )
                        ppo_metrics.update({
                            "accelerated/actor_lr_multiplier": applied_actor_multiplier,
                            "accelerated/value_lr_multiplier": (
                                transfer_controller.config.value_multiplier
                            ),
                            "accelerated/actor_lr_cap_next": transfer_controller.actor_cap,
                            "accelerated/health_warning": float(health.warning),
                            "accelerated/lr_reduction_triggered": float(
                                health.reduce_actor_lr
                            ),
                            "accelerated/rollback_retries": float(rollback_retries),
                        })
                        break
                ppo_seconds = time.perf_counter() - ppo_started
                checkpoint = paths["checkpoint"] / f"update-{update:06d}.pt"
                digest = save_model_only(
                    model,
                    checkpoint,
                    update=update,
                    metadata={
                        "project": PROJECT,
                        "version": config.version,
                        "source_policy_update": source_update,
                        "source_checkpoint_sha256": identity.checkpoint_sha256,
                        "representation_sha256": representation,
                        "opponent_snapshot": "artifact/opponent_snapshot.json",
                        "parity_report": "artifact/large_model_0806_runtime_parity.json",
                        "accelerated_transfer": (
                            transfer_controller.metadata()
                            if transfer_controller is not None else None
                        ),
                        "actor_lr_multiplier": applied_actor_multiplier,
                        "value_lr_multiplier": (
                            transfer_controller.config.value_multiplier
                            if transfer_controller is not None else 1.0
                        ),
                        "feature_preprocessing_version": FEATURE_PREPROCESSING_VERSION,
                        "action_schema_version": ACTION_BOUNDARY_SCHEMA_VERSION,
                        "decision_gate_version": DECISION_GATE_VERSION,
                        "canonicalizer_version": CANONICALIZER_VERSION,
                        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                        "official_protocol_adapter_version": OFFICIAL_PROTOCOL_ADAPTER_VERSION,
                        **source_identity,
                    },
                )
                completed_update = update
                metrics: dict[str, Any] = {
                    "trainer/update": update,
                    "rollout/source_policy_update": source_update,
                    "checkpoint/update": update,
                    "env/episodes": cumulative_episodes,
                    "env/decisions": cumulative_decisions,
                    "checkpoint/sha256_present": float(bool(digest)),
                    "representation/sha256_unchanged": float(
                        model.representation_sha256() == representation
                    ),
                    "ppo/wall_seconds": ppo_seconds,
                    "system/end_to_end/wall_seconds": rollout_seconds + ppo_seconds,
                    "system/disk/free_gib": shutil.disk_usage(paths["root"]).free / 1024**3,
                    "system/cpu/max_rss_bytes": float(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    ),
                    "system/gpu/max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "system/gpu/max_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    **rollout_metrics,
                    **ppo_metrics,
                }
                if update % config.eval_every == 0:
                    evaluator = build_collector(model, opponent, config, mode="greedy")
                    eval_started = time.perf_counter()
                    evaluation_jobs, evaluation_schedule_sha = build_frozen_jobs(
                        focal_deck_id=FOCAL_DECK_ID,
                        focal_deck=focal_deck(), runtime_root=runtime_root(),
                        source_policy_update=update,
                    )
                    if evaluation_schedule_sha != baseline_schedule_sha:
                        raise RuntimeError("canonical Frozen schedule changed across checkpoints")
                    if _schedule_payload(
                        evaluation_jobs,
                        source_checkpoint_sha256=identity.checkpoint_sha256,
                    )["environment_sha256"] != json.loads(
                        (paths["artifact"] / "schedules/eval_frozen_2048.json").read_text()
                    )["environment_sha256"]:
                        raise RuntimeError("fixed evaluation schedule changed across checkpoints")
                    evaluation = evaluator.collect(evaluation_jobs)
                    evaluation_collector_metrics = evaluator.metrics()
                    evaluation_health_metrics = (
                        _assert_acceptance_episode_health(
                            evaluation, evaluation_collector_metrics,
                            scope="eval/core",
                        )
                        if transfer_controller is not None else {}
                    )
                    checkpoint_outcomes = _persist_frozen_results(
                        paths["artifact"] / f"frozen_results/core-update-{update:06d}.json",
                        evaluation, checkpoint_update=update,
                        panel_version=CANONICAL_CONTRACT_ID,
                    )
                    metrics.update(_episode_metrics(evaluation, "eval/core"))
                    metrics.update(_paired_frozen_metrics(
                        baseline_core_outcomes, checkpoint_outcomes, "eval/core"
                    ))
                    metrics.update(evaluation_collector_metrics)
                    metrics.update(evaluation_health_metrics)
                    metrics["eval/checkpoint_update"] = update
                    metrics["eval/wall_seconds"] = time.perf_counter() - eval_started
                    frozen_win_rate = sum(
                        value == 1 for value in checkpoint_outcomes.values()
                    ) / len(checkpoint_outcomes)
                    if frozen_win_rate > best_frozen_win_rate:
                        best_frozen_win_rate = frozen_win_rate
                        best_frozen_update = update
                        frozen_below_best_streak = 0
                        _atomic_json(paths["artifact"] / "champion.json", {
                            "schema": "0038_cuda_frozen_champion_v1",
                            "checkpoint_update": update,
                            "win_rate": frozen_win_rate,
                            "delta_vs_update0": frozen_win_rate - baseline_win_rate,
                            "checkpoint": f"checkpoint/update-{update:06d}.pt",
                            "selection_runtime": "accelerated:cuda_resident",
                            "cpu_evaluation_run": False,
                        })
                    else:
                        frozen_below_best_streak += 1
                    metrics.update({
                        "eval/champion_update": float(best_frozen_update),
                        "eval/champion_win_rate": best_frozen_win_rate,
                        "eval/below_best_consecutive_points": float(
                            frozen_below_best_streak
                        ),
                        "eval/degradation_warning": float(
                            frozen_below_best_streak >= 3
                        ),
                    })
                logger.log(update, metrics)
                _merge_status(
                    paths["artifact"] / "status.json",
                    {
                        "state": "running",
                        "checkpoint_update": update,
                        "rollout_source_policy_update": source_update,
                        "episodes": cumulative_episodes,
                        "decisions": cumulative_decisions,
                        "representation_sha256_unchanged": True,
                        "decoder_sha256": model.decoder_sha256(),
                        "elapsed_seconds": time.time() - started,
                    },
                )
                # Do not retain the previous update's high-dimensional feature
                # tensors while the next on-policy rollout is being collected.
                del batch, episodes, collector
                if update % config.eval_every == 0:
                    del evaluation, evaluator
                torch.cuda.empty_cache()
                if (paths["artifact"] / "STOP_REQUESTED").exists():
                    stop_requested = True
                    break
        summary = {
            "state": "stopped_by_request" if stop_requested else "complete",
            "version": config.version,
            "updates": update,
            "episodes": cumulative_episodes,
            "decisions": cumulative_decisions,
            "elapsed_seconds": time.time() - started,
            "representation_sha256": representation,
            "representation_unchanged": model.representation_sha256() == representation,
            "initial_decoder_sha256": initial_decoder,
            "final_decoder_sha256": model.decoder_sha256(),
            "checkpoint_retention": "all",
            "wandb_url": wandb_url,
            "cuda_champion_update": best_frozen_update,
            "cuda_champion_win_rate": best_frozen_win_rate,
            "cuda_champion_delta_vs_update0": (
                best_frozen_win_rate - baseline_win_rate
            ),
            "cpu_evaluation_scheduled": False,
            "accelerated_transfer": (
                transfer_controller.metadata()
                if transfer_controller is not None else None
            ),
        }
        _atomic_json(paths["artifact"] / "training_summary.json", summary)
        _merge_status(paths["artifact"] / "status.json", summary)
        return summary
    except BaseException as error:
        _merge_status(
            paths["artifact"] / "status.json",
            {
                "state": "failed",
                "checkpoint_update": completed_update,
                "attempted_update": update,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.time() - started,
                "wandb_sync_status": "failed_or_interrupted",
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--gate-games", type=int, default=4)
    parser.add_argument("--gate-workers", type=int, default=2)
    parser.add_argument("--gate-engines-per-worker", type=int, default=1)
    parser.add_argument("--gate-inference-channels-per-role", type=int, default=1)
    parser.add_argument("--gate-coalesce-ms", type=float, default=5.0)
    parser.add_argument("--gate-mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--gate-ppo", action="store_true")
    parser.add_argument("--gate-preset", choices=tuple(PRESETS), default="BASE")
    parser.add_argument("--version", default=FORMAL_VERSION)
    parser.add_argument(
        "--updates", type=int, default=None,
        help="optional finite update limit; omit to run until artifact/STOP_REQUESTED",
    )
    parser.add_argument("--worker-processes", type=int, default=16)
    parser.add_argument("--engines-per-worker", type=int, default=8)
    parser.add_argument("--inference-channels-per-role", type=int, default=8)
    parser.add_argument("--coalesce-ms", type=float, default=5.0)
    parser.add_argument("--games-per-update", type=int, default=512)
    parser.add_argument(
        "--engine-backend",
        choices=("official", "accelerated:cuda_resident"),
        default="accelerated:cuda_resident",
    )
    parser.add_argument("--cuda-lane-count", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--trajectory-games-per-update", type=int, default=512)
    parser.add_argument("--ppo-minibatch-size", type=int, default=1024)
    parser.add_argument("--ppo-gradient-accumulation", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--optimizer-steps-per-update", type=int, default=32)
    parser.add_argument("--optimization-mode", choices=("fixed_epochs", "fixed_optimizer_budget"),
                        default="fixed_optimizer_budget")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--adaptation-arm", choices=("lora", "lora_layernorm"), default="lora"
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="INTEGRATED")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--credit-clock", choices=("turn",), default="turn")
    parser.add_argument(
        "--loss-weighting",
        choices=("episode_equal_decisions", "episode_equal_turns"),
        default="episode_equal_decisions",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--resume-update0", action="store_true")
    parser.add_argument("--accelerated-transfer-acceptance", action="store_true")
    args = parser.parse_args()
    if args.gate_output is not None:
        report = run_gate(
            output=args.gate_output,
            games=args.gate_games,
            worker_processes=args.gate_workers,
            engines_per_worker=args.gate_engines_per_worker,
            inference_channels_per_role=args.gate_inference_channels_per_role,
            run_ppo=args.gate_ppo,
            mode=args.gate_mode,
            coalesce_ms=args.gate_coalesce_ms,
            gae_lambda=args.gae_lambda,
            credit_clock=args.credit_clock,
            loss_weighting=args.loss_weighting,
            adaptation_arm=args.adaptation_arm,
            preset_name=args.gate_preset,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    run(
        RunConfig(
            version=args.version,
            updates=args.updates,
            worker_processes=args.worker_processes,
            engines_per_worker=args.engines_per_worker,
            inference_channels_per_role=args.inference_channels_per_role,
            coalesce_ms=args.coalesce_ms,
            games_per_update=args.games_per_update,
            engine_backend=args.engine_backend,
            cuda_lane_count=args.cuda_lane_count,
            rollout_batch_size=args.rollout_batch_size,
            trajectory_games_per_update=args.trajectory_games_per_update,
            ppo_minibatch_size=args.ppo_minibatch_size,
            ppo_gradient_accumulation=args.ppo_gradient_accumulation,
            ppo_epochs=args.ppo_epochs,
            optimizer_steps_per_update=args.optimizer_steps_per_update,
            optimization_mode=args.optimization_mode,
            eval_every=args.eval_every,
            adaptation_arm=args.adaptation_arm,
            preset_name=args.preset,
            wandb_mode=args.wandb_mode,
            launch_formal=args.launch_formal,
            resume_update0=args.resume_update0,
            accelerated_transfer_acceptance=args.accelerated_transfer_acceptance,
            ppo=PPOConfig(
                gae_lambda=args.gae_lambda,
                credit_clock=args.credit_clock,
                loss_weighting=args.loss_weighting,
                batch_size=args.ppo_minibatch_size,
                gradient_accumulation=args.ppo_gradient_accumulation,
                epochs=args.ppo_epochs,
                optimizer_steps_per_update=args.optimizer_steps_per_update,
                optimization_mode=args.optimization_mode,
            ),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
