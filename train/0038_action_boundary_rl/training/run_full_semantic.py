"""Formal full-0031 decoder PPO over paired Frozen-0806 seed scenarios."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any

import torch

from rl_environment.logging import TrainingLogger
from evaluation.runtime.seeded import build_seeded_runtime

from ..league import load_frozen_catalog
from ..parity import assert_large_model_0806_runtime_parity, collect_official_observations
from ..policy import AdaptationConfig, load_actor_critic
from ..rollout import FullSemanticRolloutCollector, RolloutJob
from .batch_full_semantic import prepare_episodes
from .ppo_full_semantic import PPOConfig, PPOTrainer
from .storage_full_semantic import save_model_only
from ..semantic_policy.deployment.inference import PortableSemanticPolicy
from ..evaluation.frozen_jobs import build_frozen_jobs
from ..evaluation.frozen_panel import paired_summary, wilson_interval
from ..initialization import (
    COMMON_UPDATE0_CHECKPOINT,
    COMMON_UPDATE0_SHA256,
    build_preset_from_common_update0,
)
from ..integrated.presets import PRESETS, preset


ROOT = Path(__file__).resolve().parents[3]
PROJECT = "0038_action_boundary_rl"
WANDB_DISPLAY_PREFIX = "0038 · action_boundary"
FORMAL_VERSION = "V4_integrated_fresh_rl"
SOURCE_CHECKPOINT = ROOT / "rl_runs/0037_dragapult_value_initialized_rl/source/friend_0806_epoch11/model.pt"
CANDIDATE_ROOT = ROOT / "evaluation/arena/candidates/0034_dragapult_third_large_model_zero_shot"
FOCAL_DECK_ID = "dragapult_ex_07bedfffbfad"
FOCAL_EXACT_DECK_SHA256 = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
FOCAL_DECK_PATH = ROOT / "train" / PROJECT / "league/decks" / FOCAL_DECK_ID / "deck.csv"
FROZEN_PANEL = ROOT / "experiments/0038_action_boundary_rl/frozen_panel_2048_v1.json"


@dataclass(frozen=True, slots=True)
class RunConfig:
    version: str = FORMAL_VERSION
    updates: int = 50
    worker_processes: int = 16
    engines_per_worker: int = 8
    inference_channels_per_role: int = 8
    coalesce_ms: float = 5.0
    device: str = "cuda:0"
    seed: int = 330031001
    games_per_update: int = 512
    engine_backend: str = "official"
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
    ppo: PPOConfig = PPOConfig()

    def validate(self) -> None:
        if re.fullmatch(r"V[1-9]\d*_[a-z0-9]+(?:_[a-z0-9]+)*", self.version) is None:
            raise ValueError("version must match V<n>_<ascii_snake_case>")
        if min(
            self.updates,
            self.worker_processes,
            self.engines_per_worker,
            self.inference_channels_per_role,
            self.eval_every,
        ) < 1 or self.coalesce_ms < 0:
            raise ValueError("updates, rollout topology, and eval_every must be positive")
        if self.inference_channels_per_role > self.engines_per_worker:
            raise ValueError(
                "inference_channels_per_role cannot exceed engines_per_worker"
            )
        if self.games_per_update < 2 or self.games_per_update % 2:
            raise ValueError("games_per_update must be a positive seat-balanced count")
        if self.engine_backend != "official" and not self.engine_backend.startswith("accelerated:"):
            raise ValueError("invalid engine backend")
        if self.optimization_mode not in {"fixed_epochs", "fixed_optimizer_budget"}:
            raise ValueError("invalid optimization mode")
        if min(self.inference_batch_size, self.max_inflight_requests,
               self.rollout_queue_depth, self.seed_shard_count,
               self.optimizer_steps_per_update, self.ppo_gradient_accumulation) < 1:
            raise ValueError("capacity settings must be positive")
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


def build_jobs(
    *,
    source_policy_update: int,
    seed: int,
    count: int = 512,
    greedy: bool = False,
) -> list[RolloutJob]:
    catalog = load_frozen_catalog()
    if count < 2 or count % 2:
        raise ValueError("job count must be positive and seat-balanced")
    fixed_slots = [opponent for opponent in catalog for _ in range(opponent.games)]
    scenario_count = count // 2
    update_offset = 0 if greedy else source_policy_update * 1_000_003
    scenario_rng = random.Random(seed + update_offset)
    selected = []
    while len(selected) < scenario_count:
        indices = list(range(len(fixed_slots)))
        scenario_rng.shuffle(indices)
        selected.extend(fixed_slots[index] for index in indices[:scenario_count - len(selected)])
    engine_seeds = scenario_rng.sample(range(1, 0x80000000), scenario_count)
    jobs: list[RolloutJob] = []
    deck = focal_deck()
    root = runtime_root()
    runtime = build_seeded_runtime()
    for pair_index, (opponent, engine_seed) in enumerate(
        zip(selected, engine_seeds, strict=True)
    ):
        search_seed = (
            (engine_seed + 900_000_007 + update_offset) & 0x7FFFFFFF
        ) or 1
        for seat_index, focal_first in enumerate((True, False)):
            game_index = pair_index * 2 + seat_index
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
                    action_boundary_mode="enabled",
                )
            )
    if scenario_count >= len(fixed_slots) and len({job.opponent_id for job in jobs}) != len(catalog):
        raise RuntimeError("formal schedule omitted a Frozen opponent")
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
        f"{prefix}/decisions": float(sum(len(ep.decisions) for ep in episodes)),
        f"{prefix}/policy_transitions": float(
            sum(len(ep.policy_transitions) for ep in episodes)
        ),
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


def _persist_frozen_results(path: Path, episodes: list[Any], *, checkpoint_update: int) -> dict[int, int]:
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
    if len(rows) != 2048 or len(outcomes) != 2048:
        raise RuntimeError("Frozen result persistence requires 2,048 unique games")
    _atomic_json(path, {
        "schema_version": "0038_frozen_per_game_results_v1",
        "frozen_panel_version": "0038_frozen_2048_v1",
        "checkpoint_update": checkpoint_update, "entries": rows,
    })
    return outcomes


def _paired_frozen_metrics(baseline: dict[int, int], checkpoint: dict[int, int]) -> dict[str, float]:
    summary = paired_summary(baseline, checkpoint)
    return {
        "eval/win_rate": float(summary["win_rate"]),
        "eval/wilson_low": float(summary["wilson_95"][0]),
        "eval/wilson_high": float(summary["wilson_95"][1]),
        "eval/baseline_loss_to_win": float(summary["baseline_loss_to_checkpoint_win"]),
        "eval/baseline_win_to_loss": float(summary["baseline_win_to_checkpoint_loss"]),
        "eval/mcnemar_chi2": float(summary["mcnemar_continuity_corrected_chi2"]),
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
    } for job in jobs]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _opponent_snapshot() -> dict[str, Any]:
    catalog = load_frozen_catalog()
    return {
        "schema": "0038_frozen0806_pool_snapshot_v2",
        "opponent_count": len(catalog),
        "total_games": sum(item.games for item in catalog),
        "seat_contract": "configured_by_rollout_or_frozen_panel_manifest",
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
    raise RuntimeError(
        "0038 formal PPO remains fail-closed pending explicit user approval of the "
        "V3 common update-0; tests/smoke/evaluation entry points remain available"
    )
    # The implementation below is the audited rollout/evaluation harness. The guard
    # is removed only after explicit approval of the common update-0.
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
    paths = assert_fresh_version(config.version)
    for name in ("artifact", "checkpoint", "tensorboard", "wandb"):
        paths[name].mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference/training requested but unavailable")
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
                "0038,action_boundary,official_cpu,ppo,frozen_panel_2048,"
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
        "rollout_seed_contract": {
            "sampled_engine_seeds": config.games_per_update // 2,
            "episodes_per_seed": 2,
            "fixed_physical_deck_slots": True,
            "only_seat_is_swapped_within_pair": True,
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
        "adaptation": asdict(adaptation),
        "value_diagnostics": {
            "source": "on_policy_rollout_old_value",
            "absolute_turn_bins": ["00_03", "04_07", "08_11", "12_plus"],
            "remaining_turn_bins": ["00_01", "02_03", "04_plus"],
            "outcomes": ["focal_win", "focal_loss", "draw"],
            "targets": ["gae_return", "terminal_outcome"],
        },
        "reward": "terminal_only_minus_one_zero_plus_one",
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
            "full_round_draw_limit": 50,
            "engine_turn_limit": 99,
        },
    }
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
    started = time.time()
    update = 0
    cumulative_episodes = 0
    cumulative_decisions = 0
    try:
        model, identity = build_preset_from_common_update0(
            focal_deck(), flags, device=device
        )
        opponent = load_frozen_opponent(device)
        parity = _run_parity(model, paths["artifact"] / "large_model_0806_runtime_parity.json")
        if not parity["passed"]:
            raise RuntimeError("Large Model 0806 runtime parity did not pass")
        trainer = PPOTrainer(model, device=device, config=config.ppo)
        representation = model.representation_sha256()
        initial_decoder = model.decoder_sha256()
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
            baseline_evaluator = FullSemanticRolloutCollector(
                model,
                opponent,
                device=device,
                worker_processes=config.worker_processes,
                engines_per_worker=config.engines_per_worker,
                inference_channels_per_role=config.inference_channels_per_role,
                mode="greedy",
                coalesce_ms=config.coalesce_ms,
            )
            baseline_started = time.perf_counter()
            baseline_jobs = build_frozen_jobs(
                FROZEN_PANEL, focal_deck=focal_deck(), runtime_root=runtime_root(),
                source_policy_update=0,
            )
            _atomic_json(
                paths["artifact"] / "schedules/eval_frozen_2048.json",
                _schedule_payload(
                    baseline_jobs,
                    source_checkpoint_sha256=identity.checkpoint_sha256,
                ),
            )
            baseline = baseline_evaluator.collect(baseline_jobs)
            baseline_outcomes = _persist_frozen_results(
                paths["artifact"] / "frozen_results/update-000000.json",
                baseline, checkpoint_update=0,
            )
            diagnostic_batch = prepare_episodes(
                baseline[:2],
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
                    **_episode_metrics(baseline, "eval"),
                },
            )
            for update in range(1, config.updates + 1):
                source_update = update - 1
                collector = FullSemanticRolloutCollector(
                    model,
                    opponent,
                    device=device,
                    worker_processes=config.worker_processes,
                    engines_per_worker=config.engines_per_worker,
                    inference_channels_per_role=config.inference_channels_per_role,
                    mode="sample",
                    coalesce_ms=config.coalesce_ms,
                )
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
                rollout_seconds = time.perf_counter() - rollout_started
                rollout_metrics = _episode_metrics(episodes, "rollout")
                rollout_metrics.update(collector.metrics())
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
                ppo_metrics.update(trainer.update(batch, update=update))
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
                    },
                )
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
                    "system/gpu/max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "system/gpu/max_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    **rollout_metrics,
                    **ppo_metrics,
                }
                if update % config.eval_every == 0:
                    evaluator = FullSemanticRolloutCollector(
                        model,
                        opponent,
                        device=device,
                        worker_processes=config.worker_processes,
                        engines_per_worker=config.engines_per_worker,
                        inference_channels_per_role=config.inference_channels_per_role,
                        mode="greedy",
                        coalesce_ms=config.coalesce_ms,
                    )
                    eval_started = time.perf_counter()
                    evaluation_jobs = build_frozen_jobs(
                        FROZEN_PANEL, focal_deck=focal_deck(), runtime_root=runtime_root(),
                        source_policy_update=update,
                    )
                    if _schedule_payload(
                        evaluation_jobs,
                        source_checkpoint_sha256=identity.checkpoint_sha256,
                    )["environment_sha256"] != json.loads(
                        (paths["artifact"] / "schedules/eval_frozen_2048.json").read_text()
                    )["environment_sha256"]:
                        raise RuntimeError("fixed evaluation schedule changed across checkpoints")
                    evaluation = evaluator.collect(evaluation_jobs)
                    checkpoint_outcomes = _persist_frozen_results(
                        paths["artifact"] / f"frozen_results/update-{update:06d}.json",
                        evaluation, checkpoint_update=update,
                    )
                    metrics.update(_episode_metrics(evaluation, "eval"))
                    metrics.update(_paired_frozen_metrics(baseline_outcomes, checkpoint_outcomes))
                    metrics["eval/checkpoint_update"] = update
                    metrics["eval/wall_seconds"] = time.perf_counter() - eval_started
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
        summary = {
            "state": "complete",
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
        }
        _atomic_json(paths["artifact"] / "training_summary.json", summary)
        _merge_status(paths["artifact"] / "status.json", summary)
        return summary
    except BaseException as error:
        _merge_status(
            paths["artifact"] / "status.json",
            {
                "state": "failed",
                "checkpoint_update": update,
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
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--worker-processes", type=int, default=16)
    parser.add_argument("--engines-per-worker", type=int, default=8)
    parser.add_argument("--inference-channels-per-role", type=int, default=8)
    parser.add_argument("--coalesce-ms", type=float, default=5.0)
    parser.add_argument("--games-per-update", type=int, default=512)
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
            ppo_minibatch_size=args.ppo_minibatch_size,
            ppo_gradient_accumulation=args.ppo_gradient_accumulation,
            ppo_epochs=args.ppo_epochs,
            optimizer_steps_per_update=args.optimizer_steps_per_update,
            optimization_mode=args.optimization_mode,
            eval_every=args.eval_every,
            adaptation_arm=args.adaptation_arm,
            preset_name=args.preset,
            wandb_mode=args.wandb_mode,
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
