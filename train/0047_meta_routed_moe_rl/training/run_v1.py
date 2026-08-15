"""Self-contained 0045 V1 deck-007 minimal-LoRA PPO entrypoint.

The default command is a read-only readiness check. The long run can start only
with ``--launch-formal``, preserving the user's explicit launch authority.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from evaluation.runtime.seeded import build_seeded_runtime
from rl_environment.logging import TrainingLogger

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..initial_run import balanced_focal_schedule
from ..integrated.presets import preset
from ..league.pfsp import Curriculum, PFSPConfig, PFSPState
from ..league.aggressive_meta_quota import aggressive_meta_quota_schedule
from ..league.meta_balanced import (
    balanced_meta_deck_schedule, core_deck_half_meta_balanced_schedule,
)
from ..league.sampler import LeagueLane, _seed, build_schedule, build_uniform_schedule
from ..own_archetype import OwnArchetypeVocabulary
from ..policy import AdaptationConfig, load_actor_critic
from ..policy.actor_critic import DEFAULT_FOCAL_CHECKPOINT
from ..policy_identity import materialize_policy_bundle
from ..rollout import DEFAULT_FULL_ROUND_DRAW_LIMIT, ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import load_policy
from ..telemetry import RolloutHistory, aggregate_training_rollout
from .batch_full_semantic import prepare_episodes
from .lr_profiles import LearningRateProfile, resolve_learning_rate_profile
from .ppo_full_semantic import PPOConfig, PPOTrainer
from .periodic_evaluation import is_due as periodic_evaluation_due
from .periodic_evaluation import wandb_metrics as periodic_evaluation_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
OPPONENT_POLICY_IDS = ("Champion-G2",)
FOCAL_DECK_ID = "007"
PROJECT = "0047_meta_routed_moe_rl"
VERSION = "V1_minimal_lora_dragapult_007"
RULES = ROOT / ".tmp/cuda_0032_rules/official_rules.bin"
CONTRACT = ROOT / "experiments/0047_meta_routed_moe_rl/manifest.json"
U0_CHECKPOINT = (
    ROOT / "rl_runs/0047_meta_routed_moe_rl/versions/"
    "V1_minimal_lora_dragapult_007/checkpoint/update-000000.pt"
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _cards(registry: AssetRegistry, deck_id: str) -> tuple[int, ...]:
    asset = next(row for row in registry.decks if row.deck_id == deck_id)
    return tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))


def _runtime_root() -> Path:
    roots = sorted((ROOT / "evaluation/arena/opponents").glob("*/cg/game.py"))
    if not roots:
        raise FileNotFoundError("official engine runtime is unavailable")
    return roots[0].parents[1].resolve()


def _config(
    *, forward_microbatch_size: int = 1024,
    behavior_probe_batch_size: int = 512,
    offload_reference_after_cache: bool = False,
    learning_rate_profile: LearningRateProfile | None = None,
    actor_learning_rate_scale: float | None = None,
    batch_size: int = 2048,
    value_learning_rate: float | None = None,
    prize_learning_rate: float | None = None,
    entropy_coefficient: float | None = None,
) -> PPOConfig:
    profile = resolve_learning_rate_profile(
        learning_rate_profile=learning_rate_profile,
        actor_learning_rate_scale=actor_learning_rate_scale,
        value_learning_rate=value_learning_rate,
        prize_learning_rate=prize_learning_rate,
    )
    return PPOConfig(
        **profile.rates(),
        batch_size=batch_size,
        forward_microbatch_size=forward_microbatch_size,
        behavior_probe_batch_size=behavior_probe_batch_size,
        offload_reference_after_cache=offload_reference_after_cache,
        entropy_coefficient=(
            PPOConfig().entropy_coefficient
            if entropy_coefficient is None else float(entropy_coefficient)
        ),
    )


def _paths(version: str = VERSION) -> dict[str, Path]:
    root = ROOT / "rl_runs" / PROJECT / "versions" / version
    return {name: root / name for name in ("artifact", "checkpoint", "tensorboard", "wandb")}


def _pristine_restart_allowed(
    paths: dict[str, Path], *, start_update: int, parent_checkpoint: Path,
    focal_deck_ids: tuple[str, ...],
) -> bool:
    """Accept only a launch that failed before its first rollout metric/update."""
    metrics = paths["artifact"] / "training_metrics.jsonl"
    config = paths["artifact"] / "training_config.json"
    checkpoints = sorted(paths["checkpoint"].glob("update-*.pt"))
    if not metrics.is_file() or not config.is_file():
        return False
    if checkpoints != [paths["checkpoint"] / f"update-{start_update:06d}.pt"]:
        return False
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
        metric_rows = [
            json.loads(line)
            for line in metrics.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, KeyError):
        return False
    metrics_are_pristine = not metric_rows or (
        len(metric_rows) == 1
        and metric_rows[0].get("eval/checkpoint_update") == start_update
        and metric_rows[0].get("eval/benchmark_tiny_v2") == 1.0
        and "trainer/update" not in metric_rows[0]
        and "ppo/reference_kl" not in metric_rows[0]
    )
    return (
        metrics_are_pristine
        and
        checkpoint.get("update") == start_update
        and payload.get("start_update") == start_update
        and payload.get("parent_checkpoint_sha256") == sha256_file(parent_checkpoint)
        and payload.get("focal_deck_ids") == list(focal_deck_ids)
        and payload.get("optimizer_initialization") == "fresh"
    )


def _audited_u0_launch_allowed(paths: dict[str, Path], parent_checkpoint: Path) -> bool:
    """Allow the immutable migrated U0 plus pre-launch audit evidence exactly once."""
    expected = paths["checkpoint"] / "update-000000.pt"
    checkpoints = sorted(paths["checkpoint"].glob("update-*.pt"))
    forbidden_runtime = (
        paths["artifact"] / "training_metrics.jsonl",
        paths["artifact"] / "training_config.json",
    )
    return (
        parent_checkpoint.resolve() == expected.resolve()
        and checkpoints == [expected]
        and all(not path.exists() for path in forbidden_runtime)
        and not any(any(paths[name].iterdir()) for name in ("tensorboard", "wandb"))
    )


def _run_id() -> str:
    return "0045-v1-minimal-lora-dragapult-007"


def _group_jobs_by_opponent_policy(
    jobs: list[RolloutJob],
) -> dict[str, list[RolloutJob]]:
    """Keep policy identities isolated while allowing mixed focal decks per batch."""
    grouped: dict[str, list[RolloutJob]] = defaultdict(list)
    for job in jobs:
        grouped[job.opponent_policy_id].append(job)
    return grouped


def readiness() -> dict[str, Any]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    audit = registry.validate_all()
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    paths = _paths()
    g2 = PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin"
    missing = [str(path) for path in (RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so", g2) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"0045 launch artifacts missing: {missing}")
    if "Champion-G2" not in {row.policy_id for row in registry.policies}:
        raise RuntimeError("0045 V1 requires registered immutable Champion-G2")
    return {
        "schema_version": "0045_v1_launch_readiness_v1",
        "status": "READY_AWAITING_GPU_HANDOFF" if not U0_CHECKPOINT.is_file() else "READY_AWAITING_USER_LAUNCH",
        "long_training_started": False,
        "launch_token_required": "--launch-formal",
        "project": PROJECT, "version": VERSION,
        "asset_audit": asdict(audit),
        "focal_decks": [FOCAL_DECK_ID],
        "opponent_decks": [f"{value:03d}" for value in range(1, 71)],
        "opponent_policies": list(OPPONENT_POLICY_IDS),
        "opponent_sampling": {
            "mode": "meta_balanced_training_pool", "games": 512,
            "meta_weight_3x": [0, 1, 2, 3, 5, 27],
        },
        "periodic_evaluation": {"contract": "Benchmark-Tiny-V2", "every_updates": 5, "cuda_games": 512, "formal_evaluate_every_updates": 10},
        "frozen_0045_init": str(U0_CHECKPOINT),
        "frozen_0045_init_exists": U0_CHECKPOINT.is_file(),
        "own_taxonomy": {"version": vocabulary.taxonomy_version, "classes": 29, "embedding_width": 16},
        "opponent_meta": {"classes": 15, "trainable": False},
        "cuda_engine": {"version": "2.0", "extension_sha256": sha256_file(DEFAULT_BUILD_DIR / "_ptcg_cuda.so")},
        "wandb": {"entity": "dragon_bra", "project": "pokemon-tcg-policy-learning", "run_id": _run_id()},
    }


def _jobs(
    update: int, registry: AssetRegistry, *, pfsp_state: PFSPState | None = None,
    focal_deck_ids: tuple[str, ...] = (FOCAL_DECK_ID,),
    focal_deck_id: str = FOCAL_DECK_ID,
    focal_schedule_mode: str = "fixed",
    opponent_sampling_mode: str = "pfsp_mixture",
    opponent_policy_ids: tuple[str, ...] = OPPONENT_POLICY_IDS,
    latest_champion_policy_id: str = "Champion-G2",
    rollout_games: int = 256,
    opponent_meta_weights: dict[int, float] | None = None,
    opponent_meta_quotas: dict[int, int] | None = None,
    core_deck_ids: tuple[str, ...] | None = None,
) -> tuple[list[RolloutJob], dict[str, float], dict[str, float], str]:
    training_deck_ids = tuple(
        deck.deck_id for deck in registry.decks if "training" in deck.roles
    )
    legacy_deck_ids = tuple(f"{value:03d}" for value in range(1, 68))
    deck_ids = (
        legacy_deck_ids
        if opponent_sampling_mode in {"uniform_001_067", "meta_balanced_001_067"}
        else training_deck_ids
    )
    policy_ids = opponent_policy_ids
    if opponent_sampling_mode in {
        "uniform_001_067", "meta_balanced_001_067",
        "meta_balanced_training_pool", "aggressive_meta_quota_training_pool",
        "core_deck_half_meta_balanced_half",
    }:
        curriculum = _uniform_curriculum(update, deck_ids, policy_ids)
    else:
        state = PFSPState() if pfsp_state is None else pfsp_state
        curriculum = state.curriculum_for(
            update, deck_ids=deck_ids, policy_ids=policy_ids, config=PFSPConfig(),
            training_deck_pool_hash=hashlib.sha256("".join(deck_ids).encode()).hexdigest(),
            active_policy_pool_hash=hashlib.sha256("".join(policy_ids).encode()).hexdigest(),
        )
    schedule_builder = {
        "pfsp_mixture": build_schedule,
        "uniform_001_067": build_uniform_schedule,
    }.get(opponent_sampling_mode)
    if schedule_builder is None and opponent_sampling_mode not in {
        "meta_balanced_001_067", "meta_balanced_training_pool",
        "aggressive_meta_quota_training_pool",
        "core_deck_half_meta_balanced_half",
    }:
        raise ValueError(f"unsupported opponent sampling mode: {opponent_sampling_mode}")
    if opponent_sampling_mode in {
        "meta_balanced_001_067", "meta_balanced_training_pool",
        "aggressive_meta_quota_training_pool",
        "core_deck_half_meta_balanced_half",
    }:
        if policy_ids != (latest_champion_policy_id,):
            raise ValueError("Meta-balanced opponent schedule requires one latest champion")
        own = OwnArchetypeVocabulary.load_version(
            "own_archetypes_v2", project_root=PROJECT_ROOT
        )
        training_mappings = tuple(
            row for row in own.mappings if row.deck_id in set(deck_ids)
        )
        quota_seed = 440_120_000 + update
        schedule_audit: dict[str, Any] = {}
        if opponent_sampling_mode == "aggressive_meta_quota_training_pool":
            if opponent_meta_weights is not None or opponent_meta_quotas is None:
                raise ValueError("aggressive Meta schedule requires quotas, not weights")
            opponent_slots = aggressive_meta_quota_schedule(
                quota_seed=quota_seed,
                shuffle_seed=440_120_200 + update,
                mappings=training_mappings,
                lanes=rollout_games,
                fixed_meta_quotas=opponent_meta_quotas,
            )
            branch_name = "aggressive_meta_quota"
        elif opponent_sampling_mode == "core_deck_half_meta_balanced_half":
            if opponent_meta_weights is not None or opponent_meta_quotas is not None:
                raise ValueError("core-deck half schedule does not accept Meta weights/quotas")
            if core_deck_ids is None:
                raise ValueError("core-deck half schedule requires explicit core deck IDs")
            opponent_slots, schedule_audit = core_deck_half_meta_balanced_schedule(
                quota_seed=quota_seed,
                shuffle_seed=440_120_200 + update,
                mappings=training_mappings,
                core_deck_ids=core_deck_ids,
                lanes=rollout_games,
            )
            branch_name = "core_deck_half_meta_balanced_half"
        else:
            if opponent_meta_quotas is not None:
                raise ValueError("Meta quotas require the aggressive schedule")
            opponent_slots = balanced_meta_deck_schedule(
                quota_seed=quota_seed,
                shuffle_seed=440_120_200 + update,
                mappings=training_mappings,
                lanes=rollout_games,
                meta_weights=opponent_meta_weights,
            )
            branch_name = "meta_balanced_uniform"
        scheduled_deck_counts: dict[str, int] = defaultdict(int)
        for slot in opponent_slots:
            scheduled_deck_counts[slot.deck_id] += 1
        scheduled_deck_weights = {
            deck_id: scheduled_deck_counts[deck_id] / rollout_games
            for deck_id in deck_ids
        }
        master = 430_044_001 + update
        league = tuple(
            LeagueLane(
                lane_id=row.lane_id, branch=getattr(row, "branch", branch_name),
                opponent_deck_id=row.deck_id,
                opponent_policy_id=latest_champion_policy_id,
                seat_slot=row.lane_id,
                coin_winner_seed=(coin_seed := _seed(master, row.lane_id, "coin-winner")),
                focal_won_toss=bool(coin_seed & 1),
                engine_seed=_seed(master, row.lane_id, "engine"),
                search_seed=_seed(master, row.lane_id, "search"),
                policy_seed=_seed(master, row.lane_id, "policy"),
                curriculum_version=f"{branch_name}-u{update:06d}",
            )
            for row in opponent_slots
        )
        curriculum = Curriculum(
            **{
                **asdict(curriculum),
                "curriculum_version": f"{branch_name}-u{update:06d}",
                "deck_weights": scheduled_deck_weights,
                "config_hash": hashlib.sha256(json.dumps({
                    "sampling_mode": opponent_sampling_mode,
                    "opponent_meta_weights": opponent_meta_weights,
                    "opponent_meta_quotas": opponent_meta_quotas,
                    "core_deck_ids": core_deck_ids,
                    "rollout_games": rollout_games,
                }, sort_keys=True).encode()).hexdigest(),
                "stats_snapshot": {
                    "sampling_mode": opponent_sampling_mode,
                    "pfsp_enabled": False,
                    "opponent_meta_weights": opponent_meta_weights,
                    "opponent_meta_quotas": opponent_meta_quotas,
                    "core_deck_ids": core_deck_ids,
                    "schedule_audit": schedule_audit,
                },
            }
        )
    else:
        league = schedule_builder(
            update=update, seed=430044001 + update, deck_ids=deck_ids,
            policy_ids=policy_ids,
            latest_champion_policy_id=latest_champion_policy_id,
            curriculum=curriculum,
        )
        if len(league) != rollout_games:
            raise ValueError("legacy rollout size does not match requested games")
    if focal_schedule_mode == "fixed":
        if focal_deck_ids != (focal_deck_id,):
            raise ValueError("fixed-focal 0045 run requires one matching focal deck ID")
        focal_by_lane = tuple(focal_deck_id for _ in league)
    elif focal_schedule_mode == "seeded_frequency_balanced_random_v1":
        focal_by_lane = tuple(
            row.deck_id for row in balanced_focal_schedule(
                430044711 + update, deck_ids=focal_deck_ids, lanes=len(league)
            )
        )
    elif focal_schedule_mode == "meta_balanced_001_067":
        if focal_deck_ids != legacy_deck_ids:
            raise ValueError("Meta-balanced focal schedule requires exact pool 001-067")
        own = OwnArchetypeVocabulary.load_version(
            "own_archetypes_v2", project_root=PROJECT_ROOT
        )
        training_mappings = tuple(
            row for row in own.mappings if row.deck_id in set(deck_ids)
        )
        focal_by_lane = tuple(
            row.deck_id for row in balanced_meta_deck_schedule(
                quota_seed=440_120_000 + update,
                shuffle_seed=440_120_100 + update,
                mappings=training_mappings,
                lanes=len(league),
            )
        )
    elif focal_schedule_mode == "meta_balanced_training_pool":
        if focal_deck_ids != training_deck_ids:
            raise ValueError("Meta-balanced focal schedule requires the exact training pool")
        own = OwnArchetypeVocabulary.load_version(
            "own_archetypes_v2", project_root=PROJECT_ROOT
        )
        training_id_set = set(training_deck_ids)
        training_mappings = tuple(
            row for row in own.mappings if row.deck_id in training_id_set
        )
        focal_by_lane = tuple(
            row.deck_id for row in balanced_meta_deck_schedule(
                quota_seed=440_120_000 + update,
                shuffle_seed=440_120_100 + update,
                mappings=training_mappings,
                lanes=len(league),
            )
        )
    else:
        raise ValueError(f"unsupported focal schedule mode: {focal_schedule_mode}")
    own = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=PROJECT_ROOT)
    own_by_deck = {row.deck_id: row.archetype_id for row in own.mappings}
    runtime = build_seeded_runtime()
    root = _runtime_root()
    jobs = [
        RolloutJob(
            game_id=f"rollout-u{update:06d}-{lane.lane_id:03d}",
            opponent_id=lane.opponent_deck_id,
            focal_first=lane.focal_won_toss,
            focal_won_toss=lane.focal_won_toss,
            coin_winner_seed=lane.coin_winner_seed,
            seed=lane.engine_seed,
            source_policy_update=update,
            focal_deck=_cards(registry, focal_by_lane[lane.lane_id]),
            opponent_deck=_cards(registry, lane.opponent_deck_id),
            runtime_root=root, opponent_policy_id=lane.opponent_policy_id,
            focal_deck_id=focal_by_lane[lane.lane_id],
            focal_own_archetype_id=own_by_deck[focal_by_lane[lane.lane_id]],
            policy_seed=lane.policy_seed, search_seed=lane.search_seed,
            engine_library=runtime.library_path,
            full_round_draw_limit=DEFAULT_FULL_ROUND_DRAW_LIMIT,
            action_boundary_mode="enabled",
        )
        for lane in league
    ]
    return jobs, dict(curriculum.deck_weights), dict(curriculum.policy_weights), curriculum.curriculum_version


def _uniform_curriculum(
    update: int, deck_ids: tuple[str, ...], policy_ids: tuple[str, ...],
) -> Curriculum:
    """A stateless schedule descriptor; it is not a PFSP curriculum/update."""
    return Curriculum(
        curriculum_version=f"uniform-u{update:06d}", created_at_update=update,
        deck_weights={deck_id: 1.0 / len(deck_ids) for deck_id in deck_ids},
        policy_weights={policy_id: 1.0 / len(policy_ids) for policy_id in policy_ids},
        stats_snapshot={"sampling_mode": "uniform_001_067", "pfsp_enabled": False},
        config_hash=hashlib.sha256(b"0044-uniform-001-067-v1").hexdigest(),
        training_deck_pool_hash=hashlib.sha256("".join(deck_ids).encode()).hexdigest(),
        active_policy_pool_hash=hashlib.sha256("".join(policy_ids).encode()).hexdigest(),
    )


def _load_opponent(policy_id: str, deck_id: str, device: torch.device):
    bundle = materialize_policy_bundle(PROJECT_ROOT, policy_id, purpose="0045_formal_rollout")
    policy = load_policy(policy_id, deck_id=deck_id)
    if hasattr(policy, "model"):
        modules = (policy.model,)
    else:
        values = [
            policy.actor, policy.value_head, policy.allocation_head,
            policy.value_adapter, policy.policy_strategy_adapter,
        ]
        values.extend(
            module for name in ("policy_option_lora", "meta_actor_residual")
            if (module := getattr(policy, name, None)) is not None
        )
        modules = tuple(values)
    for module in modules:
        module.to(device).eval().requires_grad_(False)
    return policy, bundle.audit


def _episode_rows(episodes: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        remaining = episode.diagnostics
        result = "win" if episode.reward == 1 else "loss" if episode.reward == -1 else "draw"
        rows.append({
            "result": result,
            "focal_prizes_taken": 6 - int(remaining["focal_prizes_remaining"]),
            "opponent_prizes_taken": 6 - int(remaining["opponent_prizes_remaining"]),
            "full_turns": (episode.turns + 1) // 2,
            "opponent_deck_id": episode.job.opponent_id,
            "opponent_policy_id": episode.job.opponent_policy_id,
            "focal_deck_id": episode.job.focal_deck_id,
            "coin_winner_seed": episode.job.coin_winner_seed,
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": remaining.get("first_player_choice"),
            "focal_first": (
                remaining["first_player_choice"]["focal_first"]
                if isinstance(remaining.get("first_player_choice"), dict)
                else None
            ),
            "branch": episode.diagnostics["sampling_branch"],
            "error": episode.error, "unfinished": not episode.valid,
            "engine_decisions": int(remaining.get("engine_selections", 0)),
        })
    return rows


def _checkpoint(
    model, update: int, *, version: str = VERSION,
    focal_deck_id: str = FOCAL_DECK_ID,
    focal_deck_ids: tuple[str, ...] | None = None,
    focal_schedule_mode: str = "fixed",
    source_parent_version: str | None = None,
    source_parent_update: int | None = None,
    parent_policy_id: str = "Champion-G2",
    parent_policy_update: int | None = 407,
) -> dict[str, Any]:
    return {
        "schema_version": "0045_minimal_lora_model_only_v1", "update": update,
        "actor_schema": "0031_rule_faithful_semantic_decision_v2",
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items() if value.requires_grad or not name.startswith("actor.") or name.startswith("actor.action_decoder.")},
        "adaptation": {
            "policy_only_option_lora": True,
            "option_block": 1,
            "attention_targets": ["self_attn.qv", "cross_attn.qv"],
            "rank": 4,
            "alpha": 8.0,
            "parameters": 10_240,
            "value_branch": "lora_free",
            "strategy_adapter": "removed",
            "meta_actor_residual": "removed",
        },
        "integrated_flags": model.integrated_flags.metadata(),
        "metadata": {
            "project_id": PROJECT, "version": version,
            "checkpoint_retention": "all", "own_archetype_class_count": 29,
            "opponent_meta_class_count": 15,
            "focal_initialization_deck_id": focal_deck_id,
            "focal_deck_id": focal_deck_id if focal_schedule_mode == "fixed" else None,
            "focal_deck_ids": list(focal_deck_ids or (focal_deck_id,)),
            "focal_schedule": (
                f"fixed_{focal_deck_id}_v1"
                if focal_schedule_mode == "fixed" else focal_schedule_mode
            ),
            "parent_policy_id": parent_policy_id,
            "parent_checkpoint_update": parent_policy_update,
            "source_parent_version": source_parent_version,
            "source_parent_update": source_parent_update,
        },
    }


def _validate_formal_deck_scope(
    registry: AssetRegistry, *, focal_deck_ids: tuple[str, ...],
    focal_deck_id: str, focal_schedule_mode: str,
) -> tuple[str, ...]:
    """Validate focal identities against the current formal training pool."""
    training_deck_ids = tuple(
        deck.deck_id for deck in registry.decks if "training" in deck.roles
    )
    if not focal_deck_ids or len(set(focal_deck_ids)) != len(focal_deck_ids):
        raise RuntimeError("formal focal deck IDs must be non-empty and unique")
    if any(deck_id not in training_deck_ids for deck_id in focal_deck_ids):
        raise RuntimeError("formal focal deck IDs escaped the 0045 training pool")
    if focal_deck_id not in training_deck_ids:
        raise RuntimeError("focal initialization deck escaped the 0045 training pool")
    if focal_schedule_mode == "fixed":
        if focal_deck_ids != (focal_deck_id,):
            raise RuntimeError("fixed-focal run requires one exact matching focal deck")
    elif focal_schedule_mode == "seeded_frequency_balanced_random_v1":
        pass
    elif focal_schedule_mode == "meta_balanced_001_067":
        if focal_deck_ids != tuple(f"{value:03d}" for value in range(1, 68)):
            raise RuntimeError("legacy generalist focal run requires exact pool 001-067")
    elif focal_schedule_mode == "meta_balanced_training_pool":
        if focal_deck_ids != training_deck_ids:
            raise RuntimeError("generalist focal run requires the exact training pool")
    else:
        raise RuntimeError("formal focal schedule mode is unsupported")
    return training_deck_ids


def run(
    *, updates: int | None, wandb_mode: str, launch_formal: bool,
    version: str = VERSION, start_update: int = 0,
    parent_checkpoint: Path | None = None, parent_pfsp_state: Path | None = None,
    reference_checkpoint: Path | None = None,
    baseline_evaluation_checkpoint: int | None = None,
    baseline_evaluation_provenance: Path | None = None,
    periodic_evaluation_enabled: bool = True,
    periodic_evaluation_profile: str = "benchmark_tiny_v2_core16_g2",
    periodic_evaluation_interval_updates: int = 5,
    wandb_run_id: str | None = None, wandb_name: str | None = None,
    focal_deck_ids: tuple[str, ...] = (FOCAL_DECK_ID,),
    focal_deck_id: str = FOCAL_DECK_ID,
    focal_schedule_mode: str = "fixed",
    evaluation_focal_deck_id: str | None = None,
    source_parent_version: str | None = None,
    source_parent_update: int | None = None,
    reference_anchor_update: int | None = None,
    reference_anchor_identity: str | None = None,
    allow_pristine_restart: bool = False,
    forward_microbatch_size: int = 256,
    behavior_probe_batch_size: int = 256,
    offload_reference_after_cache: bool = True,
    opponent_sampling_mode: str = "meta_balanced_training_pool",
    learning_rate_profile: LearningRateProfile | None = None,
    actor_learning_rate_scale: float | None = None,
    opponent_policy_ids: tuple[str, ...] = OPPONENT_POLICY_IDS,
    latest_champion_policy_id: str = "Champion-G2",
    rollout_games: int = 512,
    opponent_meta_weights: dict[int, float] | None = None,
    opponent_meta_quotas: dict[int, int] | None = None,
    ppo_batch_size: int = 4096,
    value_learning_rate: float | None = None,
    prize_learning_rate: float | None = None,
    entropy_coefficient: float | None = None,
    focal_base_checkpoint: Path | None = None,
) -> None:
    if not launch_formal:
        raise RuntimeError("formal 0045 PPO requires explicit --launch-formal")
    if periodic_evaluation_profile not in {
        "benchmark_tiny_v2_core16_g2",
        "policy0809_three_pool_cuda512",
    }:
        raise RuntimeError("unknown periodic evaluation profile")
    if periodic_evaluation_interval_updates <= 0:
        raise RuntimeError("periodic evaluation interval must be positive")
    if parent_checkpoint is None:
        raise RuntimeError("0045 formal PPO requires exact migrated Frozen-0045-Init U0")
    reference_checkpoint = reference_checkpoint or parent_checkpoint
    baseline_provenance: dict[str, Any] | None = None
    if baseline_evaluation_provenance is not None:
        if baseline_evaluation_checkpoint is not None:
            raise RuntimeError("baseline evaluation cannot be both rerun and reused")
        from ..evaluation.run_benchmark_tiny_v2_three_pool import validate_report

        baseline_provenance = json.loads(
            baseline_evaluation_provenance.read_text(encoding="utf-8")
        )
        validate_report(baseline_provenance)
        audit = baseline_provenance["focal_policy_identity_audit"]
        if (
            baseline_provenance.get("focal_checkpoint_update") != start_update
            or audit.get("source_checkpoint_sha256") != sha256_file(parent_checkpoint)
        ):
            raise RuntimeError("baseline provenance is not the exact parent checkpoint")
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "")):
        raise RuntimeError("set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("0045 formal run requires CUDA")
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    training_deck_ids = _validate_formal_deck_scope(
        registry,
        focal_deck_ids=focal_deck_ids,
        focal_deck_id=focal_deck_id,
        focal_schedule_mode=focal_schedule_mode,
    )
    evaluation_focal_deck_id = evaluation_focal_deck_id or focal_deck_id
    reference_anchor_update = (
        source_parent_update
        if reference_anchor_update is None else reference_anchor_update
    )
    reference_anchor_identity = reference_anchor_identity or "Frozen-0045-Init"
    if evaluation_focal_deck_id not in training_deck_ids:
        raise RuntimeError("evaluation focal deck escaped the 0045 training pool")
    focal_schedule = (
        f"fixed_{focal_deck_id}_v1"
        if focal_schedule_mode == "fixed" else focal_schedule_mode
    )
    if opponent_sampling_mode not in {
        "pfsp_mixture", "uniform_001_067", "meta_balanced_001_067",
        "meta_balanced_training_pool", "aggressive_meta_quota_training_pool",
    }:
        raise RuntimeError("formal opponent sampling mode is unsupported")
    if opponent_meta_weights is not None and opponent_sampling_mode not in {
        "meta_balanced_001_067", "meta_balanced_training_pool",
    }:
        raise RuntimeError("opponent Meta weights require a Meta-balanced schedule")
    if opponent_meta_quotas is not None and opponent_sampling_mode != (
        "aggressive_meta_quota_training_pool"
    ):
        raise RuntimeError("opponent Meta quotas require the aggressive schedule")
    if opponent_sampling_mode == "aggressive_meta_quota_training_pool":
        if opponent_meta_quotas is None or opponent_meta_weights is not None:
            raise RuntimeError("aggressive schedule requires quotas and forbids weights")
    if opponent_meta_weights is None and opponent_sampling_mode == "meta_balanced_training_pool":
        opponent_meta_weights = {0: 3.0, 1: 3.0, 2: 3.0, 3: 3.0, 5: 3.0, 27: 3.0}
    if opponent_sampling_mode in {
        "uniform_001_067", "meta_balanced_001_067",
        "meta_balanced_training_pool", "aggressive_meta_quota_training_pool",
    } and parent_pfsp_state is not None:
        raise RuntimeError("uniform training must not load a PFSP state")
    opponent_deck_ids = (
        tuple(f"{value:03d}" for value in range(1, 68))
        if opponent_sampling_mode in {"uniform_001_067", "meta_balanced_001_067"}
        else training_deck_ids
    )
    paths = _paths(version)
    used = [str(path) for path in paths.values() if path.exists() and any(path.iterdir())]
    pristine_restart = bool(
        used and allow_pristine_restart and parent_checkpoint is not None
        and _pristine_restart_allowed(
            paths, start_update=start_update, parent_checkpoint=parent_checkpoint,
            focal_deck_ids=focal_deck_ids,
        )
    )
    audited_u0_launch = bool(
        used and parent_checkpoint is not None
        and start_update == 0
        and _audited_u0_launch_allowed(paths, parent_checkpoint)
    )
    if used and not pristine_restart and not audited_u0_launch:
        raise FileExistsError(f"formal version paths are already used: {used}")
    for name in ("tensorboard", "wandb"):
        paths[name].mkdir(parents=True, exist_ok=True)
    focal_deck = _cards(registry, focal_deck_id)
    model, source = load_actor_critic(
        checkpoint=focal_base_checkpoint or DEFAULT_FOCAL_CHECKPOINT,
        deck=focal_deck, deck_id=focal_deck_id, device=device,
        integrated_flags=preset("FULL_MODEL"),
    )
    if parent_checkpoint is not None:
        parent = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
        if parent.get("schema_version") != "0045_minimal_lora_model_only_v1":
            raise RuntimeError("0045 requires an audited model-only parent checkpoint")
        incompatible = model.load_state_dict(parent["state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"0045 U0 checkpoint has unexpected tensors: {incompatible}")
    effective_lr_profile = resolve_learning_rate_profile(
        learning_rate_profile=learning_rate_profile,
        actor_learning_rate_scale=actor_learning_rate_scale,
        value_learning_rate=value_learning_rate,
        prize_learning_rate=prize_learning_rate,
    )
    trainer = PPOTrainer(
        model, device=device,
        config=_config(
            forward_microbatch_size=forward_microbatch_size,
            behavior_probe_batch_size=behavior_probe_batch_size,
            offload_reference_after_cache=offload_reference_after_cache,
            learning_rate_profile=effective_lr_profile,
            batch_size=ppo_batch_size,
            entropy_coefficient=entropy_coefficient,
        ),
    )
    if parent_checkpoint is not None:
        reference = torch.load(
            reference_checkpoint, map_location="cpu", weights_only=True
        )
        if reference.get("schema_version") != "0045_minimal_lora_model_only_v1":
            raise RuntimeError("0045 requires an audited model-only reference checkpoint")
        reference_model, _ = load_actor_critic(
            checkpoint=focal_base_checkpoint or DEFAULT_FOCAL_CHECKPOINT,
            deck=focal_deck, deck_id=focal_deck_id, device=device,
            integrated_flags=preset("FULL_MODEL"),
        )
        reference_incompatible = reference_model.load_state_dict(
            reference["state_dict"], strict=False
        )
        if reference_incompatible.unexpected_keys:
            raise RuntimeError(
                "0045 reference U0 has unexpected tensors: "
                f"{reference_incompatible}"
            )
        trainer.set_reference_model(reference_model)
        del reference_model
    opponents = {
        policy_id: _load_opponent(policy_id, "001", device)
        for policy_id in opponent_policy_ids
    }
    os.environ.update({
        "WANDB_MODE": wandb_mode, "WANDB_ENTITY": "dragon_bra",
        "WANDB_PROJECT": "pokemon-tcg-policy-learning",
        "WANDB_RUN_ID": wandb_run_id or _run_id(),
        "WANDB_NAME": wandb_name or "0045 · V1 · deck 007 minimal LoRA",
        "WANDB_RUN_GROUP": PROJECT, "WANDB_DIR": str(paths["wandb"]),
        "WANDB_JOB_TYPE": "ppo_champion_league",
    })
    history = RolloutHistory()
    pfsp_state = (
        PFSPState.load(parent_pfsp_state)
        if parent_pfsp_state is not None else PFSPState()
    )
    cumulative_decisions = 0
    status = paths["artifact"] / "status.json"
    effective_run_id = wandb_run_id or _run_id()
    _atomic_json(paths["artifact"] / "training_config.json", {
        "schema_version": "0045_ppo_training_config_v1",
        "project_id": PROJECT,
        "version": version,
        "configured_update_limit": updates,
        "start_update": start_update,
        "rollout_games": rollout_games,
        "focal_deck_ids": list(focal_deck_ids),
        "focal_deck_id": focal_deck_id if focal_schedule_mode == "fixed" else None,
        "focal_initialization_deck_id": focal_deck_id,
        "focal_schedule": focal_schedule,
        "opponent_deck_ids": list(opponent_deck_ids),
        "opponent_policy_ids": list(opponent_policy_ids),
        "opponent_sampling": (
            {"pfsp": 0, "uniform": rollout_games, "latest": 0}
            if opponent_sampling_mode in {
                "uniform_001_067", "meta_balanced_001_067",
                "meta_balanced_training_pool", "aggressive_meta_quota_training_pool",
            }
            else {"pfsp": 128, "uniform": 64, "latest": 64}
        ),
        "opponent_sampling_mode": opponent_sampling_mode,
        "opponent_meta_weights": opponent_meta_weights,
        "opponent_meta_quotas": opponent_meta_quotas,
        "first_player_contract": {
            "id": "seeded_toss_winner_agent_context_41_choice_v1",
            "schedule_controls": "coin_winner_seed_only",
            "actual_seat_source": "complete_toss_winning_policy_context_41_choice",
            "actual_seat_quota": None,
            "hard_gate": True,
        },
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "parent_checkpoint_sha256": (
            sha256_file(parent_checkpoint) if parent_checkpoint else None
        ),
        "reference_checkpoint": str(reference_checkpoint),
        "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
        "source_parent_version": source_parent_version,
        "source_parent_update": source_parent_update,
        "parent_pfsp_state": str(parent_pfsp_state) if parent_pfsp_state else None,
        "parent_pfsp_state_sha256": (
            sha256_file(parent_pfsp_state) if parent_pfsp_state else None
        ),
        "optimizer_initialization": "fresh",
        "learning_rate_profile": effective_lr_profile.metadata(),
        "ppo": asdict(_config(
            forward_microbatch_size=forward_microbatch_size,
            behavior_probe_batch_size=behavior_probe_batch_size,
            offload_reference_after_cache=offload_reference_after_cache,
            learning_rate_profile=effective_lr_profile,
            batch_size=ppo_batch_size,
            entropy_coefficient=entropy_coefficient,
        )),
        "reference_anchor_policy_id": reference_anchor_identity,
        "reference_anchor_update": reference_anchor_update,
        "reference_anchor_checkpoint_sha256": (
            sha256_file(reference_checkpoint)
        ),
        "baseline_evaluation_provenance": (
            {
                "report": str(baseline_evaluation_provenance),
                "report_sha256": sha256_file(baseline_evaluation_provenance),
                "checkpoint_update": baseline_provenance["focal_checkpoint_update"],
                "deployment_effective_sha256": baseline_provenance[
                    "focal_deployment_effective_sha256"
                ],
                "status": baseline_provenance["status"],
            }
            if baseline_provenance is not None else None
        ),
        "periodic_evaluation": {
            "enabled": periodic_evaluation_enabled,
            "profile": periodic_evaluation_profile,
            "contract_id": (
                "0045_policy0809_three_meta_pools_common_seeds_cuda512_v1"
                if periodic_evaluation_profile == "policy0809_three_pool_cuda512"
                else "0045_benchmark_tiny_v2_core16_meta_balanced_common_seeds_cuda512_v1"
            ),
            "interval_updates": (
                periodic_evaluation_interval_updates
                if periodic_evaluation_enabled else None
            ),
            "games_per_pool": 512 if periodic_evaluation_enabled else 0,
            "pool_count": (
                3 if periodic_evaluation_enabled
                and periodic_evaluation_profile == "policy0809_three_pool_cuda512"
                else int(periodic_evaluation_enabled)
            ),
            "games": (
                1536 if periodic_evaluation_enabled
                and periodic_evaluation_profile == "policy0809_three_pool_cuda512"
                else 512 if periodic_evaluation_enabled else 0
            ),
            "opponent_policy_id": (
                "Policy-0809"
                if periodic_evaluation_profile == "policy0809_three_pool_cuda512"
                else "Champion-G2"
            ),
            "focal_deck_id": evaluation_focal_deck_id,
            "role": "longitudinal_sentinel",
            "wandb_namespace": "eval",
            "updates_pfsp": False,
        },
        "checkpoint_retention": "all",
        "checkpoint_contents": "model_only",
        "cuda_engine": "2.0",
        "wandb": {
            "entity": "dragon_bra",
            "project": "pokemon-tcg-policy-learning",
            "run_id": effective_run_id,
            "mode": wandb_mode,
        },
    })
    _atomic_json(status, {
        "state": "running", "checkpoint_update": start_update,
        "wandb_run_id": effective_run_id,
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "reference_anchor_policy_id": reference_anchor_identity,
        "reference_anchor_update": reference_anchor_update,
        "focal_deck_ids": list(focal_deck_ids),
        "focal_deck_id": focal_deck_id if focal_schedule_mode == "fixed" else None,
        "focal_initialization_deck_id": focal_deck_id,
        "focal_schedule": focal_schedule,
        "source_parent_version": source_parent_version,
        "source_parent_update": source_parent_update,
        "opponent_sampling_mode": opponent_sampling_mode,
    })
    initial_checkpoint = paths["checkpoint"] / f"update-{start_update:06d}.pt"
    if not audited_u0_launch:
        _atomic_torch(
            initial_checkpoint,
            _checkpoint(
                model, start_update, version=version, focal_deck_id=focal_deck_id,
                focal_deck_ids=focal_deck_ids,
                focal_schedule_mode=focal_schedule_mode,
                source_parent_version=source_parent_version,
                source_parent_update=source_parent_update,
                parent_policy_id=latest_champion_policy_id,
                parent_policy_update=reference_anchor_update,
            ),
        )
    if opponent_sampling_mode == "pfsp_mixture":
        pfsp_state.save(paths["artifact"] / "pfsp_state.json")
    with TrainingLogger(paths["artifact"] / "training_metrics.jsonl", paths["tensorboard"]) as logger:
        logger.initialize_wandb({
            "trainer/update": start_update,
            "checkpoint/update": start_update,
        })
        if baseline_evaluation_checkpoint is not None:
            if not periodic_evaluation_enabled:
                raise RuntimeError(
                    "baseline evaluation cannot run when periodic evaluation is disabled"
                )
            if baseline_evaluation_checkpoint != start_update:
                raise RuntimeError("baseline evaluation must bind the exact start checkpoint")
            if periodic_evaluation_profile == "policy0809_three_pool_cuda512":
                from ..evaluation.run_benchmark_tiny_v2_three_pool import (
                    run as run_periodic_evaluation,
                )
                from .periodic_evaluation import (
                    wandb_metrics_three_pool as periodic_metrics,
                )
            else:
                from ..evaluation.run_benchmark_tiny_v2 import (
                    run as run_periodic_evaluation,
                )
                periodic_metrics = periodic_evaluation_metrics

            evaluation_root = (
                paths["artifact"] / "periodic_evaluation"
                / f"update-{start_update:06d}"
            )
            report_path = evaluation_root / "report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = run_periodic_evaluation(
                    deck_id=evaluation_focal_deck_id,
                    output_root=evaluation_root,
                    checkpoint=paths["checkpoint"] / f"update-{start_update:06d}.pt",
                    checkpoint_update=start_update,
                )
            periodic_kwargs = {"initial_baseline_update": start_update}
            if periodic_evaluation_profile == "policy0809_three_pool_cuda512":
                periodic_kwargs["interval_updates"] = periodic_evaluation_interval_updates
            eval_metrics = periodic_metrics(report, **periodic_kwargs)
            logger.log(start_update, eval_metrics)
        update = start_update
        while updates is None or update < updates:
            jobs, deck_weights, policy_weights, curriculum_version = _jobs(
                update, registry, pfsp_state=pfsp_state,
                focal_deck_ids=focal_deck_ids,
                focal_deck_id=focal_deck_id,
                focal_schedule_mode=focal_schedule_mode,
                opponent_sampling_mode=opponent_sampling_mode,
                opponent_policy_ids=opponent_policy_ids,
                latest_champion_policy_id=latest_champion_policy_id,
                rollout_games=rollout_games,
                opponent_meta_weights=opponent_meta_weights,
                opponent_meta_quotas=opponent_meta_quotas,
            )
            by_group = _group_jobs_by_opponent_policy(jobs)
            if opponent_sampling_mode in {
                "meta_balanced_001_067", "meta_balanced_training_pool",
                "aggressive_meta_quota_training_pool",
            }:
                branch = {
                    int(job.game_id.rsplit("-", 1)[1]): (
                        "aggressive_meta_quota"
                        if opponent_sampling_mode == "aggressive_meta_quota_training_pool"
                        else "meta_balanced_uniform"
                    )
                    for job in jobs
                }
            else:
                schedule_builder = (
                    build_uniform_schedule
                    if opponent_sampling_mode == "uniform_001_067"
                    else build_schedule
                )
                schedule_curriculum = (
                    _uniform_curriculum(
                        update, tuple(f"{i:03d}" for i in range(1, 68)),
                        opponent_policy_ids,
                    )
                    if opponent_sampling_mode == "uniform_001_067"
                    else pfsp_state.curricula[curriculum_version]
                )
                league = schedule_builder(
                    update=update, seed=430044001 + update,
                    deck_ids=tuple(f"{i:03d}" for i in range(1, 68)),
                    policy_ids=opponent_policy_ids,
                    latest_champion_policy_id=latest_champion_policy_id,
                    curriculum=schedule_curriculum,
                )
                branch = {row.lane_id: row.branch for row in league}
            episodes, runtime_rows = [], []
            for policy_id, group in sorted(by_group.items()):
                opponent, audit = opponents[policy_id]
                own_ids = None
                if policy_id.startswith("Champion-G"):
                    vocabulary = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=PROJECT_ROOT)
                    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
                    own_ids = torch.tensor([own_by_deck[job.opponent_id] for job in group], device=device)
                collector = ChunkedCudaRolloutCollector(
                    model, opponent, rollout_batch_size=len(group), trajectory_games_per_update=len(group),
                    device=device, rules_path=RULES, extension_dir=DEFAULT_BUILD_DIR,
                    lane_count=len(group), mode="sample", record_trajectory=True,
                    agent_selects_first_player=True,
                    opponent_policy_id=policy_id, opponent_identity_audit=audit,
                    opponent_own_archetype_ids=own_ids,
                )
                part = collector.collect(group)
                for episode in part:
                    lane = int(episode.job.game_id.rsplit("-", 1)[1])
                    episode.diagnostics["sampling_branch"] = branch[lane]
                episodes.extend(part)
                runtime_rows.append(collector.metrics())
            if len(episodes) != rollout_games or any(not episode.valid for episode in episodes):
                raise RuntimeError(
                    f"formal rollout did not return {rollout_games} valid terminal episodes"
                )
            if opponent_sampling_mode == "pfsp_mixture":
                for episode in episodes:
                    result = "win" if episode.reward == 1 else "loss" if episode.reward == -1 else "draw"
                    pfsp_state.observe(
                        deck_id=episode.job.opponent_id,
                        policy_id=episode.job.opponent_policy_id,
                        result=result, update=update,
                    )
            batch = prepare_episodes(
                episodes, gamma=1.0, gae_lambda=0.95, credit_clock="turn",
                loss_weighting="episode_equal_decisions", prize_mode="directional",
                require_policy_identity=True,
            )
            ppo = trainer.update(batch, update=update + 1)
            model.set_runtime_own_archetype_ids(None)
            checkpoint_update = update + 1
            _atomic_torch(
                paths["checkpoint"] / f"update-{checkpoint_update:06d}.pt",
                _checkpoint(
                    model, checkpoint_update, version=version,
                    focal_deck_id=focal_deck_id,
                    focal_deck_ids=focal_deck_ids,
                    focal_schedule_mode=focal_schedule_mode,
                    source_parent_version=source_parent_version,
                    source_parent_update=source_parent_update,
                    parent_policy_id=latest_champion_policy_id,
                    parent_policy_update=reference_anchor_update,
                ),
            )
            # PFSP observations become canonical only after the corresponding
            # PPO update and checkpoint succeed. A failed PPO attempt must not
            # bias a future version's opponent curriculum.
            if opponent_sampling_mode == "pfsp_mixture":
                pfsp_state.save(paths["artifact"] / "pfsp_state.json")
            runtime_metrics: dict[str, float] = {}
            additive = {
                "rollout/deck_static_cache_hits",
                "rollout/deck_static_cache_misses",
                "rollout/lane_routing_audit_failures",
                "rollout/preseat_choices_excluded_from_ppo",
            }
            for key in set().union(*(row.keys() for row in runtime_rows)):
                values = [row[key] for row in runtime_rows if key in row]
                runtime_metrics[key] = sum(values) if key in additive else min(values) if key in {"rollout/cuda_features_device_resident", "rollout/lane_routing_audit_pass"} else sum(values) / len(values)
            runtime_metrics["rollout/policy_weight_loads"] = 1.0
            # Collector groups are an execution detail; only the same 256-game
            # aggregate is authoritative for rollout W/L/D and chronology.
            for key in (
                "rollout/wins", "rollout/losses", "rollout/draws",
                "rollout/win_rate", "rollout/source_policy_update",
            ):
                runtime_metrics.pop(key, None)
            metrics = aggregate_training_rollout(
                _episode_rows(episodes), source_policy_update=update,
                checkpoint_update=checkpoint_update, curriculum_version=curriculum_version,
                deck_weights=deck_weights, policy_weights=policy_weights,
                history=history, runtime_metrics=runtime_metrics,
                sampling_mode=opponent_sampling_mode,
                expected_games=rollout_games,
            )
            metrics.update(ppo)
            if opponent_sampling_mode == "aggressive_meta_quota_training_pool":
                vocabulary = OwnArchetypeVocabulary.load_version(
                    "own_archetypes_v2", project_root=PROJECT_ROOT
                )
                meta_by_deck = {
                    row.deck_id: row.archetype_id for row in vocabulary.mappings
                }
                realized_meta_counts: dict[int, int] = defaultdict(int)
                for job in jobs:
                    realized_meta_counts[meta_by_deck[job.opponent_id]] += 1
                for meta_id, games in sorted(realized_meta_counts.items()):
                    metrics[f"sampling/opponent_meta/{meta_id:02d}/games"] = games
            metrics["trainer/update"] = checkpoint_update
            metrics["env/episodes"] = checkpoint_update * rollout_games
            cumulative_decisions += sum(len(ep.policy_transitions) for ep in episodes)
            metrics["env/decisions"] = cumulative_decisions
            logger.log(checkpoint_update, metrics)
            if periodic_evaluation_enabled and periodic_evaluation_due(
                checkpoint_update,
                interval_updates=periodic_evaluation_interval_updates,
            ):
                # Run only after the model-only checkpoint is durable. This is
                # synchronous by design so a missing/failed Benchmark V2 cannot
                # be silently skipped while the formal run advances.
                if periodic_evaluation_profile == "policy0809_three_pool_cuda512":
                    from ..evaluation.run_benchmark_tiny_v2_three_pool import (
                        run as run_periodic_evaluation,
                    )
                    from .periodic_evaluation import (
                        wandb_metrics_three_pool as periodic_metrics,
                    )
                else:
                    from ..evaluation.run_benchmark_tiny_v2 import (
                        run as run_periodic_evaluation,
                    )
                    periodic_metrics = periodic_evaluation_metrics

                evaluation_root = (
                    paths["artifact"] / "periodic_evaluation"
                    / f"update-{checkpoint_update:06d}"
                )
                report = run_periodic_evaluation(
                    deck_id=evaluation_focal_deck_id,
                    output_root=evaluation_root,
                    checkpoint=paths["checkpoint"] / f"update-{checkpoint_update:06d}.pt",
                    checkpoint_update=checkpoint_update,
                )
                periodic_kwargs = {}
                if periodic_evaluation_profile == "policy0809_three_pool_cuda512":
                    periodic_kwargs["interval_updates"] = periodic_evaluation_interval_updates
                eval_metrics = periodic_metrics(report, **periodic_kwargs)
                logger.log(checkpoint_update, eval_metrics)
            _atomic_json(status, {"state": "running", "checkpoint_update": checkpoint_update, "rollout_source_policy_update": update, "wandb_run_id": effective_run_id, "reference_anchor_policy_id": reference_anchor_identity, "reference_anchor_update": reference_anchor_update, "focal_deck_ids": list(focal_deck_ids), "focal_deck_id": focal_deck_id if focal_schedule_mode == "fixed" else None, "focal_initialization_deck_id": focal_deck_id, "focal_schedule": focal_schedule, "evaluation_focal_deck_id": evaluation_focal_deck_id, "source_parent_version": source_parent_version, "source_parent_update": source_parent_update})
            update = checkpoint_update
    if updates is not None and update >= updates:
        completed = json.loads(status.read_text(encoding="utf-8"))
        completed.update({
            "state": "complete",
            "checkpoint_update": update,
            "completion_reason": "configured_update_limit_reached",
        })
        _atomic_json(status, completed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--readiness-output", type=Path)
    parser.add_argument("--u0-checkpoint", type=Path)
    args = parser.parse_args()
    if not args.launch_formal:
        result = readiness()
        if args.readiness_output:
            _atomic_json(args.readiness_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.u0_checkpoint is None:
        parser.error("--u0-checkpoint is required for formal 0045 training")
    run(
        updates=args.updates, wandb_mode=args.wandb_mode, launch_formal=True,
        parent_checkpoint=args.u0_checkpoint.resolve(),
        baseline_evaluation_checkpoint=0,
        source_parent_version="0044_V24_final_handoff",
        source_parent_update=21,
        reference_anchor_update=0,
        reference_anchor_identity="Frozen-0045-Init",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
