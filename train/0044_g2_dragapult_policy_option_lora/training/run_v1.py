"""Self-contained 0044 V1 CUDA-2.0 PPO entrypoint.

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
from ..league.meta_balanced import balanced_meta_deck_schedule
from ..league.sampler import LeagueLane, _seed, build_schedule, build_uniform_schedule
from ..own_archetype import OwnArchetypeVocabulary
from ..policy import AdaptationConfig, load_actor_critic
from ..policy.actor_critic import DEFAULT_FOCAL_CHECKPOINT
from ..policy_identity import materialize_policy_bundle
from ..rollout import DEFAULT_FULL_ROUND_DRAW_LIMIT, ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import load_policy
from ..telemetry import RolloutHistory, aggregate_training_rollout
from .batch_full_semantic import prepare_episodes
from .ppo_full_semantic import PPOConfig, PPOTrainer
from .periodic_evaluation import is_due as periodic_evaluation_due
from .periodic_evaluation import wandb_metrics as periodic_evaluation_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
OPPONENT_POLICY_IDS = ("Champion-G2",)
FOCAL_DECK_ID = "007"
PROJECT = "0044_g2_dragapult_policy_option_lora"
VERSION = "V1_g2_dragapult_policy_option_lora"
RULES = ROOT / ".tmp/cuda_0032_rules/official_rules.bin"
CONTRACT = ROOT / "experiments/0044_g2_dragapult_policy_option_lora/initial_run_contract.json"


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
    actor_learning_rate_scale: float = 1.0,
    batch_size: int = 2048,
    value_learning_rate: float = 1.0e-4,
    prize_learning_rate: float = 1.0e-4,
    meta_actor_residual_learning_rate: float = 5.0e-6,
) -> PPOConfig:
    if actor_learning_rate_scale <= 0:
        raise ValueError("actor learning-rate scale must be positive")
    return PPOConfig(
        decoder_learning_rate=1e-5 * actor_learning_rate_scale,
        policy_adapter_learning_rate=1e-5 * actor_learning_rate_scale,
        allocation_learning_rate=1e-5 * actor_learning_rate_scale,
        option_lora_learning_rate=2e-5 * actor_learning_rate_scale,
        meta_actor_residual_learning_rate=meta_actor_residual_learning_rate,
        value_learning_rate=value_learning_rate,
        prize_learning_rate=prize_learning_rate,
        batch_size=batch_size,
        forward_microbatch_size=forward_microbatch_size,
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
    if not metrics.is_file() or metrics.stat().st_size != 0 or not config.is_file():
        return False
    if checkpoints != [paths["checkpoint"] / f"update-{start_update:06d}.pt"]:
        return False
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    except (OSError, ValueError, KeyError):
        return False
    return (
        checkpoint.get("update") == start_update
        and payload.get("start_update") == start_update
        and payload.get("parent_checkpoint_sha256") == sha256_file(parent_checkpoint)
        and payload.get("focal_deck_ids") == list(focal_deck_ids)
        and payload.get("optimizer_initialization") == "fresh"
    )


def _run_id() -> str:
    return "0044-v1-g2-dragapult-policy-option-lora"


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
    source = PROJECT_ROOT / "assets/policies/definitions/champion_g002/source_update_000407.pt"
    missing = [str(path) for path in (RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so", g2, source) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"0044 launch artifacts missing: {missing}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    meta = state["value_head.heads.archetype.1.weight"]
    own = (state["value_adapter.own_embedding.weight"], state["policy_strategy_adapter.own_embedding.weight"])
    if meta.shape != (15, 320) or any(row.shape != (29, 16) for row in own):
        raise RuntimeError("0044 own/opponent taxonomy tensor boundary changed")
    if "Champion-G2" not in {row.policy_id for row in registry.policies}:
        raise RuntimeError("0044 V1 requires registered immutable Champion-G2")
    return {
        "schema_version": "0044_v1_launch_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "long_training_started": False,
        "launch_token_required": "--launch-formal",
        "project": PROJECT, "version": VERSION,
        "asset_audit": asdict(audit),
        "focal_decks": [FOCAL_DECK_ID],
        "opponent_decks": ["001", "067"],
        "opponent_policies": list(OPPONENT_POLICY_IDS),
        "opponent_sampling": {"pfsp": 128, "uniform": 64, "latest_champion": 64},
        "periodic_evaluation": {"contract": "Benchmark-V2", "every_updates": 10, "cuda_games": 2048},
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
) -> tuple[list[RolloutJob], dict[str, float], dict[str, float], str]:
    deck_ids = tuple(deck.deck_id for deck in registry.decks if "training" in deck.roles)
    policy_ids = opponent_policy_ids
    if opponent_sampling_mode in {"uniform_001_067", "meta_balanced_001_067"}:
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
    if schedule_builder is None and opponent_sampling_mode != "meta_balanced_001_067":
        raise ValueError(f"unsupported opponent sampling mode: {opponent_sampling_mode}")
    if opponent_sampling_mode == "meta_balanced_001_067":
        if policy_ids != (latest_champion_policy_id,):
            raise ValueError("Meta-balanced opponent schedule requires one latest champion")
        own = OwnArchetypeVocabulary.load_version(
            "own_archetypes_v2", project_root=PROJECT_ROOT
        )
        quota_seed = 440_120_000 + update
        opponent_slots = balanced_meta_deck_schedule(
            quota_seed=quota_seed,
            shuffle_seed=440_120_200 + update,
            mappings=own.mappings,
            lanes=rollout_games,
        )
        master = 430_044_001 + update
        league = tuple(
            LeagueLane(
                lane_id=row.lane_id, branch="meta_balanced_uniform",
                opponent_deck_id=row.deck_id,
                opponent_policy_id=latest_champion_policy_id,
                seat_slot=row.lane_id,
                coin_winner_seed=(coin_seed := _seed(master, row.lane_id, "coin-winner")),
                focal_won_toss=bool(coin_seed & 1),
                engine_seed=_seed(master, row.lane_id, "engine"),
                search_seed=_seed(master, row.lane_id, "search"),
                policy_seed=_seed(master, row.lane_id, "policy"),
                curriculum_version=f"meta-balanced-u{update:06d}",
            )
            for row in opponent_slots
        )
        curriculum = Curriculum(
            **{
                **asdict(curriculum),
                "curriculum_version": f"meta-balanced-u{update:06d}",
                "stats_snapshot": {
                    "sampling_mode": "meta_balanced_001_067",
                    "pfsp_enabled": False,
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
            raise ValueError("fixed-focal 0044 run requires one matching focal deck ID")
        focal_by_lane = tuple(focal_deck_id for _ in league)
    elif focal_schedule_mode == "seeded_frequency_balanced_random_v1":
        focal_by_lane = tuple(
            row.deck_id for row in balanced_focal_schedule(
                430044711 + update, deck_ids=focal_deck_ids, lanes=len(league)
            )
        )
    elif focal_schedule_mode == "meta_balanced_001_067":
        if focal_deck_ids != deck_ids:
            raise ValueError("Meta-balanced focal schedule requires exact pool 001-067")
        own = OwnArchetypeVocabulary.load_version(
            "own_archetypes_v2", project_root=PROJECT_ROOT
        )
        focal_by_lane = tuple(
            row.deck_id for row in balanced_meta_deck_schedule(
                quota_seed=440_120_000 + update,
                shuffle_seed=440_120_100 + update,
                mappings=own.mappings,
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
    bundle = materialize_policy_bundle(PROJECT_ROOT, policy_id, purpose="0044_formal_rollout")
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
        "schema_version": "0044_focal_v1_model_only_v1", "update": update,
        "actor_schema": "0031_rule_faithful_semantic_decision_v2",
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items() if value.requires_grad or not name.startswith("actor.") or name.startswith("actor.action_decoder.")},
        "adaptation": {
            "no_option_lora": False,
            "policy_only_option_lora": True,
            "option_block": 1,
            "attention_targets": ["self_attn.qv", "cross_attn.qv"],
            "rank": 4,
            "alpha": 8.0,
            "parameters": 10_240,
            "value_branch": "lora_free",
            "meta_actor_residual": {
                "scope": "policy_only", "classes": 29, "rank": 4,
                "parameters": 74_240,
            },
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


def run(
    *, updates: int | None, wandb_mode: str, launch_formal: bool,
    version: str = VERSION, start_update: int = 0,
    parent_checkpoint: Path | None = None, parent_pfsp_state: Path | None = None,
    baseline_evaluation_checkpoint: int | None = None,
    periodic_evaluation_enabled: bool = True,
    wandb_run_id: str | None = None, wandb_name: str | None = None,
    focal_deck_ids: tuple[str, ...] = (FOCAL_DECK_ID,),
    focal_deck_id: str = FOCAL_DECK_ID,
    focal_schedule_mode: str = "fixed",
    evaluation_focal_deck_id: str | None = None,
    source_parent_version: str | None = None,
    source_parent_update: int | None = None,
    reference_anchor_update: int | None = None,
    allow_pristine_restart: bool = False,
    forward_microbatch_size: int = 1024,
    opponent_sampling_mode: str = "pfsp_mixture",
    actor_learning_rate_scale: float = 1.0,
    opponent_policy_ids: tuple[str, ...] = OPPONENT_POLICY_IDS,
    latest_champion_policy_id: str = "Champion-G2",
    rollout_games: int = 256,
    ppo_batch_size: int = 2048,
    value_learning_rate: float = 1.0e-4,
    prize_learning_rate: float = 1.0e-4,
    meta_actor_residual_learning_rate: float = 5.0e-6,
    focal_base_checkpoint: Path | None = None,
) -> None:
    if not launch_formal:
        raise RuntimeError("formal 0044 PPO requires explicit --launch-formal")
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "")):
        raise RuntimeError("set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("0044 formal run requires CUDA")
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    training_deck_ids = tuple(
        deck.deck_id for deck in registry.decks if "training" in deck.roles
    )
    if not focal_deck_ids or len(set(focal_deck_ids)) != len(focal_deck_ids):
        raise RuntimeError("formal focal deck IDs must be non-empty and unique")
    if any(deck_id not in training_deck_ids for deck_id in focal_deck_ids):
        raise RuntimeError("formal focal deck IDs escaped the 0044 training pool")
    if focal_deck_id not in training_deck_ids:
        raise RuntimeError("focal initialization deck escaped the 0044 training pool")
    if focal_schedule_mode == "fixed":
        if focal_deck_ids != (focal_deck_id,):
            raise RuntimeError("fixed-focal run requires one exact matching focal deck")
    elif focal_schedule_mode in {
        "seeded_frequency_balanced_random_v1", "meta_balanced_001_067"
    }:
        if focal_deck_ids != training_deck_ids:
            raise RuntimeError("generalist focal run requires exact ordered pool 001-067")
    else:
        raise RuntimeError("formal focal schedule mode is unsupported")
    evaluation_focal_deck_id = evaluation_focal_deck_id or focal_deck_id
    reference_anchor_update = (
        source_parent_update
        if reference_anchor_update is None else reference_anchor_update
    )
    if evaluation_focal_deck_id not in training_deck_ids:
        raise RuntimeError("evaluation focal deck escaped the 0044 training pool")
    focal_schedule = (
        f"fixed_{focal_deck_id}_v1"
        if focal_schedule_mode == "fixed" else focal_schedule_mode
    )
    if opponent_sampling_mode not in {
        "pfsp_mixture", "uniform_001_067", "meta_balanced_001_067"
    }:
        raise RuntimeError("formal opponent sampling mode is unsupported")
    if opponent_sampling_mode in {
        "uniform_001_067", "meta_balanced_001_067"
    } and parent_pfsp_state is not None:
        raise RuntimeError("uniform training must not load a PFSP state")
    paths = _paths(version)
    used = [str(path) for path in paths.values() if path.exists() and any(path.iterdir())]
    pristine_restart = bool(
        used and allow_pristine_restart and parent_checkpoint is not None
        and _pristine_restart_allowed(
            paths, start_update=start_update, parent_checkpoint=parent_checkpoint,
            focal_deck_ids=focal_deck_ids,
        )
    )
    if used and not pristine_restart:
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
        if parent.get("schema_version") != "0044_focal_v1_model_only_v1":
            raise RuntimeError("unsupported 0044 parent checkpoint")
        incompatible = model.load_state_dict(parent["state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"0044 parent checkpoint has unexpected tensors: {incompatible}")
    trainer = PPOTrainer(
        model, device=device,
        config=_config(
            forward_microbatch_size=forward_microbatch_size,
            actor_learning_rate_scale=actor_learning_rate_scale,
            batch_size=ppo_batch_size,
            value_learning_rate=value_learning_rate,
            prize_learning_rate=prize_learning_rate,
            meta_actor_residual_learning_rate=meta_actor_residual_learning_rate,
        ),
    )
    if parent_checkpoint is not None:
        reference_model, _ = load_actor_critic(
            checkpoint=focal_base_checkpoint or DEFAULT_FOCAL_CHECKPOINT,
            deck=focal_deck, deck_id=focal_deck_id, device=device,
            integrated_flags=preset("FULL_MODEL"),
        )
        reference_incompatible = reference_model.load_state_dict(
            parent["state_dict"], strict=False
        )
        if reference_incompatible.unexpected_keys:
            raise RuntimeError(
                "0044 reference parent has unexpected tensors: "
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
        "WANDB_NAME": wandb_name or "0044 · V1 G2 → 007 · policy-only Option LoRA",
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
        "schema_version": "0044_ppo_training_config_v1",
        "project_id": PROJECT,
        "version": version,
        "start_update": start_update,
        "rollout_games": rollout_games,
        "focal_deck_ids": list(focal_deck_ids),
        "focal_deck_id": focal_deck_id if focal_schedule_mode == "fixed" else None,
        "focal_initialization_deck_id": focal_deck_id,
        "focal_schedule": focal_schedule,
        "opponent_deck_ids": list(training_deck_ids),
        "opponent_policy_ids": list(opponent_policy_ids),
        "opponent_sampling": (
            {"pfsp": 0, "uniform": rollout_games, "latest": 0}
            if opponent_sampling_mode in {"uniform_001_067", "meta_balanced_001_067"}
            else {"pfsp": 128, "uniform": 64, "latest": 64}
        ),
        "opponent_sampling_mode": opponent_sampling_mode,
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
        "source_parent_version": source_parent_version,
        "source_parent_update": source_parent_update,
        "parent_pfsp_state": str(parent_pfsp_state) if parent_pfsp_state else None,
        "parent_pfsp_state_sha256": (
            sha256_file(parent_pfsp_state) if parent_pfsp_state else None
        ),
        "optimizer_initialization": "fresh",
        "ppo": asdict(_config(
            forward_microbatch_size=forward_microbatch_size,
            actor_learning_rate_scale=actor_learning_rate_scale,
            batch_size=ppo_batch_size,
            value_learning_rate=value_learning_rate,
            prize_learning_rate=prize_learning_rate,
            meta_actor_residual_learning_rate=meta_actor_residual_learning_rate,
        )),
        "reference_anchor_policy_id": latest_champion_policy_id,
        "reference_anchor_update": reference_anchor_update,
        "periodic_evaluation": {
            "enabled": periodic_evaluation_enabled,
            "contract_id": "0044_benchmark_v2_core16_meta_balanced_policy0809_common_seeds_cuda2048_v2",
            "interval_updates": 10 if periodic_evaluation_enabled else None,
            "games": 2048 if periodic_evaluation_enabled else 0,
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
        "reference_anchor_policy_id": latest_champion_policy_id,
        "reference_anchor_update": reference_anchor_update,
        "focal_deck_ids": list(focal_deck_ids),
        "focal_deck_id": focal_deck_id if focal_schedule_mode == "fixed" else None,
        "focal_initialization_deck_id": focal_deck_id,
        "focal_schedule": focal_schedule,
        "source_parent_version": source_parent_version,
        "source_parent_update": source_parent_update,
        "opponent_sampling_mode": opponent_sampling_mode,
    })
    _atomic_torch(
        paths["checkpoint"] / f"update-{start_update:06d}.pt",
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
            from ..evaluation.run_benchmark_v2 import run as run_benchmark_v2

            evaluation_root = (
                paths["artifact"] / "periodic_evaluation"
                / f"update-{start_update:06d}"
            )
            report_path = evaluation_root / "report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = run_benchmark_v2(
                    deck_id=evaluation_focal_deck_id,
                    output_root=evaluation_root,
                    checkpoint=paths["checkpoint"] / f"update-{start_update:06d}.pt",
                    checkpoint_update=start_update,
                )
            eval_metrics = periodic_evaluation_metrics(
                report, initial_baseline_update=start_update,
            )
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
            )
            by_group = _group_jobs_by_opponent_policy(jobs)
            if opponent_sampling_mode == "meta_balanced_001_067":
                branch = {
                    int(job.game_id.rsplit("-", 1)[1]): "meta_balanced_uniform"
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
            metrics["trainer/update"] = checkpoint_update
            metrics["env/episodes"] = checkpoint_update * rollout_games
            cumulative_decisions += sum(len(ep.policy_transitions) for ep in episodes)
            metrics["env/decisions"] = cumulative_decisions
            logger.log(checkpoint_update, metrics)
            if periodic_evaluation_enabled and periodic_evaluation_due(checkpoint_update):
                # Run only after the model-only checkpoint is durable. This is
                # synchronous by design so a missing/failed Benchmark V2 cannot
                # be silently skipped while the formal run advances.
                from ..evaluation.run_benchmark_v2 import run as run_benchmark_v2

                evaluation_root = (
                    paths["artifact"] / "periodic_evaluation"
                    / f"update-{checkpoint_update:06d}"
                )
                report = run_benchmark_v2(
                    deck_id=evaluation_focal_deck_id,
                    output_root=evaluation_root,
                    checkpoint=paths["checkpoint"] / f"update-{checkpoint_update:06d}.pt",
                    checkpoint_update=checkpoint_update,
                )
                eval_metrics = periodic_evaluation_metrics(report)
                logger.log(checkpoint_update, eval_metrics)
            _atomic_json(status, {"state": "running", "checkpoint_update": checkpoint_update, "rollout_source_policy_update": update, "wandb_run_id": effective_run_id, "reference_anchor_policy_id": latest_champion_policy_id, "reference_anchor_update": reference_anchor_update, "focal_deck_ids": list(focal_deck_ids), "focal_deck_id": focal_deck_id if focal_schedule_mode == "fixed" else None, "focal_initialization_deck_id": focal_deck_id, "focal_schedule": focal_schedule, "evaluation_focal_deck_id": evaluation_focal_deck_id, "source_parent_version": source_parent_version, "source_parent_update": source_parent_update})
            update = checkpoint_update


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--readiness-output", type=Path)
    args = parser.parse_args()
    if not args.launch_formal:
        result = readiness()
        if args.readiness_output:
            _atomic_json(args.readiness_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    run(updates=args.updates, wandb_mode=args.wandb_mode, launch_formal=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
