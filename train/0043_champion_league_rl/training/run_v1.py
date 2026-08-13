"""Self-contained 0043 V1 CUDA-2.0 PPO entrypoint.

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
from ..league.pfsp import PFSPConfig, PFSPState
from ..league.sampler import build_schedule
from ..own_archetype import OwnArchetypeVocabulary
from ..policy import AdaptationConfig, load_actor_critic
from ..policy_identity import materialize_policy_bundle
from ..rollout import DEFAULT_FULL_ROUND_DRAW_LIMIT, ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import load_policy
from ..telemetry import RolloutHistory, aggregate_training_rollout
from .batch_full_semantic import prepare_episodes
from .ppo_full_semantic import PPOConfig, PPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
PROJECT = "0043_champion_league_rl"
VERSION = "V1_focal_002_007"
RULES = ROOT / ".tmp/cuda_0032_rules/official_rules.bin"
CONTRACT = ROOT / "experiments/0043_champion_league_rl/initial_run_contract.json"


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


def _config() -> PPOConfig:
    return PPOConfig(
        decoder_learning_rate=2e-5,
        policy_adapter_learning_rate=4e-5,
        allocation_learning_rate=2e-5,
        value_learning_rate=1e-4,
        prize_learning_rate=1e-4,
    )


def _paths(version: str = VERSION) -> dict[str, Path]:
    root = ROOT / "rl_runs" / PROJECT / "versions" / version
    return {name: root / name for name in ("artifact", "checkpoint", "tensorboard", "wandb")}


def _run_id() -> str:
    return "0043-v1-focal-002-007"


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
    seed = paths["checkpoint"] / "update-000000.pt"
    portable = paths["artifact"] / "focal_seed/model.bin"
    missing = [str(path) for path in (RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so", seed, portable) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"0043 launch artifacts missing: {missing}")
    payload = torch.load(seed, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    meta = state["value_head.heads.archetype.1.weight"]
    own = (state["value_adapter.own_embedding.weight"], state["policy_strategy_adapter.own_embedding.weight"])
    if meta.shape != (15, 320) or any(row.shape != (29, 16) for row in own):
        raise RuntimeError("0043 own/opponent taxonomy tensor boundary changed")
    if registry.active_policy_ids != ("Policy-0809", "Champion-G1"):
        raise RuntimeError("0043 active opponent pool changed")
    return {
        "schema_version": "0043_v1_launch_readiness_v1",
        "status": "READY_AWAITING_USER_LAUNCH",
        "long_training_started": False,
        "launch_token_required": "--launch-formal",
        "project": PROJECT, "version": VERSION,
        "asset_audit": asdict(audit),
        "focal_decks": ["002", "007"],
        "opponent_decks": ["001", "067"],
        "opponent_policies": list(registry.active_policy_ids),
        "own_taxonomy": {"version": vocabulary.taxonomy_version, "classes": 29, "embedding_width": 16},
        "opponent_meta": {"classes": 15, "trainable": False},
        "cuda_engine": {"version": "2.0", "extension_sha256": sha256_file(DEFAULT_BUILD_DIR / "_ptcg_cuda.so")},
        "wandb": {"entity": "dragon_bra", "project": "pokemon-tcg-policy-learning", "run_id": _run_id()},
    }


def _jobs(
    update: int, registry: AssetRegistry, *, pfsp_state: PFSPState | None = None,
    focal_deck_ids: tuple[str, ...] = ("002", "007"),
) -> tuple[list[RolloutJob], dict[str, float], dict[str, float], str]:
    deck_ids = tuple(deck.deck_id for deck in registry.decks if "training" in deck.roles)
    policy_ids = registry.active_policy_ids
    state = PFSPState() if pfsp_state is None else pfsp_state
    curriculum = state.curriculum_for(
        update, deck_ids=deck_ids, policy_ids=policy_ids, config=PFSPConfig(),
        training_deck_pool_hash=hashlib.sha256("".join(deck_ids).encode()).hexdigest(),
        active_policy_pool_hash=hashlib.sha256("".join(policy_ids).encode()).hexdigest(),
    )
    league = build_schedule(
        update=update, seed=430043001 + update, deck_ids=deck_ids,
        policy_ids=policy_ids, latest_champion_policy_id="Champion-G1",
        curriculum=curriculum,
    )
    focal = balanced_focal_schedule(
        430043711 + update, deck_ids=focal_deck_ids, lanes=len(league)
    )
    own = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=PROJECT_ROOT)
    own_by_deck = {row.deck_id: row.archetype_id for row in own.mappings}
    runtime = build_seeded_runtime()
    root = _runtime_root()
    jobs = [
        RolloutJob(
            game_id=f"rollout-u{update:06d}-{lane.lane_id:03d}",
            opponent_id=lane.opponent_deck_id,
            focal_first=lane.focal_goes_first, seed=lane.engine_seed,
            source_policy_update=update,
            focal_deck=_cards(registry, focal[lane.lane_id].deck_id),
            opponent_deck=_cards(registry, lane.opponent_deck_id),
            runtime_root=root, opponent_policy_id=lane.opponent_policy_id,
            focal_deck_id=focal[lane.lane_id].deck_id,
            focal_own_archetype_id=own_by_deck[focal[lane.lane_id].deck_id],
            policy_seed=lane.policy_seed, search_seed=lane.search_seed,
            engine_library=runtime.library_path,
            full_round_draw_limit=DEFAULT_FULL_ROUND_DRAW_LIMIT,
            action_boundary_mode="enabled",
        )
        for lane in league
    ]
    return jobs, dict(curriculum.deck_weights), dict(curriculum.policy_weights), curriculum.curriculum_version


def _load_opponent(policy_id: str, deck_id: str, device: torch.device):
    bundle = materialize_policy_bundle(PROJECT_ROOT, policy_id, purpose="0043_formal_rollout")
    policy = load_policy(policy_id, deck_id=deck_id)
    modules = (policy.model,) if hasattr(policy, "model") else (
        policy.actor, policy.value_head, policy.allocation_head,
        policy.value_adapter, policy.policy_strategy_adapter,
    )
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
            "branch": episode.diagnostics["sampling_branch"],
            "error": episode.error, "unfinished": not episode.valid,
            "engine_decisions": int(remaining.get("engine_selections", 0)),
        })
    return rows


def _checkpoint(model, update: int, *, version: str = VERSION) -> dict[str, Any]:
    return {
        "schema_version": "0043_focal_v1_model_only_v1", "update": update,
        "actor_schema": "0031_rule_faithful_semantic_decision_v2",
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items() if value.requires_grad or not name.startswith("actor.") or name.startswith("actor.action_decoder.")},
        "adaptation": {"no_option_lora": True},
        "integrated_flags": model.integrated_flags.metadata(),
        "metadata": {"project_id": PROJECT, "version": version, "checkpoint_retention": "all", "own_archetype_class_count": 29, "opponent_meta_class_count": 15},
    }


def run(
    *, updates: int | None, wandb_mode: str, launch_formal: bool,
    version: str = VERSION, start_update: int = 0,
    parent_checkpoint: Path | None = None, parent_pfsp_state: Path | None = None,
    wandb_run_id: str | None = None, wandb_name: str | None = None,
    focal_deck_ids: tuple[str, ...] = ("002", "007"),
) -> None:
    if not launch_formal:
        raise RuntimeError("formal 0043 PPO requires explicit --launch-formal")
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "")):
        raise RuntimeError("set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("0043 formal run requires CUDA")
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    training_deck_ids = tuple(
        deck.deck_id for deck in registry.decks if "training" in deck.roles
    )
    if not focal_deck_ids or len(set(focal_deck_ids)) != len(focal_deck_ids):
        raise RuntimeError("formal focal deck IDs must be non-empty and unique")
    if any(deck_id not in training_deck_ids for deck_id in focal_deck_ids):
        raise RuntimeError("formal focal deck IDs escaped the 0043 training pool")
    paths = _paths(version)
    used = [str(path) for path in paths.values() if path.exists() and any(path.iterdir())]
    if used:
        raise FileExistsError(f"formal version paths are already used: {used}")
    for name in ("tensorboard", "wandb"):
        paths[name].mkdir(parents=True, exist_ok=True)
    deck002 = _cards(registry, "002")
    model, source = load_actor_critic(deck=deck002, device=device, integrated_flags=preset("FULL_MODEL"))
    if parent_checkpoint is not None:
        parent = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
        if parent.get("schema_version") != "0043_focal_v1_model_only_v1":
            raise RuntimeError("unsupported 0043 parent checkpoint")
        incompatible = model.load_state_dict(parent["state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"0043 parent checkpoint has unexpected tensors: {incompatible}")
    trainer = PPOTrainer(model, device=device, config=_config())
    if parent_checkpoint is not None:
        reference_model, _ = load_actor_critic(
            deck=deck002, device=device, integrated_flags=preset("FULL_MODEL")
        )
        trainer.set_reference_model(reference_model)
        del reference_model
    opponents = {
        policy_id: _load_opponent(policy_id, "001", device)
        for policy_id in registry.active_policy_ids
    }
    os.environ.update({
        "WANDB_MODE": wandb_mode, "WANDB_ENTITY": "dragon_bra",
        "WANDB_PROJECT": "pokemon-tcg-policy-learning",
        "WANDB_RUN_ID": wandb_run_id or _run_id(),
        "WANDB_NAME": wandb_name or "0043 · V1 focal 002/007",
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
        "schema_version": "0043_ppo_training_config_v1",
        "project_id": PROJECT,
        "version": version,
        "start_update": start_update,
        "rollout_games": 256,
        "focal_deck_ids": list(focal_deck_ids),
        "focal_schedule": "seeded_frequency_balanced_random_v1",
        "opponent_deck_ids": list(training_deck_ids),
        "opponent_policy_ids": list(registry.active_policy_ids),
        "opponent_sampling": {"pfsp": 128, "uniform": 64, "latest": 64},
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "parent_checkpoint_sha256": (
            sha256_file(parent_checkpoint) if parent_checkpoint else None
        ),
        "parent_pfsp_state": str(parent_pfsp_state) if parent_pfsp_state else None,
        "parent_pfsp_state_sha256": (
            sha256_file(parent_pfsp_state) if parent_pfsp_state else None
        ),
        "optimizer_initialization": "fresh",
        "ppo": asdict(_config()),
        "reference_anchor_update": 0,
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
        "reference_anchor_update": 0,
        "focal_deck_ids": list(focal_deck_ids),
        "focal_schedule": "seeded_frequency_balanced_random_v1",
    })
    if parent_checkpoint is not None:
        _atomic_torch(
            paths["checkpoint"] / f"update-{start_update:06d}.pt",
            _checkpoint(model, start_update, version=version),
        )
        pfsp_state.save(paths["artifact"] / "pfsp_state.json")
    with TrainingLogger(paths["artifact"] / "training_metrics.jsonl", paths["tensorboard"]) as logger:
        logger.initialize_wandb({
            "trainer/update": start_update,
            "checkpoint/update": start_update,
        })
        update = start_update
        while updates is None or update < updates:
            jobs, deck_weights, policy_weights, curriculum_version = _jobs(
                update, registry, pfsp_state=pfsp_state,
                focal_deck_ids=focal_deck_ids,
            )
            by_group = _group_jobs_by_opponent_policy(jobs)
            league = build_schedule(
                update=update, seed=430043001 + update,
                deck_ids=tuple(f"{i:03d}" for i in range(1, 68)),
                policy_ids=registry.active_policy_ids,
                latest_champion_policy_id="Champion-G1",
                curriculum=pfsp_state.curricula[curriculum_version],
            )
            branch = {row.lane_id: row.branch for row in league}
            episodes, runtime_rows = [], []
            for policy_id, group in sorted(by_group.items()):
                opponent, audit = opponents[policy_id]
                own_ids = None
                if policy_id == "Champion-G1":
                    vocabulary = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=PROJECT_ROOT)
                    parent = {row.deck_id: vocabulary.classes[row.archetype_id].embedding_init_from for row in vocabulary.mappings}
                    own_ids = torch.tensor([parent[job.opponent_id] for job in group], device=device)
                collector = ChunkedCudaRolloutCollector(
                    model, opponent, rollout_batch_size=len(group), trajectory_games_per_update=len(group),
                    device=device, rules_path=RULES, extension_dir=DEFAULT_BUILD_DIR,
                    lane_count=len(group), mode="sample", record_trajectory=True,
                    opponent_policy_id=policy_id, opponent_identity_audit=audit,
                    opponent_own_archetype_ids=own_ids,
                )
                part = collector.collect(group)
                for episode in part:
                    lane = int(episode.job.game_id.rsplit("-", 1)[1])
                    episode.diagnostics["sampling_branch"] = branch[lane]
                episodes.extend(part)
                runtime_rows.append(collector.metrics())
            if len(episodes) != 256 or any(not episode.valid for episode in episodes):
                raise RuntimeError("formal rollout did not return 256 valid terminal episodes")
            for episode in episodes:
                result = "win" if episode.reward == 1 else "loss" if episode.reward == -1 else "draw"
                pfsp_state.observe(
                    deck_id=episode.job.opponent_id,
                    policy_id=episode.job.opponent_policy_id,
                    result=result, update=update,
                )
            pfsp_state.save(paths["artifact"] / "pfsp_state.json")
            batch = prepare_episodes(
                episodes, gamma=1.0, gae_lambda=0.95, credit_clock="turn",
                loss_weighting="episode_equal_decisions", prize_mode="directional",
                require_policy_identity=True,
            )
            ppo = trainer.update(batch, update=update + 1)
            model.set_runtime_own_archetype_ids(None)
            checkpoint_update = update + 1
            _atomic_torch(paths["checkpoint"] / f"update-{checkpoint_update:06d}.pt", _checkpoint(model, checkpoint_update, version=version))
            runtime_metrics: dict[str, float] = {}
            additive = {"rollout/deck_static_cache_hits", "rollout/deck_static_cache_misses", "rollout/lane_routing_audit_failures"}
            for key in set().union(*(row.keys() for row in runtime_rows)):
                values = [row[key] for row in runtime_rows if key in row]
                runtime_metrics[key] = sum(values) if key in additive else min(values) if key in {"rollout/cuda_features_device_resident", "rollout/lane_routing_audit_pass"} else sum(values) / len(values)
            runtime_metrics["rollout/policy_weight_loads"] = 3.0
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
            )
            metrics.update(ppo)
            metrics["trainer/update"] = checkpoint_update
            metrics["env/episodes"] = checkpoint_update * 256
            cumulative_decisions += sum(len(ep.policy_transitions) for ep in episodes)
            metrics["env/decisions"] = cumulative_decisions
            logger.log(checkpoint_update, metrics)
            _atomic_json(status, {"state": "running", "checkpoint_update": checkpoint_update, "rollout_source_policy_update": update, "wandb_run_id": effective_run_id, "reference_anchor_update": 0, "focal_deck_ids": list(focal_deck_ids), "focal_schedule": "seeded_frequency_balanced_random_v1"})
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
