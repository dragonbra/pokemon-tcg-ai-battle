"""Formal indefinite 0047 deck-007 Meta-routed MoE PPO run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from rl_environment.logging import TrainingLogger

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..evaluation.moe_three_pool import evaluate_model, wandb_metrics
from ..integrated.presets import preset
from ..policy.moe_actor_critic import load_moe_actor_critic, save_router_table
from ..policy_identity import materialize_policy_bundle
from ..rollout.cuda_collector import ChunkedCudaRolloutCollector
from ..rollout.moe_runtime import MoEFocalRuntime, MoEResidentRouter
from ..runtime import _modules as policy_modules, load_policy
from ..telemetry import RolloutHistory, aggregate_training_rollout
from .batch_full_semantic import prepare_episodes
from .moe_checkpoint import (
    RETENTION_POLICY,
    atomic_save_compact_checkpoint,
    load_compact_checkpoint,
    prune_non_eval_checkpoints,
)
from .ppo_moe import PPOConfig, PPOTrainer
from .run_v1 import PROJECT_ROOT, ROOT, RULES, _cards, _jobs


PROJECT = "0047_meta_routed_moe_rl"
VERSION = "V16_deck007_soft_moe_memory_fix_resume_u5"
VERSION_ROOT = ROOT / "rl_runs" / PROJECT / "versions" / VERSION
RUN_ID = "0047-v16-deck007-soft-moe-memory-fix-resume-u5"
PARENT_VERSION = "V15_deck007_public_meta29_router29x7_cuda512_micro256"
PARENT_UPDATE = 5
PARENT_CHECKPOINT = (
    ROOT / "rl_runs" / PROJECT / "versions" / PARENT_VERSION
    / "checkpoint" / "update-000005.pt"
)
PARENT_CHECKPOINT_SHA256 = "e033c2cbc63f9f33e1eae582d8407493688694bcda6ea3c40097492cff930d78"
FOCAL_DECK_ID = "007"
FOCAL_OWN_ARCHETYPE_ID = 0
WARMUP_UPDATES = 5
ROLLOUT_GAMES = 512
EVAL_INTERVAL = 5
EVAL_AT_U0 = False
PRIZE_AUX_ACTOR_WEIGHT = 0.0
CUDA_LANES = 512
ROLLOUT_CHUNK_GAMES = 512
PPO_FORWARD_MICROBATCH = 256
CORE_OPPONENT_DECK_IDS = ("007", "003", "001", "002", "009", "011", "023")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _checkpoint_metadata(model) -> dict[str, Any]:
    return {
        "project_id": PROJECT, "version": VERSION,
        "focal_deck_id": FOCAL_DECK_ID,
        "focal_own_archetype_id": FOCAL_OWN_ARCHETYPE_ID,
        "source_policy_id": "Policy-0814",
        "experts": 7, "expert_labels": list(model.expert_labels),
        "priority_meta_ids": list(model.priority_meta_ids),
        "router_topology": "public_meta29_lookup_softmax_29x7",
        "warmup_updates": WARMUP_UPDATES,
        "routing_phase": "soft" if model.soft_routing else "hard_warmup",
        "router_probabilities": model.router_table(),
        "meta_identifier_version": "0047_public_meta29_priority_rules_v1",
        "retention_authorization": "explicit_user_authorization_2026-08-16",
    }


def _episode_rows(episodes):
    rows = []
    for episode in episodes:
        d = episode.diagnostics
        rows.append({
            "result": "win" if episode.reward == 1 else "loss" if episode.reward == -1 else "draw",
            "focal_prizes_taken": 6 - int(d["focal_prizes_remaining"]),
            "opponent_prizes_taken": 6 - int(d["opponent_prizes_remaining"]),
            "full_turns": (episode.turns + 1) // 2,
            "opponent_deck_id": episode.job.opponent_id,
            "opponent_policy_id": episode.job.opponent_policy_id,
            "focal_deck_id": FOCAL_DECK_ID, "branch": "meta_balanced_uniform",
            "error": episode.error, "unfinished": not episode.valid,
            "engine_decisions": int(d.get("engine_selections", 0)),
            "coin_winner_seed": episode.job.coin_winner_seed,
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": d["first_player_choice"],
            "focal_first": d["first_player_choice"]["focal_first"],
        })
    return rows


def _router_metrics(model, telemetry):
    output = {
        "router/unknown_decision_fraction": telemetry["unknown_decision_fraction"],
        "router/mean_identification_decision": telemetry["mean_identification_decision"],
        "router/identified_game_fraction": telemetry["identified_game_fraction"],
        "router/classification_change_total": float(
            sum(telemetry["classification_change_count"])
        ),
        "router/soft_phase": float(model.soft_routing),
    }
    for label, value in zip(model.expert_labels, telemetry["effective_usage"], strict=True):
        output[f"router/effective_usage/{label}"] = value
    probabilities = model.router_table()
    for row in probabilities:
        for expert, value in enumerate(row["probabilities"]):
            output[f"router/meta_{row['meta_id']}/{model.expert_labels[expert]}"] = value
    return output


def _combine_router_telemetry(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        raise RuntimeError("0047 rollout produced no Router telemetry chunks")
    combined = {
        name: [item for chunk in chunks for item in chunk[name]]
        for name in (
            "current_meta", "first_identification_decision", "decision_count",
            "classification_change_count", "seen_trigger_card_ids",
        )
    }
    total_decisions = sum(int(chunk["total_decisions"]) for chunk in chunks)
    combined["effective_usage"] = [
        sum(
            float(chunk["effective_usage"][index]) * int(chunk["total_decisions"])
            for chunk in chunks
        ) / max(1, total_decisions)
        for index in range(7)
    ]
    combined["unknown_decision_fraction"] = sum(
        float(chunk["unknown_decision_fraction"]) * int(chunk["total_decisions"])
        for chunk in chunks
    ) / max(1, total_decisions)
    identified_timings = [
        value for value in combined["first_identification_decision"] if value >= 0
    ]
    combined.update({
        "identifier_version": chunks[0]["identifier_version"],
        "identified_game_fraction": len(identified_timings)
        / max(1, len(combined["current_meta"])),
        "mean_identification_decision": sum(identified_timings)
        / max(1, len(identified_timings)),
        "total_decisions": total_decisions,
    })
    return combined


def _router_heatmap(model, update: int, path: Path) -> Path:
    import matplotlib.pyplot as plt
    probabilities = [row["probabilities"] for row in model.router_table()]
    figure, axis = plt.subplots(figsize=(8, 12))
    image = axis.imshow(
        probabilities, vmin=0, vmax=1, aspect="auto", cmap="viridis"
    )
    axis.set_xticks(range(model.expert_count), model.expert_labels)
    axis.set_yticks(range(29), [f"{i:02d}" for i in range(29)])
    axis.set_title(f"0047 Router · U{update}")
    figure.colorbar(image, ax=axis)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def readiness() -> dict[str, Any]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    audit = registry.validate_all()
    required = [
        RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
        PROJECT_ROOT / "assets/policies/definitions/policy_0814/model.pt",
        PROJECT_ROOT / "assets/policies/definitions/policy_0814/value_head.pt",
        PARENT_CHECKPOINT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return {
        "status": "READY", "project": PROJECT, "version": VERSION,
        "focal_deck_id": FOCAL_DECK_ID,
        "focal_own_archetype_id": FOCAL_OWN_ARCHETYPE_ID,
        "opponent_policy_id": "Policy-0814",
        "rollout_games": ROLLOUT_GAMES, "warmup_updates": WARMUP_UPDATES,
        "opponent_sampling_mode": "core_deck_half_meta_balanced_half",
        "core_opponent_deck_ids": list(CORE_OPPONENT_DECK_IDS),
        "core_opponent_fraction": 0.5,
        "cuda_lanes": CUDA_LANES,
        "rollout_chunk_games": ROLLOUT_CHUNK_GAMES,
        "ppo_forward_microbatch_size": PPO_FORWARD_MICROBATCH,
        "eval_interval": EVAL_INTERVAL,
        "eval_at_u0": EVAL_AT_U0,
        "parent_version": PARENT_VERSION,
        "parent_update": PARENT_UPDATE,
        "parent_checkpoint": str(PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "optimizer_initialization": "fresh",
        "checkpoint_contents": "fp32_effective_delta_only",
        "checkpoint_retention": RETENTION_POLICY,
        "retention_authorization": "explicit_user_authorization_2026-08-16",
        "actor_advantage": "terminal_win_loss_only",
        "prize_aux_actor_weight": PRIZE_AUX_ACTOR_WEIGHT,
        "eval_pools": {
            "focus_seven": {
                "games": 512,
                "old_three": {"games": 256, "decks": ["001", "002", "011"]},
                "new_four": {"games": 256, "decks": ["007", "003", "009", "023"]},
            },
            "remain_meta": "all_exact_decks_except_the_seven_focus_decks",
        },
        "asset_audit": asdict(audit),
        "cuda_extension_sha256": sha256_file(DEFAULT_BUILD_DIR / "_ptcg_cuda.so"),
        "wandb_run_id": RUN_ID,
    }


def run(*, updates: int | None, wandb_mode: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("0047 formal PPO requires CUDA")
    paths = {name: VERSION_ROOT / name for name in ("artifact", "checkpoint", "tensorboard", "wandb")}
    metrics_path = paths["artifact"] / "training_metrics.jsonl"
    if VERSION_ROOT.exists() and any(VERSION_ROOT.rglob("*")):
        raise FileExistsError(f"0047 formal version path is already used: {VERSION_ROOT}")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    device = torch.device("cuda:0")
    deck = _cards(registry, FOCAL_DECK_ID)
    model, source = load_moe_actor_critic(
        deck=deck, own_archetype_id=FOCAL_OWN_ARCHETYPE_ID, device=device,
        integrated_flags=preset("WIN_ONLY_ACTOR"),
    )
    parent_payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("0047 V16 parent checkpoint hash mismatch")
    if int(parent_payload.get("update", -1)) != PARENT_UPDATE:
        raise RuntimeError("0047 V16 parent checkpoint update mismatch")
    load_compact_checkpoint(model, parent_payload)
    opponent_bundle = materialize_policy_bundle(
        PROJECT_ROOT, "Policy-0814", purpose="0047_formal_rollout"
    )
    opponent = load_policy("Policy-0814", deck_id="001")
    for module in policy_modules(opponent):
        module.to(device).eval().requires_grad_(False)
    focal_ptr = {p.untyped_storage().data_ptr() for p in model.parameters()}
    opponent_ptr = {
        p.untyped_storage().data_ptr()
        for module in policy_modules(opponent) for p in module.parameters()
    }
    if focal_ptr & opponent_ptr:
        raise RuntimeError("FATAL: focal/opponent mutable storage alias")
    config = PPOConfig(
        forward_microbatch_size=PPO_FORWARD_MICROBATCH,
        prize_aux_actor_weight=PRIZE_AUX_ACTOR_WEIGHT,
    )
    trainer = PPOTrainer(model, device=device, config=config)
    reference_model, _ = load_moe_actor_critic(
        deck=deck, own_archetype_id=FOCAL_OWN_ARCHETYPE_ID, device="cpu",
        integrated_flags=preset("WIN_ONLY_ACTOR"),
    )
    trainer.set_reference_model(reference_model)
    del reference_model
    os.environ.update({
        "WANDB_MODE": wandb_mode, "WANDB_ENTITY": "dragon_bra",
        "WANDB_PROJECT": "pokemon-tcg-policy-learning", "WANDB_RUN_ID": RUN_ID,
        "WANDB_NAME": "0047 · V16 · deck 007 · soft-MoE memory fix · resume U5",
        "WANDB_RUN_GROUP": PROJECT, "WANDB_DIR": str(paths["wandb"]),
        "WANDB_JOB_TYPE": "ppo_meta_routed_moe",
    })
    _atomic_json(paths["artifact"] / "training_config.json", {
        **readiness(), "ppo": asdict(config), "source_identity": asdict(source),
        "reference_anchor": "Frozen-0047-U0",
        "checkpoint_retention": RETENTION_POLICY,
        "checkpoint_contents": "fp32_effective_delta_only",
        "retention_authorization": "explicit_user_authorization_2026-08-16",
        "configured_update_limit": updates, "wandb_mode": wandb_mode,
    })
    _atomic_json(paths["artifact"] / "status.json", {
        "state": "running", "checkpoint_update": PARENT_UPDATE,
        "parent_version": PARENT_VERSION, "parent_update": PARENT_UPDATE,
        "wandb_run_id": RUN_ID,
    })
    checkpoint_path = atomic_save_compact_checkpoint(
        paths["checkpoint"] / f"update-{PARENT_UPDATE:06d}.pt", model, PARENT_UPDATE,
        metadata=_checkpoint_metadata(model),
    )
    save_router_table(
        model, paths["checkpoint"] / f"router-{PARENT_UPDATE:06d}.json"
    )
    history = RolloutHistory()
    update = PARENT_UPDATE
    with TrainingLogger(metrics_path, paths["tensorboard"]) as logger:
        logger.initialize_wandb({
            "trainer/update": PARENT_UPDATE,
            "checkpoint/update": PARENT_UPDATE,
        })
        if EVAL_AT_U0:
            eval_root = paths["artifact"] / "evaluation" / "update-000000"
            report = evaluate_model(
                model, checkpoint_update=0,
                source_checkpoint=checkpoint_path,
                output_root=eval_root, games_per_pool=512,
            )
            u0_metrics = {
                "trainer/update": 0,
                "checkpoint/update": 0,
                **wandb_metrics(report),
            }
            heatmap = _router_heatmap(
                model, 0,
                paths["artifact"] / "router_heatmaps" / "update-000000.png",
            )
            try:
                import wandb
                wandb.log(
                    {"router/meta_expert_heatmap": wandb.Image(str(heatmap))},
                    step=0,
                )
            except Exception as error:
                u0_metrics["router/heatmap_wandb_error"] = str(error)
            logger.log(0, u0_metrics)
            _atomic_json(paths["artifact"] / "status.json", {
                "state": "running", "checkpoint_update": 0,
                "routing_phase": "hard_warmup", "u0_evaluation": "PASS",
                "wandb_run_id": RUN_ID, "last_metric_time": time.time(),
            })
        while updates is None or update < updates:
            if update == WARMUP_UPDATES:
                model.set_soft_routing(True)
            jobs, deck_weights, policy_weights, curriculum = _jobs(
                update, registry, focal_deck_ids=(FOCAL_DECK_ID,),
                focal_deck_id=FOCAL_DECK_ID,
                opponent_sampling_mode="core_deck_half_meta_balanced_half",
                opponent_policy_ids=("Policy-0814",),
                latest_champion_policy_id="Policy-0814",
                rollout_games=ROLLOUT_GAMES,
                core_deck_ids=CORE_OPPONENT_DECK_IDS,
            )
            runtime = MoEFocalRuntime(model, job_count=ROLLOUT_CHUNK_GAMES, device=device)
            router_chunks: list[dict[str, Any]] = []

            def before_chunk(begin: int, stop: int) -> None:
                runtime.reset(stop - begin)

            def after_chunk(begin: int, stop: int, _collector: Any) -> None:
                del begin, stop, _collector
                router_chunks.append(runtime.telemetry())
                runtime.contexts.clear()

            collector = ChunkedCudaRolloutCollector(
                model, opponent, device=device, rules_path=RULES,
                rollout_batch_size=ROLLOUT_CHUNK_GAMES,
                before_chunk=before_chunk, after_chunk=after_chunk,
                extension_dir=DEFAULT_BUILD_DIR, lane_count=CUDA_LANES, mode="sample",
                record_trajectory=True, agent_selects_first_player=True,
                opponent_policy_id="Policy-0814",
                opponent_identity_audit=opponent_bundle.audit,
                role_compacted=True,
                focal_compacted_policy_fn=runtime.decode_compacted,
                focal_allocation_planner_for_job=runtime.plan_allocation,
                focal_resident_router_cls=MoEResidentRouter,
            )
            episodes = collector.collect(jobs)
            if len(episodes) != ROLLOUT_GAMES or any(not row.valid for row in episodes):
                raise RuntimeError("0047 formal rollout is incomplete")
            collector_metrics = collector.metrics()
            runtime_telemetry = _combine_router_telemetry(router_chunks)
            runtime.contexts.clear()
            del collector, runtime
            torch.cuda.empty_cache()
            batch = prepare_episodes(
                episodes, gamma=1.0, gae_lambda=0.95, credit_clock="turn",
                loss_weighting="episode_equal_decisions", prize_mode="directional",
                require_policy_identity=True,
            )
            ppo = trainer.update(batch, update=update + 1)
            checkpoint_update = update + 1
            checkpoint_path = paths["checkpoint"] / f"update-{checkpoint_update:06d}.pt"
            atomic_save_compact_checkpoint(
                checkpoint_path, model, checkpoint_update,
                metadata=_checkpoint_metadata(model),
            )
            save_router_table(model, paths["checkpoint"] / f"router-{checkpoint_update:06d}.json")
            rows = _episode_rows(episodes)
            rollout = aggregate_training_rollout(
                rows, source_policy_update=update, checkpoint_update=checkpoint_update,
                curriculum_version=curriculum, deck_weights=deck_weights,
                policy_weights=policy_weights, history=history,
                runtime_metrics={**collector_metrics, "rollout/policy_weight_loads": 1.0},
                sampling_mode="meta_balanced_training_pool", expected_games=ROLLOUT_GAMES,
            )
            router = _router_metrics(model, runtime_telemetry)
            metrics = {"trainer/update": checkpoint_update, **rollout, **ppo, **router}
            if checkpoint_update % EVAL_INTERVAL == 0:
                eval_root = paths["artifact"] / "evaluation" / f"update-{checkpoint_update:06d}"
                report = evaluate_model(
                    model, checkpoint_update=checkpoint_update,
                    source_checkpoint=checkpoint_path,
                    output_root=eval_root, games_per_pool=512,
                )
                metrics.update(wandb_metrics(report))
                heatmap = _router_heatmap(
                    model, checkpoint_update,
                    paths["artifact"] / "router_heatmaps" / f"update-{checkpoint_update:06d}.png",
                )
                try:
                    import wandb
                    wandb.log({"router/meta_expert_heatmap": wandb.Image(str(heatmap))}, step=checkpoint_update)
                except Exception as error:
                    metrics["router/heatmap_wandb_error"] = str(error)
            logger.log(checkpoint_update, metrics)
            removed_checkpoints: list[str] = []
            if checkpoint_update % EVAL_INTERVAL == 0:
                removed_checkpoints = [
                    path.name for path in prune_non_eval_checkpoints(
                        paths["checkpoint"], through_update=checkpoint_update,
                        eval_interval=EVAL_INTERVAL,
                        evaluation_status=report["status"],
                    )
                ]
            _atomic_json(paths["artifact"] / "status.json", {
                "state": "running", "checkpoint_update": checkpoint_update,
                "routing_phase": "soft" if model.soft_routing else "hard_warmup",
                "checkpoint_retention": RETENTION_POLICY,
                "last_pruned_checkpoints": removed_checkpoints,
                "wandb_run_id": RUN_ID, "last_metric_time": time.time(),
            })
            model.set_runtime_own_archetype_ids(None)
            update = checkpoint_update


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    if not args.launch_formal:
        print(json.dumps(readiness(), indent=2, sort_keys=True))
        return 0
    try:
        run(updates=args.updates, wandb_mode=args.wandb_mode)
    except Exception as error:
        status_path = VERSION_ROOT / "artifact" / "status.json"
        current = {}
        if status_path.is_file():
            current = json.loads(status_path.read_text(encoding="utf-8"))
        _atomic_json(status_path, {
            **current,
            "state": "failed",
            "failure_type": type(error).__name__,
            "failure": str(error),
            "failed_at": time.time(),
        })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
