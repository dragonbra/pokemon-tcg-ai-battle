"""Fixed two-pool official CUDA evaluation for a deployable 0047 snapshot."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from evaluation.runtime.seeded import build_seeded_runtime

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..policy.moe_actor_critic import MetaRoutedMoEActorCritic, load_moe_actor_critic
from ..policy_identity import materialize_policy_bundle
from ..rollout.cuda_collector import CudaFullSemanticRolloutCollector
from ..rollout.moe_runtime import MoEFocalRuntime, MoEResidentRouter
from ..rollout.protocol import RolloutJob
from ..runtime import _modules as policy_modules, load_policy
from ..training.run_v1 import PROJECT_ROOT, RULES, _cards, _runtime_root
from ..training.moe_checkpoint import load_compact_checkpoint
from .moe_three_pool_schedule import materialize as materialize_schedule


ROUTER_TOPOLOGY = "public_meta29_lookup_softmax_29x7"
ROUTER_IDENTIFIER = "0047_public_meta29_priority_rules_v1"


def _tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(b"\0")
        digest.update(str(tensor.dtype).encode()); digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def materialize_fp16_fp32(
    model: MetaRoutedMoEActorCritic, *, deck: tuple[int, ...],
    source_checkpoint: Path, output: Path, checkpoint_update: int,
) -> tuple[MetaRoutedMoEActorCritic, dict[str, Any]]:
    source_payload = torch.load(
        source_checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if source_payload.get("update") != checkpoint_update:
        raise RuntimeError("candidate source checkpoint update mismatch")
    reconstructed, _ = load_moe_actor_critic(
        deck=deck, own_archetype_id=0, device="cpu"
    )
    load_compact_checkpoint(reconstructed, source_payload)
    live_hash = _tensor_hash(model.state_dict())
    source_effective_hash = _tensor_hash(reconstructed.state_dict())
    if live_hash != source_effective_hash:
        raise RuntimeError("FATAL: live policy differs from saved compact checkpoint")
    storage = {
        name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu().clone()
        for name, value in reconstructed.state_dict().items()
    }
    payload = {
        "schema_version": "0047_moe_fp16_storage_fp32_runtime_v1",
        "state_dict": storage,
        "metadata": {
            "checkpoint_update": int(checkpoint_update),
            "source_policy_id": "Policy-0814",
            "focal_deck_id": "007",
            "runtime_dtype": "float32",
            "experts": model.expert_count,
            "expert_labels": list(model.expert_labels),
            "router_topology": ROUTER_TOPOLOGY,
            "router_identifier": ROUTER_IDENTIFIER,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    loaded = torch.load(output, map_location="cpu", weights_only=True)
    candidate, _ = load_moe_actor_critic(deck=deck, device="cpu")
    runtime_state = {
        name: value.float() if value.is_floating_point() else value
        for name, value in loaded["state_dict"].items()
    }
    candidate.load_state_dict(runtime_state, strict=True)
    candidate.eval()
    audit = {
        "status": "PASS", "source_checkpoint_update": checkpoint_update,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_effective_sha256": source_effective_hash,
        "portable_checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "deployment_effective_sha256": _tensor_hash(candidate.state_dict()),
        "storage_dtype": "fp16", "runtime_dtype": "fp32",
        "contract_id": "kaggle_fp16_storage_fp32_runtime_v1",
        "conversion_order": "immutable_fp32_base_plus_fp32_delta_to_fp16_storage_then_strict_fp32_runtime",
        "candidate_deployment_identity_audit": "PASS",
    }
    return candidate, audit


def evaluate_model(
    model: MetaRoutedMoEActorCritic, *, checkpoint_update: int,
    source_checkpoint: Path, output_root: Path, games_per_pool: int = 512,
) -> dict[str, Any]:
    if not 1 <= games_per_pool <= 512:
        raise ValueError("0047 evaluation games_per_pool must be in [1, 512]")
    output_root.mkdir(parents=True, exist_ok=False)
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    deck = _cards(registry, "007")
    candidate, candidate_audit = materialize_fp16_fp32(
        model, deck=deck, output=output_root / "candidate_fp16.pt",
        source_checkpoint=source_checkpoint,
        checkpoint_update=checkpoint_update,
    )
    device = torch.device("cuda:0")
    candidate.to(device).eval()
    opponent_audit = materialize_policy_bundle(
        PROJECT_ROOT, "Policy-0814", purpose="0047_two_pool_evaluation"
    ).audit
    opponent = load_policy("Policy-0814", deck_id="001")
    for module in policy_modules(opponent):
        module.to(device).eval().requires_grad_(False)
    focal_pointers = {p.untyped_storage().data_ptr() for p in candidate.parameters()}
    opponent_pointers = {
        p.untyped_storage().data_ptr()
        for module in policy_modules(opponent) for p in module.parameters()
    }
    if focal_pointers & opponent_pointers:
        raise RuntimeError("FATAL: 0047 focal/opponent storage alias")
    schedule = materialize_schedule(
        PROJECT_ROOT, focal_deck_id="007",
        focal_deployment_identity=candidate_audit["deployment_effective_sha256"],
    )
    decks = {row.deck_id: _cards(registry, row.deck_id) for row in registry.decks}
    official = build_seeded_runtime()
    all_rows: list[dict[str, Any]] = []
    pool_summaries: dict[str, dict[str, Any]] = {}
    router_telemetry: dict[str, Any] = {}
    for pool_name, pool in schedule["pools"].items():
        scheduled = pool["jobs"][:games_per_pool]
        jobs = [RolloutJob(
            game_id=row["game_id"], opponent_id=row["opponent_deck_id"],
            focal_first=row["focal_won_toss"], focal_won_toss=row["focal_won_toss"],
            coin_winner_seed=row["coin_winner_seed"], seed=row["engine_seed"],
            search_seed=row["search_seed"], policy_seed=row["policy_seed"],
            source_policy_update=checkpoint_update, focal_deck=deck,
            opponent_deck=decks[row["opponent_deck_id"]],
            runtime_root=_runtime_root(), opponent_policy_id="Policy-0814",
            focal_deck_id="007", focal_own_archetype_id=0,
            engine_library=official.library_path, action_boundary_mode="enabled",
            trace_policy="errors_and_sample",
        ) for row in scheduled]
        runtime = MoEFocalRuntime(candidate, job_count=len(jobs), device=device)
        collector = CudaFullSemanticRolloutCollector(
            candidate, opponent, device=device, rules_path=RULES,
            extension_dir=DEFAULT_BUILD_DIR, lane_count=min(512, len(jobs)),
            mode="greedy", record_trajectory=False,
            agent_selects_first_player=True, opponent_policy_id="Policy-0814",
            opponent_identity_audit=opponent_audit,
            role_compacted=True,
            focal_compacted_policy_fn=runtime.decode_compacted,
            focal_allocation_planner_for_job=runtime.plan_allocation,
            focal_resident_router_cls=MoEResidentRouter,
        )
        started = time.perf_counter()
        episodes = collector.collect(jobs)
        elapsed = time.perf_counter() - started
        if len(episodes) != len(jobs) or any(not row.valid for row in episodes):
            raise RuntimeError(f"0047 evaluation pool {pool_name} is incomplete")
        wins = sum(row.reward == 1 for row in episodes)
        pool_summaries[pool_name] = {
            "games": len(episodes), "wins": wins,
            "losses": sum(row.reward == -1 for row in episodes),
            "draws": sum(row.reward == 0 for row in episodes),
            "win_rate": wins / len(episodes), "elapsed_seconds": elapsed,
        }
        router_telemetry[pool_name] = runtime.telemetry()
        all_rows.extend({
            "pool": pool_name, "meta_id": int(source["opponent_meta_archetype_id"]),
            "focus_group": source.get("focus_group"),
            "deck_id": episode.job.opponent_id, "outcome": int(episode.reward),
            "focal_first": bool(episode.diagnostics["first_player_choice"]["focal_first"]),
        } for episode, source in zip(episodes, scheduled, strict=True))
    per_meta: dict[str, dict[str, Any]] = {}
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in all_rows:
        grouped[row["meta_id"]].append(row["outcome"])
    for meta_id, outcomes in sorted(grouped.items()):
        per_meta[f"{meta_id:02d}"] = {
            "games": len(outcomes), "wins": outcomes.count(1),
            "win_rate": outcomes.count(1) / len(outcomes),
        }
    per_deck: dict[str, dict[str, Any]] = {}
    deck_grouped: dict[str, list[int]] = defaultdict(list)
    for row in all_rows:
        deck_grouped[row["deck_id"]].append(row["outcome"])
    for deck_id, outcomes in sorted(deck_grouped.items()):
        per_deck[deck_id] = {
            "games": len(outcomes), "wins": outcomes.count(1),
            "losses": outcomes.count(-1), "draws": outcomes.count(0),
            "win_rate": outcomes.count(1) / len(outcomes),
        }
    focus_group_summaries: dict[str, dict[str, Any]] = {}
    for group_name in ("old_three", "new_four"):
        outcomes = [
            row["outcome"] for row in all_rows
            if row.get("focus_group") == group_name
        ]
        if outcomes:
            focus_group_summaries[group_name] = {
                "games": len(outcomes), "wins": outcomes.count(1),
                "losses": outcomes.count(-1), "draws": outcomes.count(0),
                "win_rate": outcomes.count(1) / len(outcomes),
            }
    report = {
        "schema_version": "0047_two_pool_cuda_report_v1", "status": "PASS",
        "checkpoint_update": checkpoint_update, "opponent_policy_id": "Policy-0814",
        "candidate_identity": candidate_audit,
        "schedule_sha256": schedule["schedule_sha256"],
        "games_per_pool": games_per_pool, "pool_summaries": pool_summaries,
        "focus_group_summaries": focus_group_summaries,
        "per_deck": per_deck, "per_meta": per_meta,
        "router_telemetry": router_telemetry,
        "entries": all_rows,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def wandb_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "eval/checkpoint_update": float(report["checkpoint_update"]),
    }
    for pool, row in report["pool_summaries"].items():
        metrics[f"eval/{pool}/win_rate"] = float(row["win_rate"])
        metrics[f"eval/{pool}/n_games"] = float(row["games"])
    for group, row in report["focus_group_summaries"].items():
        metrics[f"eval/{group}/win_rate"] = float(row["win_rate"])
        metrics[f"eval/{group}/n_games"] = float(row["games"])
    for deck_id, row in report["per_deck"].items():
        metrics[f"eval/deck_{deck_id}/win_rate"] = float(row["win_rate"])
        metrics[f"eval/deck_{deck_id}/n_games"] = float(row["games"])
    for meta, row in report["per_meta"].items():
        metrics[f"eval/meta_{meta}/win_rate"] = float(row["win_rate"])
        metrics[f"eval/meta_{meta}/n_games"] = float(row["games"])
    return metrics


__all__ = ["evaluate_model", "materialize_fp16_fp32", "wandb_metrics"]
