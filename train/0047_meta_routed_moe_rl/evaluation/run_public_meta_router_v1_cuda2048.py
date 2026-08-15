"""Run the public-information 0045 expert router against Policy-0809 CUDA-2048."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from evaluation.runtime.seeded import build_seeded_runtime

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..cuda_engine_2.identity import CudaEngineIdentity
from ..own_archetype import OwnArchetypeVocabulary
from ..policy_identity import materialize_policy_bundle
from ..rollout import CudaFullSemanticRolloutCollector, RolloutJob
from ..runtime import load_policy
from ..runtime import _modules as policy_modules
from ..training.run_v1 import RULES, _runtime_root
from .benchmark_v2_schedule import (
    CONTRACT_ID as BENCHMARK_CONTRACT_ID,
    GAMES,
    materialize as materialize_schedule,
)
from .public_meta_router_v1 import (
    PUBLIC_POLICY_ID, PublicMetaResidentRouter, materialize_public_router,
)
from .run_meta_oracle_v1_cuda2048 import CHECKPOINTS, _merge_metrics, _wilson


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
VERSION = "V12_dragapult_007_public_meta_router_v3_grass_fold_u200_cuda2048"
DEFAULT_OUTPUT = (
    ROOT / "rl_runs/0047_meta_routed_moe_rl/versions" / VERSION
    / "artifact/public_router_evaluation"
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_report(report: Mapping[str, Any]) -> None:
    entries = report.get("entries")
    focal = report.get("focal_public_router_identity_audit") or {}
    opponent = report.get("opponent_policy_identity_audit") or {}
    schedule = report.get("schedule") or {}
    if (
        report.get("status") != "PASS"
        or report.get("benchmark_id") != "Benchmark-V2-Public-MetaRouter-V3-GrassFoldU200"
        or report.get("focal_policy_id") != PUBLIC_POLICY_ID
        or focal.get("status") != "PASS"
        or focal.get("policy_id") != PUBLIC_POLICY_ID
        or focal.get("critic_outputs_consumed_by_actor_or_router") is not False
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != "Policy-0809"
        or schedule.get("contract_id") != BENCHMARK_CONTRACT_ID
        or schedule.get("games") != GAMES
        or not isinstance(entries, list)
        or len(entries) != GAMES
        or any(row.get("valid") is not True or row.get("error") is not None for row in entries)
    ):
        raise RuntimeError("Public Meta Router V1 identity/completion gate failed")
    if any(
        row.get("final_routed_checkpoint_update") not in (40, 90, 200, 282)
        or not isinstance(row.get("seen_public_trigger_card_ids"), list)
        for row in entries
    ):
        raise RuntimeError("Public Meta Router V1 route telemetry gate failed")
    metrics = report.get("collector_metrics") or {}
    if (
        metrics.get("rollout/lane_routing_audit_failures", 0) != 0
        or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
        or metrics.get("rollout/lane_routing_audit_pass") != 1
        or metrics.get("rollout/unfinished_games", 0) != 0
    ):
        raise RuntimeError("Public Meta Router V1 CUDA residency/routing gate failed")


def run(*, output_root: Path = DEFAULT_OUTPUT, smoke_games: int | None = None) -> dict[str, Any]:
    formal = smoke_games is None
    game_count = GAMES if formal else int(smoke_games)
    if not 1 <= game_count <= GAMES:
        raise ValueError("smoke games must be within Benchmark V2")
    if output_root.exists():
        raise FileExistsError(output_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Public Meta Router CUDA evaluation requires CUDA")
    if any(not path.is_file() for path in CHECKPOINTS.values()):
        raise FileNotFoundError([str(path) for path in CHECKPOINTS.values() if not path.is_file()])

    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    deck_asset = next(row for row in registry.decks if row.deck_id == "007")
    cards = tuple(map(int, (PROJECT_ROOT / deck_asset.deck_path).read_text().splitlines()))
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_id = next(row.archetype_id for row in vocabulary.mappings if row.deck_id == "007")
    device = torch.device("cuda:0")
    output_root.mkdir(parents=True)
    public = materialize_public_router(
        checkpoints=CHECKPOINTS,
        base_portable=PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin",
        deck=cards,
        deck_id="007",
        own_archetype_id=own_id,
        output_root=output_root / "materialization",
        device=device,
        job_count=min(256, game_count),
    )
    opponent_bundle = materialize_policy_bundle(
        PROJECT_ROOT, "Policy-0809", purpose="benchmark_v2_public_meta_router_opponent"
    )
    opponent = load_policy("Policy-0809", deck_id="001")
    opponent_modules = policy_modules(opponent)
    for module in opponent_modules:
        module.to(device).eval().requires_grad_(False)
    focal_pointers = {
        parameter.untyped_storage().data_ptr()
        for parameter in public.model.parameters()
    } | {
        parameter.untyped_storage().data_ptr()
        for bundle in public.heads.values()
        for module in (bundle.action_decoder, bundle.policy_option_lora, bundle.allocation_head)
        for parameter in module.parameters()
    }
    opponent_pointers = {
        parameter.untyped_storage().data_ptr()
        for module in opponent_modules for parameter in module.parameters()
    }
    if focal_pointers & opponent_pointers:
        raise RuntimeError("FATAL: Public Meta Router focal/opponent storage alias")

    schedule = materialize_schedule(
        PROJECT_ROOT,
        focal_deck_id="007",
        focal_deployment_identity=public.composite_effective_sha256,
    )
    schedule_rows = schedule["jobs"][:game_count]
    decks = {
        row.deck_id: tuple(map(int, (PROJECT_ROOT / row.deck_path).read_text().splitlines()))
        for row in registry.decks
    }
    seeded_runtime = build_seeded_runtime()
    jobs = [
        RolloutJob(
            game_id=row["game_id"], opponent_id=row["opponent_deck_id"],
            focal_first=row["focal_won_toss"], focal_won_toss=row["focal_won_toss"],
            coin_winner_seed=row["coin_winner_seed"], seed=row["engine_seed"],
            search_seed=row["search_seed"], policy_seed=row["policy_seed"],
            source_policy_update=282, focal_deck=cards,
            opponent_deck=decks[row["opponent_deck_id"]],
            runtime_root=_runtime_root(), opponent_policy_id="Policy-0809",
            focal_deck_id="007", focal_own_archetype_id=own_id,
            engine_library=seeded_runtime.library_path,
            action_boundary_mode="enabled", trace_policy="errors_and_sample",
        )
        for row in schedule_rows
    ]

    episodes = []
    telemetry_rows: list[dict[str, Any]] = []
    chunk_metrics: dict[int, Mapping[str, float]] = {}
    started = time.perf_counter()
    for begin in range(0, game_count, 256):
        chunk_jobs = jobs[begin:begin + 256]
        public.reset_memory(len(chunk_jobs))
        collector = CudaFullSemanticRolloutCollector(
            public.model, opponent, device=device, rules_path=RULES,
            extension_dir=DEFAULT_BUILD_DIR, lane_count=256, mode="greedy",
            check_interval=8, record_trajectory=False,
            agent_selects_first_player=True, opponent_policy_id="Policy-0809",
            opponent_identity_audit=opponent_bundle.audit,
            focal_compacted_policy_fn=public.decode_compacted,
            focal_allocation_head_for_job=public.allocation_head_for_job,
            focal_resident_router_cls=PublicMetaResidentRouter,
        )
        chunk_episodes = collector.collect(chunk_jobs)
        episodes.extend(chunk_episodes)
        telemetry = public.memory.telemetry()
        telemetry_rows.extend({
            "final_routed_checkpoint_update": telemetry["route_updates"][index],
            "locked_public_meta_id": telemetry["locked_meta"][index],
            "provisional_public_meta_id": telemetry["provisional_meta"][index],
            "first_lock_observation": telemetry["first_lock_observation"][index],
            "public_observation_count": telemetry["observation_count"][index],
            "seen_public_trigger_card_ids": telemetry["seen_trigger_card_ids"][index],
        } for index in range(len(chunk_jobs)))
        chunk_metrics[begin // 256] = collector.metrics()
        wins = sum(episode.reward == 1 for episode in episodes)
        print(
            f"[0045 PUBLIC ROUTER] games {len(episodes)}/{game_count} "
            f"wins={wins} win_rate={wins / len(episodes):.4f} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del collector
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    metrics = _merge_metrics(chunk_metrics, games_per_second=game_count / elapsed)

    entries: list[dict[str, Any]] = []
    for episode, row, route in zip(episodes, schedule_rows, telemetry_rows, strict=True):
        choice = episode.diagnostics.get("first_player_choice")
        if not isinstance(choice, dict) or type(choice.get("focal_first")) is not bool:
            raise RuntimeError(f"public router game lacks seat evidence: {episode.job.game_id}")
        entries.append({
            "game_id": episode.job.game_id,
            "opponent_id": episode.job.opponent_id,
            "opponent_meta_archetype_id": row["opponent_meta_archetype_id"],
            **route,
            "outcome": 1 if episode.reward == 1 else -1 if episode.reward == -1 else 0,
            "turns": episode.turns, "valid": episode.valid, "error": episode.error,
            "engine_seed": row["engine_seed"], "search_seed": row["search_seed"],
            "policy_seed": row["policy_seed"], "coin_winner_seed": row["coin_winner_seed"],
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": choice, "focal_first": bool(choice["focal_first"]),
            "lane_routing_audit_status": episode.diagnostics.get("lane_routing_audit_status"),
        })
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    draws = game_count - wins - losses
    first = [row for row in entries if row["focal_first"]]
    second = [row for row in entries if not row["focal_first"]]
    summary = {
        "games": game_count, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / game_count, "wilson_95": _wilson(wins, game_count),
        "focal_first_games": len(first),
        "focal_first_win_rate": sum(row["outcome"] == 1 for row in first) / len(first),
        "focal_second_games": len(second),
        "focal_second_win_rate": sum(row["outcome"] == 1 for row in second) / len(second),
        "elapsed_seconds": elapsed, "games_per_second": game_count / elapsed,
        "locked_games": sum(row["locked_public_meta_id"] >= 0 for row in entries),
        "unresolved_games": sum(row["locked_public_meta_id"] < 0 for row in entries),
        "final_route_counts": dict(sorted(Counter(
            str(row["final_routed_checkpoint_update"]) for row in entries
        ).items())),
    }
    cuda = CudaEngineIdentity.resolve(
        ROOT, rule_pack=RULES, binary=DEFAULT_BUILD_DIR / "ptcg_cuda_smoke",
        extension=DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
        require_gpu=True, require_extension=True,
    ).to_manifest()
    report = {
        "schema_version": "0045_public_meta_router_v3_grass_fold_u200_cuda2048_report_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if formal else "SMOKE_PASS",
        "benchmark_id": "Benchmark-V2-Public-MetaRouter-V3-GrassFoldU200",
        "focal_policy_id": PUBLIC_POLICY_ID,
        "focal_policy_role": "public_information_lane_local_expert_router",
        "evidence_class": public.audit["evidence_class"],
        "non_cheating_disclosure": (
            "Routing consumes only certain public opponent Pokemon identities already "
            "encoded in the focal actor-visible semantic observation."
        ),
        "focal_public_router_identity_audit": public.audit,
        "source_checkpoints": {
            str(update): {
                "path": str(CHECKPOINTS[update].relative_to(ROOT)),
                "sha256": sha256_file(CHECKPOINTS[update]),
            }
            for update in sorted(CHECKPOINTS)
        },
        "focal_deck_id": "007", "focal_deck_display_name": deck_asset.name,
        "focal_exact_deck_sha256": deck_asset.content_sha256,
        "focal_opponent_shared_parameter_storages": 0,
        "opponent_policy_identity_audit": asdict(opponent_bundle.audit),
        "schedule": {key: value for key, value in schedule.items() if key != "jobs"},
        "selection_mode": "greedy", "official_engine": True,
        "cuda_engine_identity_audit": cuda,
        "summary": summary, "collector_metrics": metrics,
        "per_chunk_collector_metrics": {
            str(key): dict(value) for key, value in chunk_metrics.items()
        },
        "entries": entries,
    }
    if formal:
        validate_report(report)
    _atomic_json(output_root / "report.json", report)
    _atomic_json(output_root.parent / "status.json", {
        "version": VERSION, "status": "complete" if formal else "smoke_complete",
        "evaluation_status": report["status"], "evidence_class": report["evidence_class"],
        "summary": summary, "report": str((output_root / "report.json").relative_to(ROOT)),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-games", type=int)
    args = parser.parse_args()
    report = run(
        output_root=args.output_root.resolve(), smoke_games=args.smoke_games
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
