"""Run public Deck Router V2 on the comparable Policy-0814 CUDA-512 contract."""

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
from ..policy.actor_critic import DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT
from ..policy_identity import materialize_policy_bundle
from ..rollout import CudaFullSemanticRolloutCollector, RolloutJob
from ..runtime import _modules as policy_modules
from ..runtime import load_policy
from ..training.run_v1 import RULES, _runtime_root
from .policy0814_exact_deck_schedule import (
    CONTRACT_ID as EVAL_CONTRACT_ID,
    GAMES,
    OPPONENT_POLICY_ID,
    materialize as materialize_schedule,
)
from .public_deck_router_v2 import (
    PublicDeckResidentRouter,
    materialize_public_deck_router_v2,
)
from .public_deck_router_v2_rules import (
    DEFAULT_UPDATE,
    PUBLIC_POLICY_ID,
    ROUTED_UPDATES,
    ROUTE_MANIFEST,
)
from .run_meta_oracle_v1_cuda2048 import _merge_metrics, _wilson
from .run_policy0814_exact_deck_cuda512 import validate_report as validate_static_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
VERSION = "V15_public_deck_router_v2_policy0814"
BENCHMARK_ID = "V15-Public-DeckRouter-V2-Policy0814-Eval512"
V14_ROOT = (
    ROOT / "runs/versions"
    / "V14_policy0814_identity_gate_fix"
)
CHECKPOINTS = {
    update: V14_ROOT / "checkpoint" / f"update-{update:06d}.pt"
    for update in ROUTED_UPDATES
}
BASELINE_REPORT = (
    V14_ROOT / "artifact/periodic_evaluation/update-000070/report.json"
)
DEFAULT_OUTPUT = (
    ROOT / "runs/versions" / VERSION
    / "artifact/public_deck_router_eval512"
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["outcome"] == 1 for row in rows)
    losses = sum(row["outcome"] == -1 for row in rows)
    draws = len(rows) - wins - losses
    return {
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(rows) if rows else 0.0,
    }


def validate_report(report: Mapping[str, Any]) -> None:
    entries = report.get("entries")
    focal = report.get("focal_public_router_identity_audit") or {}
    opponent = report.get("opponent_policy_identity_audit") or {}
    schedule = report.get("schedule") or {}
    baseline = report.get("baseline") or {}
    candidate_rows = focal.get("candidate_materializations") or {}
    if (
        report.get("status") != "PASS"
        or report.get("benchmark_id") != BENCHMARK_ID
        or report.get("focal_policy_id") != PUBLIC_POLICY_ID
        or focal.get("status") != "PASS"
        or focal.get("policy_id") != PUBLIC_POLICY_ID
        or focal.get("rules") != ROUTE_MANIFEST
        or focal.get("critic_outputs_consumed_by_actor_or_router") is not False
        or focal.get("historical_checkpoint_boundary", {}).get(
            "omitted_trainable_tensor_count"
        ) != 16
        or set(map(int, candidate_rows)) != set(ROUTED_UPDATES)
        or any(
            row.get("status") != "PASS"
            or row.get("storage_dtype") != "fp16"
            or row.get("runtime_dtype") != "fp32"
            for row in candidate_rows.values()
        )
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != OPPONENT_POLICY_ID
        or schedule.get("contract_id") != EVAL_CONTRACT_ID
        or schedule.get("games") != GAMES
        or schedule.get("common_random_numbers") is not True
        or baseline.get("checkpoint_update") != DEFAULT_UPDATE
        or baseline.get("games") != GAMES
        or not isinstance(entries, list)
        or len(entries) != GAMES
        or report.get("focal_opponent_shared_parameter_storages") != 0
        or any(row.get("valid") is not True or row.get("error") is not None for row in entries)
    ):
        raise RuntimeError("Public Deck Router V2 identity/completion gate failed")
    if any(
        row.get("final_routed_checkpoint_update") not in ROUTED_UPDATES
        or not isinstance(row.get("seen_public_trigger_card_ids"), list)
        or not isinstance(row.get("route_decision_counts"), dict)
        for row in entries
    ):
        raise RuntimeError("Public Deck Router V2 telemetry gate failed")
    metrics = report.get("collector_metrics") or {}
    if (
        metrics.get("rollout/lane_routing_audit_failures", 0) != 0
        or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
        or metrics.get("rollout/lane_routing_audit_pass") != 1
        or metrics.get("rollout/unfinished_games", 0) != 0
    ):
        raise RuntimeError("Public Deck Router V2 CUDA residency/routing gate failed")


def run(
    *, output_root: Path = DEFAULT_OUTPUT, smoke_games: int | None = None,
) -> dict[str, Any]:
    formal = smoke_games is None
    game_count = GAMES if formal else int(smoke_games)
    if not 1 <= game_count <= GAMES:
        raise ValueError("smoke games must be within eval512")
    if output_root.exists():
        raise FileExistsError(output_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Public Deck Router V2 evaluation requires CUDA")
    required = [*CHECKPOINTS.values(), BASELINE_REPORT]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.is_file()])

    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    deck_asset = next(row for row in registry.decks if row.deck_id == "007")
    cards = tuple(map(int, (PROJECT_ROOT / deck_asset.deck_path).read_text().splitlines()))
    decks = {
        row.deck_id: tuple(map(int, (PROJECT_ROOT / row.deck_path).read_text().splitlines()))
        for row in registry.decks
    }
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    device = torch.device("cuda:0")
    output_root.mkdir(parents=True)
    public = materialize_public_deck_router_v2(
        checkpoints=CHECKPOINTS,
        base_portable=DEFAULT_0814_ACTOR_CHECKPOINT,
        base_value_checkpoint=DEFAULT_0814_VALUE_CHECKPOINT,
        deck=cards,
        deck_id="007",
        own_archetype_id=own_by_deck["007"],
        output_root=output_root / "materialization",
        device=device,
        job_count=min(256, game_count),
    )
    opponent_bundle = materialize_policy_bundle(
        PROJECT_ROOT, OPPONENT_POLICY_ID, purpose="0045_v15_public_deck_router_opponent"
    )
    opponent = load_policy(OPPONENT_POLICY_ID, deck_id="001")
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
    shared = focal_pointers & opponent_pointers
    if shared:
        raise RuntimeError("FATAL: Public Deck Router focal/opponent storage alias")

    schedule = materialize_schedule(
        PROJECT_ROOT,
        focal_deck_id="007",
        focal_deployment_identity=public.composite_effective_sha256,
    )
    schedule_rows = schedule["jobs"][:game_count]
    baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    validate_static_report(baseline_report)
    baseline_by_game = {row["game_id"]: row for row in baseline_report["entries"]}
    for row in schedule_rows:
        baseline_row = baseline_by_game.get(row["game_id"])
        if baseline_row is None or any(
            baseline_row[key] != row[key]
            for key in (
                "engine_seed", "search_seed", "policy_seed", "coin_winner_seed",
            )
        ) or baseline_row["opponent_id"] != row["opponent_deck_id"]:
            raise RuntimeError("U70 baseline does not share the eval512 game schedule")

    seeded_runtime = build_seeded_runtime()
    jobs = [
        RolloutJob(
            game_id=row["game_id"], opponent_id=row["opponent_deck_id"],
            focal_first=row["focal_won_toss"], focal_won_toss=row["focal_won_toss"],
            coin_winner_seed=row["coin_winner_seed"], seed=row["engine_seed"],
            search_seed=row["search_seed"], policy_seed=row["policy_seed"],
            source_policy_update=DEFAULT_UPDATE, focal_deck=cards,
            opponent_deck=decks[row["opponent_deck_id"]],
            runtime_root=_runtime_root(), opponent_policy_id=OPPONENT_POLICY_ID,
            focal_deck_id="007", focal_own_archetype_id=own_by_deck["007"],
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
        opponent_own_ids = torch.tensor(
            [own_by_deck[job.opponent_id] for job in chunk_jobs],
            dtype=torch.long,
            device=device,
        )
        collector = CudaFullSemanticRolloutCollector(
            public.model, opponent, device=device, rules_path=RULES,
            extension_dir=DEFAULT_BUILD_DIR, lane_count=len(chunk_jobs), mode="greedy",
            check_interval=8, record_trajectory=False,
            agent_selects_first_player=True, opponent_policy_id=OPPONENT_POLICY_ID,
            opponent_identity_audit=opponent_bundle.audit,
            opponent_own_archetype_ids=opponent_own_ids,
            focal_compacted_policy_fn=public.decode_compacted,
            focal_allocation_head_for_job=public.allocation_head_for_job,
            focal_resident_router_cls=PublicDeckResidentRouter,
        )
        chunk_episodes = collector.collect(chunk_jobs)
        episodes.extend(chunk_episodes)
        telemetry = public.memory.telemetry()
        telemetry_rows.extend({
            "final_routed_checkpoint_update": telemetry["route_updates"][index],
            "predicted_deck_code": telemetry["predicted_deck_codes"][index],
            "provisional": telemetry["provisional"][index],
            "conflict": telemetry["conflict"][index],
            "first_classification_observation": (
                telemetry["first_classification_observation"][index]
            ),
            "first_specialist_route_observation": (
                telemetry["first_specialist_route_observation"][index]
            ),
            "public_observation_count": telemetry["observation_count"][index],
            "route_change_count": telemetry["route_change_count"][index],
            "route_decision_counts": telemetry["route_decision_counts"][index],
            "seen_public_trigger_card_ids": telemetry["seen_trigger_card_ids"][index],
        } for index in range(len(chunk_jobs)))
        chunk_metrics[begin // 256] = collector.metrics()
        wins = sum(episode.reward == 1 for episode in episodes)
        print(
            f"[0045 PUBLIC DECK V2] games {len(episodes)}/{game_count} "
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
            raise RuntimeError(f"public deck game lacks seat evidence: {episode.job.game_id}")
        baseline_row = baseline_by_game[episode.job.game_id]
        outcome = 1 if episode.reward == 1 else -1 if episode.reward == -1 else 0
        entries.append({
            "game_id": episode.job.game_id,
            "opponent_id": episode.job.opponent_id,
            "opponent_meta_archetype_id": row["opponent_meta_archetype_id"],
            **route,
            "classification_matches_exact_deck": (
                route["predicted_deck_code"] == int(episode.job.opponent_id)
            ),
            "outcome": outcome,
            "baseline_u70_outcome": int(baseline_row["outcome"]),
            "paired_outcome_delta": outcome - int(baseline_row["outcome"]),
            "turns": episode.turns, "valid": episode.valid, "error": episode.error,
            "engine_seed": row["engine_seed"], "search_seed": row["search_seed"],
            "policy_seed": row["policy_seed"], "coin_winner_seed": row["coin_winner_seed"],
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": choice, "focal_first": bool(choice["focal_first"]),
            "lane_routing_audit_status": episode.diagnostics.get(
                "lane_routing_audit_status"
            ),
        })

    overall = _summary(entries)
    first = [row for row in entries if row["focal_first"]]
    second = [row for row in entries if not row["focal_first"]]
    baseline_rows = [
        {"outcome": int(baseline_by_game[row["game_id"]]["outcome"])}
        for row in schedule_rows
    ]
    summary = {
        **overall,
        "wilson_95": _wilson(overall["wins"], overall["games"]),
        "focal_first_games": len(first),
        "focal_first_win_rate": _summary(first)["win_rate"],
        "focal_second_games": len(second),
        "focal_second_win_rate": _summary(second)["win_rate"],
        "elapsed_seconds": elapsed,
        "games_per_second": game_count / elapsed,
        "classified_exact_games": sum(
            row["classification_matches_exact_deck"] for row in entries
        ),
        "provisional_games": sum(row["provisional"] for row in entries),
        "conflict_games": sum(row["conflict"] for row in entries),
        "ever_specialist_games": sum(
            row["first_specialist_route_observation"] >= 0 for row in entries
        ),
        "final_route_counts": dict(sorted(Counter(
            str(row["final_routed_checkpoint_update"]) for row in entries
        ).items())),
        "route_decision_counts": dict(sorted({
            str(update): sum(
                row["route_decision_counts"].get(str(update), 0) for row in entries
            )
            for update in ROUTED_UPDATES
        }.items())),
    }
    baseline_summary = _summary(baseline_rows)
    paired = {
        "router_wins_baseline_losses": sum(
            row["outcome"] == 1 and row["baseline_u70_outcome"] != 1 for row in entries
        ),
        "baseline_wins_router_losses": sum(
            row["baseline_u70_outcome"] == 1 and row["outcome"] != 1 for row in entries
        ),
        "net_wins": overall["wins"] - baseline_summary["wins"],
        "win_rate_delta": overall["win_rate"] - baseline_summary["win_rate"],
    }
    per_deck = {}
    for deck_id in sorted(schedule["realized_deck_counts"]):
        rows = [row for row in entries if row["opponent_id"] == deck_id]
        router_summary = _summary(rows)
        baseline_deck = _summary([
            {"outcome": row["baseline_u70_outcome"]} for row in rows
        ])
        per_deck[deck_id] = {
            "router": router_summary,
            "baseline_u70": baseline_deck,
            "net_wins": router_summary["wins"] - baseline_deck["wins"],
            "win_rate_delta": (
                router_summary["win_rate"] - baseline_deck["win_rate"]
            ),
            "classified_exact_games": sum(
                row["classification_matches_exact_deck"] for row in rows
            ),
            "ever_specialist_games": sum(
                row["first_specialist_route_observation"] >= 0 for row in rows
            ),
            "final_route_counts": dict(sorted(Counter(
                str(row["final_routed_checkpoint_update"]) for row in rows
            ).items())),
        }

    report = {
        "schema_version": "0045_public_deck_router_v2_policy0814_cuda512_report_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if formal else "SMOKE_PASS",
        "benchmark_id": BENCHMARK_ID,
        "focal_policy_id": PUBLIC_POLICY_ID,
        "focal_policy_role": "public_information_lane_local_checkpoint_router",
        "evidence_class": public.audit["evidence_class"],
        "non_cheating_disclosure": (
            "Routing consumes only certain or remembered public opponent card identities "
            "already encoded in the focal actor-visible semantic observation."
        ),
        "focal_public_router_identity_audit": public.audit,
        "source_checkpoints": {
            str(update): {
                "path": str(CHECKPOINTS[update].relative_to(ROOT)),
                "sha256": sha256_file(CHECKPOINTS[update]),
            }
            for update in ROUTED_UPDATES
        },
        "focal_deck_id": "007",
        "focal_deck_display_name": deck_asset.name,
        "focal_exact_deck_sha256": deck_asset.content_sha256,
        "focal_opponent_shared_parameter_storages": len(shared),
        "opponent_policy_identity_audit": asdict(opponent_bundle.audit),
        "schedule": {key: value for key, value in schedule.items() if key != "jobs"},
        "selection_mode": "greedy",
        "official_engine": True,
        "cuda_engine_identity_audit": CudaEngineIdentity.resolve(
            ROOT, rule_pack=RULES, binary=DEFAULT_BUILD_DIR / "ptcg_cuda_smoke",
            extension=DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
            require_gpu=True, require_extension=True,
        ).to_manifest(),
        "baseline": {
            "checkpoint_update": DEFAULT_UPDATE,
            "report": str(BASELINE_REPORT.relative_to(ROOT)),
            "report_sha256": sha256_file(BASELINE_REPORT),
            **baseline_summary,
        },
        "summary": summary,
        "paired_vs_u70": paired,
        "per_deck": per_deck,
        "collector_metrics": metrics,
        "per_chunk_collector_metrics": {
            str(key): dict(value) for key, value in chunk_metrics.items()
        },
        "entries": entries,
    }
    if formal:
        validate_report(report)
    _atomic_json(output_root / "report.json", report)
    _atomic_json(output_root / "summary.json", {
        "summary": summary,
        "baseline": report["baseline"],
        "paired_vs_u70": paired,
        "per_deck": per_deck,
    })
    _atomic_json(output_root.parent / "status.json", {
        "version": VERSION,
        "status": "complete" if formal else "smoke_complete",
        "evaluation_status": report["status"],
        "evidence_class": report["evidence_class"],
        "summary": summary,
        "report": str((output_root / "report.json").relative_to(ROOT)),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-games", type=int)
    args = parser.parse_args()
    report = run(output_root=args.output_root.resolve(), smoke_games=args.smoke_games)
    print(json.dumps({
        "summary": report["summary"],
        "baseline": report["baseline"],
        "paired_vs_u70": report["paired_vs_u70"],
        "per_deck": report["per_deck"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
