"""Run the explicitly cheating 29-way Meta-routed 0045 oracle on CUDA-2048."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
import json
import math
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
from ..rollout import ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import load_policy
from ..runtime import _modules as policy_modules
from ..training.run_v1 import RULES, _runtime_root
from .benchmark_v2_schedule import (
    CONTRACT_ID as BENCHMARK_CONTRACT_ID,
    GAMES,
    SELECTED_CLASS_IDS,
    materialize as materialize_schedule,
)
from .meta_oracle_v1 import (
    ORACLE_POLICY_ID, ROUTED_UPDATES, materialize_oracle, partition_rows,
    route_update_for_meta, scatter_by_game_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
VERSION = "V9_dragapult_007_meta_oracle_v1_cuda2048"
DEFAULT_OUTPUT = (
    ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions" / VERSION
    / "artifact/oracle_evaluation"
)
CHECKPOINTS = {
    40: ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions/"
    "V2_dragapult_007_expert_cold_start_lr/checkpoint/update-000040.pt",
    90: ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions/"
    "V3_dragapult_007_expert_continue_u45/checkpoint/update-000090.pt",
    200: ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions/"
    "V3_dragapult_007_expert_continue_u45/checkpoint/update-000200.pt",
    282: ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions/"
    "V7_dragapult_007_u276_aggressive_meta_quota_policy0809/checkpoint/update-000282.pt",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _wilson(wins: int, games: int) -> list[float]:
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / games + z * z / (4 * games * games)
    ) / denominator
    return [center - half, center + half]


def validate_report(report: Mapping[str, Any]) -> None:
    entries = report.get("entries")
    oracle = report.get("focal_oracle_identity_audit") or {}
    opponent = report.get("opponent_policy_identity_audit") or {}
    schedule = report.get("schedule") or {}
    if (
        report.get("status") != "PASS"
        or report.get("benchmark_id") != "Benchmark-V2-Meta-Oracle-V1"
        or report.get("focal_policy_id") != ORACLE_POLICY_ID
        or report.get("evidence_class") != "diagnostic_oracle_not_promote_not_kaggle"
        or oracle.get("status") != "PASS"
        or oracle.get("policy_id") != ORACLE_POLICY_ID
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != "Policy-0809"
        or schedule.get("contract_id") != BENCHMARK_CONTRACT_ID
        or schedule.get("games") != GAMES
        or not isinstance(entries, list)
        or len(entries) != GAMES
        or any(row.get("valid") is not True or row.get("error") is not None for row in entries)
    ):
        raise RuntimeError("Meta Oracle V1 identity/completion gate failed")
    if any(
        row.get("routed_checkpoint_update")
        != route_update_for_meta(int(row.get("opponent_meta_archetype_id", -1)))
        for row in entries
    ):
        raise RuntimeError("Meta Oracle V1 per-game checkpoint route mismatch")
    route_counts = Counter(int(row["routed_checkpoint_update"]) for row in entries)
    expected_counts = {40: 384, 90: 128, 200: 512, 282: 1024}
    if dict(sorted(route_counts.items())) != expected_counts:
        raise RuntimeError(f"Meta Oracle V1 route allocation mismatch: {route_counts}")
    metrics = report.get("collector_metrics") or {}
    if (
        metrics.get("rollout/lane_routing_audit_failures", 0) != 0
        or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
        or metrics.get("rollout/lane_routing_audit_pass") != 1
        or metrics.get("rollout/unfinished_games", 0) != 0
    ):
        raise RuntimeError("Meta Oracle V1 CUDA routing/device-residency gate failed")


def _merge_metrics(
    rows: Mapping[int, Mapping[str, float]], *, games_per_second: float
) -> dict[str, float]:
    keys = set().union(*(row.keys() for row in rows.values()))
    maxima = {
        "rollout/max_batch_size", "rollout/cuda_lane_count",
        "rollout/cuda_peak_allocated_bytes", "rollout/cuda_peak_reserved_bytes",
        "rollout/unique_opponent_exact_decks",
    }
    booleans = {
        "rollout/cuda_features_device_resident", "rollout/lane_routing_audit_pass",
    }
    ignored_rates = {
        "rollout/cuda_games_per_second", "rollout/strategic_decisions_per_second",
        "rollout/mean_batch_size",
    }
    result: dict[str, float] = {}
    for key in keys - ignored_rates:
        values = [float(row.get(key, 0.0)) for row in rows.values()]
        result[key] = (
            min(values) if key in booleans
            else max(values) if key in maxima
            else sum(values)
        )
    result["rollout/cuda_games_per_second"] = games_per_second
    result["rollout/unfinished_games"] = 0.0
    return result


def run(*, output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if any(not path.is_file() for path in CHECKPOINTS.values()):
        missing = [str(path) for path in CHECKPOINTS.values() if not path.is_file()]
        raise FileNotFoundError(missing)
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    deck_asset = next(row for row in registry.decks if row.deck_id == "007")
    cards = tuple(
        map(int, (PROJECT_ROOT / deck_asset.deck_path).read_text(encoding="utf-8").splitlines())
    )
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_id = next(row.archetype_id for row in vocabulary.mappings if row.deck_id == "007")
    device = torch.device("cuda:0")

    output_root.mkdir(parents=True)
    oracle = materialize_oracle(
        checkpoints=CHECKPOINTS,
        base_portable=PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin",
        deck=cards, deck_id="007", own_archetype_id=own_id,
        output_root=output_root / "materialization", device=device,
    )
    opponent_bundle = materialize_policy_bundle(
        PROJECT_ROOT, "Policy-0809", purpose="benchmark_v2_meta_oracle_opponent"
    )
    opponent = load_policy("Policy-0809", deck_id="001")
    opponent_modules = policy_modules(opponent)
    for module in opponent_modules:
        module.to(device).eval().requires_grad_(False)
    focal_pointers = {
        parameter.untyped_storage().data_ptr()
        for parameter in oracle.model.parameters()
    } | {
        parameter.untyped_storage().data_ptr()
        for bundle in oracle.heads.values()
        for module in (
            bundle.action_decoder, bundle.policy_option_lora, bundle.allocation_head
        )
        for parameter in module.parameters()
    }
    opponent_pointers = {
        parameter.untyped_storage().data_ptr()
        for module in opponent_modules for parameter in module.parameters()
    }
    if focal_pointers & opponent_pointers:
        raise RuntimeError("FATAL: Meta Oracle focal/opponent storage alias")

    schedule = materialize_schedule(
        PROJECT_ROOT, focal_deck_id="007",
        focal_deployment_identity=str(oracle.audit["composite_effective_sha256"]),
    )
    decks = {
        row.deck_id: tuple(
            map(int, (PROJECT_ROOT / row.deck_path).read_text(encoding="utf-8").splitlines())
        )
        for row in registry.decks
    }
    runtime = build_seeded_runtime()
    jobs_by_game_id: dict[str, RolloutJob] = {}
    for row in schedule["jobs"]:
        update = route_update_for_meta(int(row["opponent_meta_archetype_id"]))
        jobs_by_game_id[row["game_id"]] = RolloutJob(
            game_id=row["game_id"], opponent_id=row["opponent_deck_id"],
            focal_first=row["focal_won_toss"], focal_won_toss=row["focal_won_toss"],
            coin_winner_seed=row["coin_winner_seed"], seed=row["engine_seed"],
            search_seed=row["search_seed"], policy_seed=row["policy_seed"],
            source_policy_update=update, focal_deck=cards,
            opponent_deck=decks[row["opponent_deck_id"]],
            runtime_root=_runtime_root(), opponent_policy_id="Policy-0809",
            focal_deck_id="007", focal_own_archetype_id=own_id,
            engine_library=runtime.library_path, action_boundary_mode="enabled",
            trace_policy="errors_and_sample",
        )

    partitions = partition_rows(schedule["jobs"])
    episodes_by_game_id: dict[str, Any] = {}
    per_route_metrics: dict[int, Mapping[str, float]] = {}
    started = time.perf_counter()
    for update in ROUTED_UPDATES:
        oracle.activate(update)
        routed_jobs = [jobs_by_game_id[str(row["game_id"])] for row in partitions[update]]
        print(
            f"[0045 ORACLE] route U{update}: {len(routed_jobs)} games; "
            f"Meta={sorted({int(row['opponent_meta_archetype_id']) for row in partitions[update]})}",
            flush=True,
        )
        collector = ChunkedCudaRolloutCollector(
            oracle.model, opponent, rollout_batch_size=256,
            trajectory_games_per_update=None, device=device, rules_path=RULES,
            extension_dir=DEFAULT_BUILD_DIR, lane_count=256, mode="greedy",
            check_interval=8, record_trajectory=False,
            agent_selects_first_player=True, opponent_policy_id="Policy-0809",
            opponent_identity_audit=opponent_bundle.audit,
        )
        episodes = collector.collect(routed_jobs)
        for episode in episodes:
            if episode.job.game_id in episodes_by_game_id:
                raise RuntimeError(f"duplicate oracle episode: {episode.job.game_id}")
            episodes_by_game_id[episode.job.game_id] = episode
        per_route_metrics[update] = collector.metrics()
        del collector
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    episodes = scatter_by_game_id(schedule["jobs"], episodes_by_game_id)
    metrics = _merge_metrics(per_route_metrics, games_per_second=GAMES / elapsed)

    entries: list[dict[str, Any]] = []
    for episode, row in zip(episodes, schedule["jobs"], strict=True):
        diagnostics = episode.diagnostics
        choice = diagnostics.get("first_player_choice")
        if not isinstance(choice, dict) or type(choice.get("focal_first")) is not bool:
            raise RuntimeError(f"oracle game lacks Agent seat evidence: {episode.job.game_id}")
        entries.append({
            "game_id": episode.job.game_id,
            "opponent_id": episode.job.opponent_id,
            "opponent_meta_archetype_id": row["opponent_meta_archetype_id"],
            "routed_checkpoint_update": route_update_for_meta(
                int(row["opponent_meta_archetype_id"])
            ),
            "outcome": 1 if episode.reward == 1 else -1 if episode.reward == -1 else 0,
            "turns": episode.turns, "valid": episode.valid, "error": episode.error,
            "engine_seed": row["engine_seed"], "search_seed": row["search_seed"],
            "policy_seed": row["policy_seed"], "coin_winner_seed": row["coin_winner_seed"],
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": choice, "focal_first": bool(choice["focal_first"]),
            "lane_routing_audit_status": diagnostics.get("lane_routing_audit_status"),
        })
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    draws = GAMES - wins - losses
    first = [row for row in entries if row["focal_first"]]
    second = [row for row in entries if not row["focal_first"]]
    route_summary = {}
    for update in ROUTED_UPDATES:
        rows = [row for row in entries if row["routed_checkpoint_update"] == update]
        route_wins = sum(row["outcome"] == 1 for row in rows)
        route_summary[str(update)] = {
            "games": len(rows), "wins": route_wins,
            "losses": sum(row["outcome"] == -1 for row in rows),
            "draws": sum(row["outcome"] == 0 for row in rows),
            "win_rate": route_wins / len(rows),
            "meta_ids": sorted({row["opponent_meta_archetype_id"] for row in rows}),
        }
    summary = {
        "games": GAMES, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / GAMES, "wilson_95": _wilson(wins, GAMES),
        "focal_first_games": len(first),
        "focal_first_win_rate": sum(row["outcome"] == 1 for row in first) / len(first),
        "focal_second_games": len(second),
        "focal_second_win_rate": sum(row["outcome"] == 1 for row in second) / len(second),
        "elapsed_seconds": elapsed, "games_per_second": GAMES / elapsed,
    }
    cuda = CudaEngineIdentity.resolve(
        ROOT, rule_pack=RULES, binary=DEFAULT_BUILD_DIR / "ptcg_cuda_smoke",
        extension=DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
        require_gpu=True, require_extension=True,
    ).to_manifest()
    report = {
        "schema_version": "0045_meta_oracle_v1_cuda2048_report_v1",
        "created_at": datetime.now(UTC).isoformat(), "status": "PASS",
        "benchmark_id": "Benchmark-V2-Meta-Oracle-V1",
        "focal_policy_id": ORACLE_POLICY_ID,
        "focal_policy_role": "experimental_exact_opponent_meta_oracle",
        "evidence_class": "diagnostic_oracle_not_promote_not_kaggle",
        "cheating_disclosure": (
            "The focal runtime reads the opponent exact deck's true own_archetypes_v2 "
            "class before the game and routes checkpoint-specific trainable heads."
        ),
        "focal_oracle_identity_audit": dict(oracle.audit),
        "source_checkpoints": {
            str(update): {"path": str(CHECKPOINTS[update].relative_to(ROOT)),
                          "sha256": sha256_file(CHECKPOINTS[update])}
            for update in ROUTED_UPDATES
        },
        "focal_deck_id": "007", "focal_deck_display_name": deck_asset.name,
        "focal_exact_deck_sha256": deck_asset.content_sha256,
        "focal_opponent_shared_parameter_storages": 0,
        "opponent_policy_identity_audit": asdict(opponent_bundle.audit),
        "schedule": {key: value for key, value in schedule.items() if key != "jobs"},
        "selection_mode": "greedy", "official_engine": True,
        "cuda_engine_identity_audit": cuda,
        "route_summary": route_summary,
        "summary": summary, "collector_metrics": metrics,
        "per_route_collector_metrics": {
            str(update): dict(per_route_metrics[update]) for update in ROUTED_UPDATES
        },
        "entries": entries,
    }
    validate_report(report)
    _atomic_json(output_root / "report.json", report)
    _atomic_json(output_root.parent / "status.json", {
        "version": VERSION, "status": "complete", "evaluation_status": "PASS",
        "evidence_class": report["evidence_class"], "summary": summary,
        "report": str((output_root / "report.json").relative_to(ROOT)),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(output_root=args.output_root.resolve())
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
