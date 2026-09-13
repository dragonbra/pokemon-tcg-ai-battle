"""Run the 0045 Policy-0809 three-pool CUDA-512 periodic evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from evaluation.runtime.seeded import build_seeded_runtime

from ..assets import AssetRegistry
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..cuda_engine_2.identity import CudaEngineIdentity
from ..own_archetype import OwnArchetypeVocabulary
from ..policy_identity import materialize_policy_bundle
from ..rollout import ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import _modules as policy_modules
from ..runtime import load_policy
from ..training.run_v1 import RULES, _runtime_root
from .benchmark_tiny_v2_three_pool_schedule import (
    CONTRACT_ID,
    GAMES_PER_POOL,
    OPPONENT_POLICY_ID,
    materialize as materialize_schedule,
)
from .candidate import materialize as materialize_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
BENCHMARK_ID = "Benchmark-Tiny-V2-Policy0809-Three-Pool"


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


def _summary(entries: list[dict[str, Any]], *, elapsed: float) -> dict[str, Any]:
    games = len(entries)
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    draws = games - wins - losses
    first = [row for row in entries if row["focal_first"]]
    second = [row for row in entries if not row["focal_first"]]
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games,
        "wilson_95": _wilson(wins, games),
        "focal_first_games": len(first),
        "focal_first_win_rate": (
            sum(row["outcome"] == 1 for row in first) / len(first)
        ),
        "focal_second_games": len(second),
        "focal_second_win_rate": (
            sum(row["outcome"] == 1 for row in second) / len(second)
        ),
        "elapsed_seconds": elapsed,
        "games_per_second": games / elapsed,
    }


def validate_report(report: dict[str, Any]) -> None:
    schedule = report.get("schedule") or {}
    entries = report.get("entries") or []
    opponent = report.get("opponent_policy_identity_audit") or {}
    metrics = report.get("collector_metrics") or {}
    if (
        report.get("status") != "PASS"
        or report.get("benchmark_id") != BENCHMARK_ID
        or report.get("focal_policy_identity_audit", {}).get("status") != "PASS"
        or report.get("focal_opponent_shared_parameter_storages") != 0
        or len(entries) != 3 * GAMES_PER_POOL
        or any(row.get("valid") is not True or row.get("error") is not None for row in entries)
    ):
        raise RuntimeError("three-pool evaluation lacks 1,536 valid deployment-audited games")
    if (
        schedule.get("contract_id") != CONTRACT_ID
        or schedule.get("opponent_policy_id") != OPPONENT_POLICY_ID
        or schedule.get("games_per_pool") != GAMES_PER_POOL
        or schedule.get("common_random_numbers") is not True
        or schedule.get("seed_derivation_excludes") != ["focal_deployment_identity"]
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != OPPONENT_POLICY_ID
        or opponent.get("effective_policy_sha256")
        != schedule.get("opponent_effective_policy_sha256")
    ):
        raise RuntimeError("three-pool Policy-0809/common-seed identity gate failed")
    for pool_name, pool in schedule.get("pools", {}).items():
        pool_entries = [row for row in entries if row.get("pool") == pool_name]
        if len(pool_entries) != GAMES_PER_POOL:
            raise RuntimeError(f"pool {pool_name} is not CUDA-512")
        actual = {
            class_id: sum(
                row["opponent_meta_archetype_id"] == class_id
                for row in pool_entries
            )
            for class_id in pool["selected_class_ids"]
        }
        if set(actual.values()) != set(map(int, pool["class_counts"].values())):
            raise RuntimeError(f"pool {pool_name} Meta counts changed")
        if max(actual.values()) - min(actual.values()) > 1:
            raise RuntimeError(f"pool {pool_name} is not Meta-balanced")
    required = (
        "engine_seed", "search_seed", "policy_seed", "coin_winner_seed",
        "focal_won_toss", "first_player_choice",
    )
    if any(any(key not in row for key in required) for row in entries):
        raise RuntimeError("three-pool per-game seat/random evidence is incomplete")
    if (
        metrics.get("rollout/lane_routing_audit_failures", 0) != 0
        or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
        or metrics.get("rollout/lane_routing_audit_pass") != 1
        or metrics.get("rollout/unfinished_games", 0) != 0
    ):
        raise RuntimeError("three-pool routing/device-residency gate failed")


def run(
    *, deck_id: str, output_root: Path, checkpoint: Path,
    checkpoint_update: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if not torch.cuda.is_available():
        raise RuntimeError("three-pool periodic evaluation requires CUDA")
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    asset = next(row for row in registry.decks if row.deck_id == deck_id)
    cards = tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
    focal_own_archetype_id = own_by_deck[deck_id]
    device = torch.device("cuda:0")

    opponent_bundle = materialize_policy_bundle(
        PROJECT_ROOT, OPPONENT_POLICY_ID,
        purpose="0045_three_pool_periodic_evaluation_opponent",
    )
    opponent = load_policy(OPPONENT_POLICY_ID, deck_id="001")
    output_root.mkdir(parents=True)
    focal, focal_audit = materialize_candidate(
        checkpoint=checkpoint,
        base_portable=(
            PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin"
        ),
        deck=cards,
        deck_id=deck_id,
        own_archetype_id=focal_own_archetype_id,
        output=output_root / "materialization/model.bin",
        device=device,
    )
    opponent_modules = policy_modules(opponent)
    for module in opponent_modules:
        module.to(device).eval().requires_grad_(False)
    focal_ptr = {parameter.untyped_storage().data_ptr() for parameter in focal.parameters()}
    opponent_ptr = {
        parameter.untyped_storage().data_ptr()
        for module in opponent_modules for parameter in module.parameters()
    }
    shared = focal_ptr & opponent_ptr
    if shared:
        raise RuntimeError("FATAL: three-pool focal/opponent storage alias")

    schedule = materialize_schedule(
        PROJECT_ROOT,
        focal_deck_id=deck_id,
        focal_deployment_identity=focal_audit.effective_candidate_sha256,
    )
    schedule_jobs = [
        row for pool in schedule["pools"].values() for row in pool["jobs"]
    ]
    decks = {
        row.deck_id: tuple(
            map(int, (PROJECT_ROOT / row.deck_path).read_text().splitlines())
        )
        for row in registry.decks
    }
    runtime = build_seeded_runtime()
    jobs = [RolloutJob(
        game_id=row["game_id"],
        opponent_id=row["opponent_deck_id"],
        focal_first=row["focal_won_toss"],
        focal_won_toss=row["focal_won_toss"],
        coin_winner_seed=row["coin_winner_seed"],
        seed=row["engine_seed"],
        search_seed=row["search_seed"],
        policy_seed=row["policy_seed"],
        source_policy_update=checkpoint_update,
        focal_deck=cards,
        opponent_deck=decks[row["opponent_deck_id"]],
        runtime_root=_runtime_root(),
        opponent_policy_id=OPPONENT_POLICY_ID,
        focal_deck_id=deck_id,
        focal_own_archetype_id=focal_own_archetype_id,
        engine_library=runtime.library_path,
        action_boundary_mode="enabled",
        trace_policy="errors_and_sample",
    ) for row in schedule_jobs]
    opponent_own_ids = torch.tensor(
        [own_by_deck[row["opponent_deck_id"]] for row in schedule_jobs],
        dtype=torch.long,
        device=device,
    )
    collector = ChunkedCudaRolloutCollector(
        focal,
        opponent,
        rollout_batch_size=256,
        trajectory_games_per_update=None,
        device=device,
        rules_path=RULES,
        extension_dir=DEFAULT_BUILD_DIR,
        lane_count=256,
        mode="greedy",
        check_interval=8,
        record_trajectory=False,
        agent_selects_first_player=True,
        opponent_policy_id=OPPONENT_POLICY_ID,
        opponent_identity_audit=opponent_bundle.audit,
        opponent_own_archetype_ids=opponent_own_ids,
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    elapsed = time.perf_counter() - started
    entries: list[dict[str, Any]] = []
    for episode, row in zip(episodes, schedule_jobs, strict=True):
        diagnostics = episode.diagnostics
        choice = diagnostics.get("first_player_choice")
        if not isinstance(choice, dict) or type(choice.get("focal_first")) is not bool:
            raise RuntimeError(f"game lacks Agent seat evidence: {episode.job.game_id}")
        entries.append({
            "pool": row["pool"],
            "game_id": episode.job.game_id,
            "opponent_id": episode.job.opponent_id,
            "opponent_meta_archetype_id": row["opponent_meta_archetype_id"],
            "outcome": 1 if episode.reward == 1 else -1 if episode.reward == -1 else 0,
            "turns": episode.turns,
            "valid": episode.valid,
            "error": episode.error,
            "engine_seed": row["engine_seed"],
            "search_seed": row["search_seed"],
            "policy_seed": row["policy_seed"],
            "coin_winner_seed": row["coin_winner_seed"],
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": choice,
            "focal_first": bool(choice["focal_first"]),
            "lane_routing_audit_status": diagnostics.get("lane_routing_audit_status"),
        })
    pool_summaries = {
        pool_name: _summary(
            [row for row in entries if row["pool"] == pool_name], elapsed=elapsed / 3
        )
        for pool_name in schedule["pools"]
    }
    per_meta = {
        str(class_id): _summary(
            [row for row in entries if row["opponent_meta_archetype_id"] == class_id],
            elapsed=elapsed * sum(
                row["opponent_meta_archetype_id"] == class_id for row in entries
            ) / len(entries),
        )
        for class_id in schedule["active_class_ids"]
    }
    cuda = CudaEngineIdentity.resolve(
        ROOT,
        rule_pack=RULES,
        binary=DEFAULT_BUILD_DIR / "ptcg_cuda_smoke",
        extension=DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
        require_gpu=True,
        require_extension=True,
    ).to_manifest()
    schedule_manifest = {
        **{key: value for key, value in schedule.items() if key != "pools"},
        "pools": {
            name: {key: value for key, value in pool.items() if key != "jobs"}
            for name, pool in schedule["pools"].items()
        },
    }
    report = {
        "schema_version": "0045_policy0809_three_pool_report_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "benchmark_id": BENCHMARK_ID,
        "focal_policy_id": f"0045-single-deck-expert-{deck_id}-update-{checkpoint_update:06d}",
        "focal_policy_role": "periodic_training_candidate",
        "focal_checkpoint_update": checkpoint_update,
        "focal_policy_identity_audit": focal_audit.to_manifest(),
        "focal_deployment_effective_sha256": focal_audit.effective_candidate_sha256,
        "focal_deck_id": deck_id,
        "focal_deck_display_name": asset.name,
        "focal_exact_deck_sha256": asset.content_sha256,
        "focal_deck_cards": list(cards),
        "focal_own_archetype_id": focal_own_archetype_id,
        "focal_opponent_shared_parameter_storages": len(shared),
        "opponent_policy_identity_audit": asdict(opponent_bundle.audit),
        "schedule": schedule_manifest,
        "selection_mode": "greedy",
        "official_engine": True,
        "cuda_engine_identity_audit": cuda,
        "summary": _summary(entries, elapsed=elapsed),
        "pool_summaries": pool_summaries,
        "per_meta": per_meta,
        "collector_metrics": collector.metrics(),
        "entries": entries,
    }
    validate_report(report)
    _atomic_json(output_root / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-id", default="007")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-update", required=True, type=int)
    args = parser.parse_args()
    report = run(
        deck_id=args.deck_id,
        output_root=args.output_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        checkpoint_update=args.checkpoint_update,
    )
    print(json.dumps({
        "summary": report["summary"],
        "pool_summaries": report["pool_summaries"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
