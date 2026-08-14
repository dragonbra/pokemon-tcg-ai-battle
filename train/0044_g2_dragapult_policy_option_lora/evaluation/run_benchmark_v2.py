"""Run formal Policy-0809 Benchmark V2 CUDA-2048 with common random numbers."""

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

from ..assets import AssetRegistry, canonical_deck_sha256
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..cuda_engine_2.identity import CudaEngineIdentity
from ..own_archetype import OwnArchetypeVocabulary
from ..policy_identity import materialize_policy_bundle
from ..rollout import ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import load_policy
from ..runtime import _modules as policy_modules
from ..training.run_v1 import RULES, _runtime_root
from .candidate import materialize as materialize_candidate
from .benchmark_v2_schedule import (
    CONTRACT_ID, GAMES, MASTER_SEED, SELECTED_CLASS_IDS,
    materialize as materialize_schedule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]


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


def validate_report(report: dict[str, Any]) -> None:
    entries = report.get("entries")
    schedule = report.get("schedule") or {}
    opponent = report.get("opponent_policy_identity_audit") or {}
    focal = report.get("focal_policy_identity_audit") or {}
    metrics = report.get("collector_metrics") or {}
    if (
        report.get("status") != "PASS"
        or report.get("benchmark_id") != "Benchmark-V2"
        or focal.get("status") != "PASS"
        or report.get("focal_opponent_shared_parameter_storages") != 0
        or not isinstance(entries, list)
        or len(entries) != GAMES
        or any(row.get("valid") is not True or row.get("error") is not None for row in entries)
    ):
        raise RuntimeError("Benchmark V2 is not 2,048 valid deployment-audited games")
    if (
        schedule.get("contract_id") != CONTRACT_ID
        or schedule.get("games") != GAMES
        or schedule.get("selected_class_ids") != list(SELECTED_CLASS_IDS)
        or schedule.get("excluded_class_ids") != [
            class_id for class_id in range(29) if class_id not in SELECTED_CLASS_IDS
        ]
        or schedule.get("opponent_policy_id") != "Policy-0809"
        or schedule.get("common_random_numbers") is not True
        or schedule.get("seed_derivation_excludes") != ["focal_deployment_identity"]
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != "Policy-0809"
        or opponent.get("effective_policy_sha256")
        != schedule.get("opponent_effective_policy_sha256")
    ):
        raise RuntimeError("Benchmark V2 Policy-0809/common-seed identity gate failed")
    required = (
        "engine_seed", "search_seed", "policy_seed", "coin_winner_seed",
        "focal_won_toss", "first_player_choice",
    )
    if any(any(key not in row for key in required) for row in entries):
        raise RuntimeError("Benchmark V2 per-game random/seat evidence is incomplete")
    class_counts = {
        class_id: sum(
            row.get("opponent_meta_archetype_id") == class_id for row in entries
        )
        for class_id in SELECTED_CLASS_IDS
    }
    if (
        set(row.get("opponent_meta_archetype_id") for row in entries)
        != set(SELECTED_CLASS_IDS)
        or set(class_counts.values()) != {128}
    ):
        raise RuntimeError("Benchmark V2 Core-16 Meta allocation gate failed")
    if (
        metrics.get("rollout/lane_routing_audit_failures", 0) != 0
        or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
        or metrics.get("rollout/lane_routing_audit_pass") != 1
        or metrics.get("rollout/unfinished_games", 0) != 0
    ):
        raise RuntimeError("Benchmark V2 routing/device-residency gate failed")


def run(
    *, deck_id: str, output_root: Path, checkpoint: Path,
    checkpoint_update: int, deck_path: Path | None = None,
    deck_display_name: str | None = None, own_archetype_id: int | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    if deck_path is None:
        asset = next(row for row in registry.decks if row.deck_id == deck_id)
        cards = tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
        focal_display_name = asset.name
        focal_exact_deck_sha256 = asset.content_sha256
        focal_own_archetype_id = own_by_deck[deck_id]
        focal_deck_source = str(PROJECT_ROOT / asset.deck_path)
        focal_deck_registry_member = True
    else:
        cards = tuple(map(int, deck_path.read_text(encoding="utf-8").splitlines()))
        if len(cards) != 60:
            raise ValueError("external Benchmark V2 focal deck must contain exact 60 cards")
        if own_archetype_id is None or not 0 <= own_archetype_id < vocabulary.class_count:
            raise ValueError("external Benchmark V2 focal deck requires a valid own_archetype_id")
        focal_display_name = deck_display_name or deck_id
        focal_exact_deck_sha256 = canonical_deck_sha256(cards)
        focal_own_archetype_id = own_archetype_id
        focal_deck_source = str(deck_path)
        focal_deck_registry_member = False
    device = torch.device("cuda:0")
    opponent_bundle = materialize_policy_bundle(
        PROJECT_ROOT, "Policy-0809", purpose="benchmark_v2_opponent"
    )
    opponent = load_policy("Policy-0809", deck_id="001")
    output_root.mkdir(parents=True)
    base_portable = PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin"
    focal, focal_audit = materialize_candidate(
        checkpoint=checkpoint, base_portable=base_portable,
        deck=cards, deck_id=deck_id, own_archetype_id=focal_own_archetype_id,
        output=output_root / "materialization/model.bin", device=device,
    )
    opponent_modules = policy_modules(opponent)
    for module in opponent_modules:
        module.to(device).eval().requires_grad_(False)
    focal_ptr = {p.untyped_storage().data_ptr() for p in focal.parameters()}
    opponent_ptr = {
        p.untyped_storage().data_ptr()
        for module in opponent_modules for p in module.parameters()
    }
    shared = focal_ptr & opponent_ptr
    if shared:
        raise RuntimeError("FATAL: Benchmark V2 focal/opponent storage alias")
    schedule = materialize_schedule(
        PROJECT_ROOT, focal_deck_id=deck_id,
        focal_deployment_identity=focal_audit.effective_candidate_sha256,
    )
    decks = {
        row.deck_id: tuple(map(int, (PROJECT_ROOT / row.deck_path).read_text().splitlines()))
        for row in registry.decks
    }
    runtime = build_seeded_runtime()
    jobs = [RolloutJob(
        game_id=row["game_id"], opponent_id=row["opponent_deck_id"],
        focal_first=row["focal_won_toss"], focal_won_toss=row["focal_won_toss"],
        coin_winner_seed=row["coin_winner_seed"],
        seed=row["engine_seed"], search_seed=row["search_seed"],
        policy_seed=row["policy_seed"], source_policy_update=checkpoint_update,
        focal_deck=cards, opponent_deck=decks[row["opponent_deck_id"]],
        runtime_root=_runtime_root(), opponent_policy_id="Policy-0809",
        focal_deck_id=deck_id, focal_own_archetype_id=focal_own_archetype_id,
        engine_library=runtime.library_path, action_boundary_mode="enabled",
        trace_policy="errors_and_sample",
    ) for row in schedule["jobs"]]
    collector = ChunkedCudaRolloutCollector(
        focal, opponent, rollout_batch_size=256, trajectory_games_per_update=None,
        device=device, rules_path=RULES, extension_dir=DEFAULT_BUILD_DIR,
        lane_count=256, mode="greedy", check_interval=8, record_trajectory=False,
        agent_selects_first_player=True, opponent_policy_id="Policy-0809",
        opponent_identity_audit=opponent_bundle.audit,
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    elapsed = time.perf_counter() - started
    metrics = collector.metrics()
    entries = []
    for episode, row in zip(episodes, schedule["jobs"], strict=True):
        diagnostics = episode.diagnostics
        choice = diagnostics.get("first_player_choice")
        if not isinstance(choice, dict) or type(choice.get("focal_first")) is not bool:
            raise RuntimeError(f"Benchmark V2 game lacks Agent seat evidence: {episode.job.game_id}")
        entries.append({
            "game_id": episode.job.game_id,
            "opponent_id": episode.job.opponent_id,
            "opponent_meta_archetype_id": row["opponent_meta_archetype_id"],
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
    summary = {
        "games": GAMES, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / GAMES, "wilson_95": _wilson(wins, GAMES),
        "focal_first_games": len(first),
        "focal_first_win_rate": sum(r["outcome"] == 1 for r in first) / len(first),
        "focal_second_games": len(second),
        "focal_second_win_rate": sum(r["outcome"] == 1 for r in second) / len(second),
        "elapsed_seconds": elapsed, "games_per_second": GAMES / elapsed,
    }
    cuda = CudaEngineIdentity.resolve(
        ROOT, rule_pack=RULES, binary=DEFAULT_BUILD_DIR / "ptcg_cuda_smoke",
        extension=DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
        require_gpu=True, require_extension=True,
    ).to_manifest()
    report = {
        "schema_version": "0044_benchmark_v2_report_v1",
        "created_at": datetime.now(UTC).isoformat(), "status": "PASS",
        "benchmark_id": "Benchmark-V2", "focal_policy_id": "0044-candidate",
        "focal_policy_role": "periodic_training_candidate",
        "focal_checkpoint_update": checkpoint_update,
        "focal_policy_identity_audit": focal_audit.to_manifest(),
        "focal_deployment_effective_sha256": focal_audit.effective_candidate_sha256,
        "focal_deck_id": deck_id, "focal_deck_display_name": focal_display_name,
        "focal_exact_deck_sha256": focal_exact_deck_sha256,
        "focal_deck_cards": list(cards),
        "focal_own_archetype_id": focal_own_archetype_id,
        "focal_deck_source": focal_deck_source,
        "focal_deck_registry_member": focal_deck_registry_member,
        "focal_opponent_shared_parameter_storages": len(shared),
        "opponent_policy_identity_audit": asdict(opponent_bundle.audit),
        "schedule": {key: value for key, value in schedule.items() if key != "jobs"},
        "selection_mode": "greedy", "official_engine": True,
        "cuda_engine_identity_audit": cuda,
        "summary": summary, "collector_metrics": metrics, "entries": entries,
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
    parser.add_argument("--deck-path", type=Path)
    parser.add_argument("--deck-display-name")
    parser.add_argument("--own-archetype-id", type=int)
    args = parser.parse_args()
    report = run(
        deck_id=args.deck_id, output_root=args.output_root.resolve(),
        checkpoint=args.checkpoint.resolve(), checkpoint_update=args.checkpoint_update,
        deck_path=args.deck_path.resolve() if args.deck_path else None,
        deck_display_name=args.deck_display_name,
        own_archetype_id=args.own_archetype_id,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
