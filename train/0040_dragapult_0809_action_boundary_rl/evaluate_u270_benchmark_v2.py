"""Evaluate immutable 0040 U270 on the common-seed Policy-0809 Benchmark V2."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import torch

from evaluation.runtime.seeded import build_seeded_runtime

# This evaluation targets today's Benchmark V2 contract, whose admitted runtime
# is CUDA Engine 2.0.  Preload that project-independent engine package before
# importing the historical 0040 collector, which otherwise resolves its legacy
# engine_cuda Python package by default.
ROOT = Path(__file__).resolve().parents[2]
CURRENT_CUDA_PYTHON = ROOT / "engine_cuda_2_0/python"
sys.path.insert(0, str(CURRENT_CUDA_PYTHON))
import ptcg_cuda_engine  # noqa: E402,F401

from .candidate_deployment import (
    load_kaggle_evaluation_candidate,
    require_kaggle_candidate_deployment,
)
from . import evaluate_u270_policy0809_cuda as legacy_evaluation
from .rollout.protocol import DEFAULT_FULL_ROUND_DRAW_LIMIT, RolloutJob
from .training.run_full_semantic import runtime_root


legacy_evaluation.CUDA_EXTENSION = ROOT / "engine_cuda_2_0/build/native"
CHECKPOINT = legacy_evaluation.CHECKPOINT
CHECKPOINT_SHA256 = legacy_evaluation.CHECKPOINT_SHA256
OPPONENT_EFFECTIVE_SHA256 = legacy_evaluation.OPPONENT_EFFECTIVE_SHA256
OPPONENT_POLICY_ID = legacy_evaluation.OPPONENT_POLICY_ID
_collect_grouped_jobs = legacy_evaluation._collect_grouped_jobs
_opponent = legacy_evaluation._opponent
_sha256 = legacy_evaluation._sha256
PACKAGE = ROOT / "archive/submission/0040_dragapult_ex_007_0809_rl_update270"
REFERENCE_REPORT = ROOT / (
    "rl_runs/0045_single_deck_expert_minimal_lora/versions/"
    "V7_dragapult_007_u276_aggressive_meta_quota_policy0809/artifact/"
    "formal_evaluation/update-000282_benchmark_v2_cuda2048/report.json"
)
SOURCE_DECK_REGISTRY = ROOT / "train/0045_single_deck_expert_minimal_lora/assets/decks/registry.json"
VERSION = "V6_u270_benchmark_v2_policy0809_engine2_cuda2048"
VERSION_ROOT = ROOT / "rl_runs/0040_dragapult_0809_action_boundary_rl/versions" / VERSION
ARTIFACT = VERSION_ROOT / "artifact"
SNAPSHOT = ARTIFACT / "input_snapshot"
RESULT = ARTIFACT / "report.json"
FORMAL_HTML = ROOT / "experiments/0040_dragapult_0809_action_boundary_rl/evaluation" / f"{VERSION}.html"
TEMP = ROOT / ".tmp/evaluation/0040_u270_benchmark_v2_policy0809"
EXPECTED_COMMON_SCHEDULE = "80d4c8c76089504ba288a4e9de0a6b185aa50d15e2b731f915e029c01f7fc53f"
EXPECTED_OPPONENT = "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"
EXPECTED_META = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 27)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def canonical_deck(cards: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(map(str, sorted(cards))).encode("ascii")).hexdigest()


def load_deck(path: Path) -> tuple[int, ...]:
    cards = tuple(int(line) for line in path.read_text(encoding="utf-8").splitlines())
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise RuntimeError(f"invalid exact deck: {path}")
    return cards


def materialize_snapshot(destination: Path) -> dict:
    reference = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    schedule = reference["schedule"]
    opponent = reference["opponent_policy_identity_audit"]
    entries = reference["entries"]
    if (
        reference.get("status") != "PASS"
        or reference.get("benchmark_id") != "Benchmark-V2"
        or schedule.get("common_random_schedule_sha256") != EXPECTED_COMMON_SCHEDULE
        or schedule.get("opponent_policy_id") != OPPONENT_POLICY_ID
        or opponent.get("effective_policy_sha256") != EXPECTED_OPPONENT
        or len(entries) != 2048
        or Counter(row["opponent_meta_archetype_id"] for row in entries)
        != Counter({meta: 128 for meta in EXPECTED_META})
    ):
        raise RuntimeError("reference Benchmark V2 identity failed")
    ids = sorted({str(row["opponent_id"]) for row in entries})
    source_registry = json.loads(SOURCE_DECK_REGISTRY.read_text(encoding="utf-8"))
    source_by_id = {str(row["deck_id"]): row for row in source_registry["decks"]}
    decks = {}
    for deck_id in ids:
        source = source_by_id[deck_id]
        cards = tuple(
            int(card["card_id"])
            for card in source["cards"]
            for _ in range(int(card["count"]))
        )
        if len(cards) != 60 or canonical_deck(cards) != source["content_sha256"]:
            raise RuntimeError(f"source registry exact-deck identity failed: {deck_id}")
        target = destination / "decks" / f"{deck_id}.txt"
        atomic_text(target, "\n".join(map(str, cards)) + "\n")
        decks[deck_id] = {
            "path": str(target.relative_to(destination.parents[1])),
            "exact_deck_sha256": canonical_deck(cards),
        }
    jobs = [{
        "game_id": row["game_id"],
        "opponent_id": str(row["opponent_id"]),
        "opponent_meta_archetype_id": int(row["opponent_meta_archetype_id"]),
        "engine_seed": int(row["engine_seed"]),
        "search_seed": int(row["search_seed"]),
        "policy_seed": int(row["policy_seed"]),
        "coin_winner_seed": int(row["coin_winner_seed"]),
        "focal_won_toss": bool(row["focal_won_toss"]),
    } for row in entries]
    payload = {
        "schema_version": "0040_u270_benchmark_v2_input_snapshot_v1",
        "source_report": str(REFERENCE_REPORT.relative_to(ROOT)),
        "source_report_sha256": _sha256(REFERENCE_REPORT),
        "source_deck_registry": str(SOURCE_DECK_REGISTRY.relative_to(ROOT)),
        "source_deck_registry_sha256": _sha256(SOURCE_DECK_REGISTRY),
        "contract_id": schedule["contract_id"],
        "common_random_schedule_sha256": EXPECTED_COMMON_SCHEDULE,
        "opponent_policy_id": OPPONENT_POLICY_ID,
        "opponent_effective_policy_sha256": EXPECTED_OPPONENT,
        "selected_class_ids": list(EXPECTED_META),
        "games": len(jobs),
        "decks": decks,
        "jobs": jobs,
    }
    atomic_json(destination / "schedule.json", payload)
    return payload


def rollout_jobs(snapshot: dict) -> list[RolloutJob]:
    seeded_runtime = build_seeded_runtime()
    root = runtime_root()
    focal_deck = load_deck(PACKAGE / "deck.csv")
    decks = {
        deck_id: load_deck(VERSION_ROOT / item["path"])
        for deck_id, item in snapshot["decks"].items()
    }
    return [RolloutJob(
        game_id=row["game_id"], opponent_id=row["opponent_id"], focal_first=False,
        seed=row["engine_seed"], source_policy_update=270,
        focal_deck=focal_deck, opponent_deck=decks[row["opponent_id"]],
        runtime_root=root, opponent_policy_id=OPPONENT_POLICY_ID,
        policy_seed=row["policy_seed"], search_seed=row["search_seed"],
        engine_library=seeded_runtime.library_path,
        full_round_draw_limit=DEFAULT_FULL_ROUND_DRAW_LIMIT,
        ability_repeat_limit=20, action_boundary_mode="enabled",
        focal_won_toss=row["focal_won_toss"],
    ) for row in snapshot["jobs"]]


def wilson(wins: int, games: int) -> list[float]:
    z = 1.959963984540054
    p = wins / games
    d = 1 + z * z / games
    c = (p + z * z / (2 * games)) / d
    h = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / d
    return [c - h, c + h]


def collect_grouped_jobs_with_progress(model, opponent, jobs, device):
    groups = legacy_evaluation._group_jobs_by_opponent_deck(jobs)
    episodes_by_id = {}
    group_metrics = []
    completed = 0
    for index, group in enumerate(groups, start=1):
        collector = legacy_evaluation.CudaFullSemanticRolloutCollector(
            model, opponent, device=device,
            rules_path=legacy_evaluation.CUDA_RULES,
            extension_dir=legacy_evaluation.CUDA_EXTENSION,
            lane_count=min(256, len(group)), mode="greedy", check_interval=8,
            ability_repeat_limit=20, record_trajectory=False,
            agent_selects_first_player=True,
            opponent_policy_id=OPPONENT_POLICY_ID,
            opponent_identity_audit=opponent._policy_identity_audit,
        )
        collected = collector.collect(group)
        for episode in collected:
            if episode.job.game_id in episodes_by_id:
                raise RuntimeError("duplicate CUDA game result")
            episodes_by_id[episode.job.game_id] = episode
        metrics = collector.metrics()
        group_metrics.append({
            "opponent_id": group[0].opponent_id,
            "games": len(group), "metrics": metrics,
        })
        completed += len(group)
        print(
            f"[0040 U270 Benchmark V2] group {index:02d}/{len(groups):02d} "
            f"deck={group[0].opponent_id} games={completed}/{len(jobs)}",
            flush=True,
        )
    if set(episodes_by_id) != {job.game_id for job in jobs}:
        raise RuntimeError("grouped CUDA results do not conserve the schedule")
    return [episodes_by_id[job.game_id] for job in jobs], group_metrics


def run(*, smoke_games: int | None, diagnose_opponent_id: str | None = None) -> dict:
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("U270 source checkpoint identity failed")
    if _sha256(PACKAGE / "strategy/model.bin") != json.loads(
        (PACKAGE / "manifest.json").read_text(encoding="utf-8")
    )["portable_checkpoint_sha256"]:
        raise RuntimeError("U270 archive portable identity failed")
    output = (
        TEMP / f"diagnose-engine2-{diagnose_opponent_id}"
        if diagnose_opponent_id
        else TEMP / f"smoke-{smoke_games}"
        if smoke_games
        else VERSION_ROOT
    )
    if output.exists():
        raise FileExistsError(output)
    destination = output / "artifact/input_snapshot"
    snapshot = materialize_snapshot(destination)
    if diagnose_opponent_id:
        snapshot["jobs"] = [
            row for row in snapshot["jobs"]
            if row["opponent_id"] == diagnose_opponent_id
        ]
        if not snapshot["jobs"]:
            raise RuntimeError(f"unknown diagnostic opponent deck: {diagnose_opponent_id}")
        snapshot["games"] = len(snapshot["jobs"])
    elif smoke_games:
        snapshot["jobs"] = snapshot["jobs"][:smoke_games]
        snapshot["games"] = smoke_games
    device = torch.device("cuda:0")
    focal_deck = load_deck(PACKAGE / "deck.csv")
    model, candidate_audit = load_kaggle_evaluation_candidate(
        package=PACKAGE, checkpoint=CHECKPOINT, deck=focal_deck, device=device,
    )
    require_kaggle_candidate_deployment(model)
    first_deck = load_deck(output / snapshot["decks"][snapshot["jobs"][0]["opponent_id"]]["path"])
    opponent = _opponent(device, first_deck)
    if opponent._policy_identity_audit.effective_policy_sha256 != EXPECTED_OPPONENT:
        raise RuntimeError("Policy-0809 effective identity differs from Benchmark V2")
    jobs = (
        rollout_jobs(snapshot)
        if not smoke_games and not diagnose_opponent_id
        else _rollout_jobs_at(output, snapshot)
    )
    started = time.perf_counter()
    episodes, group_metrics = collect_grouped_jobs_with_progress(
        model, opponent, jobs, device
    )
    elapsed = time.perf_counter() - started
    by_game = {row["game_id"]: row for row in snapshot["jobs"]}
    entries = []
    for episode in episodes:
        source = by_game[episode.job.game_id]
        choice = episode.diagnostics.get("first_player_choice")
        fallback = bool(episode.diagnostics.get("macro_fallback"))
        reason = episode.diagnostics.get("macro_fallback_reason")
        semantic_fallback = fallback and reason != "chance_boundary_before_allocation"
        entries.append({
            **source,
            "outcome": 1 if episode.reward == 1 else -1 if episode.reward == -1 else 0,
            "turns": episode.turns, "valid": episode.valid, "error": episode.error,
            "first_player_choice": choice,
            "focal_first": choice.get("focal_first") if isinstance(choice, dict) else None,
            "semantic_fallback": semantic_fallback,
        })
    games = len(entries)
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    draws = games - wins - losses
    first = [row for row in entries if row["focal_first"] is True]
    second = [row for row in entries if row["focal_first"] is False]
    report = {
        "schema_version": "0040_u270_benchmark_v2_report_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS", "benchmark_id": "Benchmark-V2",
        "focal_policy_id": "0040-U270", "focal_checkpoint_update": 270,
        "focal_deck_id": "007", "focal_deck_cards": list(focal_deck),
        "focal_exact_deck_sha256": canonical_deck(focal_deck),
        "candidate_deployment_identity_audit": candidate_audit.to_manifest(),
        "opponent_policy_identity_audit": opponent._policy_identity_audit.to_manifest(),
        "schedule": {key: value for key, value in snapshot.items() if key not in {"jobs", "decks"}},
        "summary": {
            "games": games, "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / games, "wilson_95": wilson(wins, games),
            "focal_first_games": len(first),
            "focal_first_win_rate": sum(row["outcome"] == 1 for row in first) / len(first),
            "focal_second_games": len(second),
            "focal_second_win_rate": sum(row["outcome"] == 1 for row in second) / len(second),
            "elapsed_seconds": elapsed, "games_per_second": games / elapsed,
        },
        "collector": {"opponent_deck_groups": len(group_metrics), "group_metrics": group_metrics},
        "selection_mode": "greedy", "official_engine": True, "entries": entries,
    }
    expected_games = len(snapshot["jobs"])
    if (
        len(entries) != expected_games
        or any(row["valid"] is not True or row["error"] not in (None, "") for row in entries)
        or any(type(row["focal_first"]) is not bool for row in entries)
        or any(row["semantic_fallback"] for row in entries)
        or candidate_audit.status != "PASS"
        or opponent._policy_identity_audit.status != "PASS"
        or sum(item["games"] for item in group_metrics) != expected_games
        or any(item["metrics"].get("rollout/lane_routing_audit_failures", 0) for item in group_metrics)
        or any(item["metrics"].get("rollout/cuda_feature_d2h_bytes", 0) for item in group_metrics)
    ):
        report["status"] = "FAIL"
        raise RuntimeError("0040 U270 Benchmark V2 hard gate failed")
    atomic_json(output / "artifact/report.json", report)
    atomic_json(output / "artifact/status.json", {
        "version": VERSION if not smoke_games else f"smoke-{smoke_games}",
        "state": "completed", "phase": "evaluation_only",
        "checkpoint_update": 270, "opponent_policy_id": OPPONENT_POLICY_ID,
        "games": games, "errors": 0, "unfinished": 0,
    })
    return report


def _rollout_jobs_at(output: Path, snapshot: dict) -> list[RolloutJob]:
    seeded_runtime = build_seeded_runtime()
    focal_deck = load_deck(PACKAGE / "deck.csv")
    decks = {
        deck_id: load_deck(output / item["path"])
        for deck_id, item in snapshot["decks"].items()
    }
    return [RolloutJob(
        game_id=row["game_id"], opponent_id=row["opponent_id"], focal_first=False,
        seed=row["engine_seed"], source_policy_update=270,
        focal_deck=focal_deck, opponent_deck=decks[row["opponent_id"]],
        runtime_root=runtime_root(), opponent_policy_id=OPPONENT_POLICY_ID,
        policy_seed=row["policy_seed"], search_seed=row["search_seed"],
        engine_library=seeded_runtime.library_path,
        full_round_draw_limit=DEFAULT_FULL_ROUND_DRAW_LIMIT,
        ability_repeat_limit=20, action_boundary_mode="enabled",
        focal_won_toss=row["focal_won_toss"],
    ) for row in snapshot["jobs"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-games", type=int)
    parser.add_argument("--diagnose-opponent-id")
    args = parser.parse_args()
    if args.smoke_games is not None and not 1 <= args.smoke_games < 2048:
        parser.error("--smoke-games must be in [1, 2047]")
    if args.smoke_games is not None and args.diagnose_opponent_id:
        parser.error("--smoke-games and --diagnose-opponent-id are mutually exclusive")
    report = run(
        smoke_games=args.smoke_games,
        diagnose_opponent_id=args.diagnose_opponent_id,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
