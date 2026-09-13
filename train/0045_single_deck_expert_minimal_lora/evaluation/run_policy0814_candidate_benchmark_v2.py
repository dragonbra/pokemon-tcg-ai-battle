"""Evaluate a Policy-0814 descendant against Policy-0809 Benchmark V2."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import torch

from evaluation.runtime.seeded import build_seeded_runtime

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..cuda_engine_2.identity import CudaEngineIdentity
from ..own_archetype import OwnArchetypeVocabulary
from ..policy import AdaptationConfig
from ..policy.actor_critic import (
    DEFAULT_0814_ACTOR_CHECKPOINT,
    DEFAULT_0814_VALUE_CHECKPOINT,
)
from ..policy_identity import materialize_policy_bundle
from ..rollout import ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import _modules as policy_modules
from ..runtime import load_policy
from ..training.run_v1 import RULES, _runtime_root
from .benchmark_v2_schedule import (
    CONTRACT_ID,
    GAMES,
    MASTER_SEED,
    SELECTED_CLASS_IDS,
    materialize as materialize_schedule,
)
from .candidate import CONTRACT_ID as CANDIDATE_DEPLOYMENT_CONTRACT
from .candidate import materialize as materialize_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
BENCHMARK_ID = "Benchmark-V2"
FOCAL_BASE_POLICY_ID = "Policy-0814"
OPPONENT_POLICY_ID = "Policy-0809"
DEFAULT_FOCAL_ACTOR_CHECKPOINT = DEFAULT_0814_ACTOR_CHECKPOINT
DEFAULT_FOCAL_VALUE_CHECKPOINT = DEFAULT_0814_VALUE_CHECKPOINT
SUPPORTED_LANE_COUNTS = (256, 512)
DEFAULT_LANE_COUNT = 512
DEFAULT_ADAPTATION = AdaptationConfig(
    rank=16, alpha=16.0, output_projection=True, shared_state_encoder=True,
)
_SOURCE_VERSION = re.compile(r"V[1-9][0-9]*")


def candidate_policy_id(source_version: str, deck_id: str, checkpoint_update: int) -> str:
    if _SOURCE_VERSION.fullmatch(source_version) is None:
        raise ValueError("source version must be an ASCII V<number> identity")
    if re.fullmatch(r"[0-9]{3}", deck_id) is None:
        raise ValueError("deck ID must contain exactly three digits")
    if checkpoint_update < 0:
        raise ValueError("checkpoint update must be non-negative")
    return (
        f"0045-policy0814-{source_version.lower()}-deck{deck_id}"
        f"-update-{checkpoint_update:06d}"
    )


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


def _summary(entries: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    games = len(entries)
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    draws = games - wins - losses
    first = [row for row in entries if row["focal_first"]]
    second = [row for row in entries if not row["focal_first"]]
    result: dict[str, Any] = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games,
        "elapsed_seconds": elapsed,
        "games_per_second": games / elapsed if elapsed else 0.0,
    }
    if games:
        result["wilson_95"] = _wilson(wins, games)
    if first and second:
        result.update({
            "focal_first_games": len(first),
            "focal_first_win_rate": sum(row["outcome"] == 1 for row in first) / len(first),
            "focal_second_games": len(second),
            "focal_second_win_rate": sum(row["outcome"] == 1 for row in second) / len(second),
        })
    return result


def validate_report(report: dict[str, Any]) -> None:
    entries = report.get("entries")
    schedule = report.get("schedule") or {}
    opponent = report.get("opponent_policy_identity_audit") or {}
    focal = report.get("focal_policy_identity_audit") or {}
    cuda = report.get("cuda_engine_identity_audit") or {}
    metrics = report.get("collector_metrics") or {}
    reported_lane_count = report.get(
        "cuda_lane_count", metrics.get("rollout/cuda_lane_count")
    )
    source_version = report.get("focal_source_version")
    checkpoint_update = report.get("focal_checkpoint_update")
    deck_id = report.get("focal_deck_id")
    expected_policy_id = None
    if isinstance(source_version, str) and isinstance(deck_id, str) and isinstance(checkpoint_update, int):
        expected_policy_id = candidate_policy_id(source_version, deck_id, checkpoint_update)
    if (
        report.get("status") != "PASS"
        or report.get("benchmark_id") != BENCHMARK_ID
        or report.get("focal_base_policy_id") != FOCAL_BASE_POLICY_ID
        or report.get("focal_policy_id") != expected_policy_id
        or report.get("focal_policy_role") != "policy0814_descendant_candidate"
        or report.get("focal_opponent_shared_parameter_storages") != 0
        or reported_lane_count not in SUPPORTED_LANE_COUNTS
        or not isinstance(entries, list)
        or len(entries) != GAMES
        or any(row.get("valid") is not True or row.get("error") is not None for row in entries)
        or any(row.get("outcome") not in (-1, 0, 1) for row in entries)
    ):
        raise RuntimeError("Policy-0814 Benchmark V2 completion/identity gate failed")
    if (
        focal.get("status") != "PASS"
        or focal.get("contract_id") != CANDIDATE_DEPLOYMENT_CONTRACT
        or focal.get("storage_dtype") != "fp16"
        or focal.get("runtime_dtype") != "fp32"
        or focal.get("checkpoint_update") != checkpoint_update
        or focal.get("source_checkpoint_sha256") != report.get("focal_source_checkpoint_sha256")
        or focal.get("effective_candidate_sha256")
        != report.get("focal_deployment_effective_sha256")
        or focal.get("focal_exact_deck_sha256") != report.get("focal_exact_deck_sha256")
    ):
        raise RuntimeError("Policy-0814 candidate deployment identity gate failed")
    if (
        schedule.get("contract_id") != CONTRACT_ID
        or schedule.get("games") != GAMES
        or schedule.get("master_seed") != MASTER_SEED
        or schedule.get("selected_class_ids") != list(SELECTED_CLASS_IDS)
        or schedule.get("opponent_policy_id") != OPPONENT_POLICY_ID
        or schedule.get("common_random_numbers") is not True
        or schedule.get("seed_derivation_excludes") != ["focal_deployment_identity"]
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != OPPONENT_POLICY_ID
        or opponent.get("effective_policy_sha256")
        != schedule.get("opponent_effective_policy_sha256")
        or cuda.get("status") != "PASS"
    ):
        raise RuntimeError("Policy-0809 Benchmark V2 opponent/schedule/CUDA gate failed")
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
    if set(class_counts.values()) != {128}:
        raise RuntimeError("Benchmark V2 Core-16 Meta allocation gate failed")
    if (
        metrics.get("rollout/lane_routing_audit_failures", 0) != 0
        or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
        or metrics.get("rollout/lane_routing_audit_pass") != 1
        or metrics.get("rollout/unfinished_games", 0) != 0
    ):
        raise RuntimeError("Benchmark V2 routing/device-residency gate failed")


def run(
    *,
    source_version: str,
    deck_id: str,
    output_root: Path,
    checkpoint: Path,
    checkpoint_update: int,
    expected_source_checkpoint_sha256: str,
    lane_count: int = DEFAULT_LANE_COUNT,
) -> dict[str, Any]:
    policy_id = candidate_policy_id(source_version, deck_id, checkpoint_update)
    if output_root.exists():
        raise FileExistsError(output_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Policy-0814 Benchmark V2 requires CUDA")
    if lane_count not in SUPPORTED_LANE_COUNTS:
        raise ValueError(f"lane count must be one of {SUPPORTED_LANE_COUNTS}")
    observed_source_sha256 = sha256_file(checkpoint)
    if observed_source_sha256 != expected_source_checkpoint_sha256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    deck_assets = {row.deck_id: row for row in registry.decks}
    if deck_id not in deck_assets or deck_id not in own_by_deck:
        raise RuntimeError("focal deck is not fully registered")
    decks = {
        row.deck_id: tuple(
            map(int, (PROJECT_ROOT / row.deck_path).read_text().splitlines())
        )
        for row in registry.decks
    }
    cards = decks[deck_id]
    device = torch.device("cuda:0")
    opponent_bundle = materialize_policy_bundle(
        PROJECT_ROOT, OPPONENT_POLICY_ID, purpose="0045_v20_benchmark_v2_opponent"
    )
    opponent = load_policy(OPPONENT_POLICY_ID, deck_id="001")
    stage = output_root.with_name(f".{output_root.name}.partial-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    focal, focal_audit = materialize_candidate(
        checkpoint=checkpoint,
        base_portable=DEFAULT_FOCAL_ACTOR_CHECKPOINT,
        base_value_checkpoint=DEFAULT_FOCAL_VALUE_CHECKPOINT,
        adaptation=DEFAULT_ADAPTATION,
        deck=cards,
        deck_id=deck_id,
        own_archetype_id=own_by_deck[deck_id],
        output=stage / "materialization/model.pt",
        device=device,
    )
    if (
        focal_audit.checkpoint_update != checkpoint_update
        or focal_audit.source_checkpoint_sha256 != expected_source_checkpoint_sha256
        or focal_audit.focal_exact_deck_sha256 != deck_assets[deck_id].content_sha256
    ):
        raise RuntimeError("materialized focal candidate does not match request")
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
        raise RuntimeError("FATAL: Policy-0814 focal / Policy-0809 opponent storage alias")
    schedule = materialize_schedule(
        PROJECT_ROOT,
        focal_deck_id=deck_id,
        focal_deployment_identity=focal_audit.effective_candidate_sha256,
    )
    runtime = build_seeded_runtime()
    jobs = [
        RolloutJob(
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
            focal_own_archetype_id=own_by_deck[deck_id],
            engine_library=runtime.library_path,
            action_boundary_mode="enabled",
            trace_policy="errors_and_sample",
        )
        for row in schedule["jobs"]
    ]
    opponent_own_ids = torch.tensor(
        [own_by_deck[row["opponent_deck_id"]] for row in schedule["jobs"]],
        dtype=torch.long,
        device=device,
    )
    collector = ChunkedCudaRolloutCollector(
        focal,
        opponent,
        rollout_batch_size=lane_count,
        trajectory_games_per_update=None,
        device=device,
        rules_path=RULES,
        extension_dir=DEFAULT_BUILD_DIR,
        lane_count=lane_count,
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
    entries = []
    for episode, row in zip(episodes, schedule["jobs"], strict=True):
        choice = episode.diagnostics.get("first_player_choice")
        if not isinstance(choice, dict) or type(choice.get("focal_first")) is not bool:
            raise RuntimeError(f"game lacks Agent seat evidence: {episode.job.game_id}")
        entries.append({
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
            "lane_routing_audit_status": episode.diagnostics.get(
                "lane_routing_audit_status"
            ),
        })
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["opponent_meta_archetype_id"]].append(entry)
    per_meta = {
        str(meta_id): _summary(rows, elapsed * len(rows) / GAMES)
        for meta_id, rows in sorted(grouped.items())
    }
    schedule_manifest = {key: value for key, value in schedule.items() if key != "jobs"}
    report = {
        "schema_version": "0045_policy0814_candidate_benchmark_v2_report_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "benchmark_id": BENCHMARK_ID,
        "focal_policy_id": policy_id,
        "focal_policy_role": "policy0814_descendant_candidate",
        "focal_base_policy_id": FOCAL_BASE_POLICY_ID,
        "focal_source_version": source_version,
        "focal_checkpoint_update": checkpoint_update,
        "focal_source_checkpoint_sha256": expected_source_checkpoint_sha256,
        "focal_policy_identity_audit": focal_audit.to_manifest(),
        "focal_deployment_effective_sha256": focal_audit.effective_candidate_sha256,
        "focal_deck_id": deck_id,
        "focal_deck_display_name": deck_assets[deck_id].name,
        "focal_exact_deck_sha256": deck_assets[deck_id].content_sha256,
        "focal_own_archetype_id": own_by_deck[deck_id],
        "focal_opponent_shared_parameter_storages": len(shared),
        "opponent_policy_identity_audit": asdict(opponent_bundle.audit),
        "schedule": schedule_manifest,
        "selection_mode": "greedy",
        "official_engine": True,
        "cuda_engine_identity_audit": CudaEngineIdentity.resolve(
            ROOT,
            rule_pack=RULES,
            binary=DEFAULT_BUILD_DIR / "ptcg_cuda_smoke",
            extension=DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
            require_gpu=True,
            require_extension=True,
        ).to_manifest(),
        "cuda_lane_count": lane_count,
        "summary": _summary(entries, elapsed),
        "per_meta": per_meta,
        "collector_metrics": collector.metrics(),
        "entries": entries,
    }
    validate_report(report)
    _atomic_json(stage / "report.json", report)
    os.replace(stage, output_root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--deck-id", default="067")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-update", required=True, type=int)
    parser.add_argument("--expected-source-checkpoint-sha256", required=True)
    parser.add_argument(
        "--lane-count", type=int, choices=SUPPORTED_LANE_COUNTS,
        default=DEFAULT_LANE_COUNT,
    )
    args = parser.parse_args()
    report = run(
        source_version=args.source_version,
        deck_id=args.deck_id,
        output_root=args.output_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        checkpoint_update=args.checkpoint_update,
        expected_source_checkpoint_sha256=args.expected_source_checkpoint_sha256,
        lane_count=args.lane_count,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
