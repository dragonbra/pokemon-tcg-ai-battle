"""Run the V13 official-engine greedy eval512 against immutable Policy-0814."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
import json
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
from ..policy import AdaptationConfig
from ..policy.actor_critic import DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT
from ..policy_identity import materialize_policy_bundle
from ..rollout import ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import _modules as policy_modules
from ..runtime import load_policy
from ..training.runner import RULES, _runtime_root
from .candidate import (
    load_portable_candidate,
    materialize as materialize_candidate,
)
from .policy0814_exact_deck_schedule import (
    CONTRACT_ID, GAMES, OPPONENT_POLICY_ID, materialize as materialize_schedule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
BENCHMARK_ID = "V13-Policy0814-Exact-Deck-Eval512"
V13_ADAPTATION = AdaptationConfig(
    rank=16, alpha=16.0, output_projection=True, shared_state_encoder=True,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _summary(entries: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    games = len(entries)
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    draws = games - wins - losses
    return {
        "games": games, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / games,
        "elapsed_seconds": elapsed,
        "games_per_second": games / elapsed if elapsed else 0.0,
    }


def validate_report(report: dict[str, Any]) -> None:
    entries = report.get("entries") or []
    schedule = report.get("schedule") or {}
    opponent = report.get("opponent_policy_identity_audit") or {}
    metrics = report.get("collector_metrics") or {}
    if (
        report.get("status") != "PASS"
        or report.get("benchmark_id") != BENCHMARK_ID
        or len(entries) != GAMES
        or report.get("focal_policy_identity_audit", {}).get("status") != "PASS"
        or report.get("focal_opponent_shared_parameter_storages") != 0
        or any(row.get("valid") is not True or row.get("error") is not None for row in entries)
    ):
        raise RuntimeError("V13 eval512 lacks 512 valid deployment-audited games")
    if (
        schedule.get("contract_id") != CONTRACT_ID
        or schedule.get("opponent_policy_id") != OPPONENT_POLICY_ID
        or schedule.get("games") != GAMES
        or schedule.get("common_random_numbers") is not True
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != OPPONENT_POLICY_ID
        or opponent.get("effective_policy_sha256")
        != schedule.get("opponent_effective_policy_sha256")
    ):
        raise RuntimeError("V13 Policy-0814 eval identity gate failed")
    actual: dict[str, int] = defaultdict(int)
    for row in entries:
        actual[row["opponent_id"]] += 1
    if dict(sorted(actual.items())) != schedule.get("realized_deck_counts"):
        raise RuntimeError("V13 eval per-deck schedule changed")
    if (
        metrics.get("rollout/lane_routing_audit_failures", 0) != 0
        or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
        or metrics.get("rollout/lane_routing_audit_pass") != 1
        or metrics.get("rollout/unfinished_games", 0) != 0
    ):
        raise RuntimeError("V13 eval routing/device-residency gate failed")


def run(
    *, deck_id: str, output_root: Path, checkpoint_update: int,
    checkpoint: Path | None = None, portable_candidate: Path | None = None,
    expected_source_checkpoint_sha256: str | None = None,
    expected_portable_checkpoint_sha256: str | None = None,
    expected_effective_candidate_sha256: str | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if not torch.cuda.is_available():
        raise RuntimeError("V13 eval512 requires CUDA")
    if (checkpoint is None) == (portable_candidate is None):
        raise ValueError("provide exactly one of checkpoint or portable_candidate")
    if portable_candidate is not None and any(value is None for value in (
        expected_source_checkpoint_sha256,
        expected_portable_checkpoint_sha256,
        expected_effective_candidate_sha256,
    )):
        raise ValueError("portable candidate evaluation requires all expected hashes")
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    deck_assets = {row.deck_id: row for row in registry.decks}
    decks = {
        row.deck_id: tuple(map(int, (PROJECT_ROOT / row.deck_path).read_text().splitlines()))
        for row in registry.decks
    }
    cards = decks[deck_id]
    device = torch.device("cuda:0")
    opponent_bundle = materialize_policy_bundle(
        PROJECT_ROOT, OPPONENT_POLICY_ID, purpose="0045_v13_periodic_eval_opponent"
    )
    opponent = load_policy(OPPONENT_POLICY_ID, deck_id="001")
    output_root.mkdir(parents=True)
    if portable_candidate is not None:
        assert expected_source_checkpoint_sha256 is not None
        assert expected_portable_checkpoint_sha256 is not None
        assert expected_effective_candidate_sha256 is not None
        focal, focal_audit = load_portable_candidate(
            portable=portable_candidate,
            deck=cards,
            deck_id=deck_id,
            own_archetype_id=own_by_deck[deck_id],
            device=device,
            expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
            expected_portable_checkpoint_sha256=expected_portable_checkpoint_sha256,
            expected_effective_candidate_sha256=expected_effective_candidate_sha256,
        )
    else:
        assert checkpoint is not None
        focal, focal_audit = materialize_candidate(
            checkpoint=checkpoint,
            base_portable=DEFAULT_0814_ACTOR_CHECKPOINT,
            base_value_checkpoint=DEFAULT_0814_VALUE_CHECKPOINT,
            adaptation=V13_ADAPTATION,
            deck=cards,
            deck_id=deck_id,
            own_archetype_id=own_by_deck[deck_id],
            output=output_root / "materialization/model.pt",
            device=device,
        )
    if focal_audit.checkpoint_update != checkpoint_update:
        raise RuntimeError("requested checkpoint update does not match candidate identity")
    opponent_modules = policy_modules(opponent)
    for module in opponent_modules:
        module.to(device).eval().requires_grad_(False)
    focal_ptr = {p.untyped_storage().data_ptr() for p in focal.parameters()}
    opponent_ptr = {
        p.untyped_storage().data_ptr() for module in opponent_modules for p in module.parameters()
    }
    shared = focal_ptr & opponent_ptr
    if shared:
        raise RuntimeError("FATAL: V13 focal/opponent storage alias")
    schedule = materialize_schedule(
        PROJECT_ROOT,
        focal_deck_id=deck_id,
        focal_deployment_identity=focal_audit.effective_candidate_sha256,
    )
    runtime = build_seeded_runtime()
    jobs = [RolloutJob(
        game_id=row["game_id"], opponent_id=row["opponent_deck_id"],
        focal_first=row["focal_won_toss"], focal_won_toss=row["focal_won_toss"],
        coin_winner_seed=row["coin_winner_seed"], seed=row["engine_seed"],
        search_seed=row["search_seed"], policy_seed=row["policy_seed"],
        source_policy_update=checkpoint_update, focal_deck=cards,
        opponent_deck=decks[row["opponent_deck_id"]], runtime_root=_runtime_root(),
        opponent_policy_id=OPPONENT_POLICY_ID, focal_deck_id=deck_id,
        focal_own_archetype_id=own_by_deck[deck_id], engine_library=runtime.library_path,
        action_boundary_mode="enabled", trace_policy="errors_and_sample",
    ) for row in schedule["jobs"]]
    opponent_own_ids = torch.tensor(
        [own_by_deck[row["opponent_deck_id"]] for row in schedule["jobs"]],
        dtype=torch.long, device=device,
    )
    collector = ChunkedCudaRolloutCollector(
        focal, opponent, rollout_batch_size=512, trajectory_games_per_update=None,
        device=device, rules_path=RULES, extension_dir=DEFAULT_BUILD_DIR,
        lane_count=512, mode="greedy", check_interval=8, record_trajectory=False,
        agent_selects_first_player=True, opponent_policy_id=OPPONENT_POLICY_ID,
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
            "game_id": episode.job.game_id, "opponent_id": episode.job.opponent_id,
            "outcome": 1 if episode.reward == 1 else -1 if episode.reward == -1 else 0,
            "turns": episode.turns, "valid": episode.valid, "error": episode.error,
            "engine_seed": row["engine_seed"], "search_seed": row["search_seed"],
            "policy_seed": row["policy_seed"], "coin_winner_seed": row["coin_winner_seed"],
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": choice, "focal_first": bool(choice["focal_first"]),
        })
    per_deck = {}
    for opponent_id in sorted(schedule["realized_deck_counts"]):
        rows = [row for row in entries if row["opponent_id"] == opponent_id]
        per_deck[opponent_id] = _summary(rows, elapsed * len(rows) / GAMES)
    schedule_manifest = {key: value for key, value in schedule.items() if key != "jobs"}
    report = {
        "schema_version": "0045_v13_policy0814_exact_deck_eval512_report_v1",
        "created_at": datetime.now(UTC).isoformat(), "status": "PASS",
        "benchmark_id": BENCHMARK_ID, "focal_checkpoint_update": checkpoint_update,
        "focal_deck_id": deck_id,
        "focal_exact_deck_sha256": deck_assets[deck_id].content_sha256,
        "focal_policy_identity_audit": focal_audit.to_manifest(),
        "focal_deployment_effective_sha256": focal_audit.effective_candidate_sha256,
        "focal_opponent_shared_parameter_storages": len(shared),
        "opponent_policy_identity_audit": asdict(opponent_bundle.audit),
        "schedule": schedule_manifest, "selection_mode": "greedy", "official_engine": True,
        "cuda_engine_identity_audit": CudaEngineIdentity.resolve(
            ROOT, rule_pack=RULES, binary=DEFAULT_BUILD_DIR / "ptcg_cuda_smoke",
            extension=DEFAULT_BUILD_DIR / "_ptcg_cuda.so", require_gpu=True,
            require_extension=True,
        ).to_manifest(),
        "summary": _summary(entries, elapsed), "per_deck": per_deck,
        "collector_metrics": collector.metrics(), "entries": entries,
    }
    validate_report(report)
    _atomic_json(output_root / "report.json", report)
    _atomic_json(output_root / "summary.json", {
        "checkpoint_update": checkpoint_update,
        "summary": report["summary"], "per_deck": per_deck,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-id", default="007")
    parser.add_argument("--output-root", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--portable-candidate", type=Path)
    parser.add_argument("--checkpoint-update", required=True, type=int)
    parser.add_argument("--expected-source-checkpoint-sha256")
    parser.add_argument("--expected-portable-checkpoint-sha256")
    parser.add_argument("--expected-effective-candidate-sha256")
    args = parser.parse_args()
    report = run(
        deck_id=args.deck_id, output_root=args.output_root.resolve(),
        checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
        portable_candidate=(
            args.portable_candidate.resolve() if args.portable_candidate else None
        ),
        checkpoint_update=args.checkpoint_update,
        expected_source_checkpoint_sha256=args.expected_source_checkpoint_sha256,
        expected_portable_checkpoint_sha256=args.expected_portable_checkpoint_sha256,
        expected_effective_candidate_sha256=args.expected_effective_candidate_sha256,
    )
    print(json.dumps({"summary": report["summary"], "per_deck": report["per_deck"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
