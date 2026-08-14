"""Run one formal Champion-G2 focal Benchmark V1 CUDA-2048 evaluation."""

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

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..cuda_engine_2.identity import CudaEngineIdentity
from ..own_archetype import OwnArchetypeVocabulary
from ..policy_identity import materialize_policy_bundle
from ..rollout import ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import load_policy
from ..training.run_v1 import RULES, _runtime_root
from .candidate import materialize as materialize_candidate
from .benchmark_v1_schedule import GAMES, MASTER_SEED, materialize as materialize_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
VERSION = "V1_g2_dragapult_policy_option_lora"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
REPORT_ROOT = VERSION_ROOT / "artifact/reports"


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
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return [center - half, center + half]


def validate_report(report: dict[str, Any]) -> None:
    entries = report.get("entries")
    schedule = report.get("schedule") or {}
    opponent = report.get("opponent_policy_identity_audit") or {}
    metrics = report.get("collector_metrics") or {}
    if (
        report.get("status") != "PASS"
        or report.get("focal_policy_id") not in {"Champion-G2", "0044-V1-candidate"}
        or report.get("focal_opponent_shared_parameter_storages") != 0
        or not isinstance(entries, list)
        or len(entries) != GAMES
        or any(row.get("valid") is not True or row.get("error") is not None for row in entries)
    ):
        raise RuntimeError("Benchmark V1 is not 2,048 valid terminal focal games")
    if (
        schedule.get("opponent_policy_id") != "Champion-G2"
        or schedule.get("games") != GAMES
        or schedule.get("empty_class_ids") != [14]
        or opponent.get("requested_policy_id") != "Champion-G2"
        or opponent.get("effective_policy_sha256")
        != schedule.get("opponent_effective_policy_sha256")
    ):
        raise RuntimeError("Benchmark V1 fixed Champion-G2 opponent identity mismatch")
    if (
        metrics.get("rollout/lane_routing_audit_failures", 0) != 0
        or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
        or metrics.get("rollout/lane_routing_audit_pass") != 1
    ):
        raise RuntimeError("Benchmark V1 routing/device-residency gate failed")


def run(
    *, deck_id: str, output_root: Path, checkpoint: Path | None = None,
    checkpoint_update: int | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    asset = next(row for row in registry.decks if row.deck_id == deck_id)
    cards = tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
    vocabulary = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=PROJECT_ROOT)
    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    device = torch.device("cuda:0")
    opponent_bundle = materialize_policy_bundle(PROJECT_ROOT, "Champion-G2", purpose="benchmark_v1_opponent")
    opponent = load_policy("Champion-G2", deck_id="001")
    output_root.mkdir(parents=True)
    g2_policy = next(row for row in registry.policies if row.policy_id == "Champion-G2")
    source_checkpoint = checkpoint or next(
        PROJECT_ROOT / row.path for row in g2_policy.artifacts
        if row.purpose == "model_only_checkpoint"
    )
    base_portable = PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin"
    focal, focal_audit = materialize_candidate(
        checkpoint=source_checkpoint, base_portable=base_portable,
        deck=cards, deck_id=deck_id, own_archetype_id=own_by_deck[deck_id],
        output=output_root / "materialization/model.bin", device=device,
    )
    for module in (
        opponent.actor, opponent.value_head, opponent.allocation_head,
        opponent.value_adapter, opponent.policy_strategy_adapter,
    ):
        module.to(device).eval().requires_grad_(False)
    focal_ptr = {parameter.untyped_storage().data_ptr() for parameter in focal.parameters()}
    opponent_ptr = {
        parameter.untyped_storage().data_ptr()
        for module in (
            opponent.actor, opponent.value_head, opponent.allocation_head,
            opponent.value_adapter, opponent.policy_strategy_adapter,
        ) for parameter in module.parameters()
    }
    shared = focal_ptr & opponent_ptr
    if shared:
        raise RuntimeError("FATAL: Benchmark V1 focal/opponent storage alias")
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
        policy_seed=row["policy_seed"], source_policy_update=(checkpoint_update or 407),
        focal_deck=cards, opponent_deck=decks[row["opponent_deck_id"]],
        runtime_root=_runtime_root(), opponent_policy_id="Champion-G2",
        focal_deck_id=deck_id, focal_own_archetype_id=own_by_deck[deck_id],
        engine_library=runtime.library_path, action_boundary_mode="enabled",
        trace_policy="errors_and_sample",
    ) for row in schedule["jobs"]]
    opponent_own_ids = torch.tensor(
        [own_by_deck[job.opponent_id] for job in jobs], dtype=torch.long, device=device
    )
    collector = ChunkedCudaRolloutCollector(
        focal, opponent, rollout_batch_size=256, trajectory_games_per_update=None,
        device=device, rules_path=RULES, extension_dir=DEFAULT_BUILD_DIR,
        lane_count=256, mode="greedy", check_interval=8, record_trajectory=False,
        agent_selects_first_player=True, opponent_policy_id="Champion-G2",
        opponent_identity_audit=opponent_bundle.audit,
        opponent_own_archetype_ids=opponent_own_ids,
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    elapsed = time.perf_counter() - started
    metrics = collector.metrics()
    entries = []
    for episode, schedule_row in zip(episodes, schedule["jobs"], strict=True):
        diagnostics = episode.diagnostics
        choice = diagnostics.get("first_player_choice")
        if not isinstance(choice, dict) or type(choice.get("focal_first")) is not bool:
            raise RuntimeError(f"Benchmark game lacks Agent seat evidence: {episode.job.game_id}")
        entries.append({
            "game_id": episode.job.game_id,
            "opponent_id": episode.job.opponent_id,
            "opponent_meta_archetype_id": schedule_row["opponent_meta_archetype_id"],
            "outcome": 1 if episode.reward == 1 else -1 if episode.reward == -1 else 0,
            "turns": episode.turns, "valid": episode.valid, "error": episode.error,
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
        "schema_version": "0044_benchmark_v1_report_v1",
        "created_at": datetime.now(UTC).isoformat(), "status": "PASS",
        "benchmark_id": "Benchmark-V1", "version": VERSION,
        "focal_policy_id": (
            "Champion-G2" if checkpoint is None else "0044-V1-candidate"
        ),
        "focal_policy_role": (
            "promoted_immutable" if checkpoint is None else "periodic_training_candidate"
        ),
        "focal_checkpoint_update": checkpoint_update or 407,
        "focal_policy_identity_audit": focal_audit.to_manifest(),
        "focal_deployment_effective_sha256": focal_audit.effective_candidate_sha256,
        "focal_deck_id": deck_id, "focal_deck_display_name": asset.name,
        "focal_exact_deck_sha256": asset.content_sha256,
        "focal_deck_cards": list(cards),
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
    parser.add_argument("--deck-id", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-update", type=int)
    args = parser.parse_args()
    output = args.output_root or REPORT_ROOT / args.deck_id
    report = run(
        deck_id=args.deck_id, output_root=output.resolve(),
        checkpoint=args.checkpoint, checkpoint_update=args.checkpoint_update,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
