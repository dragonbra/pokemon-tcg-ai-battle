"""Raw-FP32 U-checkpoint diagnostic on canonical 0042 CUDA-2048.

This entry point is deliberately ineligible for Kaggle, Frozen Promote, and
package-strength evidence because it skips the mandatory FP16 storage round-trip.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from ..evaluation.frozen_jobs import (
    CANONICAL_CONTRACT_ID,
    EVALUATION_UNITS,
    UNIT_GAMES,
    build_frozen_jobs,
)
from ..evaluation.frozen_panel import wilson_interval
from ..rollout import ChunkedCudaRolloutCollector
from ..rollout.deck_routing import exact_deck_sha256
from ..training import run_full_semantic as runner
from .fp32_cpu2048_policy0809 import (
    CONTRACT_ID as RAW_FP32_CONTRACT_ID,
    _load_package,
    _sha256,
    export_fp32_diagnostic,
)


SCHEMA = "0042_raw_fp32_cuda2048_policy0809_diagnostic_v1"
EXPECTED_GAMES = EVALUATION_UNITS * UNIT_GAMES


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_update(checkpoint: Path) -> int:
    digest = _sha256(checkpoint)
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != digest:
        raise RuntimeError("checkpoint SHA-256 sidecar mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    update = payload.get("update")
    if not isinstance(update, int) or update < 0:
        raise RuntimeError("checkpoint update identity is missing")
    return update


def _episode_rows(episodes: list[Any], opponent_hash: str) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        focal_first = runner._episode_focal_first(episode)
        choice = episode.diagnostics.get("first_player_choice")
        rows.append({
            "game_id": episode.job.game_id,
            "replica": int(episode.job.game_id[1:3]) - 1,
            "opponent_id": episode.job.opponent_id,
            "opponent_exact_deck_sha256": exact_deck_sha256(
                episode.job.opponent_deck
            ),
            "opponent_effective_policy_sha256": opponent_hash,
            "engine_seed": episode.job.seed,
            "search_seed": episode.job.search_seed,
            "policy_seed": episode.job.policy_seed,
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": choice,
            "focal_first": focal_first,
            "outcome": (
                1 if episode.reward == 1.0
                else -1 if episode.reward == -1.0 else 0
            ),
            "turns": episode.turns,
            "valid": episode.valid,
            "error": episode.error,
            "fallback_reason": episode.diagnostics.get("macro_fallback_reason"),
            "termination": episode.diagnostics.get("termination"),
            "lane_routing_audit_status": episode.diagnostics.get(
                "lane_routing_audit_status"
            ),
        })
    return rows


def _health_gate(
    episodes: list[Any], metrics: dict[str, float], *, expected_games: int,
) -> dict[str, Any]:
    if len(episodes) != expected_games:
        raise RuntimeError(
            f"CUDA diagnostic returned {len(episodes)} games, expected {expected_games}"
        )
    invalid = [item for item in episodes if not item.valid or item.error]
    duplicate_ids = len({item.job.game_id for item in episodes}) != expected_games
    duplicate_seeds = len({item.job.seed for item in episodes}) != expected_games
    semantic_fallbacks = [
        item for item in episodes
        if item.diagnostics.get("macro_fallback_reason")
        not in {None, runner.CHANCE_BOUNDARY_FALLBACK}
    ]
    unsupported = sum(
        int(item.diagnostics.get("unsupported_effect", 0)) for item in episodes
    )
    pending_resets = sum(
        int(item.diagnostics.get("pending_macro_reset", 0)) for item in episodes
    )
    missing_first_player_evidence = sum(
        not isinstance(item.diagnostics.get("first_player_choice"), dict)
        for item in episodes
    )
    routing_pass = metrics.get("rollout/lane_routing_audit_pass") == 1.0
    routing_failures = metrics.get("rollout/lane_routing_audit_failures") == 0.0
    resident_features = metrics.get("rollout/cuda_features_device_resident") == 1.0
    no_feature_d2h = metrics.get("rollout/cuda_feature_d2h_bytes") == 0.0
    if any((
        invalid, duplicate_ids, duplicate_seeds, semantic_fallbacks, unsupported,
        pending_resets, missing_first_player_evidence, not routing_pass,
        not routing_failures, not resident_features, not no_feature_d2h,
    )):
        raise RuntimeError(
            "raw-FP32 CUDA-2048 health gate failed: "
            f"invalid={len(invalid)} duplicate_ids={duplicate_ids} "
            f"duplicate_seeds={duplicate_seeds} "
            f"semantic_fallbacks={len(semantic_fallbacks)} "
            f"unsupported={unsupported} pending_resets={pending_resets} "
            f"missing_first_player_evidence={missing_first_player_evidence} "
            f"routing_pass={routing_pass} routing_failures={routing_failures} "
            f"resident_features={resident_features} no_feature_d2h={no_feature_d2h}"
        )
    return {
        "status": "PASS",
        "terminal_games": expected_games,
        "errors": 0,
        "unfinished": 0,
        "semantic_fallbacks": 0,
        "unsupported_effects": 0,
        "pending_macro_resets": 0,
        "unique_game_ids": expected_games,
        "unique_engine_seeds": expected_games,
        "first_player_evidence_games": expected_games,
        "lane_routing_audit": "PASS",
        "cuda_features_device_resident": True,
        "cuda_feature_d2h_bytes": 0,
    }


def _summary(rows: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    wins = sum(row["outcome"] == 1 for row in rows)
    losses = sum(row["outcome"] == -1 for row in rows)
    draws = sum(row["outcome"] == 0 for row in rows)
    first = [row for row in rows if row["focal_first"]]
    second = [row for row in rows if not row["focal_first"]]
    replicas: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        replicas[int(row["replica"])].append(row)
    low, high = wilson_interval(wins, len(rows))
    return {
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(rows),
        "wilson_95": [low, high],
        "focal_first_games": len(first),
        "focal_first_win_rate": (
            sum(row["outcome"] == 1 for row in first) / len(first)
        ),
        "focal_second_games": len(second),
        "focal_second_win_rate": (
            sum(row["outcome"] == 1 for row in second) / len(second)
        ),
        "replicas": {
            str(index): {
                "games": len(items),
                "wins": sum(row["outcome"] == 1 for row in items),
                "losses": sum(row["outcome"] == -1 for row in items),
                "draws": sum(row["outcome"] == 0 for row in items),
                "win_rate": sum(row["outcome"] == 1 for row in items) / len(items),
            }
            for index, items in sorted(replicas.items())
        },
        "elapsed_seconds": elapsed_seconds,
        "games_per_second": len(rows) / max(elapsed_seconds, 1e-9),
    }


def run(
    *, checkpoint: Path, output_root: Path, device_name: str = "cuda:0",
    lane_count: int = 256, chunk_games: int = 256,
    evaluation_units: int = EVALUATION_UNITS,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if evaluation_units not in {1, EVALUATION_UNITS}:
        raise ValueError("evaluation units must be 1 for smoke or 8 for CUDA-2048")
    if chunk_games < 1 or (UNIT_GAMES * evaluation_units) % chunk_games:
        raise ValueError("chunk games must divide the requested game count")
    if lane_count < 1 or lane_count > chunk_games:
        raise ValueError("lane count must be positive and no larger than chunk games")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("raw-FP32 CUDA diagnostic requires an available CUDA device")
    update = _checkpoint_update(checkpoint)
    output_root.mkdir(parents=True)
    package = output_root / f"u{update:03d}_fp32_diagnostic_package"
    manifest = export_fp32_diagnostic(
        checkpoint=checkpoint,
        output=package,
        evaluation_inference_device=str(device),
    )
    if (
        manifest.get("checkpoint_update") != update
        or manifest.get("storage_dtype") != "fp32"
        or manifest.get("runtime_dtype") != "fp32"
        or manifest.get("diagnostic_contract_id") != RAW_FP32_CONTRACT_ID
        or manifest.get("kaggle_strength_evidence") is not False
        or manifest.get("promote_evidence") is not False
        or manifest.get("promotion_eligible") is not False
    ):
        raise RuntimeError("raw-FP32 diagnostic manifest failed closed")
    model = _load_package(package, runner.focal_deck()).to(
        device=device, dtype=torch.float32
    ).eval().requires_grad_(False)
    if any(
        tensor.dtype != torch.float32
        for tensor in model.parameters() if torch.is_floating_point(tensor)
    ):
        raise RuntimeError("raw-FP32 CUDA runtime contains a non-FP32 parameter")
    opponent = runner.load_frozen_opponent(device)
    opponent_audit = opponent._policy_identity_audit
    if (
        opponent_audit.status != "PASS"
        or opponent_audit.requested_policy_id != runner.OPPONENT_POLICY_ID
    ):
        raise RuntimeError("FATAL: full Policy-0809 opponent identity audit failed")
    jobs, schedule_sha256 = build_frozen_jobs(
        focal_deck_id=runner.FOCAL_DECK_ID,
        focal_deck=runner.focal_deck(),
        runtime_root=runner.runtime_root(),
        source_policy_update=update,
        focal_deployment_identity=manifest["diagnostic_effective_sha256"],
        opponent_effective_policy_sha256=opponent_audit.effective_policy_sha256,
        evaluation_units=evaluation_units,
    )
    collector = ChunkedCudaRolloutCollector(
        model,
        opponent,
        rollout_batch_size=chunk_games,
        trajectory_games_per_update=None,
        device=device,
        rules_path=runner.CUDA_RULES,
        extension_dir=runner.CUDA_EXTENSION,
        lane_count=lane_count,
        mode="greedy",
        check_interval=8,
        record_trajectory=False,
        agent_selects_first_player=True,
        opponent_policy_id=opponent_audit.requested_policy_id,
        opponent_identity_audit=opponent_audit,
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    elapsed_seconds = time.perf_counter() - started
    metrics = collector.metrics()
    health = _health_gate(episodes, metrics, expected_games=len(jobs))
    rows = _episode_rows(episodes, opponent_audit.effective_policy_sha256)
    report = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "diagnostic_only": True,
        "kaggle_strength_evidence": False,
        "frozen_promote_evidence": False,
        "package_strength_evidence": False,
        "selection": "RAW FP32 DIAGNOSTIC ONLY; no Kaggle/Promote strength claim",
        "raw_fp32_contract_id": RAW_FP32_CONTRACT_ID,
        "canonical_schedule_contract_id": CANONICAL_CONTRACT_ID,
        "benchmark_label": (
            "raw_fp32_cuda2048_diagnostic"
            if evaluation_units == EVALUATION_UNITS
            else "raw_fp32_cuda256_smoke"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_update": update,
        "focal_deck_id": runner.FOCAL_DECK_ID,
        "focal_exact_deck_sha256": exact_deck_sha256(runner.focal_deck()),
        "candidate_diagnostic_effective_sha256": manifest[
            "diagnostic_effective_sha256"
        ],
        "candidate_storage_dtype": "fp32",
        "candidate_runtime_dtype": "fp32",
        "package": str(package),
        "package_manifest": manifest,
        "opponent_policy_id": runner.OPPONENT_POLICY_ID,
        "opponent_policy_identity_audit": opponent_audit.to_manifest(),
        "schedule_sha256": schedule_sha256,
        "evaluation_units": evaluation_units,
        "health": health,
        "summary": _summary(rows, elapsed_seconds),
        "collector_metrics": metrics,
        "entries": rows,
    }
    _atomic_json(output_root / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lane-count", type=int, default=256)
    parser.add_argument("--chunk-games", type=int, default=256)
    parser.add_argument("--evaluation-units", type=int, choices=(1, 8), default=8)
    args = parser.parse_args()
    report = run(
        checkpoint=args.checkpoint.resolve(),
        output_root=args.output_root.resolve(),
        device_name=args.device,
        lane_count=args.lane_count,
        chunk_games=args.chunk_games,
        evaluation_units=args.evaluation_units,
    )
    print(json.dumps({
        "report": str(args.output_root.resolve() / "report.json"),
        **report["summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
