"""Formal 0042 custom-focal-deck CUDA evaluation against full Policy-0809.

The focal candidate follows kaggle_fp16_storage_fp32_runtime_v1.  The supplied
exact-60 deck is bound into both the portable deployment identity and every
resident CUDA lane.  This diagnostic never promotes a checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from ..candidate_deployment import (
    CONTRACT_ID,
    materialize_kaggle_evaluation_candidate,
)
from ..evaluation.frozen_jobs import (
    CANONICAL_CONTRACT_ID,
    EVALUATION_UNITS,
    UNIT_GAMES,
    build_frozen_jobs,
)
from ..rollout import ChunkedCudaRolloutCollector
from ..rollout.deck_routing import exact_deck_sha256
from ..training import run_full_semantic as runner


SCHEMA = "0042_fp16_custom_deck_cuda2048_policy0809_diagnostic_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_exact_deck(path: Path) -> tuple[int, ...]:
    cards = tuple(
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    exact_deck_sha256(cards)
    return cards


def _shared_storage_count(left: torch.nn.Module, right: torch.nn.Module) -> int:
    left_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in left.parameters()
    }
    right_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in right.parameters()
    }
    return len(left_storage & right_storage)


def _wilson(wins: int, games: int) -> list[float]:
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    spread = z * math.sqrt(
        p * (1 - p) / games + z * z / (4 * games * games)
    ) / denominator
    return [centre - spread, centre + spread]


def _rows(episodes: list[Any], opponent_hash: str) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        fallback_reason = episode.diagnostics.get("macro_fallback_reason")
        rows.append({
            "game_id": episode.job.game_id,
            "opponent_id": episode.job.opponent_id,
            "opponent_exact_deck_sha256": exact_deck_sha256(
                episode.job.opponent_deck
            ),
            "opponent_effective_policy_sha256": opponent_hash,
            "engine_seed": episode.job.seed,
            "search_seed": episode.job.search_seed,
            "policy_seed": episode.job.policy_seed,
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": episode.diagnostics.get("first_player_choice"),
            "focal_first": runner._episode_focal_first(episode),
            "outcome": (
                1 if episode.reward == 1.0
                else -1 if episode.reward == -1.0 else 0
            ),
            "turns": int(episode.turns),
            "valid": bool(episode.valid),
            "error": episode.error,
            "fallback_reason": fallback_reason,
            "chance_boundary": fallback_reason == runner.CHANCE_BOUNDARY_FALLBACK,
            "semantic_fallback": bool(fallback_reason)
            and fallback_reason != runner.CHANCE_BOUNDARY_FALLBACK,
            "lane_routing_audit_status": episode.diagnostics.get(
                "lane_routing_audit_status"
            ),
        })
    return rows


def run(
    *, checkpoint: Path, deck_path: Path, deck_id: str, deck_display_name: str,
    deck_source: str, output_root: Path, device_name: str = "cuda:0",
    evaluation_units: int = EVALUATION_UNITS,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if evaluation_units not in {1, EVALUATION_UNITS}:
        raise ValueError("evaluation units must be 1 (smoke) or 8 (CUDA-2048)")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("custom-deck CUDA evaluation requires an available CUDA device")
    deck = _load_exact_deck(deck_path)
    focal_deck_sha256 = exact_deck_sha256(deck)
    update = _checkpoint_update(checkpoint)
    output_root.mkdir(parents=True)

    model, candidate_audit = materialize_kaggle_evaluation_candidate(
        source=runner.CANDIDATE_ROOT,
        checkpoint=checkpoint,
        deck=deck,
        deck_id=deck_id,
        deck_display_name=deck_display_name,
        deck_source=deck_source,
        device=device,
        temporary_root=output_root / "materialization",
    )
    if candidate_audit.status != "PASS" or candidate_audit.contract_id != CONTRACT_ID:
        raise RuntimeError("custom focal candidate deployment audit failed")
    opponent = runner.load_frozen_opponent(device)
    opponent_audit = opponent._policy_identity_audit
    if (
        opponent_audit.status != "PASS"
        or opponent_audit.requested_policy_id != runner.OPPONENT_POLICY_ID
    ):
        raise RuntimeError("FATAL: full Policy-0809 opponent identity audit failed")
    shared_storage = _shared_storage_count(model, opponent)
    if shared_storage:
        raise RuntimeError(
            f"FATAL: focal and opponent share {shared_storage} parameter storages"
        )

    jobs, schedule_sha256 = build_frozen_jobs(
        focal_deck_id=deck_id,
        focal_deck=deck,
        runtime_root=runner.runtime_root(),
        source_policy_update=update,
        focal_deployment_identity=candidate_audit.effective_candidate_sha256,
        opponent_effective_policy_sha256=opponent_audit.effective_policy_sha256,
        evaluation_units=evaluation_units,
    )
    expected_games = evaluation_units * UNIT_GAMES
    collector = ChunkedCudaRolloutCollector(
        model,
        opponent,
        rollout_batch_size=UNIT_GAMES,
        trajectory_games_per_update=None,
        device=device,
        rules_path=runner.CUDA_RULES,
        extension_dir=runner.CUDA_EXTENSION,
        lane_count=UNIT_GAMES,
        mode="greedy",
        check_interval=8,
        record_trajectory=False,
        agent_selects_first_player=True,
        opponent_policy_id=opponent_audit.requested_policy_id,
        opponent_identity_audit=opponent_audit,
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    elapsed = time.perf_counter() - started
    metrics = collector.metrics()
    if len(episodes) != expected_games:
        raise RuntimeError(
            f"CUDA collector returned {len(episodes)} games, expected {expected_games}"
        )
    core_metrics = runner._episode_metrics(episodes, "eval/core")
    health = runner._assert_acceptance_episode_health(
        episodes, metrics, scope="eval/core", allow_chance_boundary=True
    )
    rows = _rows(episodes, opponent_audit.effective_policy_sha256)
    wins = sum(row["outcome"] == 1 for row in rows)
    losses = sum(row["outcome"] == -1 for row in rows)
    draws = sum(row["outcome"] == 0 for row in rows)
    first = [row for row in rows if row["focal_first"]]
    second = [row for row in rows if not row["focal_first"]]
    summary = {
        "games": expected_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / expected_games,
        "wilson_95": _wilson(wins, expected_games),
        "focal_first_games": len(first),
        "focal_first_win_rate": sum(row["outcome"] == 1 for row in first) / len(first),
        "focal_second_games": len(second),
        "focal_second_win_rate": sum(row["outcome"] == 1 for row in second) / len(second),
        "elapsed_seconds": elapsed,
        "games_per_second": expected_games / max(elapsed, 1e-9),
    }
    report = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "promotion_status": "NOT_PROMOTED_DIAGNOSTIC_ONLY",
        "benchmark_kind": "cuda_2048" if evaluation_units == 8 else "cuda_256_smoke",
        "candidate_deployment_contract": CONTRACT_ID,
        "canonical_schedule_contract": CANONICAL_CONTRACT_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_update": update,
        "focal_deck_id": deck_id,
        "focal_deck_display_name": deck_display_name,
        "focal_deck_source": deck_source,
        "focal_exact_deck_sha256": focal_deck_sha256,
        "focal_deck_cards": list(deck),
        "candidate_deployment_identity_audit": candidate_audit.to_manifest(),
        "opponent_policy_id": opponent_audit.requested_policy_id,
        "opponent_policy_identity_audit": opponent_audit.to_manifest(),
        "focal_opponent_shared_parameter_storages": shared_storage,
        "schedule_sha256": schedule_sha256,
        "evaluation_units": evaluation_units,
        "summary": summary,
        "core_metrics": core_metrics,
        "health": health,
        "collector_metrics": metrics,
        "entries": rows,
    }
    _atomic_json(output_root / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--deck-id", required=True)
    parser.add_argument("--deck-display-name", required=True)
    parser.add_argument("--deck-source", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-units", type=int, choices=(1, 8), default=8)
    args = parser.parse_args()
    report = run(
        checkpoint=args.checkpoint.resolve(),
        deck_path=args.deck.resolve(),
        deck_id=args.deck_id,
        deck_display_name=args.deck_display_name,
        deck_source=args.deck_source,
        output_root=args.output_root.resolve(),
        device_name=args.device,
        evaluation_units=args.evaluation_units,
    )
    print(json.dumps({"report": str(args.output_root.resolve() / "report.json"), **report["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
