"""Independent Seeded-2048 CUDA replication for Frozen deck 002."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine_cuda_2_0.tools import evaluate_policy_0806_cuda as base


REPLICATION_SEED = 843573811
DECK_NUMBER = "002"
OUTPUT_ROOT = (
    ROOT
    / "docs/evaluation/combat_mat/policy_0806"
    / "0806_kaggle_top100_plus_v1_cuda_seeded_2048_agent_choice_replication_002_v3"
)
TEMP_ROOT = ROOT / ".tmp/evaluation/policy_0806_cuda_seeded2048_agent_choice_replication_002_v3"


def _candidate(catalog: Any, candidates: tuple[Any, ...]) -> Any:
    del catalog
    return next(
        candidate
        for candidate in candidates
        if candidate.package_manifest["frozen_deck_number"] == DECK_NUMBER
    )


def _schedule(catalog: Any, candidate: Any) -> dict[str, Any]:
    return base.build_cuda_schedule(
        focal_deck_id=candidate.name,
        entries=catalog.pool.schedule,
        evaluation_seed=REPLICATION_SEED,
    )


def _strict_result(result: dict[str, Any], schedule_path: Path) -> bool:
    guard = result.get("progress_guard", {})
    device = result.get("device", {})
    return (
        result.get("passed") is True
        and result.get("schema_version")
        == "cuda_semantic0031_resident_refill_strict_fp32_v3_agent_first_player"
        and result.get("collector", {}).get("completed_games") == base.EXPECTED_GAMES
        and result.get("collector", {}).get("errors") == 0
        and len(result.get("determinism", {}).get("game_results", []))
        == base.EXPECTED_GAMES
        and result.get("schedule", {}).get("sha256") == base._sha256(schedule_path)
        and result.get("models", {}).get("actor_checkpoint_sha256")
        == base.POLICY_SHA256
        and result.get("models", {}).get("opponent_checkpoint_sha256")
        == base.POLICY_SHA256
        and device.get("float32_matmul_precision") == "highest"
        and device.get("matmul_allow_tf32") is False
        and device.get("cudnn_allow_tf32") is False
        and guard.get("engine_turn_draw_limit") == 100
        and guard.get("full_round_draw_limit") == 50
        and result.get("per_game_diagnostics", {}).get("schema")
        == "cuda_resident_terminal_diagnostics_v2_agent_first_player"
    )


def _summarize(
    catalog: Any,
    candidate: Any,
    schedule: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    first = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    second = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    by_opponent: dict[str, dict[str, int]] = {}
    wins = losses = draws = 0
    game_records = base.build_cuda_game_records(schedule, result)
    for game in game_records:
        won = game["focal_outcome"] == "win"
        lost = game["focal_outcome"] == "loss"
        outcome = "wins" if won else "losses" if lost else "draws"
        wins += int(won)
        losses += int(lost)
        draws += int(not won and not lost)
        seat = first if game["focal_first"] else second
        seat["games"] += 1
        seat[outcome] += 1
        row = by_opponent.setdefault(
            game["opponent_id"],
            {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        )
        row["games"] += 1
        row[outcome] += 1
    if wins + losses + draws != base.EXPECTED_GAMES:
        raise RuntimeError("replication outcome total is invalid")
    if first["games"] + second["games"] != base.EXPECTED_GAMES:
        raise RuntimeError("replication actual-seat evidence is incomplete")
    return {
        "deck_number": DECK_NUMBER,
        "deck_id": candidate.name,
        "display_name": candidate.display_name,
        "representative_cards": list(candidate.representative_cards),
        "exact_deck_sha256": candidate.package_manifest["exact_deck_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "games": base.EXPECTED_GAMES,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / base.EXPECTED_GAMES,
        "first": first,
        "second": second,
        "by_opponent": by_opponent,
        "progress_guard_forfeits": result["progress_guard"][
            "forfeit_schedule_indices"
        ],
        "turn_limit_draws": result["progress_guard"][
            "turn_limit_draw_schedule_indices"
        ],
        "turn_limit_draw_contract": True,
        "wall_seconds": result["collector"]["wall_seconds"],
        "games_per_second": result["collector"]["games_per_second_wall"],
        "lane_count": int(result["collector"].get("lane_count", 0)),
        "refill_events": int(result["collector"].get("refill_events", 0)),
        "peak_reserved_bytes": result["memory"]["torch_peak_reserved_bytes"],
        "game_results_sha256": result["determinism"]["game_results_sha256"],
        "terminal_state_sha256": result["determinism"]["terminal_state_sha256"],
    }


def main() -> int:
    catalog, candidates = base._catalog()
    candidate = _candidate(catalog, candidates)
    schedule = _schedule(catalog, candidate)
    schedule_path = TEMP_ROOT / "schedule.json"
    result_path = TEMP_ROOT / "result.json"
    log_path = TEMP_ROOT / "run.log"
    base._atomic_json(schedule_path, schedule)
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if not _strict_result(cached, schedule_path):
            result_path.unlink()
    if not result_path.is_file():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(base.BENCHMARK),
            "--actor-mode",
            "0806",
            "--actor-package",
            str(base.POLICY_ROOT),
            "--opponent-model",
            str(base.POLICY_MODEL),
            "--focal-deck",
            str(base.resolve_candidate_deck_path(candidate)),
            "--deck-root",
            str(base.POOL_ROOT / "decks"),
            "--schedule",
            str(schedule_path),
            "--game-limit",
            str(base.EXPECTED_GAMES),
            "--lane-count",
            str(base.CUDA_LANE_COUNT),
            "--check-interval",
            "8",
            "--ability-repeat-limit",
            "20",
            "--engine-turn-draw-limit",
            "100",
            "--output",
            str(result_path),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            result_path.unlink(missing_ok=True)
            raise RuntimeError(f"replication failed; see {log_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not _strict_result(result, schedule_path):
        raise RuntimeError("replication failed the strict Seeded-2048 contract")
    summary = _summarize(catalog, candidate, schedule, result)
    report_path = OUTPUT_ROOT / "reports/002_alakazam_dudunsparce.html"
    base._atomic_text(
        report_path,
        base._report_html(
            catalog,
            candidate,
            summary,
            report_label="Frozen-002 · independent CUDA Seeded-2048 replication",
            back_href="../index.html",
            evaluation_seed=REPLICATION_SEED,
        ),
    )
    manifest = {
        "schema": "policy_0806_cuda_seeded2048_independent_replication_v1",
        "evaluation_seed": REPLICATION_SEED,
        "canonical_evaluation_seed": base.EVALUATION_SEED,
        "engine_seed_overlap_with_canonical": 0,
        "policy_sha256": base.POLICY_SHA256,
        "opponent_pool_id": "0806_kaggle_top100_plus_v1",
        "report": "reports/002_alakazam_dudunsparce.html",
        "summary": summary,
    }
    base._atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    base._atomic_text(
        OUTPUT_ROOT / "index.html",
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Frozen-002 独立 Seeded-2048 复测</title></head><body>'
        '<main><h1>Frozen-002 独立 Seeded-2048 复测</h1>'
        f'<p>Seed {REPLICATION_SEED}，与 canonical seed 无对局 seed 重叠。</p>'
        '<p><a href="reports/002_alakazam_dudunsparce.html">打开完整报告</a></p>'
        '</main></body></html>',
    )
    print(
        f"CUDA_002_REPLICATION_COMPLETE record={summary['wins']}-"
        f"{summary['losses']}-{summary['draws']} "
        f"wall={summary['wall_seconds']:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
