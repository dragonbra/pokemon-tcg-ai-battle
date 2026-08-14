from __future__ import annotations

import importlib
import subprocess
import sys


def test_execution_order_prioritizes_requested_decks_and_covers_full67() -> None:
    run = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.evaluation.benchmark_v2_g2_vs_g3_u110_full67"
    )
    assert run.EXECUTION_DECK_IDS[:5] == ("007", "003", "002", "066", "067")
    assert len(run.EXECUTION_DECK_IDS) == 67
    assert set(run.EXECUTION_DECK_IDS) == {f"{value:03d}" for value in range(1, 68)}
    assert run.EXECUTION_DECK_IDS == run.PRIORITY_DECK_IDS + tuple(
        deck_id for deck_id in run._remainder
        if deck_id not in run.EXPEDITED_DECK_IDS
    )
    assert run.PROJECT_ROOT.name == "0044_g2_dragapult_policy_option_lora"


def test_preflight_binds_g2_and_exact_terminal_g3_u110() -> None:
    run = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.evaluation.benchmark_v2_g2_vs_g3_u110_full67"
    )
    result = run.preflight(require_unused=False)
    assert result["status"] == "PASS"
    assert result["total_games"] == 67 * 2 * 2048
    assert [(row["arm"], row["update"], row["sha256"]) for row in result["arms"]] == [
        ("g2", 0, run.G2_CHECKPOINT_SHA256),
        ("g3_u110", 110, run.G3_CHECKPOINT_SHA256),
    ]


def test_help_is_read_only_and_does_not_launch_workers() -> None:
    result = subprocess.run([
        sys.executable, "-m",
        "train.0044_g2_dragapult_policy_option_lora.evaluation.benchmark_v2_g2_vs_g3_u110_full67",
        "--help",
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--launch-formal" in result.stdout
