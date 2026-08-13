from __future__ import annotations

import importlib

import pytest


module = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.training.periodic_evaluation"
)


def test_evaluation_is_due_exactly_every_ten_saved_updates() -> None:
    assert [value for value in range(31) if module.is_due(value)] == [10, 20, 30]


def test_wandb_metrics_are_checkpoint_bound_and_not_rollout_metrics() -> None:
    report = {
        "status": "PASS", "benchmark_id": "Benchmark-V2",
        "focal_checkpoint_update": 20,
        "summary": {
            "games": 2048, "wins": 1100, "losses": 900, "draws": 48,
            "win_rate": 1100 / 2048, "focal_first_win_rate": 0.55,
            "focal_second_win_rate": 0.52, "games_per_second": 123.0,
        },
    }
    result = module.wandb_metrics(report)
    assert result["eval/checkpoint_update"] == 20
    assert result["eval/games"] == 2048
    assert result["eval/benchmark_v2"] == 1.0
    assert result["eval/common_random_numbers"] == 1.0
    assert not any(key.startswith("rollout/") for key in result)


def test_non_pass_or_off_boundary_report_fails_closed() -> None:
    with pytest.raises(RuntimeError):
        module.wandb_metrics({"status": "FAIL"})


def test_u0_parent_baseline_is_allowed() -> None:
    report = {
        "status": "PASS", "benchmark_id": "Benchmark-V2",
        "focal_checkpoint_update": 0,
        "summary": {
            "games": 2048, "wins": 1000, "losses": 1000, "draws": 48,
            "win_rate": 1000 / 2048, "focal_first_win_rate": 0.49,
            "focal_second_win_rate": 0.48, "games_per_second": 100.0,
        },
    }
    result = module.wandb_metrics(report, initial_baseline_update=0)
    assert result["eval/checkpoint_update"] == 0


def test_inherited_non_tens_checkpoint_is_allowed_only_as_initial_baseline() -> None:
    report = {
        "status": "PASS", "benchmark_id": "Benchmark-V2",
        "focal_checkpoint_update": 4,
        "summary": {
            "games": 2048, "wins": 1000, "losses": 1000, "draws": 48,
            "win_rate": 1000 / 2048, "focal_first_win_rate": 0.49,
            "focal_second_win_rate": 0.48, "games_per_second": 100.0,
        },
    }
    with pytest.raises(RuntimeError):
        module.wandb_metrics(report)
    result = module.wandb_metrics(report, initial_baseline_update=4)
    assert result["eval/checkpoint_update"] == 4
