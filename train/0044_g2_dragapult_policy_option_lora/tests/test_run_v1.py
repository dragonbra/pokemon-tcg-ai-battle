from __future__ import annotations

import importlib

import pytest


runner = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.training.run_v1")


def test_v1_readiness_is_green_but_does_not_launch() -> None:
    result = runner.readiness()
    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["long_training_started"] is False
    assert result["focal_decks"] == ["007"]
    assert result["opponent_policies"] == ["Champion-G2"]
    assert result["opponent_sampling"] == {
        "pfsp": 128, "uniform": 64, "latest_champion": 64,
    }
    assert result["periodic_evaluation"] == {
        "contract": "Benchmark-V2", "every_updates": 10, "cuda_games": 2048,
    }
    assert result["own_taxonomy"]["classes"] == 29
    assert result["opponent_meta"] == {"classes": 15, "trainable": False}


def test_v1_long_run_requires_explicit_launch_token() -> None:
    with pytest.raises(RuntimeError, match="--launch-formal"):
        runner.run(updates=1, wandb_mode="offline", launch_formal=False)


def test_rollout_group_is_singleton_champion_g2() -> None:
    class Job:
        def __init__(self, policy: str, deck: str) -> None:
            self.opponent_policy_id = policy
            self.focal_deck_id = deck

    groups = runner._group_jobs_by_opponent_policy([
        Job("Champion-G2", "007"), Job("Champion-G2", "007"),
    ])
    assert sorted(groups) == ["Champion-G2"]
    assert {job.focal_deck_id for job in groups["Champion-G2"]} == {"007"}
