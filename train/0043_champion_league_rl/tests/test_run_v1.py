from __future__ import annotations

import importlib

import pytest


runner = importlib.import_module("train.0043_champion_league_rl.training.run_v1")


def test_v1_readiness_is_green_but_does_not_launch() -> None:
    result = runner.readiness()
    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["long_training_started"] is False
    assert result["focal_decks"] == ["002", "007"]
    assert result["opponent_policies"] == ["Policy-0809", "Champion-G1"]
    assert result["own_taxonomy"]["classes"] == 29
    assert result["opponent_meta"] == {"classes": 15, "trainable": False}


def test_v1_long_run_requires_explicit_launch_token() -> None:
    with pytest.raises(RuntimeError, match="--launch-formal"):
        runner.run(updates=1, wandb_mode="offline", launch_formal=False)


def test_rollout_groups_isolate_opponent_policy_not_focal_deck() -> None:
    class Job:
        def __init__(self, policy: str, deck: str) -> None:
            self.opponent_policy_id = policy
            self.focal_deck_id = deck

    groups = runner._group_jobs_by_opponent_policy([
        Job("Policy-0809", "002"), Job("Policy-0809", "007"),
        Job("Champion-G1", "002"), Job("Champion-G1", "007"),
    ])
    assert sorted(groups) == ["Champion-G1", "Policy-0809"]
    assert {job.focal_deck_id for job in groups["Policy-0809"]} == {"002", "007"}
    assert {job.focal_deck_id for job in groups["Champion-G1"]} == {"002", "007"}
