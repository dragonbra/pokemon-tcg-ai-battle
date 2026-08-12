from __future__ import annotations

import importlib

import pytest


aggregate_rollout = importlib.import_module(
    "train.0043_champion_league_rl.telemetry"
).aggregate_rollout


def _games():
    rows = []
    for index in range(256):
        result = ("win", "loss", "draw")[index % 3]
        rows.append({
            "result": result,
            "focal_prizes_taken": 6 if result == "win" else (4 if result == "loss" else 3),
            "opponent_prizes_taken": 2 if result == "win" else (6 if result == "loss" else 3),
            "full_turns": 8 if result == "win" else (12 if result == "loss" else 50),
            "opponent_deck_id": f"{index % 4 + 1:03d}",
            "opponent_policy_id": "Policy-0809" if index % 2 else "Champion-G1",
            "branch": ("pfsp", "uniform", "latest_champion")[index % 3],
        })
    return rows


def test_rollout_aggregation_splits_win_loss_prize_and_turn_metrics() -> None:
    result = aggregate_rollout(
        _games(), curriculum_version="C002",
        deck_weights={f"{index:03d}": 0.25 for index in range(1, 5)},
        policy_weights={"Policy-0809": 0.5, "Champion-G1": 0.5},
    )
    assert result["rollout/games"] == 256
    assert result["rollout/curriculum_version"] == "C002"
    assert result["strength/prizes_taken_on_loss"] == 4
    assert result["strength/opponent_prizes_taken_on_win"] == 2
    assert result["strength/turns_to_win"] == 8
    assert result["strength/turns_to_loss"] == 12
    assert result["pfsp/sampling_entropy"] > 0
    assert set(result["pfsp/deck_difficulty"]) == {"001", "002", "003", "004"}


def test_telemetry_rejects_non_256_or_incomplete_games() -> None:
    with pytest.raises(ValueError, match="exactly 256"):
        aggregate_rollout([], curriculum_version="C000", deck_weights={}, policy_weights={})
    rows = _games()
    rows[0].pop("full_turns")
    with pytest.raises(ValueError, match="missing telemetry fields"):
        aggregate_rollout(rows, curriculum_version="C000", deck_weights={}, policy_weights={})
