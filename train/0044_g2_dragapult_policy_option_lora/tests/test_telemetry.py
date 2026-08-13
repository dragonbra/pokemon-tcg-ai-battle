from __future__ import annotations

import importlib

import pytest


aggregate_rollout = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.telemetry"
).aggregate_rollout
telemetry = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.telemetry")


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
            "focal_deck_id": "002" if index % 2 else "007",
            "error": False, "unfinished": False, "engine_decisions": 10,
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


def test_formal_rollout_has_chronology_rolling_slices_and_runtime_health() -> None:
    history = telemetry.RolloutHistory()
    runtime = {
        "rollout/cuda_games_per_second": 123.0,
        "rollout/strategic_decisions_per_second": 456.0,
        "rollout/cuda_features_device_resident": 1.0,
        "rollout/cuda_feature_d2h_bytes": 0.0,
        "rollout/lane_routing_audit_pass": 1.0,
        "rollout/lane_routing_audit_failures": 0.0,
        "rollout/policy_weight_loads": 1.0,
        "rollout/deck_static_cache_hits": 189.0,
        "rollout/deck_static_cache_misses": 67.0,
    }
    result = telemetry.aggregate_training_rollout(
        _games(), source_policy_update=0, checkpoint_update=1,
        curriculum_version="C000", deck_weights={"001": 1.0},
        policy_weights={"Policy-0809": .5, "Champion-G1": .5},
        history=history, runtime_metrics=runtime,
    )
    assert result["rollout/source_policy_update"] == 0
    assert result["checkpoint/update"] == 1
    assert result["rollout/rolling_100/games"] == 100
    assert result["rollout/rolling_500/games"] == 256
    assert result["rollout/candidate_localization_only"] == 1
    assert result["rollout/strength_evidence"] == 0
    assert result["rollout/focal_deck/002/games"] == 128
    assert result["rollout/opponent_policy/Champion-G1/games"] == 128
    assert result["rollout/opponent_deck/001/games"] == 64


def test_uniform_rollout_does_not_emit_pfsp_namespace() -> None:
    result = aggregate_rollout(
        _games(), curriculum_version="uniform-u000000",
        deck_weights={f"{index:03d}": 1 / 67 for index in range(1, 68)},
        policy_weights={"Champion-G2": 1.0},
        sampling_mode="uniform_001_067",
    )
    assert result["sampling/mode_uniform"] == 1.0
    assert not any(key.startswith("pfsp/") for key in result)
