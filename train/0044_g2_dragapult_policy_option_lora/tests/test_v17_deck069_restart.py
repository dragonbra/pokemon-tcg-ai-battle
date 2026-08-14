from __future__ import annotations

import importlib
from pathlib import Path

def test_expanded_training_pool_contains_001_through_069() -> None:
    common = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    registry = common.AssetRegistry.load(common.PROJECT_ROOT)
    registry.validate_all()

    training_deck_ids = tuple(
        row.deck_id for row in registry.decks if "training" in row.roles
    )
    assert training_deck_ids == tuple(f"{index:03d}" for index in range(1, 70))


def test_v17_readiness_pins_u57_deck069_and_champion_g3(tmp_path: Path) -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v17"
    )

    result = runner.readiness(version_root=tmp_path / runner.VERSION)

    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["version"] == "V17_g4_u57_deck069_champion_g3_rollout_only"
    assert result["generation"] == {
        "source": "V15 G4 candidate U57",
        "target": "deck-069 specialist",
        "champion_mutation": False,
    }
    assert result["parent"] == {
        "version": "V15_g4_v14_u2_rollout_only_cuda_recovery",
        "source_update": 57,
        "local_start_update": 0,
        "checkpoint_sha256": runner.PARENT_CHECKPOINT_SHA256,
        "optimizer": "fresh",
        "pfsp_state": None,
    }
    assert result["focal"] == {
        "deck_id": "069",
        "own_archetype_id": 3,
        "schedule": "fixed_069_v1",
        "registry_role": "training",
    }
    assert result["opponents"]["policy_ids"] == ["Champion-G3"]
    assert result["opponents"]["deck_ids"] == [
        f"{index:03d}" for index in range(1, 70)
    ]
    assert result["opponents"]["sampling"] == "meta_balanced_training_pool"
    assert result["opponents"]["pfsp"] is False
    assert result["evaluation"]["enabled"] is False
    assert result["ppo"]["batch_size"] == 4096
    assert result["ppo"]["forward_microbatch_size"] == 768
    assert result["ppo"]["value_learning_rate"] == 2e-5


def test_v17_jobs_bind_069_only_against_meta_balanced_001_069() -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v17"
    )
    common = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    registry = common.AssetRegistry.load(common.PROJECT_ROOT)

    jobs, deck_weights, policy_weights, _ = common._jobs(
        0,
        registry,
        focal_deck_ids=(runner.FOCAL_DECK_ID,),
        focal_deck_id=runner.FOCAL_DECK_ID,
        focal_schedule_mode="fixed",
        opponent_sampling_mode=runner.OPPONENT_SCHEDULE,
        opponent_policy_ids=(runner.OPPONENT_POLICY_ID,),
        latest_champion_policy_id=runner.OPPONENT_POLICY_ID,
        rollout_games=runner.ROLLOUT_GAMES,
    )

    assert len(jobs) == 512
    assert {job.focal_deck_id for job in jobs} == {"069"}
    assert {job.focal_own_archetype_id for job in jobs} == {3}
    assert {job.opponent_id for job in jobs} == {
        f"{index:03d}" for index in range(1, 70)
    }
    assert len(deck_weights) == 69
    assert "068" in deck_weights and "069" in deck_weights
    assert policy_weights == {"Champion-G3": 1.0}


def test_benchmark_v2_remains_frozen_to_historical_001_067() -> None:
    common = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    schedule = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.evaluation.benchmark_v2_schedule"
    )
    payload = schedule.materialize(
        common.PROJECT_ROOT,
        focal_deck_id="069",
        focal_deployment_identity="a" * 64,
    )
    opponent_ids = {row["opponent_deck_id"] for row in payload["jobs"]}
    assert opponent_ids <= {f"{index:03d}" for index in range(1, 68)}
    assert "068" not in opponent_ids and "069" not in opponent_ids


def test_v18_retries_v17_from_original_u57_after_telemetry_fix(tmp_path: Path) -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v18"
    )
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["version"] == "V18_g4_u57_deck069_telemetry_fix"
    assert result["retry_of"] == "V17_g4_u57_deck069_champion_g3_rollout_only"
    assert result["parent"]["version"] == (
        "V15_g4_v14_u2_rollout_only_cuda_recovery"
    )
    assert result["parent"]["source_update"] == 57
    assert result["parent"]["local_start_update"] == 0
    assert result["telemetry_fix"] == {
        "root_cause": "new sampling mode omitted from telemetry closed enum",
        "accepted_mode": "meta_balanced_training_pool",
        "v17_u1_policy_evidence": "checkpoint_only_not_canonical_metrics",
    }
