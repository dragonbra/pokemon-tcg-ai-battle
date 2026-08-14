from __future__ import annotations

from collections import Counter
import importlib
from pathlib import Path


def test_v19_pins_u14_half_standard_lr_and_1024_rollout(tmp_path: Path) -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v19"
    )
    result = runner.readiness(version_root=tmp_path / runner.VERSION)

    assert result["version"] == (
        "V19_v18_u14_deck069_half_standard_lr_1024_rollout"
    )
    assert result["parent"]["version"] == "V18_g4_u57_deck069_telemetry_fix"
    assert result["parent"]["source_update"] == 14
    assert result["parent"]["checkpoint_sha256"] == runner.PARENT_CHECKPOINT_SHA256
    assert result["rollout_contract"]["games_per_update"] == 1024
    assert result["rollout_contract"]["opponent_policy_id"] == "Champion-G3"
    assert result["rollout_contract"]["opponent_meta_weights"] == {0: 2.0}
    assert result["rollout_contract"]["weighted_meta_member_decks"] == [
        "007", "018", "067",
    ]
    assert result["rollout_contract"]["evaluation_enabled"] is False

    profile = result["learning_rate_contract"]
    assert profile["profile_id"] == "0044_v19_half_standard_lr_run_override"
    assert profile["base_profile_id"] == "0044_standard_lr_v1"
    assert profile["scale"] == 0.5
    assert profile["scope"] == "run_only"
    assert profile["default_after_run"] == "0044_standard_lr_v1"
    assert profile["deck_id"] is None
    assert profile["deck_identity_bound"] is False
    assert profile["rates"] == {
        "decoder_learning_rate": 2.5e-6,
        "policy_adapter_learning_rate": 2.5e-6,
        "allocation_learning_rate": 2.5e-6,
        "option_lora_learning_rate": 5e-6,
        "meta_actor_residual_learning_rate": 2.5e-6,
        "value_learning_rate": 1e-5,
        "prize_learning_rate": 1e-5,
    }


def test_v19_doubles_display_meta_01_and_balances_its_member_decks() -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v19"
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
        opponent_meta_weights=runner.OPPONENT_META_WEIGHTS,
    )
    vocabulary = common.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=common.PROJECT_ROOT
    )
    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    meta_counts = Counter(own_by_deck[job.opponent_id] for job in jobs)
    assert len(jobs) == 1024
    assert meta_counts[0] == 71
    assert set(meta_counts.values()) <= {35, 36, 71}
    dragapult_counts = Counter(
        job.opponent_id for job in jobs if own_by_deck[job.opponent_id] == 0
    )
    assert set(dragapult_counts) == {"007", "018", "067"}
    assert max(dragapult_counts.values()) - min(dragapult_counts.values()) <= 1
    assert abs(sum(deck_weights.values()) - 1.0) < 1e-12
    assert policy_weights == {"Champion-G3": 1.0}


def test_v20_retries_same_semantics_with_only_smaller_physical_microbatch(
    tmp_path: Path,
) -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v20"
    )
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["version"] == "V20_v19_retry_microbatch512"
    assert result["retry_of"] == (
        "V19_v18_u14_deck069_half_standard_lr_1024_rollout"
    )
    assert result["parent"]["source_update"] == 14
    assert result["ppo"]["rollout_games"] == 1024
    assert result["ppo"]["batch_size"] == 4096
    assert result["ppo"]["forward_microbatch_size"] == 512
    assert result["cuda_recovery"]["semantic_change"] is False
    assert result["rollout_contract"]["opponent_meta_weights"] == {0: 2.0}


def test_v21_keeps_semantics_and_offloads_reference_only_after_cache(
    tmp_path: Path,
) -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v21"
    )
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["version"] == "V21_v20_retry_reference_offload_microbatch256"
    assert result["parent"]["source_update"] == 14
    assert result["ppo"]["rollout_games"] == 1024
    assert result["ppo"]["batch_size"] == 4096
    assert result["ppo"]["forward_microbatch_size"] == 256
    assert result["ppo"]["behavior_probe_batch_size"] == 256
    assert result["ppo"]["offload_reference_after_cache"] is True
    assert result["cuda_recovery"]["semantic_change"] is False
    assert "cached log-probs" in result["cuda_recovery"][
        "physical_changes_only"
    ]["reference_policy"]
    assert result["rollout_contract"]["games_per_update"] == 1024
    assert result["rollout_contract"]["opponent_meta_weights"] == {0: 2.0}
