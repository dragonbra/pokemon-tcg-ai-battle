from __future__ import annotations

from collections import Counter
import importlib
from pathlib import Path


def _modules():
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v22"
    )
    common = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    return runner, common


def test_v22_continues_exact_v21_u1_with_fresh_training_state(
    tmp_path: Path,
) -> None:
    runner, _ = _modules()
    result = runner.readiness(version_root=tmp_path / runner.VERSION)

    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["version"] == "V22_v21_u1_rollout512_meta6_x3"
    assert result["parent"] == {
        "version": "V21_v20_retry_reference_offload_microbatch256",
        "source_update": 1,
        "local_start_update": 0,
        "checkpoint_sha256": runner.PARENT_CHECKPOINT_SHA256,
        "optimizer": "fresh",
        "pfsp_state": None,
    }
    assert result["continuation"]["v21_status"] == "failed_after_durable_u1"
    assert result["continuation"]["discarded_partial_update"] == 2
    assert result["continuation"]["on_policy_data"] == "fresh"
    assert result["ppo"]["rollout_games"] == 512
    assert result["ppo"]["batch_size"] == 4096
    assert result["ppo"]["forward_microbatch_size"] == 256
    assert result["ppo"]["behavior_probe_batch_size"] == 256
    assert result["ppo"]["offload_reference_after_cache"] is True
    assert result["rollout_contract"]["opponent_policy_id"] == "Champion-G3"
    assert result["rollout_contract"]["opponent_meta_weights"] == {
        0: 3.0,
        1: 3.0,
        2: 3.0,
        3: 3.0,
        5: 3.0,
        27: 3.0,
    }
    assert result["rollout_contract"]["evaluation_enabled"] is False
    assert result["rollout_contract"]["pfsp_enabled"] is False


def test_v22_schedule_assigns_38_lanes_to_each_weighted_meta() -> None:
    runner, common = _modules()
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

    assert len(jobs) == 512
    assert {job.focal_deck_id for job in jobs} == {"069"}
    assert {job.opponent_id for job in jobs} == {
        f"{deck_id:03d}" for deck_id in range(1, 70)
    }
    assert {meta_counts[meta_id] for meta_id in runner.WEIGHTED_META_IDS} == {38}
    assert {
        count
        for meta_id, count in meta_counts.items()
        if meta_id not in runner.WEIGHTED_META_IDS
    } <= {12, 13}
    assert abs(sum(deck_weights.values()) - 1.0) < 1e-12
    assert policy_weights == {"Champion-G3": 1.0}


def test_v22_launch_forwards_exact_formal_contract(monkeypatch) -> None:
    runner, _ = _modules()
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "readiness", lambda: {})
    monkeypatch.setattr(runner, "run", lambda **kwargs: captured.update(kwargs))
    runner.launch(wandb_mode="online", updates=3)

    assert captured["version"] == runner.VERSION
    assert captured["start_update"] == 0
    assert captured["parent_checkpoint"] == runner.PARENT_CHECKPOINT
    assert captured["parent_pfsp_state"] is None
    assert captured["source_parent_version"] == runner.PARENT_VERSION
    assert captured["source_parent_update"] == 1
    assert captured["reference_anchor_update"] == 1
    assert captured["reference_anchor_identity"] == (
        "V21_v20_retry_reference_offload_microbatch256@update-000001"
    )
    assert captured["rollout_games"] == 512
    assert captured["opponent_meta_weights"] == runner.OPPONENT_META_WEIGHTS
    assert captured["opponent_policy_ids"] == ("Champion-G3",)
    assert captured["latest_champion_policy_id"] == "Champion-G3"
    assert captured["periodic_evaluation_enabled"] is False
    assert captured["forward_microbatch_size"] == 256
    assert captured["behavior_probe_batch_size"] == 256
    assert captured["offload_reference_after_cache"] is True
    assert captured["wandb_run_id"] == runner.WANDB_RUN_ID
    assert captured["wandb_mode"] == "online"
    assert captured["updates"] == 3


def test_v22_design_documents_record_current_contract() -> None:
    runner, common = _modules()
    markdown = (
        common.ROOT / "experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md"
    ).read_text(encoding="utf-8")
    html = (
        common.ROOT / "experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html"
    ).read_text(encoding="utf-8")

    for text in (markdown, html):
        assert runner.VERSION in text
        assert "V21 U1" in text
        assert "512" in text
        assert "00/01/02/03/05/27" in text
        assert "3.0" in text

