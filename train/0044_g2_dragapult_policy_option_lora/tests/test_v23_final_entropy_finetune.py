from __future__ import annotations

import importlib
from pathlib import Path


def _modules():
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v23"
    )
    common = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    return runner, common


def test_v23_pins_v22_u9_and_changes_only_entropy_objective(
    tmp_path: Path,
) -> None:
    runner, _ = _modules()
    result = runner.readiness(version_root=tmp_path / runner.VERSION)

    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["version"] == "V23_v22_u9_final_entropy_0015"
    assert result["parent"] == {
        "version": "V22_v21_u1_rollout512_meta6_x3",
        "source_update": 9,
        "local_start_update": 0,
        "checkpoint_sha256": runner.PARENT_CHECKPOINT_SHA256,
        "optimizer": "fresh",
        "pfsp_state": None,
    }
    assert result["final_finetune"] == {
        "on_policy_data": "fresh",
        "entropy_coefficient_before": 0.003,
        "entropy_coefficient_after": 0.0015,
        "single_semantic_change": "entropy_bonus_coefficient",
    }
    assert result["ppo"]["entropy_coefficient"] == 0.0015
    assert result["ppo"]["rollout_games"] == 512
    assert result["ppo"]["batch_size"] == 4096
    assert result["ppo"]["forward_microbatch_size"] == 256
    assert result["ppo"]["behavior_probe_batch_size"] == 256
    assert result["ppo"]["offload_reference_after_cache"] is True
    assert result["ppo"]["learning_rate_profile"]["profile_id"] == (
        "0044_v19_half_standard_lr_run_override"
    )
    assert result["rollout_contract"]["opponent_meta_weights"] == {
        0: 3.0,
        1: 3.0,
        2: 3.0,
        3: 3.0,
        5: 3.0,
        27: 3.0,
    }
    assert result["rollout_contract"]["opponent_policy_id"] == "Champion-G3"
    assert result["rollout_contract"]["evaluation_enabled"] is False


def test_v23_launch_forwards_entropy_without_changing_v22_contract(
    monkeypatch,
) -> None:
    runner, _ = _modules()
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "readiness", lambda: {})
    monkeypatch.setattr(runner, "run", lambda **kwargs: captured.update(kwargs))
    runner.launch(wandb_mode="online", updates=2)

    assert captured["version"] == runner.VERSION
    assert captured["start_update"] == 0
    assert captured["parent_checkpoint"] == runner.PARENT_CHECKPOINT
    assert captured["source_parent_version"] == runner.PARENT_VERSION
    assert captured["source_parent_update"] == 9
    assert captured["reference_anchor_update"] == 9
    assert captured["reference_anchor_identity"] == (
        "V22_v21_u1_rollout512_meta6_x3@update-000009"
    )
    assert captured["entropy_coefficient"] == 0.0015
    assert captured["rollout_games"] == 512
    assert captured["opponent_meta_weights"] == runner.OPPONENT_META_WEIGHTS
    assert captured["opponent_policy_ids"] == ("Champion-G3",)
    assert captured["periodic_evaluation_enabled"] is False
    assert captured["forward_microbatch_size"] == 256
    assert captured["behavior_probe_batch_size"] == 256
    assert captured["offload_reference_after_cache"] is True
    assert captured["learning_rate_profile"] == runner.LEARNING_RATE_PROFILE
    assert captured["wandb_run_id"] == runner.WANDB_RUN_ID
    assert captured["wandb_mode"] == "online"
    assert captured["updates"] == 2


def test_run_config_entropy_override_is_local_and_default_is_unchanged() -> None:
    _, common = _modules()
    assert common._config().entropy_coefficient == 0.003
    assert common._config(entropy_coefficient=0.0015).entropy_coefficient == 0.0015


def test_v23_design_documents_record_final_finetune_contract() -> None:
    runner, common = _modules()
    markdown = (
        common.ROOT / "experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md"
    ).read_text(encoding="utf-8")
    html = (
        common.ROOT / "experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html"
    ).read_text(encoding="utf-8")

    for text in (markdown, html):
        assert runner.VERSION in text
        assert "V22 U9" in text
        assert "0.003" in text
        assert "0.0015" in text

