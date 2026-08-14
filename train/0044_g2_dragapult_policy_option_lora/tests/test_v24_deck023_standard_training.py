from __future__ import annotations

import importlib
from pathlib import Path


def _modules():
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v24"
    )
    common = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    return runner, common


def test_v24_pins_v23_u11_and_restores_deck023_standard_contract(
    tmp_path: Path,
) -> None:
    runner, _ = _modules()
    result = runner.readiness(version_root=tmp_path / runner.VERSION)

    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["version"] == "V24_v23_u11_deck023_standard_lr_entropy"
    assert result["parent"] == {
        "version": "V23_v22_u9_final_entropy_0015",
        "source_update": 11,
        "local_start_update": 0,
        "checkpoint_sha256": runner.PARENT_CHECKPOINT_SHA256,
        "optimizer": "fresh",
        "pfsp_state": None,
    }
    assert result["focal"] == {
        "deck_id": "023",
        "display_name": "Hydrapple ex / Meganium",
        "own_archetype_id": 27,
        "schedule": "fixed_023_v1",
        "registry_roles": ["training", "evaluation"],
    }
    assert result["standard_restore"]["learning_rate_before"] == (
        "0044_v19_half_standard_lr_run_override"
    )
    assert result["standard_restore"]["learning_rate_after"] == (
        "0044_standard_lr_v1"
    )
    assert result["standard_restore"]["entropy_coefficient_before"] == 0.0015
    assert result["standard_restore"]["entropy_coefficient_after"] == 0.003
    assert result["ppo"]["learning_rate_profile"]["scale"] == 1.0
    assert result["ppo"]["learning_rate_profile"]["rates"] == {
        "allocation_learning_rate": 5e-6,
        "decoder_learning_rate": 5e-6,
        "meta_actor_residual_learning_rate": 5e-6,
        "option_lora_learning_rate": 1e-5,
        "policy_adapter_learning_rate": 5e-6,
        "prize_learning_rate": 2e-5,
        "value_learning_rate": 2e-5,
    }
    assert result["ppo"]["entropy_coefficient"] == 0.003
    assert result["ppo"]["rollout_games"] == 512
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


def test_v24_launch_forwards_exact_formal_contract(monkeypatch) -> None:
    runner, _ = _modules()
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "readiness", lambda: {})
    monkeypatch.setattr(runner, "run", lambda **kwargs: captured.update(kwargs))
    runner.launch(wandb_mode="online", updates=3)

    assert captured["version"] == runner.VERSION
    assert captured["start_update"] == 0
    assert captured["parent_checkpoint"] == runner.PARENT_CHECKPOINT
    assert captured["source_parent_version"] == runner.PARENT_VERSION
    assert captured["source_parent_update"] == 11
    assert captured["reference_anchor_update"] == 11
    assert captured["reference_anchor_identity"] == (
        "V23_v22_u9_final_entropy_0015@update-000011"
    )
    assert captured["focal_deck_ids"] == ("023",)
    assert captured["focal_deck_id"] == "023"
    assert captured["evaluation_focal_deck_id"] == "023"
    assert captured["learning_rate_profile"] == runner.LEARNING_RATE_PROFILE
    assert captured["entropy_coefficient"] == 0.003
    assert captured["rollout_games"] == 512
    assert captured["opponent_meta_weights"] == runner.OPPONENT_META_WEIGHTS
    assert captured["opponent_policy_ids"] == ("Champion-G3",)
    assert captured["periodic_evaluation_enabled"] is False
    assert captured["forward_microbatch_size"] == 256
    assert captured["behavior_probe_batch_size"] == 256
    assert captured["offload_reference_after_cache"] is True
    assert captured["wandb_run_id"] == runner.WANDB_RUN_ID
    assert captured["wandb_mode"] == "online"
    assert captured["updates"] == 3


def test_v24_design_documents_record_deck_and_standard_restore() -> None:
    runner, common = _modules()
    markdown = (
        common.ROOT / "experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md"
    ).read_text(encoding="utf-8")
    html = (
        common.ROOT / "experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html"
    ).read_text(encoding="utf-8")

    for text in (markdown, html):
        assert runner.VERSION in text
        assert "V23 U11" in text
        assert "deck 023" in text
        assert "Hydrapple ex / Meganium" in text
        assert "0044_standard_lr_v1" in text
        assert "0.003" in text
