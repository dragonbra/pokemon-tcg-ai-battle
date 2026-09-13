from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch


PKG = "train.0045_single_deck_expert_minimal_lora"
ROOT = Path(__file__).resolve().parents[3]


def test_v16_readiness_binds_stopped_v14_u105_and_doubled_state_lora_lr():
    v16 = importlib.import_module(f"{PKG}.training.run_v16_complete_delta_u105")
    ready = v16.readiness(require_pristine=False)
    assert ready["status"] == "READY_AWAITING_USER_LAUNCH"
    assert ready["parent"]["version"] == "V14_policy0814_identity_gate_fix"
    assert ready["parent"]["update"] == 105
    assert ready["parent"]["checkpoint_sha256"] == (
        "1df883f14ee026025b98ac519d25442e9497e59fda513d55663889eaa620a5a3"
    )
    assert ready["state_encoder_lora_reinitialization"]["tensor_count"] == 16
    assert ready["learning_rate_profile"]["rates"] == {
        "allocation_learning_rate": 1e-5,
        "decoder_learning_rate": 1e-5,
        "option_lora_learning_rate": 2e-5,
        "prize_learning_rate": 2e-5,
        "shared_encoder_learning_rate": 4e-5,
        "value_learning_rate": 2e-5,
    }


def test_v16_handoff_is_complete_and_exactly_reconstructable(tmp_path: Path):
    v16 = importlib.import_module(f"{PKG}.training.run_v16_complete_delta_u105")
    checkpointing = importlib.import_module(f"{PKG}.training.checkpointing")
    result = v16.materialize_sources(tmp_path)
    parent = torch.load(result["parent_checkpoint"], map_location="cpu", weights_only=True)
    reference = torch.load(
        result["reference_checkpoint"], map_location="cpu", weights_only=True
    )
    audit = json.loads(result["audit"].read_text())
    assert parent["schema_version"] == checkpointing.SCHEMA_VERSION
    assert parent["update"] == 105
    assert len(parent["state_dict"]) == 115
    assert reference["schema_version"] == checkpointing.SCHEMA_VERSION
    assert len(reference["state_dict"]) == 115
    assert audit["status"] == "PASS"
    assert audit["legacy_parent"]["preserved_trainable_tensor_count"] == 99
    assert audit["legacy_parent"]["missing_trainable_tensor_count"] == 16
    assert audit["complete_parent"]["trainable_tensor_count"] == 115
    assert audit["complete_parent"]["reconstruction_status"] == "PASS"
    assert audit["state_encoder_lora_initialization"]["all_b_tensors_zero"] is True


def test_v16_optimizer_changes_only_shared_encoder_lora_rate():
    v16 = importlib.import_module(f"{PKG}.training.run_v16_complete_delta_u105")
    ppo = importlib.import_module(f"{PKG}.training.ppo_full_semantic")
    model, _, _ = v16._seeded_model(v16.STATE_LORA_INITIALIZATION_SEED)
    trainer = ppo.PPOTrainer(
        model,
        device=torch.device("cpu"),
        config=ppo.PPOConfig(
            **v16.LEARNING_RATE_PROFILE.rates(),
            batch_size=4096,
            forward_microbatch_size=256,
            behavior_probe_batch_size=256,
            offload_reference_after_cache=True,
        ),
    )
    rates = {
        row["name"]: row["base_learning_rate"]
        for row in trainer.optimizer_group_manifest()
    }
    assert rates == {
        "action_decoder": 1e-5,
        "value_win": 2e-5,
        "value_adapter": 2e-5,
        "allocation_head": 1e-5,
        "policy_option_lora": 2e-5,
        "shared_encoder_lora": 4e-5,
        "value_prize": 2e-5,
    }


def test_v16_launch_contract_continues_from_105(monkeypatch, tmp_path: Path):
    v16 = importlib.import_module(f"{PKG}.training.run_v16_complete_delta_u105")
    captured = {}
    monkeypatch.setattr(v16, "SOURCE_ROOT", tmp_path / "source")
    monkeypatch.setattr(v16, "VERSION_ROOT", tmp_path / "version")
    monkeypatch.setattr(v16, "run", lambda **kwargs: captured.update(kwargs))
    v16.launch(wandb_mode="offline", updates=106)
    assert captured["start_update"] == 105
    assert captured["updates"] == 106
    assert captured["baseline_evaluation_checkpoint"] == 105
    assert captured["periodic_evaluation_interval_updates"] == 5
    assert captured["periodic_evaluation_profile"] == "policy0814_exact_deck_cuda512"
    assert captured["learning_rate_profile"] == v16.LEARNING_RATE_PROFILE
    assert captured["reference_anchor_identity"] == "Policy-0814-V16-Complete-U0"
    assert captured["source_parent_version"] == "V14_policy0814_identity_gate_fix"
    assert captured["source_parent_update"] == 105
