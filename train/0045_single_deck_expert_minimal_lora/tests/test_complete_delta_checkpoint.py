from __future__ import annotations

import importlib

import pytest
import torch


PKG = "train.0045_single_deck_expert_minimal_lora"


def _model():
    contract = importlib.import_module(
        f"{PKG}.training.run_v13_policy0814_shared_encoder"
    )
    model, _, audit = contract._model_and_audit()
    return model, audit


def test_legacy_checkpoint_filter_reproduces_exact_state_encoder_loss():
    model, audit = _model()
    state = model.state_dict()
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    legacy = {
        name
        for name, value in state.items()
        if value.requires_grad
        or not name.startswith("actor.")
        or name.startswith("actor.action_decoder.")
    }
    missing = sorted(trainable - legacy)
    assert audit["tensor_count"] == 115
    assert sum(value.requires_grad for value in state.values()) == 0
    assert len(missing) == 16
    assert all(name.startswith("actor.state_encoder.") for name in missing)
    assert all(".parametrizations." in name for name in missing)


def test_complete_delta_round_trip_reconstructs_full_in_memory_model():
    checkpointing = importlib.import_module(f"{PKG}.training.checkpointing")
    run_v1 = importlib.import_module(f"{PKG}.training.run_v1")
    source, audit = _model()
    payload = run_v1._checkpoint(source, 105, version="V16_test")
    assert payload["schema_version"] == checkpointing.SCHEMA_VERSION
    assert payload["update"] == 105
    assert len(payload["state_dict"]) == 115
    assert set(payload["state_dict"]) == {
        name for name, value in source.named_parameters() if value.requires_grad
    }
    assert payload["checkpoint_integrity"]["trainable_tensor_count"] == 115
    assert payload["checkpoint_integrity"]["total_trainable_params"] == audit[
        "total_trainable_params"
    ]

    reconstructed, _ = _model()
    checkpointing.load_complete_delta(reconstructed, payload)
    result = checkpointing.audit_reconstruction(source, reconstructed, payload)
    assert result["status"] == "PASS"
    assert result["full_state_tensor_count"] == len(source.state_dict())
    assert result["source_full_state_sha256"] == result[
        "reconstructed_full_state_sha256"
    ]


def test_complete_delta_rejects_missing_or_mutated_trainable_tensor():
    checkpointing = importlib.import_module(f"{PKG}.training.checkpointing")
    run_v1 = importlib.import_module(f"{PKG}.training.run_v1")
    model, _ = _model()
    payload = run_v1._checkpoint(model, 105, version="V16_test")
    name = next(iter(payload["state_dict"]))

    missing = dict(payload)
    missing["state_dict"] = dict(payload["state_dict"])
    missing["state_dict"].pop(name)
    with pytest.raises(RuntimeError, match="inventory"):
        checkpointing.validate_complete_delta(model, missing)

    mutated = dict(payload)
    mutated["state_dict"] = dict(payload["state_dict"])
    mutated["state_dict"][name] = mutated["state_dict"][name].clone()
    mutated["state_dict"][name].view(-1)[0] += 1
    with pytest.raises(RuntimeError, match="hash"):
        checkpointing.validate_complete_delta(model, mutated)


@pytest.mark.parametrize(
    "forbidden",
    ("optimizer_state_dict", "scheduler_state_dict", "rng_state", "rollout_buffer"),
)
def test_complete_delta_rejects_forbidden_resume_state(forbidden: str):
    checkpointing = importlib.import_module(f"{PKG}.training.checkpointing")
    run_v1 = importlib.import_module(f"{PKG}.training.run_v1")
    model, _ = _model()
    payload = run_v1._checkpoint(model, 105, version="V16_test")
    payload[forbidden] = {}
    with pytest.raises(RuntimeError, match="forbidden"):
        checkpointing.validate_complete_delta(model, payload)
