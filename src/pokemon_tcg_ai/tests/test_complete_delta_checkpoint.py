from __future__ import annotations

import importlib

import pytest
import torch


PKG = "pokemon_tcg_ai"


def _model():
    contract = importlib.import_module(
        f"{PKG}.training.train"
    )
    model, _, audit = contract._model_and_audit()
    return model, audit


def test_complete_delta_round_trip_reconstructs_full_in_memory_model():
    checkpointing = importlib.import_module(f"{PKG}.training.checkpointing")
    runner = importlib.import_module(f"{PKG}.training.runner")
    source, audit = _model()
    payload = runner._checkpoint(source, 105, version="public_test")
    assert payload["schema_version"] == checkpointing.SCHEMA_VERSION
    assert payload["update"] == 105
    assert len(payload["state_dict"]) == 125
    assert set(payload["state_dict"]) == {
        name for name, value in source.named_parameters() if value.requires_grad
    }
    assert payload["checkpoint_integrity"]["trainable_tensor_count"] == 125
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
    runner = importlib.import_module(f"{PKG}.training.runner")
    model, _ = _model()
    payload = runner._checkpoint(model, 105, version="public_test")
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
    runner = importlib.import_module(f"{PKG}.training.runner")
    model, _ = _model()
    payload = runner._checkpoint(model, 105, version="public_test")
    payload[forbidden] = {}
    with pytest.raises(RuntimeError, match="forbidden"):
        checkpointing.validate_complete_delta(model, payload)
