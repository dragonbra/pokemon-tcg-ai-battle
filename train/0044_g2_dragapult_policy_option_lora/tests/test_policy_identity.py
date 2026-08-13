from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


module = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.policy_identity")
PROJECT = Path(__file__).resolve().parents[1]


def test_policy_0809_full_component_audit_passes() -> None:
    bundle = module.materialize_policy_bundle(PROJECT, "Policy-0809")
    assert bundle.audit.status == "PASS"
    assert bundle.audit.tensor_count == 293
    assert bundle.audit.storage_dtype == "fp32"
    assert bundle.audit.effective_policy_sha256 == "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"


def test_champion_g1_portable_and_reconstruction_provenance_pass() -> None:
    bundle = module.materialize_policy_bundle(PROJECT, "Champion-G1")
    assert bundle.audit.status == "PASS"
    assert bundle.audit.tensor_count == 294
    assert bundle.audit.storage_dtype == "fp16"
    assert set(bundle.audit.component_sha256) == set(module.PORTABLE_FIELDS)


def test_champion_g2_portable_and_reconstruction_provenance_pass() -> None:
    bundle = module.materialize_policy_bundle(PROJECT, "Champion-G2")
    assert bundle.audit.status == "PASS"
    assert bundle.audit.tensor_count == 294
    assert bundle.audit.storage_dtype == "fp16"
    assert bundle.audit.effective_policy_sha256 == (
        "5f314275c576c1957fe011ab554ca4807cc73680fc89f443394894a7441cb082"
    )


def test_materializations_never_share_tensor_storage() -> None:
    anchor = module.materialize_policy_bundle(PROJECT, "Policy-0809")
    champion = module.materialize_policy_bundle(PROJECT, "Champion-G1")
    g2 = module.materialize_policy_bundle(PROJECT, "Champion-G2")
    second_anchor = module.materialize_policy_bundle(PROJECT, "Policy-0809")
    module.assert_storage_isolation(anchor, champion, g2, second_anchor)
    name = next(iter(anchor.tensors))
    before = second_anchor.tensors[name].clone()
    anchor.tensors[name].view(-1)[0] += 1
    assert torch.equal(second_anchor.tensors[name], before)


def test_deliberate_alias_hard_fails() -> None:
    anchor = module.materialize_policy_bundle(PROJECT, "Policy-0809")
    alias = module.MaterializedPolicyBundle("bad", dict(anchor.tensors), anchor.audit)
    with pytest.raises(module.PolicyIdentityViolation, match="storage alias"):
        module.assert_storage_isolation(anchor, alias)
