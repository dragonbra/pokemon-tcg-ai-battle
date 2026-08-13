from __future__ import annotations

import importlib
from pathlib import Path


module = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.performance")
ROOT = Path(__file__).resolve().parents[3]


def test_cuda_2_kernel_codec_are_identical_and_routing_is_pinned() -> None:
    audit = module.audit_source_hot_path(ROOT)
    assert audit["status"] == "PASS"
    assert audit["cpu_feature_compiler_imports"] == 0
    assert len(audit["hot_path_files"]) == 2
    assert len(audit["audited_routing_files"]) == 2


def test_cuda_2_extension_keeps_official_resident_api() -> None:
    audit = module.audit_extension_api(ROOT / "engine_cuda_2_0/build/native")
    assert audit["status"] == "PASS"
    assert set(audit["methods"]) == module.EXPECTED_OFFICIAL_API


def test_0044_resident_backend_is_identity_gated_and_device_only() -> None:
    resident = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.resident")
    source = Path(resident.__file__).read_text()
    assert "CudaEngineIdentity.resolve" in source
    assert "semantic0031_resident" in source
    assert "Semantic0031ResidentRouter" in source
    assert "features_device_resident: bool = True" in source
    assert "feature_d2h_bytes: int = 0" in source
    assert "semantic_runtime.features.compiler" not in source
