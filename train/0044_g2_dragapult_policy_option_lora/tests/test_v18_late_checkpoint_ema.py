from __future__ import annotations

import importlib
from pathlib import Path

import torch


builder = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation.build_v18_u14_u17_ema"
)


def test_ema_uses_audited_normalized_decay_half_weights(tmp_path: Path) -> None:
    output = tmp_path / "model.pt"
    manifest_path = tmp_path / "manifest.json"
    manifest = builder.build(
        output_checkpoint=output, output_manifest=manifest_path
    )

    assert manifest["status"] == "PASS"
    assert manifest["method"] == "finite_window_normalized_ema"
    assert manifest["decay"] == 0.5
    assert manifest["source_updates"] == [14, 15, 16, 17]
    assert manifest["normalized_weights"] == {
        "14": 1 / 15, "15": 2 / 15, "16": 4 / 15, "17": 8 / 15,
    }
    assert manifest["floating_accumulation_dtype"] == "torch.float32"
    assert manifest["non_floating_source_update"] == 17
    assert len(manifest["source_checkpoint_sha256"]) == 4
    assert len(manifest["output_checkpoint_sha256"]) == 64

    derived = torch.load(output, map_location="cpu", weights_only=True)
    assert derived["schema_version"] == "0044_focal_v1_model_only_v1"
    assert derived["update"] == 17
    assert derived["derived_checkpoint"]["not_a_true_optimizer_update"] is True
    assert derived["derived_checkpoint"]["source_updates"] == [14, 15, 16, 17]
    assert derived["metadata"]["version"] == builder.DERIVED_VERSION

    sources = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in builder.SOURCE_CHECKPOINTS
    ]
    float_key = next(
        key for key, value in derived["state_dict"].items()
        if value.dtype == torch.float32
    )
    expected = sum(
        sources[index]["state_dict"][float_key].float() * weight
        for index, weight in enumerate((1 / 15, 2 / 15, 4 / 15, 8 / 15))
    )
    torch.testing.assert_close(derived["state_dict"][float_key], expected)

    integer_key = next(
        key for key, value in derived["state_dict"].items()
        if not value.is_floating_point()
    )
    assert torch.equal(
        derived["state_dict"][integer_key],
        sources[-1]["state_dict"][integer_key],
    )
