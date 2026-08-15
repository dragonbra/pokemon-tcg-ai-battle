"""Build an audited finite-window EMA candidate from V23 U7-U10."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch

from ..assets import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_VERSION = "V23_v22_u9_final_entropy_0015"
DERIVED_VERSION = f"{SOURCE_VERSION}__ema_u7_u10_decay_0_5"
VERSION_ROOT = (
    REPO_ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions"
    / SOURCE_VERSION
)
SOURCE_UPDATES = (7, 8, 9, 10)
SOURCE_CHECKPOINTS = tuple(
    VERSION_ROOT / f"checkpoint/update-{update:06d}.pt"
    for update in SOURCE_UPDATES
)
EXPECTED_SOURCE_HASHES = {
    "7": "bb124085a990d07fe665d93cc6f2e608648d79d038aca6ae42af29dab40545e1",
    "8": "c1986e5e1e6775e4f954126b7da2a2e8744f0ce6e406dd7b3c5dad137e720885",
    "9": "4d5e08bd0e40e1071f2d22fecd44b505dc4812ac3144bd29cdd555bb0dbe076d",
    "10": "bedee3d637c44d361389cd4e8be7ad03df9cf7a174f0d0f056aab716552fe1e8",
}
NORMALIZED_WEIGHTS = (1 / 15, 2 / 15, 4 / 15, 8 / 15)
OUTPUT_ROOT = VERSION_ROOT / "artifact/derived_candidates/ema_u7_u10_decay_0_5"
OUTPUT_CHECKPOINT = OUTPUT_ROOT / "model.pt"
OUTPUT_MANIFEST = OUTPUT_ROOT / "manifest.json"


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_sources() -> list[dict[str, Any]]:
    source_hashes = {
        str(update): sha256_file(path)
        for update, path in zip(SOURCE_UPDATES, SOURCE_CHECKPOINTS, strict=True)
    }
    if source_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("V23 U7-U10 source checkpoint identity changed")
    sources = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in SOURCE_CHECKPOINTS
    ]
    reference = sources[-1]
    reference_state = reference.get("state_dict")
    if not isinstance(reference_state, dict):
        raise RuntimeError("V23 U10 checkpoint has no state_dict")
    for update, source in zip(SOURCE_UPDATES, sources, strict=True):
        if (
            source.get("schema_version") != "0044_focal_v1_model_only_v1"
            or source.get("update") != update
            or source.get("metadata", {}).get("version") != SOURCE_VERSION
            or source.get("actor_schema") != reference.get("actor_schema")
            or source.get("adaptation") != reference.get("adaptation")
            or source.get("integrated_flags") != reference.get("integrated_flags")
        ):
            raise RuntimeError(f"V23 U{update} EMA source schema mismatch")
        state = source.get("state_dict")
        if not isinstance(state, dict) or tuple(state) != tuple(reference_state):
            raise RuntimeError(f"V23 U{update} EMA source keys mismatch")
        for key, newest in reference_state.items():
            value = state[key]
            if value.shape != newest.shape or value.dtype != newest.dtype:
                raise RuntimeError(f"V23 U{update} tensor mismatch: {key}")
    return sources


def build(
    *, output_checkpoint: Path = OUTPUT_CHECKPOINT,
    output_manifest: Path = OUTPUT_MANIFEST,
) -> dict[str, Any]:
    sources = _load_sources()
    newest = sources[-1]
    newest_state = newest["state_dict"]
    ema_state: dict[str, torch.Tensor] = {}
    for key, newest_value in newest_state.items():
        values = [source["state_dict"][key] for source in sources]
        if newest_value.is_floating_point():
            if any(value.dtype != torch.float32 for value in values):
                raise RuntimeError(f"EMA floating tensor is not FP32: {key}")
            average = torch.zeros_like(newest_value, dtype=torch.float32)
            for value, weight in zip(values, NORMALIZED_WEIGHTS, strict=True):
                average.add_(value.float(), alpha=float(weight))
            ema_state[key] = average
        else:
            ema_state[key] = newest_value.clone()

    payload = deepcopy(newest)
    payload["state_dict"] = ema_state
    payload["metadata"] = {
        **deepcopy(newest["metadata"]),
        "version": DERIVED_VERSION,
        "derived_from_version": SOURCE_VERSION,
        "derived_from_updates": list(SOURCE_UPDATES),
    }
    payload["derived_checkpoint"] = {
        "method": "finite_window_normalized_ema",
        "decay": 0.5,
        "source_updates": list(SOURCE_UPDATES),
        "normalized_weights": {
            str(update): weight
            for update, weight in zip(
                SOURCE_UPDATES, NORMALIZED_WEIGHTS, strict=True
            )
        },
        "not_a_true_optimizer_update": True,
    }
    _atomic_torch(output_checkpoint, payload)
    manifest = {
        "schema_version": "0044_v23_u7_u10_ema_manifest_v1",
        "status": "PASS",
        "method": "finite_window_normalized_ema",
        "decay": 0.5,
        "source_version": SOURCE_VERSION,
        "derived_version": DERIVED_VERSION,
        "source_updates": list(SOURCE_UPDATES),
        "normalized_weights": {
            str(update): weight
            for update, weight in zip(
                SOURCE_UPDATES, NORMALIZED_WEIGHTS, strict=True
            )
        },
        "source_checkpoint_sha256": dict(EXPECTED_SOURCE_HASHES),
        "floating_accumulation_dtype": str(torch.float32),
        "non_floating_source_update": 10,
        "state_dict_key_count": len(ema_state),
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": sha256_file(output_checkpoint),
        "not_a_true_optimizer_update": True,
    }
    _atomic_json(output_manifest, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-checkpoint", type=Path, default=OUTPUT_CHECKPOINT)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(build(
        output_checkpoint=args.output_checkpoint,
        output_manifest=args.output_manifest,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
