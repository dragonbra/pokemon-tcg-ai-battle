"""Build an audited finite-window EMA candidate from V18 U14-U17."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..assets import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_VERSION = "V18_g4_u57_deck069_telemetry_fix"
DERIVED_VERSION = f"{SOURCE_VERSION}__ema_u14_u17_decay_0_5"
VERSION_ROOT = (
    REPO_ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions"
    / SOURCE_VERSION
)
SOURCE_UPDATES = (14, 15, 16, 17)
SOURCE_CHECKPOINTS = tuple(
    VERSION_ROOT / f"checkpoint/update-{update:06d}.pt"
    for update in SOURCE_UPDATES
)
EXPECTED_SOURCE_HASHES = {
    "14": "e43152d75ccbf1fd1a9c3967b3b2543f3e4ea478a339e24b69c7cc4ece00ffc9",
    "15": "5bf49520f5b0e25d1ebe06acb036e9b96f497605c865d3eae7573cb3a7a45243",
    "16": "8e377b9707ea4affe153c9c72c948ab45be6f702c81aafdec6fdfe544a33e1a0",
    "17": "e05bbcc813b1315dc66c0717c158981ac43319d16b3e000e648ba0195bd04892",
}
NORMALIZED_WEIGHTS = (1 / 15, 2 / 15, 4 / 15, 8 / 15)
OUTPUT_ROOT = VERSION_ROOT / "artifact/derived_candidates/ema_u14_u17_decay_0_5"
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
        raise RuntimeError("V18 U14-U17 source checkpoint identity changed")
    sources = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in SOURCE_CHECKPOINTS
    ]
    reference = sources[-1]
    reference_state = reference.get("state_dict")
    if not isinstance(reference_state, dict):
        raise RuntimeError("V18 U17 checkpoint has no state_dict")
    for update, source in zip(SOURCE_UPDATES, sources, strict=True):
        if (
            source.get("schema_version") != "0044_focal_v1_model_only_v1"
            or source.get("update") != update
            or source.get("metadata", {}).get("version") != SOURCE_VERSION
            or source.get("actor_schema") != reference.get("actor_schema")
            or source.get("adaptation") != reference.get("adaptation")
        ):
            raise RuntimeError(f"V18 U{update} EMA source schema mismatch")
        state = source.get("state_dict")
        if not isinstance(state, dict) or tuple(state) != tuple(reference_state):
            raise RuntimeError(f"V18 U{update} EMA source keys mismatch")
        for key, newest in reference_state.items():
            value = state[key]
            if value.shape != newest.shape or value.dtype != newest.dtype:
                raise RuntimeError(f"V18 U{update} tensor mismatch: {key}")
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
        "schema_version": "0044_v18_u14_u17_ema_manifest_v1",
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
        "non_floating_source_update": 17,
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
