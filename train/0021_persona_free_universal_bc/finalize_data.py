"""Validate corrected shards and atomically publish the 0021 data release gate."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .card_semantics import CardSemanticRegistry
from .config import MODEL_READY_DATASET
from .training.materialized import validate_materialized_dataset


AUDIT_PATH = Path("experiments/0021_persona_free_universal_bc/data_audit.json")
MANIFEST_PATH = Path("experiments/0021_persona_free_universal_bc/manifest.json")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _shape_ranges(reference: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    ranges: dict[str, dict[str, list[int]]] = {}
    for split_shards in reference["shards"].values():
        for shard in split_shards:
            for field, shape in shard["shapes"].items():
                if field not in ranges:
                    ranges[field] = {"minimum": list(shape), "maximum": list(shape)}
                    continue
                ranges[field]["minimum"] = [
                    min(old, new)
                    for old, new in zip(ranges[field]["minimum"], shape, strict=True)
                ]
                ranges[field]["maximum"] = [
                    max(old, new)
                    for old, new in zip(ranges[field]["maximum"], shape, strict=True)
                ]
    return dict(sorted(ranges.items()))


def main() -> None:
    registry = CardSemanticRegistry.from_official_csv(
        Path("data/official/EN_Card_Data.csv")
    )
    reference = validate_materialized_dataset(MODEL_READY_DATASET, registry=registry)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_episodes = audit["episode_split_counts"]
    actual_episodes = reference["split_contract"]["episode_counts"]
    if actual_episodes != expected_episodes:
        raise ValueError(
            f"materialized Episode splits changed: {actual_episodes} != {expected_episodes}"
        )
    shard_counts = {
        split: len(shards) for split, shards in reference["shards"].items()
    }
    shard_bytes = {
        split: sum(item["bytes"] for item in shards)
        for split, shards in reference["shards"].items()
    }
    audit.update(
        {
            "status": "corrected_materialization_verified_training_ready",
            "model_ready": {
                "path": str(MODEL_READY_DATASET),
                "schema_version": reference["schema_version"],
                "content_sha256": reference["content_sha256"],
                "raw_dataset_content_sha256": reference["raw_dataset_content_sha256"],
                "feature_compiler_sha256": reference["feature_compiler_sha256"],
                "materializer_sha256": reference["materializer_sha256"],
                "ontology_file_sha256": reference["ontology_file_sha256"],
                "counts": reference["counts"],
                "a0_eligible_counts": reference["a0_eligible_counts"],
                "shard_counts": shard_counts,
                "shard_bytes": shard_bytes,
                "field_shape_ranges": _shape_ranges(reference),
                "actor_source_fields": [],
                "cache_only_provenance_fields": ["source_id", "source_payload_sha256"],
            },
        }
    )
    manifest["status"] = "corrected_data_verified_training_ready"
    manifest["data"].update(
        {
            "corrected_model_ready_path": str(MODEL_READY_DATASET),
            "corrected_model_ready_content_sha256": reference["content_sha256"],
            "corrected_model_ready_schema": reference["schema_version"],
            "rematerialization_required": False,
            "rematerialization_verified": True,
        }
    )
    manifest["training"]["formal_training_allowed"] = True
    _atomic_json(AUDIT_PATH, audit)
    _atomic_json(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "content_sha256": reference["content_sha256"],
                "counts": reference["counts"],
                "a0_eligible_counts": reference["a0_eligible_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
