"""Create the V1 mutable focal seed by parent-copy expansion of G1 own embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .assets import canonical_json_sha256, sha256_file
from .own_archetype import OwnArchetypeVocabulary


PROJECT_ROOT = Path(__file__).resolve().parent
G1_ROOT = PROJECT_ROOT / "assets/policies/definitions/champion_g001"
VERSION_ROOT = PROJECT_ROOT.parents[1] / "rl_runs/0043_champion_league_rl/versions/V1_focal_002_007"
FOCAL_CHECKPOINT = VERSION_ROOT / "checkpoint/update-000000.pt"
FOCAL_PORTABLE = VERSION_ROOT / "artifact/focal_seed/model.bin"
MIGRATION_PATH = PROJECT_ROOT / "assets/taxonomy/migrations/v1_to_v2.json"
G1_EXPECTED = {
    "source_update_000010.pt": "aea408e8e140ceb28d23962cf45abac76a29be01888953bbbc5253ff6d120fe3",
    "model.bin": "cd5c05741aadf16db8bb8ca4f527e02eae431db75c33eba30f291fe89f57de84",
}
FP32_KEYS = (
    "value_adapter.own_embedding.weight",
    "policy_strategy_adapter.own_embedding.weight",
)
PORTABLE_FIELDS = {
    "value_adapter_state_dict": "own_embedding.weight",
    "policy_strategy_adapter_state_dict": "own_embedding.weight",
}


def _tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(b"\0")
        digest.update(str(tensor.dtype).encode()); digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0"); digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _expand(old: torch.Tensor, parents: tuple[int, ...], new_size: int) -> torch.Tensor:
    if tuple(old.shape) != (15, 16):
        raise RuntimeError(f"unexpected G1 own embedding shape: {tuple(old.shape)}")
    result = old.new_empty((new_size, 16))
    result[:15].copy_(old)
    for new_id, parent_id in enumerate(parents, start=15):
        result[new_id].copy_(old[parent_id])
    if not torch.equal(result[:15], old):
        raise RuntimeError("G1 rows changed during migration")
    return result


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def migrate() -> dict[str, Any]:
    for filename, expected in G1_EXPECTED.items():
        if sha256_file(G1_ROOT / filename) != expected:
            raise RuntimeError(f"Champion-G1 changed: {filename}")
    v1 = OwnArchetypeVocabulary.load_version("0042_own_archetypes_v1")
    v2 = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=PROJECT_ROOT)
    if [(row.archetype_id, row.name) for row in v2.classes[:15]] != [
        (row.archetype_id, row.name) for row in v1.classes
    ]:
        raise RuntimeError("V2 rewrites a frozen V1 ID")
    parents = tuple(row.embedding_init_from for row in v2.classes[15:])
    if any(not 0 <= parent < 15 for parent in parents):
        raise RuntimeError("new archetype lacks a valid G1 parent row")

    fp32 = torch.load(G1_ROOT / "source_update_000010.pt", map_location="cpu", weights_only=True)
    if set(fp32) != {"schema_version", "update", "actor_schema", "state_dict", "adaptation", "integrated_flags", "metadata"}:
        raise RuntimeError("G1 model-only checkpoint inventory changed")
    fp32_state = dict(fp32["state_dict"])
    for key in FP32_KEYS:
        fp32_state[key] = _expand(fp32_state[key], parents, v2.class_count)
    fp32_metadata = dict(fp32["metadata"])
    fp32_metadata.update({
        "project_id": "0043_champion_league_rl",
        "version": "V1-Focal-Seed",
        "model_schema": "0043_strategy_conditioned_own_taxonomy_v2",
        "own_archetype_class_count": v2.class_count,
        "own_embedding_dim": 16,
        "own_archetype_vocabulary_version": v2.taxonomy_version,
        "own_archetype_taxonomy_sha256": v2.taxonomy_sha256,
        "own_deck_mapping_sha256": v2.mapping_sha256,
        "parent_policy_id": "Champion-G1",
        "migration_id": "own_archetype_taxonomy_v1_to_v2",
        "optimizer_state_migrated": False,
    })
    fp32_out = {
        **fp32, "schema_version": "0043_focal_v1_model_only_v1", "update": 0,
        "state_dict": fp32_state, "metadata": fp32_metadata,
    }
    fp32_path = FOCAL_CHECKPOINT
    _atomic_torch_save(fp32_out, fp32_path)

    portable = torch.load(G1_ROOT / "model.bin", map_location="cpu", weights_only=True)
    portable_out = dict(portable)
    for field, key in PORTABLE_FIELDS.items():
        state = dict(portable[field])
        state[key] = _expand(state[key], parents, v2.class_count)
        portable_out[field] = state
    portable_metadata = dict(portable["metadata"])
    portable_metadata.update({
        "project_id": "0043_champion_league_rl", "version": "V1-Focal-Seed",
        "checkpoint_update": 0, "checkpoint_sha256": sha256_file(fp32_path),
        "own_archetype_class_count": v2.class_count,
        "own_archetype_vocabulary_version": v2.taxonomy_version,
        "own_archetype_taxonomy_sha256": v2.taxonomy_sha256,
        "own_deck_mapping_sha256": v2.mapping_sha256,
        "parent_policy_id": "Champion-G1",
        "migration_id": "own_archetype_taxonomy_v1_to_v2",
    })
    portable_out["schema_version"] = "0043_focal_v1_kaggle_candidate_v1"
    portable_out["metadata"] = portable_metadata
    portable_path = FOCAL_PORTABLE
    _atomic_torch_save(portable_out, portable_path)

    all_tensors = {
        f"{field}.{name}": tensor
        for field in (
            "actor_state_dict", "value_head_state_dict", "allocation_head_state_dict",
            "value_adapter_state_dict", "policy_strategy_adapter_state_dict",
        )
        for name, tensor in portable_out[field].items()
    }
    effective_hash = _tensor_hash(all_tensors)
    migration = {
        "schema_version": "0043_own_archetype_embedding_migration_v1",
        "migration": {"from": v1.taxonomy_version, "to": v2.taxonomy_version},
        "source_policy_id": "Champion-G1", "target_run_seed": "V1-Focal-Seed",
        "source_artifact_sha256": G1_EXPECTED,
        "old_size": 15, "new_size": v2.class_count, "embedding_width": 16,
        "preserved_archetype_ids": list(range(15)),
        "added_archetypes": [
            {"archetype_id": row.archetype_id, "name": row.name,
             "parent_id": row.embedding_init_from, "init": "copy_parent"}
            for row in v2.classes[15:]
        ],
        "tensor_migrations": {
            "value_adapter.own_embedding.weight": {"old_shape": [15,16], "new_shape": [v2.class_count,16]},
            "policy_strategy_adapter.own_embedding.weight": {"old_shape": [15,16], "new_shape": [v2.class_count,16]},
        },
        "old_rows_preserved": True, "optimizer_state_migrated": False,
        "taxonomy_sha256": v2.taxonomy_sha256, "deck_mapping_sha256": v2.mapping_sha256,
    }
    _write_json(MIGRATION_PATH, migration)
    migration_hash = sha256_file(MIGRATION_PATH)
    manifest = {
        "schema_version": "0043_focal_seed_manifest_v1",
        "run_version": "V1_focal_002_007", "role": "mutable_focal_seed",
        "promoted_policy_id": None, "generation": None,
        "frozen": False, "parent_policy_id": "Champion-G1",
        "architecture_version": "0043_strategy_conditioned_own_taxonomy_v2",
        "taxonomy": {"version": v2.taxonomy_version, "sha256": v2.taxonomy_sha256,
                     "class_count": v2.class_count, "mapping_sha256": v2.mapping_sha256},
        "migration": {"id": "own_archetype_taxonomy_v1_to_v2", "manifest_sha256": migration_hash},
        "artifacts": [
            {"purpose":"model_only_seed","path":"checkpoint/update-000000.pt","sha256":sha256_file(fp32_path)},
            {"purpose":"portable_fp16_artifact","path":"artifact/focal_seed/model.bin","sha256":sha256_file(portable_path)},
        ],
        "effective_policy_sha256": effective_hash,
        "inference_semantics": {"storage_dtype":"fp16","runtime_dtype":"fp32","deploy_mode":"kaggle_fp16_storage_fp32_runtime_v1"},
        "optimizer": {"state_migrated": False, "initialization": "fresh_from_approved_0043_groups"},
        "readiness": "PENDING_ZERO_STEP_PARITY",
    }
    manifest_path = VERSION_ROOT / "artifact/focal_seed/manifest.json"
    _write_json(manifest_path, manifest)
    return {"status":"PASS","old_size":15,"new_size":v2.class_count,
            "g1_unchanged":True,"effective_policy_sha256":effective_hash,
            "migration_sha256":migration_hash}


if __name__ == "__main__":
    print(json.dumps(migrate(), sort_keys=True))
