"""One-time import of approved assets into the self-contained 0043 project.

Historical paths are accepted only by this importer. Runtime loaders consume only
the generated project-relative registries and blobs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .assets import AssetRegistry, canonical_deck_sha256, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments/0043_champion_league_rl"
SOURCE_CATALOG_ROOT = REPOSITORY_ROOT / "train/0042_full_model_design/league"
SOURCE_CATALOG = SOURCE_CATALOG_ROOT / "frozen_catalog.json"
SOURCE_0809 = REPOSITORY_ROOT / "archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt"
SOURCE_G1_ROOT = REPOSITORY_ROOT / "archive/pretrained/0042_champion_g1"
SOURCE_G1_DELTA = SOURCE_G1_ROOT / "source_update_000010.pt"
SOURCE_G1_PORTABLE = SOURCE_G1_ROOT / "fp16_reference_048_package/strategy/model.bin"
SOURCE_G1_MANIFEST = SOURCE_G1_ROOT / "manifest.json"
SOURCE_SEMANTIC_RUNTIME = (
    SOURCE_G1_ROOT / "fp16_reference_048_package/strategy"
)
SOURCE_ACTIVE_CONFIG = (
    REPOSITORY_ROOT
    / "rl_runs/0042_full_model_design/versions/"
    "V22_archaludon_ex_cinderace_048/artifact/training_config.json"
)

EXPECTED_SOURCE_HASHES = {
    SOURCE_CATALOG: "b1147f570c1df9f9b3261f3ed89d83c5a96cc5f51931a30ce9ae57c59bbc3d2c",
    SOURCE_0809: "926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f",
    SOURCE_G1_DELTA: "aea408e8e140ceb28d23962cf45abac76a29be01888953bbbc5253ff6d120fe3",
    SOURCE_G1_PORTABLE: "cd5c05741aadf16db8bb8ca4f527e02eae431db75c33eba30f291fe89f57de84",
    SOURCE_ACTIVE_CONFIG: "fe6e1862a9f8660b3b19505898dacf55a2955a07385958639a5cb1f80d94264c",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _copy_immutable(source: Path, destination: Path) -> None:
    expected = EXPECTED_SOURCE_HASHES.get(source)
    actual = sha256_file(source)
    if expected is not None and actual != expected:
        raise RuntimeError(f"approved import source changed: {source}: {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != actual:
            raise RuntimeError(f"refusing to overwrite immutable asset: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != actual:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copied asset failed SHA-256 verification: {destination}")
    temporary.replace(destination)


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _timestamp() -> str:
    # Import provenance is immutable content. Re-running the idempotent importer
    # must not rename otherwise identical registries.
    return "2026-08-12T00:00:00+00:00"


def import_decks() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sha256_file(SOURCE_CATALOG) != EXPECTED_SOURCE_HASHES[SOURCE_CATALOG]:
        raise RuntimeError("approved 55-deck catalog hash changed")
    catalog = json.loads(SOURCE_CATALOG.read_text(encoding="utf-8"))
    if (
        catalog.get("schema_version") != "evaluation_frozen_schedule_v1"
        or len(catalog.get("entries", ())) != 55
        or catalog.get("total_games") != 256
    ):
        raise RuntimeError("approved source is not the audited 55-deck/256-game catalog")
    source_by_legacy_id: dict[str, Path] = {}
    for root in sorted((SOURCE_CATALOG_ROOT / "decks").iterdir()):
        if not root.is_dir():
            continue
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        source_by_legacy_id[manifest["deck_id"]] = root
    imported_at = _timestamp()
    decks: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    scheduled: list[tuple[str, dict[str, Any], Path]] = []
    for row in catalog["entries"]:
        legacy_deck_id = row["deck_id"]
        source_root = source_by_legacy_id.get(legacy_deck_id)
        if source_root is None:
            raise RuntimeError(f"catalog deck source is missing: {legacy_deck_id}")
        numeric_id = source_root.name.split("_", 1)[0]
        if len(numeric_id) != 3 or not numeric_id.isdigit():
            raise RuntimeError(f"source deck lacks canonical numeric identity: {source_root}")
        scheduled.append((numeric_id, row, source_root))
    if {numeric_id for numeric_id, _, _ in scheduled} != {
        f"{index:03d}" for index in range(1, 56)
    }:
        raise RuntimeError("approved source does not map one-to-one onto deck IDs 001 through 055")

    entry_by_id: dict[str, dict[str, Any]] = {}
    for numeric_id, row, source_root in scheduled:
        legacy_deck_id = row["deck_id"]
        cards = tuple(
            int(value) for value in (source_root / "deck.csv").read_text().splitlines()
        )
        content_sha = canonical_deck_sha256(cards)
        if content_sha != row["exact_deck_sha256"]:
            raise RuntimeError(f"source deck identity mismatch: {numeric_id}")
        destination = PROJECT_ROOT / "assets/decks/definitions" / numeric_id / "deck.csv"
        _copy_immutable(source_root / "deck.csv", destination)
        decks.append(
            {
                "deck_id": numeric_id,
                "name": row["archetype"],
                "archetype": row["archetype"],
                "deck_path": _relative(destination),
                "file_sha256": sha256_file(destination),
                "content_sha256": content_sha,
                "card_count": 60,
                "cards": [
                    {"card_id": card_id, "count": count}
                    for card_id, count in sorted(Counter(cards).items())
                ],
                "roles": ["training", "evaluation"],
                "tags": ["meta", row["segment"]],
                "source": {
                    "imported_from": str(source_root.relative_to(REPOSITORY_ROOT)),
                    "imported_at": imported_at,
                    "legacy_semantic_deck_id": legacy_deck_id,
                    "source_catalog_sha256": EXPECTED_SOURCE_HASHES[SOURCE_CATALOG],
                    "best_rank": row["best_rank"],
                },
            }
        )
        entry_by_id[numeric_id] = {
            "deck_id": numeric_id,
            "exact_deck_sha256": content_sha,
            "games": row["games"],
        }
    decks.sort(key=lambda deck: deck["deck_id"])
    entries = [entry_by_id[numeric_id] for numeric_id, _, _ in scheduled]
    definitions_root = PROJECT_ROOT / "assets/decks/definitions"
    canonical_names = {deck["deck_id"] for deck in decks}
    for old_root in definitions_root.iterdir():
        if old_root.is_dir() and old_root.name not in canonical_names:
            files = tuple(old_root.iterdir())
            if len(files) != 1 or files[0].name != "deck.csv":
                raise RuntimeError(f"refusing to prune unexpected legacy definition: {old_root}")
            files[0].unlink()
            old_root.rmdir()
    _write_json(
        PROJECT_ROOT / "assets/decks/registry.json",
        {
            "schema_version": "0043_deck_registry_v2_numeric_identity",
            "identity_contract": "deck_id is the unique zero-padded numeric identity; 001-055 are immutable and future decks append from 056",
            "immutability": "content change requires a new deck_id",
            "decks": decks,
        },
    )
    return decks, entries


def _policy_artifact(
    source: Path, purpose: str, policy_directory: str, name: str
) -> dict[str, str]:
    digest = sha256_file(source)
    destination = PROJECT_ROOT / "assets/policies/definitions" / policy_directory / name
    _copy_immutable(source, destination)
    return {"purpose": purpose, "path": _relative(destination), "sha256": digest}


def import_policies() -> list[dict[str, Any]]:
    g1_archive = json.loads(SOURCE_G1_MANIFEST.read_text(encoding="utf-8"))
    if (
        g1_archive.get("champion_id") != "Champion-G1"
        or g1_archive.get("source_checkpoint_sha256")
        != EXPECTED_SOURCE_HASHES[SOURCE_G1_DELTA]
    ):
        raise RuntimeError("G1 archive is not the user-designated generation anchor")
    base_0809 = _policy_artifact(
        SOURCE_0809, "complete_base_checkpoint", "policy_0809", "model.pt"
    )
    g1_delta = _policy_artifact(
        SOURCE_G1_DELTA, "model_only_delta", "champion_g001", "source_update_000010.pt"
    )
    g1_portable = _policy_artifact(
        SOURCE_G1_PORTABLE, "portable_fp16_artifact", "champion_g001", "model.bin"
    )
    legacy_blob_root = PROJECT_ROOT / "assets/policies/blobs"
    if legacy_blob_root.exists():
        for hash_root in legacy_blob_root.iterdir():
            if not hash_root.is_dir():
                raise RuntimeError(f"refusing to prune unexpected legacy policy blob: {hash_root}")
            files = tuple(hash_root.iterdir())
            if len(files) != 1 or files[0].name not in {
                "model.pt", "source_update_000010.pt", "model.bin"
            }:
                raise RuntimeError(f"refusing to prune unexpected legacy policy blob: {hash_root}")
            files[0].unlink()
            hash_root.rmdir()
        legacy_blob_root.rmdir()
    imported_at = _timestamp()
    component_rows = {
        "prototype_encoder": "c5cbea41fdae1a2991a53f29c0f37043254a997449a2281eb893ad3d8043d297",
        "state_encoder": "9f819d4dcf6c4b463651f0a50c809adb4e62eb1915be5fefd16597f27b6ec270",
        "option_input_encoder": "21db4ccf7f4faa6c436293e09b09797601c928265da72fb9731977f2ab485083",
        "option_transformer_layer_0": "fe707c99e7bef6990217ad96cb1efa493efcc9d9c7d0881cd433d909e4fc44fe",
        "option_transformer_layer_1": "7073e2f657c6276abc0490abfe994c9178438aee428bfd5680f32acb41cdebe1",
        "final_norm": "4da805d623898210c9fbdf6c81040a55ec4f802f87a7a7f8a0b03ffda44eda7d",
        "lora": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "action_decoder": "1414ab7f9a2d423e39d2098e973dda6dd2d6b1085ccbedba3246b5e33437fbff",
    }
    policy_0809 = {
        "schema_version": "0043_policy_manifest_v1",
        "policy_id": "Policy-0809",
        "role": "historical_anchor",
        "generation": None,
        "frozen": True,
        "architecture_version": "0031_rule_faithful_semantic_decision_v2",
        "parent_policy_id": None,
        "components": {
            name: {"source_policy_id": "Policy-0809", "effective_sha256": digest}
            for name, digest in component_rows.items()
        },
        "artifacts": [base_0809],
        "effective_policy_sha256": "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96",
        "inference_semantics": {
            "storage_dtype": "fp32",
            "runtime_dtype": "fp32",
            "deploy_mode": "immutable_full_policy",
        },
        "source": {"imported_at": imported_at, "imported_from": str(SOURCE_0809.relative_to(REPOSITORY_ROOT))},
    }
    g1 = {
        "schema_version": "0043_policy_manifest_v1",
        "policy_id": "Champion-G1",
        "role": "latest_champion",
        "generation": 1,
        "frozen": True,
        "architecture_version": "0042_strategy_conditioned_model_only_v1",
        "parent_policy_id": "Policy-0809",
        "components": {
            "base": {"source_policy_id": "Policy-0809", "artifact_sha256": base_0809["sha256"]},
            "trained_delta": {"source_policy_id": "Champion-G1", "artifact_sha256": g1_delta["sha256"]},
            "portable_effective": {"source_policy_id": "Champion-G1", "artifact_sha256": g1_portable["sha256"]},
        },
        "artifacts": [base_0809, g1_delta, g1_portable],
        "effective_policy_sha256": g1_archive["reference_export_manifest"]["deployment_effective_sha256"],
        "inference_semantics": {
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "deploy_mode": "kaggle_fp16_storage_fp32_runtime_v1",
        },
        "designation": g1_archive["designation"],
        "promotion_record": g1_archive["promote_champion_protocol_decision"],
        "source": {"imported_at": imported_at, "imported_from": str(SOURCE_G1_ROOT.relative_to(REPOSITORY_ROOT))},
    }
    policies = []
    for manifest, filename in ((policy_0809, "policy_0809.json"), (g1, "champion_g001.json")):
        path = PROJECT_ROOT / "assets/policies/manifests" / filename
        _write_json(path, manifest)
        policies.append(
            {
                "policy_id": manifest["policy_id"], "role": manifest["role"],
                "generation": manifest["generation"], "frozen": True,
                "manifest_path": _relative(path), "manifest_sha256": sha256_file(path),
                "effective_policy_sha256": manifest["effective_policy_sha256"],
                "artifacts": manifest["artifacts"],
            }
        )
    _write_json(
        PROJECT_ROOT / "assets/policies/registry.json",
        {
            "schema_version": "0043_policy_registry_v1",
            "latest_champion_policy_id": "Champion-G1",
            "active_policy_pool": ["Policy-0809", "Champion-G1"],
            "policies": policies,
        },
    )
    return policies


def import_evaluation(entries: list[dict[str, Any]]) -> None:
    manifest = {
        "schema_version": "0043_frozen_evaluation_manifest_v2_numeric_deck_identity",
        "evaluation_id": "FrozenMeta256-V1",
        "deck_pool_role": "evaluation",
        "policy_id": "Policy-0809",
        "deck_count": len(entries),
        "total_games": sum(row["games"] for row in entries),
        "entries": entries,
        "composition_source_sha256": EXPECTED_SOURCE_HASHES[SOURCE_CATALOG],
        "composition_semantics": "unchanged exact 55-deck integer-frequency unit",
    }
    seeds = {
        "schema_version": "0043_frozen_seed_contract_v1",
        "evaluation_id": "FrozenMeta256-V1",
        "master_seed": 341512806,
        "frequency_unit_games": 256,
        "cpu_replicas": [0],
        "cuda_replicas": list(range(8)),
        "seat_semantics": "seeded toss winner processes official context 41 and chooses first or second",
        "seed_derivation_inputs": [
            "master_seed", "focal_deployment_identity", "opponent_effective_identity",
            "exact_deck_slot", "replica", "namespace",
        ],
        "status": "MATERIALIZER_IMPLEMENTED_CPU256_CUDA2048_FREQUENCY_PARITY_PASS",
    }
    manifest_path = PROJECT_ROOT / "assets/evaluation/manifests/frozen_meta_256_v1.json"
    seed_path = PROJECT_ROOT / "assets/evaluation/seeds/frozen_meta_256_v1.json"
    _write_json(manifest_path, manifest)
    _write_json(seed_path, seeds)
    _write_json(
        PROJECT_ROOT / "assets/evaluation/registry.json",
        {
            "schema_version": "0043_evaluation_registry_v1",
            "evaluations": [
                {
                    "evaluation_id": "FrozenMeta256-V1", "deck_pool_role": "evaluation",
                    "policy_id": "Policy-0809", "manifest_path": _relative(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "seed_manifest_path": _relative(seed_path),
                    "seed_manifest_sha256": sha256_file(seed_path), "entries": entries,
                }
            ],
        },
    )


def import_phase0_config() -> None:
    destination = EXPERIMENT_ROOT / "active_training_config.json"
    _copy_immutable(SOURCE_ACTIVE_CONFIG, destination)


def import_semantic_runtime() -> None:
    """Freeze the approved inference source inside 0043 without a runtime dependency.

    The archived package is provenance only. Every executable import after this
    one-time operation resolves below ``0043_champion_league_rl.semantic_runtime``.
    Model weights remain in the semantic policy directories and are deliberately
    excluded from this source-tree copy.
    """
    destination_root = PROJECT_ROOT / "semantic_runtime"
    rows: list[dict[str, str]] = []
    for source in sorted(SOURCE_SEMANTIC_RUNTIME.rglob("*")):
        if not source.is_file() or source.name == "model.bin":
            continue
        if source.suffix not in {".py", ".json", ".md"}:
            continue
        relative = source.relative_to(SOURCE_SEMANTIC_RUNTIME)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256_file(source)
        if destination.exists():
            if sha256_file(destination) != source_hash:
                raise RuntimeError(f"refusing to overwrite frozen semantic runtime: {destination}")
        else:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        rows.append({"path": relative.as_posix(), "sha256": source_hash})
    if not rows:
        raise RuntimeError("approved semantic runtime source is empty")
    tree_hash = hashlib.sha256(
        "".join(f"{row['path']}\0{row['sha256']}\n" for row in rows).encode("utf-8")
    ).hexdigest()
    _write_json(
        destination_root / "runtime_manifest.json",
        {
            "schema_version": "0043_semantic_runtime_manifest_v1",
            "source": str(SOURCE_SEMANTIC_RUNTIME.relative_to(REPOSITORY_ROOT)),
            "imported_at": _timestamp(),
            "runtime_tree_sha256": tree_hash,
            "excluded": ["model.bin", "__pycache__", "*.pyc"],
            "files": rows,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="import every Phase 1 asset domain")
    args = parser.parse_args()
    if not args.all:
        parser.error("Phase 1 currently requires --all so registries cannot be partially published")
    import_phase0_config()
    import_semantic_runtime()
    _, entries = import_decks()
    import_policies()
    import_evaluation(entries)
    audit = AssetRegistry.load(PROJECT_ROOT).validate_all()
    print(json.dumps({
        "status": audit.status, "decks": audit.deck_count,
        "evaluation_games": audit.evaluation_games,
        "policies": list(audit.policy_ids),
        "latest_champion": audit.latest_champion_policy_id,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
