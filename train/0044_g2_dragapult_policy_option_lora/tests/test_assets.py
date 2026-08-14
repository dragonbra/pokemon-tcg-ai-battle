from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import shutil

import pytest

assets = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.assets")
AssetIntegrityError = assets.AssetIntegrityError
AssetRegistry = assets.AssetRegistry
canonical_deck_sha256 = assets.canonical_deck_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _clone_assets(destination: Path) -> None:
    def copy_asset(source: str, target: str) -> str:
        if Path(source).suffix in {".pt", ".bin"}:
            os.link(source, target)
            return target
        return shutil.copy2(source, target)

    shutil.copytree(
        PROJECT_ROOT / "assets",
        destination / "assets",
        copy_function=copy_asset,
    )


def test_complete_imported_asset_registry_passes() -> None:
    audit = AssetRegistry.load(PROJECT_ROOT).validate_all()

    assert audit.status == "PASS"
    assert audit.deck_count == 69
    assert audit.training_deck_count == 69
    assert audit.evaluation_deck_count == 0
    assert audit.evaluation_games == 0
    assert audit.policy_ids == ("Champion-G2", "Champion-G3", "Policy-0809")
    assert audit.latest_champion_policy_id == "Champion-G3"
    registry = AssetRegistry.load(PROJECT_ROOT)
    assert tuple(deck.deck_id for deck in registry.decks) == tuple(
        f"{index:03d}" for index in range(1, 70)
    )
    assert tuple(
        deck.deck_id for deck in registry.decks if "training" in deck.roles
    ) == tuple(f"{index:03d}" for index in range(1, 70))
    assert tuple(
        deck.deck_id for deck in registry.decks if deck.roles == ("evaluation",)
    ) == ()
    assert tuple(path.name for path in sorted(
        (PROJECT_ROOT / "assets/decks/definitions").iterdir()
    )) == tuple(f"{index:03d}" for index in range(1, 70))
    assert all(
        deck.deck_path == f"assets/decks/definitions/{deck.deck_id}/deck.csv"
        for deck in registry.decks
    )


def test_deck_hash_is_order_independent_and_requires_exactly_sixty_cards() -> None:
    cards = tuple(range(1, 61))
    assert canonical_deck_sha256(cards) == canonical_deck_sha256(tuple(reversed(cards)))
    with pytest.raises(AssetIntegrityError, match="exactly 60"):
        canonical_deck_sha256(cards[:-1])


def test_registry_rejects_external_asset_path(tmp_path: Path) -> None:
    copied = tmp_path / "project"
    _clone_assets(copied)
    registry_path = copied / "assets/decks/registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["decks"][0]["deck_path"] = "/tmp/external-deck.csv"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        AssetIntegrityError, match="(?:project-relative|canonical numeric ID)"
    ):
        AssetRegistry.load(copied).validate_all()


def test_registry_rejects_mutated_deck_under_same_id(tmp_path: Path) -> None:
    copied = tmp_path / "project"
    _clone_assets(copied)
    registry = AssetRegistry.load(copied)
    deck = registry.decks[0]
    path = copied / deck.deck_path
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[0] = str(int(rows[0]) + 1)
    path.unlink()
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(AssetIntegrityError, match="(?:file|content) SHA-256 mismatch"):
        AssetRegistry.load(copied).validate_all()


def test_registry_rejects_missing_policy_blob(tmp_path: Path) -> None:
    copied = tmp_path / "project"
    _clone_assets(copied)
    registry = AssetRegistry.load(copied)
    blob = copied / registry.policies[0].artifacts[0].path
    blob.unlink()

    with pytest.raises(AssetIntegrityError, match="missing asset"):
        AssetRegistry.load(copied).validate_all()


def test_policy_paths_use_semantic_directories_not_hash_directories() -> None:
    registry = AssetRegistry.load(PROJECT_ROOT)
    paths = {
        policy.policy_id: {artifact.path for artifact in policy.artifacts}
        for policy in registry.policies
    }
    assert paths["Champion-G2"] == {
        "assets/policies/definitions/champion_g002/source_update_000407.pt",
        "assets/policies/definitions/champion_g002/model.bin",
    }
    assert paths["Policy-0809"] == {
        "assets/policies/definitions/policy_0809/model.pt",
    }
    assert paths["Champion-G3"] == {
        "assets/policies/definitions/champion_g003/source_update_000110.pt",
        "assets/policies/definitions/champion_g003/model.bin",
    }
    assert set(paths) == {"Champion-G2", "Champion-G3", "Policy-0809"}
    assert not (PROJECT_ROOT / "assets/policies/blobs").exists()


def test_legacy_frozen_meta_pool_is_not_an_0044_evaluation_identity() -> None:
    registry = AssetRegistry.load(PROJECT_ROOT)
    assert registry.evaluations == ()
    contract = json.loads(
        (PROJECT_ROOT / "assets/evaluation/benchmark_v1/contract.json").read_text()
    )
    assert contract["opponent_policy_id"] == "Champion-G2"
    assert contract["cuda_games"] == 2048
