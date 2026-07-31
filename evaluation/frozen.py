"""Strict loader for the immutable shared-policy Frozen Arena."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evaluation.cards import card_image_url, load_card_catalog
from evaluation.packages.loader import (
    PackageValidationError,
    SubmissionPackage,
    _read_deck,
    load_submission_package,
)


EXPECTED_FOUNDATION_SHA256 = (
    "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"
)
EXPECTED_POOL_ID = "0019_foundation_48_exact_decks_v1"
EXPECTED_ENTRY_FIELDS = frozenset(
    {"name", "package", "display_name", "representative_card_ids", "enabled", "tags"}
)


@dataclass(frozen=True)
class FrozenCatalog:
    pool_id: str
    manifest: dict[str, Any]
    manifest_sha256: str
    policy: SubmissionPackage
    opponents: tuple[SubmissionPackage, ...]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageValidationError(f"could not read Frozen {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageValidationError(f"Frozen {label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_card_ids(evaluation_root: Path) -> set[int]:
    catalog = load_card_catalog(
        evaluation_root.parent / "data" / "official" / "EN_Card_Data.csv"
    )
    return set(catalog)


def _entry_list(catalog_path: Path) -> list[dict[str, Any]]:
    catalog = _read_json(catalog_path, "catalog")
    if set(catalog) != {"opponents"} or not isinstance(catalog["opponents"], list):
        raise PackageValidationError("Frozen catalog must contain only an opponents list")
    result: list[dict[str, Any]] = []
    for index, entry in enumerate(catalog["opponents"]):
        if not isinstance(entry, dict) or set(entry) != EXPECTED_ENTRY_FIELDS:
            raise PackageValidationError(f"Frozen catalog entry {index} fields mismatch")
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise PackageValidationError(f"Frozen catalog entry {index} has invalid name")
        if not isinstance(entry["display_name"], str) or not entry["display_name"]:
            raise PackageValidationError(f"Frozen catalog entry {index} has invalid display_name")
        if type(entry["enabled"]) is not bool:
            raise PackageValidationError(f"Frozen catalog entry {index} has invalid enabled flag")
        if not isinstance(entry["tags"], list) or "foundation_frozen" not in entry["tags"]:
            raise PackageValidationError(f"Frozen catalog entry {index} lacks foundation_frozen tag")
        cards = entry["representative_card_ids"]
        if (
            not isinstance(cards, list)
            or not 1 <= len(cards) <= 2
            or not all(type(card_id) is int and card_id > 0 for card_id in cards)
        ):
            raise PackageValidationError(
                f"Frozen catalog entry {index} has invalid representative_card_ids"
            )
        result.append(entry)
    return result


def load_frozen_catalog(path: Path, evaluation_root: Path) -> FrozenCatalog:
    """Load 48 deck identities bound to one immutable Foundation policy."""
    frozen_root = (evaluation_root / "arena" / "frozen").resolve()
    manifest_path = frozen_root / "manifest.json"
    manifest = _read_json(manifest_path, "Arena manifest")
    foundation = manifest.get("foundation")
    if (
        manifest.get("schema_version") != "evaluation_frozen_arena_v1"
        or manifest.get("pool_id") != EXPECTED_POOL_ID
        or manifest.get("deck_count") != 48
        or not isinstance(foundation, dict)
        or foundation.get("weights_sha256") != EXPECTED_FOUNDATION_SHA256
        or foundation.get("deployment_source_id") != 0
        or foundation.get("encoder_frozen") is not True
        or foundation.get("decoder_frozen") is not True
    ):
        raise PackageValidationError("Frozen Arena manifest identity mismatch")

    official_ids = _official_card_ids(evaluation_root)
    policy = load_submission_package(
        frozen_root / "_policy", official_ids, name="frozen_foundation_policy"
    )
    policy_manifest = policy.package_manifest or {}
    if (
        policy_manifest.get("foundation_sha256") != EXPECTED_FOUNDATION_SHA256
        or policy_manifest.get("update") != 0
    ):
        raise PackageValidationError("Frozen shared policy is not Foundation update 0")

    card_catalog = load_card_catalog(
        evaluation_root.parent / "data" / "official" / "EN_Card_Data.csv"
    )
    entries = _entry_list(path)
    if len(entries) != 48:
        raise PackageValidationError(f"Frozen catalog must contain 48 entries, got {len(entries)}")
    names: set[str] = set()
    deck_hashes: set[str] = set()
    opponents: list[SubmissionPackage] = []
    for entry in entries:
        name = entry["name"]
        if name in names:
            raise PackageValidationError(f"duplicate Frozen opponent name: {name}")
        names.add(name)
        expected_package = f"arena/frozen/{name}"
        if entry["package"] != expected_package:
            raise PackageValidationError(
                f"Frozen package must be {expected_package}: {entry['package']}"
            )
        deck_root = (evaluation_root / entry["package"]).resolve()
        if deck_root.parent != frozen_root or not deck_root.is_dir():
            raise PackageValidationError(f"Frozen deck path escapes pool or is missing: {name}")
        if {item.name for item in deck_root.iterdir()} != {"deck.csv", "manifest.json"}:
            raise PackageValidationError(f"Frozen deck identity must be lightweight: {name}")
        deck = _read_deck(deck_root / "deck.csv", official_ids)
        deck_manifest = _read_json(deck_root / "manifest.json", f"deck {name} manifest")
        canonical = ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
        exact_hash = hashlib.sha256(canonical).hexdigest()
        if (
            deck_manifest.get("schema_version") != "evaluation_frozen_deck_v1"
            or deck_manifest.get("deck_id") != name
            or deck_manifest.get("deck_sha256") != exact_hash
            or deck_manifest.get("decoder_ref") != "foundation"
            or deck_manifest.get("foundation_sha256") != EXPECTED_FOUNDATION_SHA256
            or deck_manifest.get("deployment_source_id") != 0
        ):
            raise PackageValidationError(f"Frozen deck manifest identity mismatch: {name}")
        if exact_hash in deck_hashes:
            raise PackageValidationError(f"duplicate Frozen exact deck: {name}")
        deck_hashes.add(exact_hash)
        representative_cards: list[dict[str, object]] = []
        for card_id in entry["representative_card_ids"]:
            metadata = card_catalog.get(card_id)
            if card_id not in deck or metadata is None or "Pokémon" not in metadata["stage_or_type"]:
                raise PackageValidationError(f"invalid Frozen representative card: {name}/{card_id}")
            representative_cards.append(
                {
                    "card_id": card_id,
                    "name": metadata["name"],
                    "image_url": card_image_url(
                        metadata["expansion"], metadata["collection_number"]
                    ),
                }
            )
        identity_hash = hashlib.sha256(
            f"{policy.package_hash}:{exact_hash}:{_sha256(deck_root / 'manifest.json')}".encode()
        ).hexdigest()
        opponents.append(
            SubmissionPackage(
                name=name,
                root=deck_root,
                deck=deck,
                entrypoint=policy.entrypoint,
                package_hash=identity_hash,
                deck_hash=_sha256(deck_root / "deck.csv"),
                cg_manifest=policy.cg_manifest,
                display_name=entry["display_name"],
                representative_cards=tuple(representative_cards),
                package_manifest=deck_manifest,
            )
        )
    return FrozenCatalog(
        pool_id=EXPECTED_POOL_ID,
        manifest=manifest,
        manifest_sha256=_sha256(manifest_path),
        policy=replace(policy, display_name="[Frozen 0019] Shared Foundation Policy"),
        opponents=tuple(opponents),
    )


__all__ = ["FrozenCatalog", "load_frozen_catalog"]
