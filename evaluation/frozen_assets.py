"""Materialize the immutable shared-policy Frozen Arena asset set."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

from evaluation.cards import load_card_catalog


load_deck_plugins = importlib.import_module(
    "train.0022_league_training.decks"
).load_deck_plugins


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "evaluation"
DEFAULT_SOURCE_DECKS = ROOT / "train" / "0022_league_training" / "deck"
DEFAULT_TARGET = EVALUATION_ROOT / "arena" / "frozen"
DEFAULT_CATALOG = EVALUATION_ROOT / "configs" / "frozen.json"
EXPECTED_FOUNDATION_SHA256 = (
    "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _representative_cards(plugin: Any, official_cards: dict[int, dict]) -> list[int]:
    manifest = json.loads((plugin.root / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest.get("cards") or []
    pokemon: list[int] = []
    for item in declared:
        card_id = item.get("card_id") if isinstance(item, dict) else None
        metadata = official_cards.get(card_id) if isinstance(card_id, int) else None
        if metadata is not None and "Pokémon" in metadata["stage_or_type"]:
            pokemon.append(card_id)
    if not pokemon:
        raise ValueError(f"Frozen deck has no declared representative Pokémon: {plugin.deck_id}")

    display_tokens = [
        token.strip().casefold()
        for token in plugin.display_name.replace("- Limitless", "").split("/")
        if token.strip()
    ]
    selected: list[int] = []
    for token in display_tokens:
        match = next(
            (
                card_id
                for card_id in pokemon
                if token in official_cards[card_id]["name"].casefold()
                or official_cards[card_id]["name"].casefold() in token
            ),
            None,
        )
        if match is not None and match not in selected:
            selected.append(match)
        if len(selected) == 2:
            break
    if len(selected) < 2:
        for card_id in pokemon:
            if card_id not in selected:
                selected.append(card_id)
            if len(selected) >= 2:
                break
    return selected[:2]


def materialize(
    *, source_decks: Path, policy_source: Path, target: Path, catalog_path: Path
) -> None:
    if target.exists() or catalog_path.exists():
        raise FileExistsError(f"Frozen Arena assets already exist: {target} or {catalog_path}")
    plugins = load_deck_plugins(source_decks)
    if len(plugins) != 48:
        raise ValueError(f"Frozen Arena requires exactly 48 deck plugins, got {len(plugins)}")
    if len({plugin.deck_sha256 for plugin in plugins}) != len(plugins):
        raise ValueError("Frozen Arena source contains duplicate exact decks")

    policy_manifest = json.loads((policy_source / "manifest.json").read_text(encoding="utf-8"))
    if policy_manifest.get("foundation_sha256") != EXPECTED_FOUNDATION_SHA256:
        raise ValueError("Frozen policy Foundation SHA mismatch")
    if policy_manifest.get("update") != 0:
        raise ValueError("Frozen policy must be exported from update 0")

    official_cards = load_card_catalog(ROOT / "data" / "official" / "EN_Card_Data.csv")
    target.mkdir(parents=True)
    shutil.copytree(
        policy_source,
        target / "_policy",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    entries: list[dict[str, object]] = []
    for plugin in plugins:
        deck_root = target / plugin.deck_id
        deck_root.mkdir()
        shutil.copy2(plugin.root / "deck.csv", deck_root / "deck.csv")
        deck_manifest = {
            "schema_version": "evaluation_frozen_deck_v1",
            "deck_id": plugin.deck_id,
            "display_name": plugin.display_name,
            "deck_sha256": plugin.deck_sha256,
            "decoder_ref": "foundation",
            "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
            "deployment_source_id": 0,
            "provenance": plugin.provenance,
            "source_plugin_sha256": plugin.plugin_sha256,
        }
        (deck_root / "manifest.json").write_text(
            _canonical_json(deck_manifest), encoding="utf-8"
        )
        entries.append(
            {
                "name": plugin.deck_id,
                "package": f"arena/frozen/{plugin.deck_id}",
                "display_name": f"[Frozen 0019] {plugin.display_name}",
                "representative_card_ids": _representative_cards(plugin, official_cards),
                "enabled": True,
                "tags": ["foundation_frozen", "source_id_0", "deck_exact_60"],
            }
        )

    policy_sha = _tree_sha256(target / "_policy")
    manifest = {
        "schema_version": "evaluation_frozen_arena_v1",
        "pool_id": "0019_foundation_48_exact_decks_v1",
        "deck_count": len(entries),
        "policy_path": "_policy",
        "policy_tree_sha256": policy_sha,
        "foundation": {
            "asset_id": "0019-0730-epoch13",
            "epoch": 13,
            "deployment_source_id": 0,
            "weights_sha256": EXPECTED_FOUNDATION_SHA256,
            "decoder_frozen": True,
            "encoder_frozen": True,
        },
        "evaluation": {
            "official_engine_required": True,
            "greedy": True,
            "equal_games_per_deck": True,
            "balanced_seats_for_even_games": True,
            "shared_gpu_policy_required": True,
        },
    }
    (target / "manifest.json").write_text(_canonical_json(manifest), encoding="utf-8")
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(_canonical_json({"opponents": entries}), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="materialize immutable Frozen Arena assets")
    parser.add_argument("--source-decks", type=Path, default=DEFAULT_SOURCE_DECKS)
    parser.add_argument("--policy-source", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args(argv)
    materialize(
        source_decks=args.source_decks,
        policy_source=args.policy_source,
        target=args.target,
        catalog_path=args.catalog,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
