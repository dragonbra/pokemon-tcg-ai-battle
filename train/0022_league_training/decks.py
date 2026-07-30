"""Fail-closed discovery for exact-deck League plugins."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


DECK_ID = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
PLUGIN_SCHEMA = "0022_league_deck_plugin_v1"
ALLOWED_FILES = frozenset({"manifest.json", "deck.csv"})
ONTOLOGY_PATH = Path(__file__).with_name("assets") / "card_ontology.json"


class DeckRole(str, Enum):
    FROZEN = "frozen"
    LIVE = "live"


@dataclass(frozen=True)
class DeckPlugin:
    deck_id: str
    display_name: str
    role: DeckRole
    focal: bool
    decoder_ref: str
    decoder_sha256: str | None
    deck: tuple[int, ...]
    deck_sha256: str
    plugin_sha256: str
    provenance: dict[str, str]
    root: Path

    def snapshot(self) -> dict[str, object]:
        value = asdict(self)
        value["role"] = self.role.value
        value["deck"] = list(self.deck)
        value.pop("root")
        return value


def _canonical_json(value: object) -> bytes:
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (text + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_deck(path: Path) -> tuple[tuple[int, ...], str]:
    if not path.is_file():
        raise ValueError(f"missing deck.csv: {path}")
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    known_card_ids = {int(card["card_id"]) for card in ontology.get("cards", [])}
    cards: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value:
            raise ValueError(f"deck.csv line {line_number} is empty")
        try:
            card_id = int(value)
        except ValueError as error:
            raise ValueError(f"deck.csv line {line_number} is not an integer") from error
        if card_id <= 0:
            raise ValueError(f"deck.csv line {line_number} is not a positive card ID")
        if card_id not in known_card_ids:
            raise ValueError(f"deck.csv contains unknown ontology card ID: {card_id}")
        cards.append(card_id)
    if len(cards) != 60:
        raise ValueError(f"deck.csv must contain exactly 60 card IDs, got {len(cards)}")
    canonical = "".join(f"{card_id}\n" for card_id in cards).encode("ascii")
    return tuple(cards), _sha256_bytes(canonical)


def _require_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest {key} must be a nonempty string")
    return value.strip()


def _load_plugin(root: Path) -> DeckPlugin:
    unknown = {path.name for path in root.iterdir()} - ALLOWED_FILES
    if unknown:
        raise ValueError(f"unknown files in deck plugin {root.name}: {sorted(unknown)}")
    try:
        manifest: dict[str, Any] = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable deck manifest: {root / 'manifest.json'}") from error
    allowed_fields = {
        "schema_version", "deck_id", "display_name", "role", "focal",
        "decoder_ref", "decoder_sha256", "provenance",
    }
    unknown_fields = set(manifest) - allowed_fields
    if unknown_fields:
        raise ValueError(f"unknown deck manifest fields: {sorted(unknown_fields)}")
    if manifest.get("schema_version") != PLUGIN_SCHEMA:
        raise ValueError(f"deck plugin schema must be {PLUGIN_SCHEMA}")
    deck_id = _require_text(manifest, "deck_id")
    if DECK_ID.fullmatch(deck_id) is None or not deck_id.isascii():
        raise ValueError(f"invalid ASCII snake-case deck_id: {deck_id}")
    if root.name != deck_id:
        raise ValueError(f"deck directory {root.name} does not match deck_id {deck_id}")
    try:
        role = DeckRole(_require_text(manifest, "role"))
    except ValueError as error:
        raise ValueError("manifest role must be frozen or live") from error
    focal = manifest.get("focal")
    if not isinstance(focal, bool):
        raise ValueError("manifest focal must be a boolean")
    if focal and role is not DeckRole.LIVE:
        raise ValueError("only a Live deck may be focal")
    decoder_ref = _require_text(manifest, "decoder_ref")
    decoder_sha = manifest.get("decoder_sha256")
    if decoder_ref == "foundation":
        if decoder_sha is not None:
            raise ValueError("foundation decoder_ref must omit decoder_sha256")
    else:
        if Path(decoder_ref).is_absolute() or ".." in Path(decoder_ref).parts:
            raise ValueError("decoder_ref must be a repository-relative immutable path")
        if not isinstance(decoder_sha, str) or re.fullmatch(r"[0-9a-f]{64}", decoder_sha) is None:
            raise ValueError("non-Foundation decoder_ref requires decoder_sha256")
    raw_provenance = manifest.get("provenance")
    if not isinstance(raw_provenance, dict):
        raise ValueError("manifest provenance must be an object")
    required_provenance = {"source", "evidence", "captured_at"}
    if set(raw_provenance) != required_provenance:
        raise ValueError(f"provenance fields must be {sorted(required_provenance)}")
    provenance = {key: _require_text(raw_provenance, key) for key in sorted(raw_provenance)}
    deck, deck_sha = _read_deck(root / "deck.csv")
    identity = dict(manifest)
    identity["deck_sha256"] = deck_sha
    plugin_sha = _sha256_bytes(_canonical_json(identity))
    return DeckPlugin(
        deck_id=deck_id,
        display_name=_require_text(manifest, "display_name"),
        role=role,
        focal=focal,
        decoder_ref=decoder_ref,
        decoder_sha256=decoder_sha,
        deck=deck,
        deck_sha256=deck_sha,
        plugin_sha256=plugin_sha,
        provenance=provenance,
        root=root,
    )


def load_deck_plugins(root: Path) -> tuple[DeckPlugin, ...]:
    """Load all immediate plugin directories in deterministic ID order."""
    if not root.is_dir():
        raise ValueError(f"deck staging directory does not exist: {root}")
    allowed_root_files = {"README.md", ".gitkeep"}
    stray_files = [
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name not in allowed_root_files
    ]
    if stray_files:
        raise ValueError(f"unknown files in deck staging root: {sorted(stray_files)}")
    plugins = tuple(_load_plugin(path) for path in sorted(root.iterdir()) if path.is_dir())
    ids = [plugin.deck_id for plugin in plugins]
    hashes = [plugin.deck_sha256 for plugin in plugins]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate deck_id in League catalog")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate exact deck hash in League catalog")
    return plugins


def write_catalog_snapshot(path: Path, plugins: tuple[DeckPlugin, ...]) -> str:
    payload = {
        "schema_version": "0022_league_catalog_snapshot_v1",
        "decks": [plugin.snapshot() for plugin in plugins],
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_bytes(encoded)


__all__ = ["DeckPlugin", "DeckRole", "load_deck_plugins", "write_catalog_snapshot"]
