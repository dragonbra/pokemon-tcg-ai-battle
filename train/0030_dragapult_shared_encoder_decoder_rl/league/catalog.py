"""Fail-closed loader for the 51 exact Frozen 0019 deck identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


CATALOG_PATH = Path(__file__).with_name("frozen_catalog.json")
DECK_ROOT = Path(__file__).with_name("decks")
EXPECTED_CATALOG_FILE_SHA256 = "296ba0891df600de0025247c064dbc8a5d086029b2387db4ef0c4d000bd5849a"
EXPECTED_COUNT = 51


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class DeckIdentity:
    deck_id: str
    display_name: str
    deck: tuple[int, ...]
    deck_sha256: str
    root: Path


def _read_deck(path: Path) -> tuple[tuple[int, ...], str]:
    try:
        cards = tuple(int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid Frozen deck file: {path}") from error
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError(f"Frozen deck must contain exactly 60 positive card IDs: {path}")
    canonical = ",".join(str(card) for card in sorted(cards)).encode("ascii")
    return cards, _sha256_bytes(canonical)


def load_frozen_catalog() -> tuple[DeckIdentity, ...]:
    actual_catalog_sha = _sha256(CATALOG_PATH)
    if actual_catalog_sha != EXPECTED_CATALOG_FILE_SHA256:
        raise ValueError(f"Frozen catalog SHA mismatch: {actual_catalog_sha}")
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = payload.get("opponents") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise ValueError(f"Frozen catalog must contain {EXPECTED_COUNT} opponents")
    output: list[DeckIdentity] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("enabled"):
            raise ValueError("Frozen catalog contains a disabled or malformed opponent")
        deck_id = str(row.get("name") or "")
        root = DECK_ROOT / deck_id
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        deck, digest = _read_deck(root / "deck.csv")
        if manifest.get("directory") != deck_id or manifest.get("deck_sha256") != digest:
            raise ValueError(f"Frozen deck identity mismatch: {deck_id}")
        output.append(DeckIdentity(deck_id, str(row.get("display_name") or deck_id), deck, digest, root))
    if len({item.deck_id for item in output}) != EXPECTED_COUNT:
        raise ValueError("Frozen catalog contains duplicate deck IDs")
    return tuple(output)


__all__ = ["DeckIdentity", "load_frozen_catalog"]
