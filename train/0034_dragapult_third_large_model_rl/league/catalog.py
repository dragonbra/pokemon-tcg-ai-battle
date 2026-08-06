"""Fail-closed loader for the 55-deck Frozen-0806 rollout schedule."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


CATALOG_PATH = Path(__file__).with_name("frozen_catalog.json")
DECK_ROOT = Path(__file__).with_name("decks")
EXPECTED_CATALOG_FILE_SHA256 = "16dbd18ce417405571c88997c9e97f9b2ec2adf96db544d1a9af988bb3c3cc3c"
EXPECTED_POOL_ID = "0806_kaggle_top100_plus_v1"
EXPECTED_COUNT = 55
EXPECTED_GAMES = 256


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
    games: int
    segment: str
    best_rank: int


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
    rows = payload.get("entries") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version") != "evaluation_frozen_schedule_v1"
        or payload.get("pool_id") != EXPECTED_POOL_ID
        or payload.get("total_games") != EXPECTED_GAMES
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_COUNT
    ):
        raise ValueError(
            f"Frozen-0806 catalog must contain {EXPECTED_COUNT} decks and {EXPECTED_GAMES} games"
        )
    output: list[DeckIdentity] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Frozen-0806 catalog contains a malformed schedule row")
        deck_id = str(row.get("deck_id") or "")
        root = DECK_ROOT / deck_id
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        deck, digest = _read_deck(root / "deck.csv")
        games = row.get("games")
        segment = row.get("segment")
        best_rank = row.get("best_rank")
        if (
            manifest.get("deck_id") != deck_id
            or manifest.get("exact_deck_sha256") != digest
            or row.get("exact_deck_sha256") != digest
            or type(games) is not int
            or games < 1
            or segment not in {"top100", "potential_101_500"}
            or type(best_rank) is not int
            or best_rank < 1
        ):
            raise ValueError(f"Frozen deck identity mismatch: {deck_id}")
        output.append(
            DeckIdentity(
                deck_id,
                str(row.get("archetype") or deck_id),
                deck,
                digest,
                root,
                games,
                segment,
                best_rank,
            )
        )
    if len({item.deck_id for item in output}) != EXPECTED_COUNT:
        raise ValueError("Frozen catalog contains duplicate deck IDs")
    if len({item.deck_sha256 for item in output}) != EXPECTED_COUNT:
        raise ValueError("Frozen catalog contains duplicate exact decks")
    if sum(item.games for item in output) != EXPECTED_GAMES:
        raise ValueError("Frozen catalog game counts do not total 256")
    return tuple(output)


__all__ = ["DeckIdentity", "load_frozen_catalog"]
