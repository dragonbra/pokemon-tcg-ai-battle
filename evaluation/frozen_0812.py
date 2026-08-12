"""Fail-closed loader for the immutable 65-deck Frozen-0812 v3 pool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.cards import load_card_catalog
from evaluation.frozen_0806 import (
    Frozen0806Deck as Frozen0812Deck,
    Frozen0806Pool as Frozen0812Pool,
    Frozen0806ScheduleEntry as Frozen0812ScheduleEntry,
    exact_deck_sha256,
)
from evaluation.packages.loader import PackageValidationError, _read_deck


ROOT = Path(__file__).resolve().parents[1]
POOL_ID = "0812_top100_top500_stadiums_mill_plus_limitless_dragapult_v3"
POOL_ROOT = ROOT / "evaluation/arena/frozen_pools" / POOL_ID
EXPECTED_DECK_COUNT = 65
EXPECTED_GAMES = 256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PackageValidationError(f"JSON object required: {path}")
    return payload


def load_frozen_0812_pool(pool_root: Path = POOL_ROOT) -> Frozen0812Pool:
    pool_root = pool_root.resolve()
    manifest_path = pool_root / "manifest.json"
    manifest = _json(manifest_path)
    if (
        manifest.get("pool_id") != POOL_ID
        or manifest.get("deck_count") != EXPECTED_DECK_COUNT
        or manifest.get("total_games") != EXPECTED_GAMES
        or manifest.get("schedule_path") != "schedule.json"
    ):
        raise PackageValidationError("Frozen-0812 manifest identity mismatch")
    source = manifest.get("source_snapshot") or {}
    source_path = (ROOT / str(source.get("path", ""))).resolve()
    if ROOT not in source_path.parents or _sha256(source_path) != source.get("sha256"):
        raise PackageValidationError("Frozen-0812 source snapshot mismatch")
    source_payload = _json(source_path)
    if len(source_payload.get("players") or []) != 100:
        raise PackageValidationError("Frozen-0812 source snapshot is not Top100")

    official_ids = set(load_card_catalog(ROOT / "data/official/EN_Card_Data.csv"))
    decks = []
    by_id = {}
    hashes = set()
    numbers = set()
    for deck_root in sorted(path for path in (pool_root / "decks").iterdir() if path.is_dir()):
        try:
            number = int(deck_root.name.split("_", 1)[0])
        except ValueError as exc:
            raise PackageValidationError(f"invalid Frozen-0812 directory: {deck_root.name}") from exc
        cards = tuple(_read_deck(deck_root / "deck.csv", official_ids))
        deck_manifest = _json(deck_root / "manifest.json")
        deck_id = deck_manifest.get("deck_id")
        deck_hash = exact_deck_sha256(cards)
        if (
            number in numbers
            or deck_id in by_id
            or deck_hash in hashes
            or deck_manifest.get("schema_version") != "evaluation_frozen_0812_deck_v1"
            or deck_manifest.get("pool_id") != POOL_ID
            or deck_manifest.get("exact_deck_sha256") != deck_hash
            or not set(deck_manifest.get("representative_card_ids") or []).issubset(cards)
        ):
            raise PackageValidationError(f"Frozen-0812 deck identity mismatch: {deck_root.name}")
        numbers.add(number)
        hashes.add(deck_hash)
        deck = Frozen0812Deck(str(deck_id), deck_root, cards, deck_hash, deck_manifest)
        decks.append(deck)
        by_id[deck.deck_id] = deck
    if numbers != set(range(1, EXPECTED_DECK_COUNT + 1)):
        raise PackageValidationError("Frozen-0812 deck numbering/count mismatch")

    schedule_path = pool_root / "schedule.json"
    if _sha256(schedule_path) != manifest.get("schedule_sha256"):
        raise PackageValidationError("Frozen-0812 schedule hash mismatch")
    schedule_payload = _json(schedule_path)
    if (
        schedule_payload.get("pool_id") != POOL_ID
        or schedule_payload.get("total_games") != EXPECTED_GAMES
        or len(schedule_payload.get("entries") or []) != EXPECTED_DECK_COUNT
    ):
        raise PackageValidationError("Frozen-0812 schedule identity mismatch")
    schedule = []
    allowed_segments = {
        "top100", "potential_101_500", "top100_new_archetype",
        "limitless_representative",
        "top500_limitless_counter_selection",
        "top500_stadium_coverage", "top500_mill_coverage",
    }
    for index, raw in enumerate(schedule_payload["entries"]):
        try:
            entry = Frozen0812ScheduleEntry(
                deck_id=raw["deck_id"], archetype=raw["archetype"],
                segment=raw["segment"], games=raw["games"],
                observed_players=raw["observed_players"], best_rank=raw["best_rank"],
                selection_rank=raw["selection_rank"], source_ranks=tuple(raw["source_ranks"]),
                exact_deck_sha256=raw["exact_deck_sha256"],
                binding_counts=dict(raw["binding_counts"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PackageValidationError(f"Frozen-0812 schedule row {index} invalid") from exc
        deck = by_id.get(entry.deck_id)
        if (
            deck is None
            or entry.exact_deck_sha256 != deck.exact_deck_sha256
            or entry.segment not in allowed_segments
            or type(entry.games) is not int or entry.games <= 0
        ):
            raise PackageValidationError(f"Frozen-0812 schedule row {index} mismatch")
        schedule.append(entry)
    if {entry.deck_id for entry in schedule} != set(by_id):
        raise PackageValidationError("Frozen-0812 schedule/deck identity set mismatch")
    if sum(entry.games for entry in schedule) != EXPECTED_GAMES:
        raise PackageValidationError("Frozen-0812 schedule total mismatch")
    additions = manifest.get("additions") or {}
    addition_ids = {
        additions.get("top100_new_archetype"),
        additions.get("limitless_self_destruct_dragapult"),
        *(additions.get("top500_limitless_counter_selections") or []),
        *(additions.get("top500_stadium_coverage_selections") or []),
        *(additions.get("top500_mill_coverage_selections") or []),
    }
    if addition_ids != {entry.deck_id for entry in schedule[-10:]}:
        raise PackageValidationError("Frozen-0812 additions identity mismatch")
    return Frozen0812Pool(
        pool_id=POOL_ID,
        root=pool_root,
        manifest=manifest,
        manifest_sha256=_sha256(manifest_path),
        policies=dict(manifest.get("policies") or {}),
        decks=tuple(decks),
        schedule=tuple(schedule),
        total_games=EXPECTED_GAMES,
    )


if __name__ == "__main__":
    pool = load_frozen_0812_pool()
    print(json.dumps({"pool_id": pool.pool_id, "decks": len(pool.decks), "games": pool.total_games}, indent=2))
