"""Fail-closed publication contract for durable Combat Mat reports."""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any


CATALOG_DECKS = 55
META_ARCHETYPES = 14


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Combat Mat {label} must be a mapping")
    return value


def _records(value: object, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list | tuple) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"Combat Mat {label} must be a record list")
    return list(value)


def _count(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"Combat Mat {key} must be a non-negative integer")
    return value


def _validate_record(record: Mapping[str, Any]) -> int:
    games = _count(record, "games")
    if sum(_count(record, key) for key in ("wins", "losses", "draws")) != games:
        raise ValueError("Combat Mat W-L-D does not conserve games")
    return games


def validate_combat_mat_index(
    payload: Mapping[str, Any], *, expected_catalog_decks: int = CATALOG_DECKS
) -> None:
    """Require the configured deck main view and 14-axis aggregate view."""
    if expected_catalog_decks <= 0:
        raise ValueError("Combat Mat expected catalog deck count must be positive")
    if payload.get("catalog_decks") != expected_catalog_decks:
        raise ValueError(
            f"Combat Mat index must declare {expected_catalog_decks} catalog decks"
        )
    meta = _records(payload.get("meta_archetype_aggregates"), "index meta rows")
    if [item.get("class_id") for item in meta] != list(range(META_ARCHETYPES)):
        raise ValueError("Combat Mat index must contain ordered meta classes 0-13")
    other = _mapping(payload.get("meta_archetype_other"), "index Other meta row")
    if other.get("class_id") != META_ARCHETYPES:
        raise ValueError("Combat Mat index must contain the class-14 Other row")

    catalog_rows = _records(payload.get("catalog"), "index catalog")
    expected_numbers = [
        f"{number:03d}" for number in range(1, expected_catalog_decks + 1)
    ]
    if [str(item.get("deck_number")) for item in catalog_rows] != expected_numbers:
        raise ValueError(
            "Combat Mat index must contain ordered decks "
            f"001-{expected_catalog_decks:03d}"
        )
    for item in catalog_rows:
        if item.get("status") not in {"tested", "pending"}:
            raise ValueError("Combat Mat index deck status must be tested or pending")
        cards = _records(item.get("representative_cards"), "representative cards")
        if not cards or any(not card.get("image_url") for card in cards):
            raise ValueError("Combat Mat index decks require representative card art")


def validate_combat_mat_detail(
    payload: Mapping[str, Any], *, expected_opponent_ids: Set[str]
) -> None:
    """Require exact-60 art, all exact opponents, and all 14 meta aggregates."""
    if payload.get("deck_total") != 60:
        raise ValueError("Combat Mat detail must declare an exact 60-card deck")
    cards = _records(payload.get("deck_cards"), "detail deck cards")
    if sum(_count(card, "count") for card in cards) != 60:
        raise ValueError("Combat Mat detail card counts must sum to 60")
    if any(not card.get("image_url") for card in cards):
        raise ValueError("Combat Mat detail requires card art for every unique card")

    opponents = _mapping(payload.get("by_opponent"), "detail opponent rows")
    if set(str(key) for key in opponents) != set(expected_opponent_ids):
        raise ValueError("Combat Mat detail must contain every catalog opponent")
    opponent_games = sum(
        _validate_record(_mapping(record, "opponent result"))
        for record in opponents.values()
    )

    meta = _records(payload.get("by_meta_archetype"), "detail meta rows")
    if [item.get("class_id") for item in meta] != list(range(META_ARCHETYPES)):
        raise ValueError("Combat Mat detail must contain ordered meta classes 0-13")
    meta_games = sum(_validate_record(item) for item in meta)
    other = _mapping(payload.get("meta_archetype_other"), "detail Other row")
    if other.get("class_id") != META_ARCHETYPES:
        raise ValueError("Combat Mat detail must contain the class-14 Other row")
    other_games = _validate_record(other)
    total_games = _count(payload, "games")
    if opponent_games != total_games or meta_games + other_games != total_games:
        raise ValueError("Combat Mat detail aggregations do not conserve total games")
    if sum(_count(payload, key) for key in ("wins", "losses", "draws")) != total_games:
        raise ValueError("Combat Mat detail total W-L-D does not conserve games")
