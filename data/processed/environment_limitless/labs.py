from __future__ import annotations

import json
from typing import Any


def _message(payload: str) -> Any:
    document = json.loads(payload)
    if not document.get("ok"):
        raise ValueError("Limitless Labs payload is not successful")
    return document.get("message")


def parse_deck_meta_payload(payload: str) -> list[dict[str, Any]]:
    rows = _message(payload)
    if not isinstance(rows, list):
        raise ValueError("Limitless Labs deck payload must contain a list")
    return [
        {
            "deck_id": row["identifier"],
            "name": row["name"],
            "primary_id": row["sup_identifier"],
            "primary_name": row["sup_name"],
            "icons": row.get("icons") or "",
            "players": int(row["players"]),
            "day2s": int(row["day2s"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "ties": int(row["ties"]),
            "matches": int(row["wins"]) + int(row["losses"]) + int(row["ties"]),
        }
        for row in rows
    ]


def parse_standings_payload(payload: str) -> list[dict[str, Any]]:
    rows = _message(payload)
    if not isinstance(rows, list):
        raise ValueError("Limitless Labs standings payload must contain a list")
    return [
        {
            "player_id": row.get("player_id"),
            "labs_player_id": row.get("tp_id"),
            "name": row["name"],
            "country": row.get("country"),
            "placement": row.get("placement"),
            "points": int(row.get("points") or 0),
            "wins": int(row.get("wins") or 0),
            "losses": int(row.get("losses") or 0),
            "ties": int(row.get("ties") or 0),
            "day2": bool(row.get("day2")),
            "topcut": bool(row.get("topcut")),
            "has_decklist": bool(row.get("decklist")),
            "deck_id": row.get("deck_id"),
            "deck_name": row.get("deck_name"),
            "icons": row.get("icons") or "",
        }
        for row in rows
    ]


def parse_decklist_payload(payload: str) -> list[dict[str, Any]]:
    message = _message(payload)
    if not isinstance(message, dict):
        raise ValueError("Limitless Labs decklist payload must contain groups")
    cards: list[dict[str, Any]] = []
    for group in ("pokemon", "trainer", "energy"):
        for row in message.get(group, []):
            cards.append(
                {
                    "group": group,
                    "count": int(row["count"]),
                    "name": row["name"],
                    "set": row["set"],
                    "number": str(row["number"]),
                }
            )
    return cards
