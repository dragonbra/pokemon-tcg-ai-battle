"""Legal fallback for exported canonical policies."""

from __future__ import annotations

from typing import Any


def legal_fallback(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option")
    if not isinstance(options, list):
        raise ValueError("select.option is not a list")
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", len(options)))
    if not 0 <= minimum <= maximum <= len(options):
        raise ValueError("invalid selection bounds")
    return list(range(minimum))


__all__ = ["legal_fallback"]
