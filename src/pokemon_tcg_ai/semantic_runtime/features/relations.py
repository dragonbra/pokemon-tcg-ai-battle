"""Small lossless helpers shared by the semantic compiler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def integer(value: Any, default: int = 0) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


def card_id(value: Any) -> int:
    if isinstance(value, Mapping):
        return max(0, integer(value.get("id", value.get("cardId", 0))))
    return max(0, integer(value))


__all__ = ["card_id", "integer"]
