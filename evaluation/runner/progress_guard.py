"""Shared evaluation progress-guard adjudication contract."""

from __future__ import annotations

from typing import Any


FORFEIT_KEY = "__evaluation_forfeit__"


def progress_guard_forfeit_reason(action: Any) -> str | None:
    if not isinstance(action, dict) or set(action) != {FORFEIT_KEY}:
        return None
    reason = action.get(FORFEIT_KEY)
    return reason if isinstance(reason, str) and reason else None


__all__ = ["FORFEIT_KEY", "progress_guard_forfeit_reason"]
