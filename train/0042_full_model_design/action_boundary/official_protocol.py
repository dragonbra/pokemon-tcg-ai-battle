"""Session-isolated expansion of pending macros into unchanged official returns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .macro_protocol import PendingMacroTransaction


class OfficialProtocolExecutor:
    def __init__(self) -> None:
        self._pending: dict[str, PendingMacroTransaction] = {}

    def begin(self, transaction: PendingMacroTransaction) -> None:
        if transaction.battle_id in self._pending:
            raise RuntimeError(f"battle already has a pending macro: {transaction.battle_id}")
        self._pending[transaction.battle_id] = transaction

    def has_pending(self, battle_id: str) -> bool:
        return battle_id in self._pending

    def select(self, battle_id: str, observation: Mapping[str, Any]) -> list[int]:
        transaction = self._pending[battle_id]
        action = transaction.next_primitive(observation, battle_id=battle_id)
        if transaction.complete:
            del self._pending[battle_id]
        return action

    def invalidate(self, battle_id: str, reason: str) -> PendingMacroTransaction | None:
        transaction = self._pending.pop(battle_id, None)
        if transaction is not None:
            transaction.invalidate(reason)
        return transaction

    def clear(self, battle_id: str) -> None:
        self._pending.pop(battle_id, None)


__all__ = ["OfficialProtocolExecutor"]
