"""Stateful adapter that expands one macro plan into official primitive selects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from .dragapult import DragapultDamageAllocation, PHANTOM_DIVE_CONTEXTS
from .dragapult import StableTargetIdentity


class MacroProtocolError(RuntimeError):
    pass


@dataclass(slots=True)
class PendingMacroTransaction:
    battle_id: str
    actor: int
    allocation: DragapultDamageAllocation
    effect_serial: int | None = None
    created_at: float = field(default_factory=monotonic)
    timeout_seconds: float = 30.0
    consumed: int = 0
    invalid_reason: str | None = None
    _visible_identity_fingerprint: tuple[Any, ...] | None = None

    @property
    def complete(self) -> bool:
        return self.consumed == self.allocation.total

    def invalidate(self, reason: str) -> None:
        self.invalid_reason = reason

    def next_primitive(self, observation: Mapping[str, Any], *, battle_id: str) -> list[int]:
        try:
            return self._next_primitive(observation, battle_id=battle_id)
        except BaseException as error:
            self.invalidate(f"{type(error).__name__}: {error}")
            raise

    def _next_primitive(self, observation: Mapping[str, Any], *, battle_id: str) -> list[int]:
        if self.invalid_reason is not None:
            raise MacroProtocolError(f"macro transaction is invalid: {self.invalid_reason}")
        if battle_id != self.battle_id:
            raise MacroProtocolError("battle/session ID drift")
        if monotonic() - self.created_at > self.timeout_seconds:
            raise MacroProtocolError("macro transaction expired")
        if self.complete:
            raise MacroProtocolError("macro transaction is already complete")
        current, select = observation.get("current"), observation.get("select")
        if not isinstance(current, Mapping) or not isinstance(select, Mapping):
            raise MacroProtocolError("macro callback lacks current/select")
        result = current.get("result")
        if isinstance(result, int) and result >= 0:
            raise MacroProtocolError("unexpected terminal inside macro transaction")
        if current.get("yourIndex") != self.actor:
            raise MacroProtocolError("macro actor or priority drift")
        if select.get("context") not in PHANTOM_DIVE_CONTEXTS:
            raise MacroProtocolError("macro selection context drift")
        if select.get("remainDamageCounter") != self.allocation.total - self.consumed:
            raise MacroProtocolError("macro remaining-counter drift")
        if select.get("minCount") != 1 or select.get("maxCount") != 1:
            raise MacroProtocolError("macro primitive selection bounds drift")
        effect = select.get("effect")
        effect_serial = effect.get("serial") if isinstance(effect, Mapping) else None
        if self.effect_serial is not None and effect_serial != self.effect_serial:
            raise MacroProtocolError("macro effect identity drift")
        if self.effect_serial is None and isinstance(effect_serial, int):
            self.effect_serial = effect_serial
        serial = self.allocation.primitive_target_serials()[self.consumed]
        options = select.get("option")
        players = current.get("players")
        if not isinstance(options, Sequence) or not isinstance(players, Sequence) or len(players) != 2:
            raise MacroProtocolError("macro callback has an invalid legal set")
        logs = observation.get("logs")
        if isinstance(logs, Sequence):
            log_types = {
                item.get("type") for item in logs if isinstance(item, Mapping)
            }
            if 22 in log_types or "Coin" in log_types:
                raise MacroProtocolError("random result inside macro transaction")
            unexpected = log_types.difference({15, 16, "Attack", "HpChange"})
            if unexpected:
                raise MacroProtocolError(f"new information/event inside macro transaction: {unexpected}")
        visible_identity = tuple(
            (player_index, zone, slot, card.get("serial"), card.get("id"))
            for player_index, player in enumerate(players)
            if isinstance(player, Mapping)
            for zone in ("active", "bench", "discard")
            for slot, card in enumerate(player.get(zone) or [])
            if isinstance(card, Mapping)
        )
        if (
            self._visible_identity_fingerprint is not None
            and visible_identity != self._visible_identity_fingerprint
        ):
            raise MacroProtocolError("visible card identity changed inside macro transaction")
        self._visible_identity_fingerprint = visible_identity
        matches: list[int] = []
        expected = next(target for target in self.allocation.target_ids if target.serial == serial)
        for index, option in enumerate(options):
            if not isinstance(option, Mapping) or option.get("playerIndex") != expected.player_index:
                continue
            slot = option.get("index")
            bench = players[expected.player_index].get("bench") if isinstance(players[expected.player_index], Mapping) else None
            if not isinstance(slot, int) or not isinstance(bench, Sequence) or not 0 <= slot < len(bench):
                continue
            target = bench[slot]
            if isinstance(target, Mapping) and target.get("serial") == serial and target.get("id") == expected.card_id:
                matches.append(index)
        if len(matches) != 1:
            raise MacroProtocolError(f"stable target serial {serial} has {len(matches)} legal matches")
        self.consumed += 1
        return [matches[0]]

    def to_wire(self) -> dict[str, Any]:
        return {
            "battle_id": self.battle_id, "actor": self.actor,
            "effect_serial": self.effect_serial,
            "targets": [
                {
                    "player_index": item.player_index, "serial": item.serial,
                    "card_id": item.card_id, "initial_bench_slot": item.initial_bench_slot,
                }
                for item in self.allocation.target_ids
            ],
            "counters": list(self.allocation.counters),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "PendingMacroTransaction":
        targets = tuple(StableTargetIdentity(**dict(item)) for item in payload["targets"])
        allocation = DragapultDamageAllocation(targets, tuple(payload["counters"]))
        return cls(
            battle_id=str(payload["battle_id"]), actor=int(payload["actor"]),
            allocation=allocation, effect_serial=payload.get("effect_serial"),
        )


__all__ = ["MacroProtocolError", "PendingMacroTransaction"]
