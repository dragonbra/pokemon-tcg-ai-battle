"""Stateless online projection for the legacy-default 0025 actor."""

from __future__ import annotations

from typing import Any, Sequence

from .base_model import IDOnlyCodec, IDOnlyConfig, collate_id_only


class OnlineCausalEncoder:
    """Expose the Frozen inference-server encoder interface without hidden history."""

    def __init__(
        self,
        actor: int,
        deck: Sequence[int],
        config: IDOnlyConfig,
    ) -> None:
        if actor not in (0, 1):
            raise ValueError("actor must be 0 or 1")
        if len(deck) != 60:
            raise ValueError("deck must contain exactly 60 cards")
        self.actor = actor
        self.deck = tuple(int(card_id) for card_id in deck)
        self.config = config
        self.codec = IDOnlyCodec(config)

    def encode(self, observation: dict[str, Any]) -> dict[str, Any]:
        current = observation.get("current") or {}
        if current.get("yourIndex") != self.actor:
            raise ValueError("observation actor changed within an inference session")
        select = observation.get("select") or {}
        options = select.get("option")
        if not isinstance(options, list) or not options:
            raise ValueError("observation has no legal options")
        minimum = int(select.get("minCount", 0))
        maximum = int(select.get("maxCount", len(options)))
        if not 0 <= minimum <= maximum <= len(options):
            raise ValueError("invalid selection bounds")
        row = self.codec.encode(observation, list(range(minimum)))
        if row is None:
            raise ValueError("observation exceeds the legacy codec contract")
        return collate_id_only([row])


__all__ = ["OnlineCausalEncoder"]
