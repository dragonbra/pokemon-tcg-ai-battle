"""Online feature encoder for 0028 canonical semantic inference."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from .features.collate import collate_canonical_records
from .features.compiler import compile_canonical_row
from .knowledge.state import CausalKnowledge


def _deck_manifest(deck: Sequence[int]) -> dict[str, object]:
    counts = Counter(int(card_id) for card_id in deck)
    return {"counts": [[card_id, counts[card_id]] for card_id in sorted(counts)]}


def _legal_target(observation: Mapping[str, Any]) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option")
    if not isinstance(options, list):
        raise ValueError("select.option is not a list")
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", len(options)))
    if not 0 <= minimum <= maximum <= len(options):
        raise ValueError("invalid selection bounds")
    return list(range(minimum))


class OnlineCausalEncoder:
    """Maintain actor-local ledger and emit a one-observation 0028 batch."""

    def __init__(self, actor: int, registered_deck: Sequence[int], config: Any, prototypes: Any | None = None) -> None:
        if len(registered_deck) != 60:
            raise ValueError("registered deck must contain 60 cards")
        self.actor = int(actor)
        self.deck = tuple(int(card_id) for card_id in registered_deck)
        self.config = config
        if prototypes is None:
            from pathlib import Path
            from .domain.prototypes import PrototypeIndex

            root = Path(__file__).resolve().parent
            prototypes = PrototypeIndex.load(
                root / "official_public_prototypes_v1.json",
                root / "official_full_engine_prototypes_v1.json",
            )
        self.prototypes = prototypes
        self.knowledge = CausalKnowledge(self.actor, self.deck)
        self._deck_manifest = _deck_manifest(self.deck)

    def encode(self, observation: Mapping[str, Any]) -> dict[str, Tensor]:
        if (observation.get("current") or {}).get("yourIndex") != self.actor:
            raise ValueError("causal actor mismatch")
        target = _legal_target(observation)
        snapshot = self.knowledge.consume(observation)
        row = {
            "actor_observation": observation,
            "ordered_action": target,
            "action_termination": "online_inference_placeholder",
            "identity": {
                "date": "online",
                "episode_id": 0,
                "player_index": self.actor,
            },
            "split": "online",
            "deck_manifest": self._deck_manifest,
        }
        record = compile_canonical_row(row, snapshot, self.prototypes)
        return collate_canonical_records([record])


__all__ = ["OnlineCausalEncoder"]
