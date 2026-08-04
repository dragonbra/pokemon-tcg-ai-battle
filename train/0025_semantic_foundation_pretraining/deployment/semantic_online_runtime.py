"""Stateful raw-observation adapter for the multi-memory semantic actor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..features.compiler import compile_row
from ..features.prototypes import PrototypeIndex
from ..knowledge.state import CausalKnowledge
from ..model.batching import collate
from ..model.multi_memory import SemanticModelConfig


def _prototype_path() -> Path:
    packaged = Path(__file__).resolve().with_name("official_public_prototypes_v1.json")
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[1] / "assets" / "official_public_prototypes_v1.json"


class OnlineCausalEncoder:
    """Compile one chronological engine session with the training-time compiler."""

    def __init__(
        self,
        actor: int,
        registered_deck: Sequence[int],
        config: SemanticModelConfig,
    ) -> None:
        if actor not in (0, 1):
            raise ValueError("actor must be 0 or 1")
        if len(registered_deck) != 60:
            raise ValueError("registered deck must contain exactly 60 cards")
        self.actor = actor
        self.deck = tuple(int(card_id) for card_id in registered_deck)
        if any(card_id <= 0 for card_id in self.deck):
            raise ValueError("registered deck contains a non-positive card ID")
        self.config = config
        self.prototypes = PrototypeIndex.load(_prototype_path())
        self.knowledge = CausalKnowledge(actor, self.deck)

    def encode(self, observation: Mapping[str, Any]):
        current = observation.get("current")
        select = observation.get("select")
        if not isinstance(current, Mapping) or current.get("yourIndex") != self.actor:
            raise ValueError("semantic online actor mismatch")
        if not isinstance(select, Mapping):
            raise ValueError("semantic online observation has no select payload")
        options = select.get("option")
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
            raise ValueError("semantic online observation has no legal options")
        minimum = int(select.get("minCount", 0))
        maximum = int(select.get("maxCount", len(options)))
        if not 0 <= minimum <= maximum <= len(options):
            raise ValueError("semantic online selection bounds are invalid")
        if len(options) > self.config.max_options or minimum > self.config.max_action_steps:
            raise RuntimeError("observation exceeds semantic inference bounds")

        row = {
            "actor_observation": dict(observation),
            "ordered_action": list(range(minimum)),
            "action_termination": "online_structural_target",
            "identity": None,
            "split": "online",
            "deck_manifest": {"counts": sorted(Counter(self.deck).items())},
        }
        snapshot = self.knowledge.consume(observation)
        return collate([compile_row(row, snapshot, self.prototypes)])


__all__ = ["OnlineCausalEncoder"]
