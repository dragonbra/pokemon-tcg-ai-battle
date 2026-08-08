"""Torch-free worker-local compiler for one chronological official battle side."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..semantic_policy.domain.prototypes import PrototypeIndex
from ..semantic_policy.features.compiler import compile_canonical_row
from ..semantic_policy.knowledge.state import CausalKnowledge


ASSETS = Path(__file__).resolve().parents[1] / "semantic_policy/assets"


@lru_cache(maxsize=1)
def _shared_prototypes() -> PrototypeIndex:
    return PrototypeIndex.load(
        ASSETS / "official_public_prototypes_v1.json",
        ASSETS / "official_full_engine_prototypes_v2.json",
    )


class WorkerLocalCompiler:
    def __init__(self, actor: int, registered_deck: Sequence[int]) -> None:
        if actor not in (0, 1):
            raise ValueError("actor must be 0 or 1")
        self.deck = tuple(int(card_id) for card_id in registered_deck)
        if len(self.deck) != 60 or any(card_id <= 0 for card_id in self.deck):
            raise ValueError("registered deck must contain exactly 60 positive card IDs")
        self.actor = actor
        self.prototypes = _shared_prototypes()
        self.knowledge = CausalKnowledge(actor, self.deck)
        self.deck_manifest = {"counts": sorted(Counter(self.deck).items())}

    def compile(self, observation: Mapping[str, Any]):
        current = observation.get("current")
        select = observation.get("select")
        if not isinstance(current, Mapping) or current.get("yourIndex") != self.actor:
            raise ValueError("0037 worker-local actor mismatch")
        if not isinstance(select, Mapping):
            raise ValueError("0037 worker-local observation has no select payload")
        options = select.get("option")
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
            raise ValueError("0037 worker-local observation has no legal options")
        minimum = select.get("minCount", 0)
        maximum = select.get("maxCount", len(options))
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 0 <= minimum <= maximum <= len(options)
        ):
            raise ValueError("0037 worker-local selection bounds are invalid")
        if len(options) > 128 or minimum > 64:
            raise RuntimeError("observation exceeds 0037 inference bounds")
        row = {
            "actor_observation": dict(observation),
            "ordered_action": list(range(minimum)),
            "action_termination": "online_structural_target",
            "identity": None,
            "split": "online",
            "deck_manifest": self.deck_manifest,
        }
        snapshot = self.knowledge.consume(observation)
        return compile_canonical_row(row, snapshot, self.prototypes)


__all__ = ["WorkerLocalCompiler"]
