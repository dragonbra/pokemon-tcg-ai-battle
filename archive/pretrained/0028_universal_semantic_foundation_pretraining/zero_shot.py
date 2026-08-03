"""Stateful zero-shot adapter from official observations to 0028 actions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from loader import ENGINE_PROTOTYPES, PUBLIC_PROTOTYPES, load_model
from model_source.domain import PrototypeIndex
from model_source.features.collate import collate_canonical_records
from model_source.features.compiler import compile_canonical_row
from model_source.knowledge.state import CausalKnowledge
from model_source.model import SemanticPolicy


def legal_fallback(observation: Mapping[str, Any]) -> list[int]:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        raise ValueError("observation has no select payload")
    options = select.get("option")
    if not isinstance(options, Sequence):
        raise ValueError("observation has no legal options")
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", len(options)))
    if not 0 <= minimum <= maximum <= len(options):
        raise ValueError("observation selection bounds are invalid")
    return list(range(minimum))


class OnlineCausalEncoder:
    """Chronologically apply the exact training-time feature compiler."""

    def __init__(self, actor: int, registered_deck: Sequence[int], model: SemanticPolicy):
        if actor not in (0, 1):
            raise ValueError("actor must be 0 or 1")
        deck = tuple(int(card_id) for card_id in registered_deck)
        if len(deck) != 60 or any(card_id <= 0 for card_id in deck):
            raise ValueError("registered_deck must contain exactly 60 positive card IDs")
        self.actor = actor
        self.deck = deck
        self.config = model.config
        self.prototypes = PrototypeIndex.load(PUBLIC_PROTOTYPES, ENGINE_PROTOTYPES)
        self.knowledge = CausalKnowledge(actor, deck)

    def encode(self, observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        current = observation.get("current")
        select = observation.get("select")
        if not isinstance(current, Mapping) or current.get("yourIndex") != self.actor:
            raise ValueError("observation actor does not match the causal session")
        if not isinstance(select, Mapping):
            raise ValueError("observation has no select payload")
        options = select.get("option")
        if not isinstance(options, Sequence):
            raise ValueError("observation has no legal options")
        minimum = int(select.get("minCount", 0))
        maximum = int(select.get("maxCount", len(options)))
        if not 0 <= minimum <= maximum <= len(options):
            raise ValueError("observation selection bounds are invalid")
        if len(options) > self.config.max_options or minimum > self.config.max_action_steps:
            raise RuntimeError("observation exceeds the released model bounds")

        counts = Counter(self.deck)
        structural_target = list(range(minimum))
        row = {
            "actor_observation": dict(observation),
            "ordered_action": structural_target,
            "action_termination": "online_structural_target",
            "identity": None,
            "split": "zero_shot",
            "deck_manifest": {"counts": sorted(counts.items())},
        }
        snapshot = self.knowledge.consume(observation)
        record = compile_canonical_row(row, snapshot, self.prototypes)
        return collate_canonical_records([record])


class ZeroShotPolicy:
    """Load once, preserve causal state, and select ordered legal option indices."""

    requires_source_identity = False

    def __init__(
        self,
        registered_deck: Sequence[int],
        *,
        device: str | torch.device = "cpu",
        model: SemanticPolicy | None = None,
    ) -> None:
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.model = model if model is not None else load_model(self.device)
        self.model.to(self.device).eval()
        self.deck = tuple(int(card_id) for card_id in registered_deck)
        if len(self.deck) != 60:
            raise ValueError("registered_deck must contain exactly 60 cards")
        self.encoder: OnlineCausalEncoder | None = None

    def reset(self) -> None:
        self.encoder = None

    def select(self, observation: Mapping[str, Any]) -> list[int]:
        current = observation.get("current")
        actor = current.get("yourIndex") if isinstance(current, Mapping) else None
        if actor not in (0, 1):
            raise ValueError("observation has no valid actor index")
        if self.encoder is None or self.encoder.actor != actor:
            self.encoder = OnlineCausalEncoder(actor, self.deck, self.model)
        batch = {
            name: value.to(self.device, non_blocking=True)
            for name, value in self.encoder.encode(observation).items()
        }
        with torch.inference_mode():
            return self.model.greedy_action(batch)

    def select_or_fallback(self, observation: Mapping[str, Any]) -> list[int]:
        try:
            return self.select(observation)
        except (IndexError, RuntimeError, ValueError):
            self.reset()
            return legal_fallback(observation)


__all__ = ["OnlineCausalEncoder", "ZeroShotPolicy", "legal_fallback"]

