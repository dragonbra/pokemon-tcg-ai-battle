"""Online causal feature encoder matching the frozen 0014 materializer."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .foundation.model_source.base_model import IDOnlyCodec, IDOnlyConfig, collate_id_only
from .foundation.model_source.features.compiler import (
    _event_features,
    _ledger_features,
    _zone_row,
)
from .foundation.model_source.knowledge.state import CausalKnowledge


def _padded_2d(
    rows: Sequence[Sequence[int | float]], width: int, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    count = max(1, len(rows))
    values = torch.zeros((1, count, width), dtype=dtype)
    mask = torch.zeros((1, count), dtype=torch.bool)
    if rows:
        values[0, : len(rows)] = torch.tensor(rows, dtype=dtype)
        mask[0, : len(rows)] = True
    return values, mask


def _padded_1d(values: Sequence[int | float], dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    count = max(1, len(values))
    tensor = torch.zeros((1, count), dtype=dtype)
    mask = torch.zeros((1, count), dtype=torch.bool)
    if values:
        tensor[0, : len(values)] = torch.tensor(values, dtype=dtype)
        mask[0, : len(values)] = True
    return tensor, mask


class OnlineCausalEncoder:
    """Maintain actor-local knowledge and emit a one-observation AC batch."""

    def __init__(self, actor: int, registered_deck: Sequence[int], config: IDOnlyConfig) -> None:
        if len(registered_deck) != 60:
            raise ValueError("registered deck must contain 60 cards")
        self.actor = actor
        self.deck = tuple(int(card) for card in registered_deck)
        self.codec = IDOnlyCodec(config)
        self.knowledge = CausalKnowledge(actor, self.deck)
        self.counts = Counter(self.deck)
        self.registered_ids = sorted(self.counts)
        self.registered_multiplicity = [
            self.counts[card_id] for card_id in self.registered_ids
        ]

    def encode(self, observation: Mapping[str, Any]) -> dict[str, Tensor]:
        encoded = self.codec.encode(dict(observation), None)
        if encoded is None:
            raise RuntimeError("observation exceeds the frozen 0014 codec contract")
        batch = collate_id_only([encoded])
        snapshot = self.knowledge.consume(observation)
        ledger_cat, ledger_num = _ledger_features(self.deck, snapshot)
        event_cat, event_num = _event_features(snapshot)

        current = observation["current"]
        players = current["players"]
        stadium = current.get("stadium", ())
        zone_rows: list[list[float]] = []
        for player_index, player in enumerate(players):
            if player_index == 1 - self.actor:
                known = len(snapshot.known_opponent_hand)
                unknown = snapshot.unknown_opponent_hand
            else:
                known = int(player.get("handCount", 0))
                unknown = 0
            stadium_count = (
                sum(
                    1
                    for item in stadium
                    if isinstance(item, Mapping)
                    and item.get("playerIndex") == player_index
                )
                if isinstance(stadium, Sequence)
                else 0
            )
            zone_rows.append(
                _zone_row(
                    player,
                    known_hand=known,
                    unknown_hand=unknown,
                    stadium_count=stadium_count,
                )
            )

        registered, registered_mask = _padded_1d(self.registered_ids, torch.long)
        multiplicity, multiplicity_mask = _padded_1d(
            self.registered_multiplicity, torch.float32
        )
        ledger_cats, ledger_mask = _padded_2d(ledger_cat, 4, torch.long)
        ledger_nums, ledger_num_mask = _padded_2d(ledger_num, 15, torch.float32)
        event_cats, event_mask = _padded_2d(event_cat, 8, torch.long)
        event_nums, event_num_mask = _padded_2d(event_num, 4, torch.float32)
        known_hand, known_hand_mask = _padded_1d(
            [item.card_id for item in snapshot.known_opponent_hand], torch.long
        )
        if not (
            torch.equal(registered_mask, multiplicity_mask)
            and torch.equal(ledger_mask, ledger_num_mask)
            and torch.equal(event_mask, event_num_mask)
        ):
            raise RuntimeError("online paired feature masks disagree")
        batch.update(
            {
                "source_id": torch.zeros(1, dtype=torch.long),
                "zone_inventory_num": torch.tensor([zone_rows], dtype=torch.float32),
                "registered_card_ids": registered,
                "registered_multiplicity": multiplicity,
                "registered_mask": registered_mask,
                "ledger_cat": ledger_cats,
                "ledger_num": ledger_nums,
                "ledger_mask": ledger_mask,
                "event_cat": event_cats,
                "event_num": event_nums,
                "event_mask": event_mask,
                "known_opponent_hand_card_ids": known_hand,
                "known_opponent_hand_mask": known_hand_mask,
                "unknown_opponent_hand_count": torch.tensor(
                    [snapshot.unknown_opponent_hand], dtype=torch.float32
                ),
            }
        )
        return batch


__all__ = ["OnlineCausalEncoder"]
