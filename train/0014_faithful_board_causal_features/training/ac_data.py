"""Full-feature model view over the one immutable 0014 cache."""
from __future__ import annotations

import json
import random
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from torch import Tensor

from . import materialized


def select_ac_batch(tensors: Mapping[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    batch = materialized.select_a0_batch(tensors, indices)

    def selected(name: str) -> Tensor:
        return tensors[name].index_select(0, indices)

    registered_mask = selected("registered_mask")
    registered_width = max(1, int(registered_mask.sum(1).max()))
    event_mask = selected("event_mask")
    event_width = max(1, int(event_mask.sum(1).max()))
    hand_mask = selected("known_opponent_hand_mask")
    hand_width = max(1, int(hand_mask.sum(1).max()))
    batch.update(
        {
            "zone_inventory_num": selected("zone_inventory_num").float(),
            "registered_card_ids": selected("registered_card_ids")[:, :registered_width].long(),
            "registered_multiplicity": selected("registered_multiplicity")[:, :registered_width].float(),
            "registered_mask": registered_mask[:, :registered_width],
            "ledger_cat": selected("ledger_cat")[:, :registered_width].long(),
            "ledger_num": selected("ledger_num")[:, :registered_width].float(),
            "ledger_mask": selected("ledger_mask")[:, :registered_width],
            "event_cat": selected("event_cat")[:, :event_width].long(),
            "event_num": selected("event_num")[:, :event_width].float(),
            "event_mask": event_mask[:, :event_width],
            "known_opponent_hand_card_ids": selected("known_opponent_hand_card_ids")[:, :hand_width].long(),
            "known_opponent_hand_mask": hand_mask[:, :hand_width],
            "unknown_opponent_hand_count": selected("unknown_opponent_hand_count").float(),
        }
    )
    if not torch.equal(batch["registered_mask"], batch["ledger_mask"]):
        raise ValueError("registered deck and ledger alignment drift")
    return batch


def iter_ac_batches(
    root: Path | str,
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, Tensor]]:
    base = Path(root)
    reference = json.loads((base / "feature_dataset_reference.json").read_text())
    shards = list(reference["shards"][split])
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(shards)
    for expected in shards:
        tensors = materialized._load_shard(base / expected["path"], expected)
        eligible = tensors["a0_eligible"].nonzero(as_tuple=False).flatten()
        if shuffle:
            generator = torch.Generator().manual_seed(rng.randrange(2**63))
            eligible = eligible[torch.randperm(eligible.numel(), generator=generator)]
        for start in range(0, eligible.numel(), batch_size):
            yield select_ac_batch(tensors, eligible[start : start + batch_size])


__all__ = ["iter_ac_batches", "select_ac_batch"]
