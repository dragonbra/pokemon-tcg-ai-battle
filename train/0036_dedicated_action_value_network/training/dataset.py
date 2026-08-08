"""Hash-verified gzip shard batching for 0036."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor

from ..features.collate import collate_canonical_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ValueBatch:
    features: dict[str, Tensor]
    value_target: Tensor
    archetype_target: Tensor
    final_diff_target: Tensor
    episode_weight: Tensor
    is_exact_007: Tensor

    @property
    def size(self) -> int:
        return int(self.value_target.shape[0])

    def to(self, device: str | torch.device, non_blocking: bool = False) -> "ValueBatch":
        return ValueBatch(
            {key: value.to(device, non_blocking=non_blocking) for key, value in self.features.items()},
            self.value_target.to(device, non_blocking=non_blocking),
            self.archetype_target.to(device, non_blocking=non_blocking),
            self.final_diff_target.to(device, non_blocking=non_blocking),
            self.episode_weight.to(device, non_blocking=non_blocking),
            self.is_exact_007.to(device, non_blocking=non_blocking),
        )


class ValueDataset:
    def __init__(self, root: Path, *, verify_hashes: bool = True) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "0036_action_value_dataset_v1" or self.manifest.get("status") != "complete":
            raise ValueError("0036 dataset is not a complete compatible artifact")
        if verify_hashes:
            for split in ("train", "validation"):
                for shard in self.manifest["shards"][split]:
                    path = root / shard["path"]
                    if _sha256(path) != shard["sha256"]:
                        raise ValueError(f"0036 dataset shard hash mismatch: {path}")

    @staticmethod
    def _collate(rows: list[dict[str, Any]]) -> ValueBatch:
        return ValueBatch(
            collate_canonical_records(rows),
            torch.tensor([row["labels"]["value_target"] for row in rows], dtype=torch.float32),
            torch.tensor([row["labels"]["archetype_target"] for row in rows], dtype=torch.long),
            torch.tensor([row["labels"]["final_diff_target"] for row in rows], dtype=torch.long),
            torch.tensor([row["labels"]["episode_weight"] for row in rows], dtype=torch.float32),
            torch.tensor([row["labels"].get("is_exact_007", False) for row in rows], dtype=torch.bool),
        )

    def batches(self, split: str, batch_size: int, *, epoch: int = 0, shuffle: bool = False,
                seed: int = 20260808) -> Iterator[ValueBatch]:
        if split not in {"train", "validation"} or batch_size < 1:
            raise ValueError("0036 split/batch size is invalid")
        shards = list(self.manifest["shards"][split])
        rng = random.Random(seed + epoch)
        if shuffle:
            rng.shuffle(shards)
        pending: list[dict[str, Any]] = []
        for shard in shards:
            with gzip.open(self.root / shard["path"], "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            if shuffle:
                rng.shuffle(rows)
            pending.extend(rows)
            while len(pending) >= batch_size:
                batch, pending = pending[:batch_size], pending[batch_size:]
                yield self._collate(batch)
        if pending:
            yield self._collate(pending)


__all__ = ["ValueBatch", "ValueDataset"]
