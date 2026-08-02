"""Verified bounded-memory batches for the 0025 semantic decision dataset."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..model.batching import collate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collate_training_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
    """Collate actor-only tensors and build the shared autoregressive target."""
    if not records:
        raise ValueError("cannot collate an empty training batch")
    actors: list[Mapping[str, Any]] = []
    actions: list[list[int]] = []
    for row in records:
        actor = row.get("actor")
        target = row.get("target")
        if not isinstance(actor, Mapping) or not isinstance(target, Mapping):
            raise ValueError("training row is missing actor or target")
        legacy = actor.get("legacy")
        if not isinstance(legacy, Mapping):
            raise ValueError("actor payload is missing legacy tensors")
        if "action" in legacy:
            raise ValueError("actor payload contains target action")
        action = target.get("ordered_action")
        if not isinstance(action, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in action
        ):
            raise ValueError("ordered action must contain integer option indices")
        option_count = len(legacy.get("option_cat", []))
        if option_count < 1:
            raise ValueError("training row has no legal options")
        if len(set(action)) != len(action) or any(
            value < 0 or value >= option_count for value in action
        ):
            raise ValueError("ordered action is not a unique legal option sequence")
        minimum = int(legacy.get("min_count", 0))
        maximum = int(legacy.get("max_count", option_count))
        if not minimum <= len(action) <= maximum:
            raise ValueError("ordered action violates minCount/maxCount")
        semantic_lengths = {
            len(actor.get("semantic_option_cat", [])),
            len(actor.get("semantic_option_num", [])),
            len(actor.get("semantic_option_state", [])),
        }
        if semantic_lengths != {option_count}:
            raise ValueError("legacy and semantic option counts disagree")
        actors.append(actor)
        actions.append(action)

    batch = collate(actors)
    option_width = batch["option_mask"].size(1)
    target_width = max(len(action) for action in actions) + 1
    targets = torch.full((len(records), target_width), -100, dtype=torch.long)
    for row_index, action in enumerate(actions):
        if action:
            targets[row_index, : len(action)] = torch.tensor(action, dtype=torch.long)
        targets[row_index, len(action)] = option_width
    batch["targets"] = targets
    return batch


class SemanticDecisionDataset:
    """An immutable split-aware semantic dataset with verified shard commitments."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise ValueError("semantic dataset is not complete")
        shards = self.manifest.get("shards")
        if not isinstance(shards, Mapping) or set(shards) != {"train", "validation"}:
            raise ValueError("semantic manifest has no split-specific shards")
        split_counts = self.manifest.get("split_counts")
        if not isinstance(split_counts, Mapping):
            raise ValueError("semantic manifest has no split counts")
        self.shards: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "validation"):
            values = shards[split]
            if not isinstance(values, list):
                raise ValueError(f"semantic shard list is invalid: {split}")
            accepted: list[dict[str, Any]] = []
            observed = 0
            for item in values:
                if not isinstance(item, Mapping):
                    raise ValueError(f"semantic shard entry is invalid: {split}")
                name = item.get("path")
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError(f"unsafe semantic shard path: {name}")
                path = self.root / name
                if not path.is_file() or _sha256(path) != item.get("sha256"):
                    raise ValueError(f"semantic shard commitment mismatch: {path}")
                count = item.get("count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError(f"semantic shard count is invalid: {path}")
                observed += count
                accepted.append(dict(item))
            if observed != int(split_counts[split]):
                raise ValueError(f"semantic split count mismatch: {split}")
            self.shards[split] = accepted
        self.split_counts = {
            split: int(split_counts[split]) for split in ("train", "validation")
        }
        self.manifest_sha256 = _sha256(manifest_path)

    def _load_shard(self, split: str, item: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        path = self.root / str(item["path"])
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                audit = row.get("audit")
                if not isinstance(audit, Mapping) or audit.get("split") != split:
                    raise ValueError(f"row split mismatch in semantic shard: {path}")
                rows.append(row)
        if len(rows) != int(item["count"]):
            raise ValueError(f"row count mismatch in semantic shard: {path}")
        return rows

    def iter_record_batches(
        self,
        split: str,
        batch_size: int,
        *,
        seed: int,
    ) -> Iterator[list[dict[str, Any]]]:
        if split not in self.shards:
            raise ValueError(f"unknown semantic split: {split}")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        rng = random.Random(seed)
        shard_items = list(self.shards[split])
        if split == "train":
            rng.shuffle(shard_items)
        pending: list[dict[str, Any]] = []
        yielded = 0
        for item in shard_items:
            rows = self._load_shard(split, item)
            if split == "train":
                rng.shuffle(rows)
            pending.extend(rows)
            while len(pending) >= batch_size:
                batch = pending[:batch_size]
                del pending[:batch_size]
                yielded += len(batch)
                yield batch
        if pending:
            yielded += len(pending)
            yield pending
        if yielded != self.split_counts[split]:
            raise ValueError(f"semantic epoch did not cover exact split: {split}")

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        seed: int,
    ) -> Iterator[dict[str, Tensor]]:
        for records in self.iter_record_batches(split, batch_size, seed=seed):
            yield collate_training_records(records)

    def batch_count(self, split: str, batch_size: int) -> int:
        if split not in self.split_counts or batch_size < 1:
            raise ValueError("invalid split or batch size")
        return math.ceil(self.split_counts[split] / batch_size)


__all__ = ["SemanticDecisionDataset", "collate_training_records"]
