"""Immutable canonical semantic shards and deterministic training batches."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from torch import Tensor

from ..features.canonical.batching import collate_canonical_records
from ..features.canonical.schema import SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CanonicalDecisionDataset:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION or self.manifest.get("status") != "complete":
            raise ValueError("canonical dataset manifest is not complete/supported")
        self.shards: dict[str, list[dict[str, Any]]] = {}
        self.split_counts: dict[str, int] = {}
        for split in ("train", "validation"):
            observed = 0
            accepted = []
            for raw in self.manifest["shards"][split]:
                item = dict(raw)
                name = item.get("path")
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError("unsafe canonical shard path")
                path = self.root / name
                if not path.is_file() or _sha256(path) != item.get("sha256"):
                    raise ValueError(f"canonical shard commitment mismatch: {path}")
                observed += int(item["count"])
                accepted.append(item)
            expected = int(self.manifest["split_counts"][split])
            if observed != expected:
                raise ValueError(f"canonical split count mismatch: {split}")
            self.shards[split] = accepted
            self.split_counts[split] = expected
        self.manifest_sha256 = _sha256(manifest_path)

    def _load(self, split: str, item: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows = []
        with gzip.open(self.root / str(item["path"]), "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("schema_version") != SCHEMA_VERSION or row.get("audit", {}).get("split") != split:
                    raise ValueError("canonical shard row contract mismatch")
                rows.append(row)
        if len(rows) != int(item["count"]):
            raise ValueError("canonical shard row count mismatch")
        return rows

    def iter_record_batches(self, split: str, batch_size: int, *, seed: int) -> Iterator[list[dict[str, Any]]]:
        if split not in self.shards or batch_size < 1:
            raise ValueError("invalid canonical split or batch size")
        rng = random.Random(seed)
        shards = list(self.shards[split])
        if split == "train":
            rng.shuffle(shards)
        pending: list[dict[str, Any]] = []
        yielded = 0
        for item in shards:
            rows = self._load(split, item)
            if split == "train":
                rng.shuffle(rows)
            pending.extend(rows)
            while len(pending) >= batch_size:
                batch, pending = pending[:batch_size], pending[batch_size:]
                yielded += len(batch)
                yield batch
        if pending:
            yielded += len(pending)
            yield pending
        if yielded != self.split_counts[split]:
            raise ValueError("canonical iteration did not cover the exact split")

    def iter_batches(self, split: str, batch_size: int, *, seed: int) -> Iterator[dict[str, Tensor]]:
        for rows in self.iter_record_batches(split, batch_size, seed=seed):
            yield collate_canonical_records(rows)

    def batch_count(self, split: str, batch_size: int) -> int:
        if split not in self.split_counts or batch_size < 1:
            raise ValueError("invalid canonical split or batch size")
        return math.ceil(self.split_counts[split] / batch_size)


__all__ = ["CanonicalDecisionDataset"]
