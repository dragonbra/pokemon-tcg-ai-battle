"""Record-backed streaming loaders for formal Semantic Goal Policy training."""
from __future__ import annotations

import json
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..data.shards import iter_dataset, iter_records
from ..features.card_semantics import CardSemanticRegistry
from ..features.compiler import COMPILER_VERSION, compiler_sha256
from .batching import collate_records, iter_prepare_record_stream


@dataclass(frozen=True, slots=True)
class DatasetContract:
    root: Path
    content_sha256: str
    compiler_sha256: str
    ontology_sha256: str
    counts: Mapping[str, int]


class RecordBatchSource:
    def __init__(
        self,
        root: Path | str,
        *,
        registry: CardSemanticRegistry,
        batch_size: int,
        shuffle_buffer: int = 2048,
        seed: int = 20260726,
        materialize_semantics: bool = True,
    ) -> None:
        self.root = Path(root)
        self.registry = registry
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.materialize_semantics = materialize_semantics
        if batch_size <= 0 or shuffle_buffer < batch_size:
            raise ValueError("batch size must be positive and fit inside shuffle buffer")
        self.reference = json.loads((self.root / "dataset_reference.json").read_text())
        self.contract = self._validate_contract()

    def _validate_contract(self) -> DatasetContract:
        reference = self.reference
        if reference.get("schema_version") != "dataset_reference_v3":
            raise ValueError("formal training requires dataset_reference_v3")
        if reference.get("feature_compiler_version") != COMPILER_VERSION:
            raise ValueError("dataset feature compiler version mismatch")
        current_compiler = compiler_sha256()
        if reference.get("feature_compiler_sha256") != current_compiler:
            raise ValueError("dataset feature compiler SHA-256 mismatch")
        if reference.get("ontology_sha256") != self.registry.sha256:
            raise ValueError("dataset ontology SHA-256 mismatch")
        observed = {"train": 0, "validation": 0}
        for record in iter_dataset(self.root):
            observed[record.split] += 1
        if observed != reference.get("counts"):
            raise ValueError("dataset counts changed after full validation")
        return DatasetContract(
            self.root,
            reference["content_sha256"],
            current_compiler,
            self.registry.sha256,
            dict(observed),
        )

    def _records(self, split: str) -> Iterator[dict[str, Any]]:
        if split not in {"train", "validation"}:
            raise ValueError("unsupported dataset split")
        for shard in self.reference["shards"][split]:
            path = self.root / shard["path"]
            for record in iter_records(path, expected_sha256=shard["sha256"]):
                if record.split != split:
                    raise ValueError("record split disagrees with requested split")
                yield record.as_dict()

    @staticmethod
    def _buffer_shuffle(
        values: Iterator[dict[str, Any]],
        *,
        size: int,
        rng: random.Random,
    ) -> Iterator[dict[str, Any]]:
        buffer: list[dict[str, Any]] = []
        for value in values:
            if len(buffer) < size:
                buffer.append(value)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = value
        rng.shuffle(buffer)
        yield from buffer

    def batches(
        self,
        split: str,
        *,
        epoch: int,
        training: bool,
    ) -> Iterator[dict[str, Tensor]]:
        prepared = iter_prepare_record_stream(
            self._records(split),
            registry=self.registry if self.materialize_semantics else None,
        )
        values: Iterator[dict[str, Any]]
        torch_generator = torch.Generator().manual_seed(self.seed + epoch * 10_000)
        if training:
            values = self._buffer_shuffle(
                prepared,
                size=self.shuffle_buffer,
                rng=random.Random(self.seed + epoch),
            )
        else:
            values = prepared
        batch: list[dict[str, Any]] = []
        for value in values:
            batch.append(value)
            if len(batch) == self.batch_size:
                yield collate_records(
                    batch,
                    registry=self.registry,
                    permute_options=training,
                    generator=torch_generator,
                )
                batch = []
        if batch:
            yield collate_records(
                batch,
                registry=self.registry,
                permute_options=training,
                generator=torch_generator,
            )


__all__ = ["DatasetContract", "RecordBatchSource"]
