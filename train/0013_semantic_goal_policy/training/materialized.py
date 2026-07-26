"""Immutable model-ready tensor shards for the shared M0-M5 feature contract."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import shutil
import threading
import uuid
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..data.shards import iter_dataset
from ..features.card_semantics import CardSemanticRegistry
from ..features.compiler import COMPILER_VERSION, compiler_sha256
from .batching import collate_records, iter_prepare_record_stream

SCHEMA_VERSION = "materialized_feature_dataset_v1"
CACHE_FIELDS = frozenset({
    "state_cat", "state_num", "entities_cat", "entities_num", "entity_semantic",
    "entity_mask", "deck_card_ids", "deck_multiplicity", "deck_semantic", "deck_mask",
    "ledger_cat", "ledger_num", "ledger_semantic", "ledger_mask", "events_cat",
    "events_num", "event_semantic", "event_mask", "relation_edges", "relation_mask",
    "options_cat", "options_num", "option_semantic", "option_mask", "min_count",
    "max_count", "targets", "target_mask",
})
_INTEGER_FIELDS = frozenset({
    "state_cat", "entities_cat", "deck_card_ids", "ledger_cat", "events_cat",
    "relation_edges", "options_cat", "min_count", "max_count", "targets",
})
_FLOAT_FIELDS = frozenset({
    "state_num", "entities_num", "entity_semantic", "deck_multiplicity",
    "deck_semantic", "ledger_num", "ledger_semantic", "events_num",
    "event_semantic", "options_num", "option_semantic",
})
_BOOL_FIELDS = CACHE_FIELDS - _INTEGER_FIELDS - _FLOAT_FIELDS
_WORKER_REGISTRY: CardSemanticRegistry | None = None


def _group_key(row: Mapping[str, Any]) -> tuple[object, object, object]:
    identity = row.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("materialization record lacks source identity")
    source = identity.get("source")
    value = source if isinstance(source, Mapping) else identity
    required = ("date", "episode_id", "player_index")
    if not all(name in value for name in required):
        raise ValueError("materialization record has incomplete source identity")
    return value["date"], value["episode_id"], value["player_index"]


def _record_groups(rows: Iterable[Mapping[str, Any]]) -> Iterator[list[Mapping[str, Any]]]:
    seen: set[tuple[object, object, object]] = set()
    current_key: tuple[object, object, object] | None = None
    current: list[Mapping[str, Any]] = []
    for row in rows:
        key = _group_key(row)
        if key != current_key:
            if current:
                if current_key in seen:
                    raise ValueError("episode-player group is not contiguous")
                seen.add(current_key)  # type: ignore[arg-type]
                yield current
            current_key = key
            current = []
        current.append(row)
    if current:
        if current_key in seen:
            raise ValueError("episode-player group is not contiguous")
        yield current


def _worker_initialize(card_data_path: str, ontology_sha256: str) -> None:
    global _WORKER_REGISTRY
    torch.set_num_threads(1)
    _WORKER_REGISTRY = CardSemanticRegistry.from_official_csv(card_data_path)
    if _WORKER_REGISTRY.sha256 != ontology_sha256:
        raise ValueError("worker ontology SHA-256 mismatch")


def _compile_group_local(
    rows: list[Mapping[str, Any]], registry: CardSemanticRegistry
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for prepared in iter_prepare_record_stream(
        rows, registry=registry, include_digest=False
    ):
        output.append({
            "split": prepared["split"],
            "ordered_action": prepared["ordered_action"],
            "action_termination": prepared["action_termination"],
            "_compiled_features": prepared["_compiled_features"],
        })
    return output


def _compile_group(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if _WORKER_REGISTRY is None:
        raise RuntimeError("materialization worker is not initialized")
    return _compile_group_local(rows, _WORKER_REGISTRY)


def _parallel_groups(
    groups: Iterable[list[Mapping[str, Any]]],
    *,
    workers: int,
    card_data_path: Path,
    ontology_sha256: str,
) -> Iterator[list[dict[str, Any]]]:
    pending: deque[Future[list[dict[str, Any]]]] = deque()
    values = iter(groups)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_initialize,
        initargs=(str(card_data_path), ontology_sha256),
    ) as executor:
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(_compile_group, next(values)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(_compile_group, next(values)))
            except StopIteration:
                pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sparse_relations(dense: Tensor) -> tuple[Tensor, Tensor]:
    per_row: list[Tensor] = []
    max_edges = 1
    for row in dense:
        endpoints = row.nonzero(as_tuple=False)
        if endpoints.numel():
            kinds = row[endpoints[:, 0], endpoints[:, 1]].unsqueeze(1)
            edges = torch.cat((endpoints, kinds), dim=1)
        else:
            edges = torch.zeros(0, 3, dtype=torch.long)
        per_row.append(edges)
        max_edges = max(max_edges, edges.size(0))
    output = torch.zeros(dense.size(0), max_edges, 3, dtype=torch.int16)
    mask = torch.zeros(dense.size(0), max_edges, dtype=torch.bool)
    for index, edges in enumerate(per_row):
        if edges.numel():
            output[index, : edges.size(0)] = edges.to(torch.int16)
            mask[index, : edges.size(0)] = True
    return output, mask


def _compact_batch(records: list[Mapping[str, Any]]) -> dict[str, Tensor]:
    batch = collate_records(records)
    relation_edges, relation_mask = _sparse_relations(batch.pop("relations"))
    batch.pop("option_permutation_old_to_new")
    batch["relation_edges"] = relation_edges
    batch["relation_mask"] = relation_mask
    compact: dict[str, Tensor] = {}
    for name, value in batch.items():
        if name in _INTEGER_FIELDS:
            if value.numel() and (int(value.min()) < -1 or int(value.max()) > 32767):
                raise ValueError(f"integer cache field outside int16 range: {name}")
            compact[name] = value.to(torch.int16).contiguous()
        elif name in _FLOAT_FIELDS:
            compact[name] = value.to(torch.float16).contiguous()
        elif name in _BOOL_FIELDS:
            compact[name] = value.to(torch.bool).contiguous()
        else:
            raise ValueError(f"undeclared cache field: {name}")
    if set(compact) != CACHE_FIELDS:
        raise ValueError("materialized cache fields do not match schema")
    return compact


def _write_shard(
    staging: Path,
    split: str,
    ordinal: int,
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    name = f"{split}-{ordinal:05d}.pt"
    partial = staging / f"{name}.partial"
    final = staging / name
    tensors = _compact_batch(records)
    torch.save(tensors, partial)
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    partial.replace(final)
    return {
        "path": name,
        "sha256": _sha256(final),
        "bytes": final.stat().st_size,
        "count": len(records),
        "shapes": {key: list(value.shape[1:]) for key, value in sorted(tensors.items())},
    }


def build_materialized_dataset(
    raw_dataset: Path | str,
    output: Path | str,
    *,
    registry: CardSemanticRegistry,
    shard_records: int = 1024,
    source_workers: int = 8,
) -> dict[str, Any]:
    raw_root = Path(raw_dataset)
    target = Path(output)
    compiler_commitment = compiler_sha256()
    materializer_commitment = _sha256(Path(__file__))
    if target.exists():
        raise FileExistsError(target)
    if source_workers <= 0:
        raise ValueError("source workers must be positive")
    raw_reference = json.loads((raw_root / "dataset_reference.json").read_text(encoding="utf-8"))
    if raw_reference.get("schema_version") != "dataset_reference_v3":
        raise ValueError("materialization requires dataset_reference_v3")
    token = uuid.uuid4().hex
    staging = target.with_name(f".{target.name}.staging-{token}")
    staging.mkdir(parents=True)
    shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    buffers: dict[str, list[Mapping[str, Any]]] = {"train": [], "validation": []}
    counts = {"train": 0, "validation": 0}
    try:
        rows = (record.as_dict() for record in iter_dataset(raw_root))
        groups = _record_groups(rows)
        if source_workers == 1:
            compiled_groups: Iterable[list[dict[str, Any]]] = (
                _compile_group_local(group, registry) for group in groups
            )
        else:
            card_data_path = Path(__file__).parents[3] / "data" / "official" / "EN_Card_Data.csv"
            compiled_groups = _parallel_groups(
                groups,
                workers=source_workers,
                card_data_path=card_data_path,
                ontology_sha256=registry.sha256,
            )
        for prepared_group in compiled_groups:
            for prepared in prepared_group:
                split = prepared.get("split")
                if split not in buffers:
                    raise ValueError("materialized record has invalid split")
                buffers[split].append(prepared)
                counts[split] += 1
                if len(buffers[split]) == shard_records:
                    shards[split].append(_write_shard(
                        staging, split, len(shards[split]), buffers[split]
                    ))
                    buffers[split].clear()
        for split in ("train", "validation"):
            if buffers[split]:
                shards[split].append(_write_shard(
                    staging, split, len(shards[split]), buffers[split]
                ))
                buffers[split].clear()
        if counts != raw_reference["counts"]:
            raise ValueError("materialized counts disagree with raw dataset")
        if compiler_sha256() != compiler_commitment:
            raise RuntimeError("compiler sources changed during materialization")
        if _sha256(Path(__file__)) != materializer_commitment:
            raise RuntimeError("materializer source changed during materialization")
        reference: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "raw_dataset_path": str(raw_root),
            "raw_dataset_content_sha256": raw_reference["content_sha256"],
            "raw_record_schema_version": raw_reference["record_schema_version"],
            "action_contract_version": raw_reference["action_contract_version"],
            "typed_input_schema_version": raw_reference["typed_input_schema_version"],
            "feature_compiler_version": COMPILER_VERSION,
            "feature_compiler_sha256": compiler_commitment,
            "materializer_sha256": materializer_commitment,
            "ontology_sha256": registry.sha256,
            "counts": counts,
            "shard_records": shard_records,
            "source_workers": source_workers,
            "storage_dtypes": {
                "categorical_and_targets": "int16",
                "numeric_and_semantic": "float16",
                "masks": "bool",
                "relations": "sparse_int16_edges",
            },
            "fields": sorted(CACHE_FIELDS),
            "shards": shards,
        }
        reference["content_sha256"] = hashlib.sha256(_canonical(reference)).hexdigest()
        (staging / "feature_dataset_reference.json").write_bytes(_canonical(reference))
        for path in staging.iterdir():
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        staging.replace(target)
        return reference
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_shard(
    path: Path,
    expected: Mapping[str, Any],
    *,
    verify_commitment: bool = False,
) -> dict[str, Tensor]:
    if verify_commitment and (
        _sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]
    ):
        raise ValueError(f"materialized shard commitment mismatch: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict) or set(value) != CACHE_FIELDS:
        raise ValueError("invalid materialized shard fields")
    tensors = {str(key): item for key, item in value.items() if isinstance(item, Tensor)}
    count = expected["count"]
    if len(tensors) != len(CACHE_FIELDS) or any(item.size(0) != count for item in tensors.values()):
        raise ValueError("invalid materialized shard leading dimension")
    return tensors


def validate_materialized_dataset(
    root: Path | str,
    *,
    registry: CardSemanticRegistry,
) -> dict[str, Any]:
    base = Path(root)
    reference_path = base / "feature_dataset_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    commitment = reference.pop("content_sha256", None)
    if reference.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported materialized dataset schema")
    if hashlib.sha256(_canonical(reference)).hexdigest() != commitment:
        raise ValueError("materialized dataset reference commitment mismatch")
    reference["content_sha256"] = commitment
    if reference["feature_compiler_version"] != COMPILER_VERSION:
        raise ValueError("materialized compiler version mismatch")
    if reference["feature_compiler_sha256"] != compiler_sha256():
        raise ValueError("materialized compiler SHA-256 mismatch")
    if reference.get("materializer_sha256") != _sha256(Path(__file__)):
        raise ValueError("materializer SHA-256 mismatch")
    if reference["ontology_sha256"] != registry.sha256:
        raise ValueError("materialized ontology SHA-256 mismatch")
    declared = {"feature_dataset_reference.json"}
    observed_counts = {"train": 0, "validation": 0}
    for split in ("train", "validation"):
        for item in reference["shards"][split]:
            name = item["path"]
            if Path(name).name != name or not name.startswith(f"{split}-"):
                raise ValueError("unsafe materialized shard path")
            declared.add(name)
            _load_shard(base / name, item, verify_commitment=True)
            observed_counts[split] += item["count"]
    if observed_counts != reference["counts"]:
        raise ValueError("materialized aggregate count mismatch")
    if {path.name for path in base.iterdir()} != declared:
        raise ValueError("materialized dataset contains undeclared files")
    return reference


def _runtime_dtypes(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    output: dict[str, Tensor] = {}
    for name, value in batch.items():
        if name in _INTEGER_FIELDS:
            output[name] = value.to(torch.long)
        elif name in _FLOAT_FIELDS:
            output[name] = value.to(torch.float32)
        else:
            output[name] = value
    return output


def _dense_relations(batch: dict[str, Tensor]) -> None:
    edges = batch.pop("relation_edges")
    mask = batch.pop("relation_mask")
    entity_width = batch["entities_cat"].size(1)
    dense = torch.zeros(edges.size(0), entity_width, entity_width, dtype=torch.long)
    for row in range(edges.size(0)):
        selected = edges[row, mask[row]].long()
        if selected.numel():
            dense[row, selected[:, 0], selected[:, 1]] = selected[:, 2]
    batch["relations"] = dense


def _permute_options(batch: dict[str, Tensor], generator: torch.Generator) -> None:
    width = batch["option_mask"].size(1)
    audit = torch.full((batch["option_mask"].size(0), width), -1, dtype=torch.long)
    for row in range(batch["option_mask"].size(0)):
        count = int(batch["option_mask"][row].sum())
        new_to_old = torch.randperm(count, generator=generator)
        old_to_new = torch.empty(count, dtype=torch.long)
        old_to_new[new_to_old] = torch.arange(count)
        audit[row, :count] = old_to_new
        for name in ("options_cat", "options_num", "option_semantic", "option_mask"):
            batch[name][row, :count] = batch[name][row, :count].clone()[new_to_old]
        target_mask = batch["target_mask"][row]
        selected = target_mask & (batch["targets"][row] < count)
        batch["targets"][row, selected] = old_to_new[batch["targets"][row, selected]]
    batch["option_permutation_old_to_new"] = audit


def _prefetch(values: Iterable[dict[str, Tensor]], depth: int = 2) -> Iterator[dict[str, Tensor]]:
    sentinel = object()
    channel: queue.Queue[object] = queue.Queue(depth)

    def produce() -> None:
        try:
            for value in values:
                channel.put(value)
        except BaseException as error:
            channel.put(error)
        finally:
            channel.put(sentinel)

    thread = threading.Thread(target=produce, name="feature-shard-prefetch", daemon=True)
    thread.start()
    while True:
        value = channel.get()
        if value is sentinel:
            break
        if isinstance(value, BaseException):
            raise value
        yield value  # type: ignore[misc]


class MaterializedBatchSource:
    def __init__(
        self,
        root: Path | str,
        *,
        registry: CardSemanticRegistry,
        batch_size: int,
        seed: int,
        pin_memory: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.root = Path(root)
        self.registry = registry
        self.batch_size = batch_size
        self.seed = seed
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.reference = validate_materialized_dataset(self.root, registry=registry)

    def batch_count(self, split: str) -> int:
        if split not in {"train", "validation"}:
            raise ValueError("invalid split")
        return sum(
            (entry["count"] + self.batch_size - 1) // self.batch_size
            for entry in self.reference["shards"][split]
        )

    def batches(self, split: str, *, epoch: int, training: bool) -> Iterator[dict[str, Tensor]]:
        if split not in {"train", "validation"}:
            raise ValueError("invalid split")
        entries = list(self.reference["shards"][split])
        if training:
            random.Random(self.seed + epoch * 1009).shuffle(entries)
        generator = torch.Generator().manual_seed(self.seed + epoch * 1009 + 17)

        def values() -> Iterator[dict[str, Tensor]]:
            for entry in entries:
                shard = _load_shard(self.root / entry["path"], entry)
                count = entry["count"]
                order = torch.randperm(count, generator=generator) if training else torch.arange(count)
                for start in range(0, count, self.batch_size):
                    rows = order[start : start + self.batch_size]
                    batch = _runtime_dtypes({
                        name: value.index_select(0, rows).clone() for name, value in shard.items()
                    })
                    _dense_relations(batch)
                    if training:
                        _permute_options(batch, generator)
                    else:
                        width = batch["option_mask"].size(1)
                        count_rows = batch["option_mask"].sum(1)
                        audit = torch.full((batch["option_mask"].size(0), width), -1, dtype=torch.long)
                        for row, option_count in enumerate(count_rows.tolist()):
                            audit[row, :option_count] = torch.arange(option_count)
                        batch["option_permutation_old_to_new"] = audit
                    if self.pin_memory:
                        batch = {name: value.pin_memory() for name, value in batch.items()}
                    yield batch

        yield from _prefetch(values())


__all__ = [
    "MaterializedBatchSource", "build_materialized_dataset", "validate_materialized_dataset",
]
