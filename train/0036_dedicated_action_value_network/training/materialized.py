"""Immutable, mmap-backed tensor shards for 0036 Value training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import queue
import random
import shutil
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .dataset import ValueBatch, ValueDataset


SCHEMA_VERSION = "0036_materialized_value_dataset_v1"
LABEL_FIELDS = frozenset({
    "value_target",
    "archetype_target",
    "final_diff_target",
    "episode_weight",
    "is_exact_007",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _flatten(batch: ValueBatch) -> dict[str, Tensor]:
    values = dict(batch.features)
    values.update({
        "value_target": batch.value_target,
        "archetype_target": batch.archetype_target,
        "final_diff_target": batch.final_diff_target,
        "episode_weight": batch.episode_weight,
        "is_exact_007": batch.is_exact_007,
    })
    return values


def _compact_value_batch(batch: ValueBatch) -> dict[str, Tensor]:
    """Use compact storage dtypes while preserving supervision labels exactly."""
    compact: dict[str, Tensor] = {}
    for name, value in _flatten(batch).items():
        if value.dtype is torch.long:
            if value.numel() and (int(value.min()) < -32768 or int(value.max()) > 32767):
                raise ValueError(f"integer cache field outside int16 range: {name}")
            compact[name] = value.to(torch.int16).contiguous()
        elif value.dtype is torch.float32 and name not in {"value_target", "episode_weight"}:
            compact[name] = value.to(torch.float16).contiguous()
        elif value.dtype in {torch.float32, torch.bool}:
            compact[name] = value.contiguous()
        else:
            raise ValueError(f"unsupported cache dtype for {name}: {value.dtype}")
    return compact


def _runtime_value_batch(values: Mapping[str, Tensor]) -> ValueBatch:
    runtime: dict[str, Tensor] = {}
    for name, value in values.items():
        if value.dtype is torch.int16:
            runtime[name] = value.to(torch.long)
        elif value.dtype is torch.float16:
            runtime[name] = value.to(torch.float32)
        else:
            runtime[name] = value
    features = {name: value for name, value in runtime.items() if name not in LABEL_FIELDS}
    return ValueBatch(
        features=features,
        value_target=runtime["value_target"],
        archetype_target=runtime["archetype_target"],
        final_diff_target=runtime["final_diff_target"],
        episode_weight=runtime["episode_weight"],
        is_exact_007=runtime["is_exact_007"],
    )


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import orjson
    except ImportError:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    with gzip.open(path, "rb") as handle:
        return [orjson.loads(line) for line in handle]


def _write_shard(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    partial = path.with_suffix(path.suffix + ".partial")
    tensors = _compact_value_batch(ValueDataset._collate(rows))
    torch.save(tensors, partial)
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    partial.replace(path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "count": len(rows),
        "shapes": {name: list(value.shape[1:]) for name, value in sorted(tensors.items())},
    }


def _materialize_source_shard(arguments: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    source_path, output_path, source_entry = arguments
    torch.set_num_threads(1)
    rows = _load_json_rows(Path(source_path))
    if len(rows) != source_entry["count"]:
        raise ValueError(f"source shard count mismatch: {source_entry['path']}")
    entry = _write_shard(Path(output_path), rows)
    entry["source_path"] = source_entry["path"]
    entry["source_sha256"] = source_entry["sha256"]
    return entry


def build_materialized_dataset(
    source_root: Path | str,
    output_root: Path | str,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Convert the verified gzip/JSON source once; never overwrite an existing cache."""
    source = Path(source_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(output)
    if workers < 1:
        raise ValueError("0036 materialization workers must be positive")
    source_dataset = ValueDataset(source, verify_hashes=True)
    source_manifest_path = source / "manifest.json"
    source_manifest_sha256 = _sha256(source_manifest_path)
    materializer_sha256 = _sha256(Path(__file__))
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for split in ("train", "validation"):
                source_entries = source_dataset.manifest["shards"][split]
                arguments = [
                    (
                        str(source / entry["path"]),
                        str(staging / f"{split}-{ordinal:05d}.pt"),
                        entry,
                    )
                    for ordinal, entry in enumerate(source_entries)
                ]
                for ordinal, entry in enumerate(
                    executor.map(_materialize_source_shard, arguments, chunksize=1)
                ):
                    shards[split].append(entry)
                    print({
                        "event": "0036_materialize_tensor_shard",
                        "split": split,
                        "shard": ordinal + 1,
                        "shards": len(source_entries),
                        "count": entry["count"],
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }, flush=True)
        counts = {
            split: sum(entry["count"] for entry in entries)
            for split, entries in shards.items()
        }
        if counts != source_dataset.manifest["split_decision_counts"]:
            raise ValueError("materialized counts disagree with source manifest")
        if _sha256(source_manifest_path) != source_manifest_sha256:
            raise RuntimeError("source manifest changed during materialization")
        if _sha256(Path(__file__)) != materializer_sha256:
            raise RuntimeError("materializer changed during materialization")
        fields = sorted(set(shards["train"][0]["shapes"]) | set(shards["validation"][0]["shapes"]))
        reference: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "source_manifest_sha256": source_manifest_sha256,
            "source_dataset_schema_version": source_dataset.manifest["schema_version"],
            "source_catalog_sha256": source_dataset.manifest["catalog_sha256"],
            "source_actor_schema_version": source_dataset.manifest["actor_schema_version"],
            "source_checkpoint_required_sha256": source_dataset.manifest["source_checkpoint_required_sha256"],
            "materializer_sha256": materializer_sha256,
            "materialization_workers": workers,
            "counts": counts,
            "fields": fields,
            "storage_dtypes": {
                "actor_integer_and_class_labels": "int16",
                "actor_numeric": "float16",
                "value_and_episode_weight_labels": "float32",
                "masks": "bool",
            },
            "build_seconds": time.perf_counter() - started,
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
        staging.replace(output)
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
        path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]
    ):
        raise ValueError(f"materialized shard commitment mismatch: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict) or not value or not all(isinstance(item, Tensor) for item in value.values()):
        raise ValueError(f"invalid materialized shard payload: {path}")
    count = int(expected["count"])
    if any(item.ndim < 1 or item.size(0) != count for item in value.values()):
        raise ValueError(f"invalid materialized shard leading dimension: {path}")
    return value


def validate_materialized_dataset(
    root: Path | str,
    *,
    source_root: Path | str | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    base = Path(root)
    reference_path = base / "feature_dataset_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    commitment = reference.pop("content_sha256", None)
    if reference.get("schema_version") != SCHEMA_VERSION or reference.get("status") != "complete":
        raise ValueError("unsupported or incomplete 0036 materialized dataset")
    if hashlib.sha256(_canonical(reference)).hexdigest() != commitment:
        raise ValueError("0036 materialized reference commitment mismatch")
    reference["content_sha256"] = commitment
    if reference.get("materializer_sha256") != _sha256(Path(__file__)):
        raise ValueError("0036 materializer source commitment mismatch")
    if source_root is not None:
        source_manifest = Path(source_root) / "manifest.json"
        if _sha256(source_manifest) != reference["source_manifest_sha256"]:
            raise ValueError("0036 source dataset commitment mismatch")
    declared = {"feature_dataset_reference.json"}
    counts = {"train": 0, "validation": 0}
    expected_fields = set(reference["fields"])
    for split in ("train", "validation"):
        for entry in reference["shards"][split]:
            name = entry["path"]
            if Path(name).name != name or not name.startswith(f"{split}-"):
                raise ValueError(f"unsafe materialized shard path: {name}")
            declared.add(name)
            values = _load_shard(base / name, entry, verify_commitment=verify_hashes)
            if set(values) != expected_fields:
                raise ValueError(f"materialized shard fields mismatch: {name}")
            counts[split] += entry["count"]
    if counts != reference["counts"]:
        raise ValueError("0036 materialized aggregate count mismatch")
    if {path.name for path in base.iterdir()} != declared:
        raise ValueError("0036 materialized dataset contains undeclared files")
    return reference


def _prefetch(values: Iterable[ValueBatch], depth: int) -> Iterator[ValueBatch]:
    if depth < 1:
        yield from values
        return
    sentinel = object()
    channel: queue.Queue[object] = queue.Queue(maxsize=depth)

    def produce() -> None:
        try:
            for value in values:
                channel.put(value)
        except BaseException as error:
            channel.put(error)
        finally:
            channel.put(sentinel)

    thread = threading.Thread(target=produce, name="0036-value-batch-prefetch", daemon=True)
    thread.start()
    while True:
        value = channel.get()
        if value is sentinel:
            return
        if isinstance(value, BaseException):
            raise value
        yield value  # type: ignore[misc]


class MaterializedValueDataset:
    def __init__(
        self,
        root: Path,
        *,
        source_root: Path | None = None,
        verify_hashes: bool = True,
        pin_memory: bool = True,
        prefetch_depth: int = 2,
    ) -> None:
        if prefetch_depth < 0:
            raise ValueError("0036 prefetch depth cannot be negative")
        self.root = root
        self.reference = validate_materialized_dataset(
            root, source_root=source_root, verify_hashes=verify_hashes
        )
        self.manifest = {"catalog_sha256": self.reference["source_catalog_sha256"]}
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.prefetch_depth = prefetch_depth

    def batch_count(self, split: str, batch_size: int) -> int:
        if split not in {"train", "validation"} or batch_size < 1:
            raise ValueError("0036 split/batch size is invalid")
        return sum(
            (entry["count"] + batch_size - 1) // batch_size
            for entry in self.reference["shards"][split]
        )

    def batches(
        self,
        split: str,
        batch_size: int,
        *,
        epoch: int = 0,
        shuffle: bool = False,
        seed: int = 20260808,
    ) -> Iterator[ValueBatch]:
        if split not in {"train", "validation"} or batch_size < 1:
            raise ValueError("0036 split/batch size is invalid")
        entries = list(self.reference["shards"][split])
        rng = random.Random(seed + epoch)
        if shuffle:
            rng.shuffle(entries)
        generator = torch.Generator().manual_seed(seed + epoch * 1009 + 17)

        def values() -> Iterator[ValueBatch]:
            for entry in entries:
                shard = _load_shard(self.root / entry["path"], entry)
                count = int(entry["count"])
                order = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
                for start in range(0, count, batch_size):
                    rows = order[start : start + batch_size]
                    selected = {
                        name: value.index_select(0, rows)
                        for name, value in shard.items()
                    }
                    batch = _runtime_value_batch(selected)
                    if self.pin_memory:
                        batch = ValueBatch(
                            {name: value.pin_memory() for name, value in batch.features.items()},
                            batch.value_target.pin_memory(),
                            batch.archetype_target.pin_memory(),
                            batch.final_diff_target.pin_memory(),
                            batch.episode_weight.pin_memory(),
                            batch.is_exact_007.pin_memory(),
                        )
                    yield batch

        yield from _prefetch(values(), self.prefetch_depth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = build_materialized_dataset(args.source, args.output, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "MaterializedValueDataset",
    "build_materialized_dataset",
    "validate_materialized_dataset",
]
