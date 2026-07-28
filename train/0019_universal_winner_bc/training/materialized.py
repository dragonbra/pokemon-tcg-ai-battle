"""Build and validate immutable 0019 full-feature shards."""
from __future__ import annotations

import hashlib
import gzip
import json
import os
import random
import shutil
import uuid
import argparse
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..features.compiler import COMPILER_VERSION, compile_row, compiler_sha256
from ..knowledge.state import CausalKnowledge
from ..card_semantics import CardSemanticRegistry
from ..storage import (
    DEFAULT_C_FREE_FLOOR_BYTES,
    DEFAULT_DATASET_LIMIT_BYTES,
    DEFAULT_LINUX_FREE_FLOOR_BYTES,
    guard_storage,
)

SCHEMA_VERSION = "faithful_board_causal_cache_v1"
TERMINATION_CODES = {"forced_max": 1, "optional_stop": 2}
SEMANTICALLY_EQUIVALENT_COMPILER_SHA256S: frozenset[str] = frozenset()
V1_COMPATIBLE_MATERIALIZER_SHA256S: frozenset[str] = frozenset()

LEGACY_FIELDS = frozenset(
    {
        "legacy_global_cat",
        "legacy_global_num",
        "legacy_entity_cat",
        "legacy_entity_num",
        "legacy_entity_mask",
        "legacy_option_cat",
        "legacy_option_mask",
        "ordered_action",
        "action_mask",
        "min_count",
        "max_count",
        "a0_eligible",
    }
)
AUXILIARY_FIELDS = frozenset(
    {
        "zone_inventory_num",
        "registered_card_ids",
        "registered_multiplicity",
        "registered_mask",
        "ledger_cat",
        "ledger_num",
        "ledger_mask",
        "event_cat",
        "event_num",
        "event_mask",
        "known_opponent_hand_card_ids",
        "known_opponent_hand_mask",
        "unknown_opponent_hand_count",
        "source_id",
    }
)
_INT32_FIELDS = frozenset({"source_id"})
METADATA_FIELDS = frozenset({"action_termination", "source_payload_sha256"})
CACHE_FIELDS = LEGACY_FIELDS | AUXILIARY_FIELDS | METADATA_FIELDS

_INT16_FIELDS = frozenset(
    {
        "legacy_global_cat",
        "legacy_entity_cat",
        "legacy_option_cat",
        "ordered_action",
        "min_count",
        "max_count",
        "registered_card_ids",
        "registered_multiplicity",
        "ledger_cat",
        "event_cat",
        "known_opponent_hand_card_ids",
        "unknown_opponent_hand_count",
    }
)
_FP32_FIELDS = frozenset({"legacy_global_num", "legacy_entity_num"})
_FP16_FIELDS = frozenset({"zone_inventory_num", "ledger_num", "event_num"})
_BOOL_FIELDS = frozenset(
    {
        "legacy_entity_mask",
        "legacy_option_mask",
        "action_mask",
        "a0_eligible",
        "registered_mask",
        "ledger_mask",
        "event_mask",
        "known_opponent_hand_mask",
    }
)

_WORKER_REGISTRY: Any = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _group_key(row: Mapping[str, Any]) -> tuple[object, object, object]:
    identity = row["identity"]
    return identity["date"], identity["episode_id"], identity["player_index"]


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


def _iter_raw_rows(
    root: Path, reference: Mapping[str, Any]
) -> Iterator[Mapping[str, Any]]:
    for split in ("train", "validation"):
        for expected in reference["shards"][split]:
            path = root / expected["path"]
            if _sha256(path) != expected["sha256"]:
                raise ValueError(f"raw shard commitment mismatch: {path}")
            count = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("split") != split:
                        raise ValueError(f"raw shard split mismatch: {path}")
                    yield row
                    count += 1
            if count != expected["count"]:
                raise ValueError(f"raw shard count mismatch: {path}")


def _registered_deck(row: Mapping[str, Any]) -> list[int]:
    result: list[int] = []
    for card_id, count in row["deck_manifest"]["counts"]:
        result.extend([int(card_id)] * int(count))
    if len(result) != 60:
        raise ValueError("registered deck is not 60 cards")
    return result


def _compile_group_local(rows: list[Mapping[str, Any]], registry: Any) -> list[dict[str, Any]]:
    first = rows[0]
    actor = int(first["identity"]["player_index"])
    knowledge = CausalKnowledge(actor, _registered_deck(first))
    output: list[dict[str, Any]] = []
    for row in rows:
        if int(row["identity"]["player_index"]) != actor:
            raise ValueError("mixed actors inside chronological group")
        snapshot = knowledge.consume(row["actor_observation"], row["event_cursor"])
        compiled = compile_row(row, snapshot, registry)
        compiled["split"] = row["split"]
        compiled["source_payload_sha256"] = row["source_payload_sha256"]
        output.append(compiled)
    return output


def _worker_initialize(card_data_path: str, ontology_sha256: str) -> None:
    global _WORKER_REGISTRY
    torch.set_num_threads(1)
    _WORKER_REGISTRY = CardSemanticRegistry.from_official_csv(card_data_path)
    if _WORKER_REGISTRY.sha256 != ontology_sha256:
        raise ValueError("worker ontology SHA-256 mismatch")


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


def _pad_2d(
    records: Sequence[Mapping[str, Any]], name: str, width: int, *, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    maximum = max(1, max(len(record[name]) for record in records))
    values = torch.zeros((len(records), maximum, width), dtype=dtype)
    mask = torch.zeros((len(records), maximum), dtype=torch.bool)
    for index, record in enumerate(records):
        rows = record[name]
        if rows:
            values[index, : len(rows)] = torch.tensor(rows, dtype=dtype)
            mask[index, : len(rows)] = True
    return values, mask


def _pad_1d(
    records: Sequence[Mapping[str, Any]], name: str, *, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    maximum = max(1, max(len(record[name]) for record in records))
    values = torch.zeros((len(records), maximum), dtype=dtype)
    mask = torch.zeros((len(records), maximum), dtype=torch.bool)
    for index, record in enumerate(records):
        items = record[name]
        if items:
            values[index, : len(items)] = torch.tensor(items, dtype=dtype)
            mask[index, : len(items)] = True
    return values, mask


def _compact_records(records: list[Mapping[str, Any]]) -> dict[str, Tensor]:
    entity_cat, entity_mask = _pad_2d(records, "entity_cat", 7, dtype=torch.int16)
    entity_num, entity_num_mask = _pad_2d(records, "entity_num", 5, dtype=torch.float32)
    option_cat, option_mask = _pad_2d(records, "option_cat", 12, dtype=torch.int16)
    ledger_cat, ledger_mask = _pad_2d(records, "ledger_cat", 4, dtype=torch.int16)
    ledger_num, ledger_num_mask = _pad_2d(records, "ledger_num", 15, dtype=torch.float16)
    event_cat, event_mask = _pad_2d(records, "event_cat", 8, dtype=torch.int16)
    event_num, event_num_mask = _pad_2d(records, "event_num", 4, dtype=torch.float16)
    action, action_mask = _pad_1d(records, "action", dtype=torch.int16)
    registered_ids, registered_mask = _pad_1d(
        records, "registered_card_ids", dtype=torch.int16
    )
    registered_multiplicity, registered_multiplicity_mask = _pad_1d(
        records, "registered_multiplicity", dtype=torch.int16
    )
    known_hand, known_hand_mask = _pad_1d(
        records, "known_opponent_hand_card_ids", dtype=torch.int16
    )
    if not (
        torch.equal(entity_mask, entity_num_mask)
        and torch.equal(ledger_mask, ledger_num_mask)
        and torch.equal(event_mask, event_num_mask)
        and torch.equal(registered_mask, registered_multiplicity_mask)
    ):
        raise ValueError("paired cache channel lengths disagree")
    tensors: dict[str, Tensor] = {
        "legacy_global_cat": torch.tensor(
            [record["global_cat"] for record in records], dtype=torch.int16
        ),
        "legacy_global_num": torch.tensor(
            [record["global_num"] for record in records], dtype=torch.float32
        ),
        "legacy_entity_cat": entity_cat,
        "legacy_entity_num": entity_num,
        "legacy_entity_mask": entity_mask,
        "legacy_option_cat": option_cat,
        "legacy_option_mask": option_mask,
        "ordered_action": action,
        "action_mask": action_mask,
        "min_count": torch.tensor([record["min_count"] for record in records], dtype=torch.int16),
        "max_count": torch.tensor([record["max_count"] for record in records], dtype=torch.int16),
        "a0_eligible": torch.tensor(
            [record["a0_eligible"] for record in records], dtype=torch.bool
        ),
        "action_termination": torch.tensor(
            [TERMINATION_CODES[record["action_termination"]] for record in records],
            dtype=torch.uint8,
        ),
        "zone_inventory_num": torch.tensor(
            [record["zone_inventory_num"] for record in records], dtype=torch.float16
        ),
        "registered_card_ids": registered_ids,
        "registered_multiplicity": registered_multiplicity,
        "registered_mask": registered_mask,
        "ledger_cat": ledger_cat,
        "ledger_num": ledger_num,
        "ledger_mask": ledger_mask,
        "event_cat": event_cat,
        "event_num": event_num,
        "event_mask": event_mask,
        "known_opponent_hand_card_ids": known_hand,
        "known_opponent_hand_mask": known_hand_mask,
        "unknown_opponent_hand_count": torch.tensor(
            [record["unknown_opponent_hand_count"] for record in records], dtype=torch.int16
        ),
        "source_id": torch.tensor(
            [record["source_id"] for record in records], dtype=torch.int32
        ),
        "source_payload_sha256": torch.tensor(
            [list(bytes.fromhex(record["source_payload_sha256"])) for record in records],
            dtype=torch.uint8,
        ),
    }
    if set(tensors) != CACHE_FIELDS:
        raise ValueError("materialized fields do not match declared schema")
    return {name: value.contiguous() for name, value in tensors.items()}


def _write_shard(
    staging: Path,
    split: str,
    ordinal: int,
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    name = f"{split}-{ordinal:05d}.pt"
    partial = staging / f"{name}.partial"
    final = staging / name
    tensors = _compact_records(records)
    torch.save(tensors, partial)
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    partial.replace(final)
    return {
        "path": name,
        "sha256": _sha256(final),
        "bytes": final.stat().st_size,
        "count": len(records),
        "a0_eligible": int(tensors["a0_eligible"].sum()),
        "shapes": {name: list(value.shape[1:]) for name, value in sorted(tensors.items())},
    }


def _ontology_payload(registry: Any) -> dict[str, Any]:
    cards = []
    for card_id in range(1, 2049):
        card = registry.lookup(card_id)
        if getattr(card.identity_state, "value", None) == "observed":
            cards.append(registry._as_dict(card))
    return {
        "schema_version": "0014_card_ontology_v1",
        "ontology_sha256": registry.sha256,
        "cards": cards,
    }


def build_materialized_dataset(
    raw_dataset: Path | str,
    output: Path | str,
    *,
    registry: Any,
    shard_records: int = 1024,
    source_workers: int = 1,
    dataset_limit_bytes: int = DEFAULT_DATASET_LIMIT_BYTES,
    linux_free_floor_bytes: int = DEFAULT_LINUX_FREE_FLOOR_BYTES,
    c_free_floor_bytes: int = DEFAULT_C_FREE_FLOOR_BYTES,
) -> dict[str, Any]:
    raw_root, target = Path(raw_dataset), Path(output)
    if target.exists():
        raise FileExistsError(target)
    if source_workers <= 0 or shard_records <= 0:
        raise ValueError("worker and shard sizes must be positive")
    raw_reference = json.loads((raw_root / "dataset_reference.json").read_text())
    compiler_commitment = compiler_sha256()
    materializer_commitment = _sha256(Path(__file__))
    staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    initial_storage = guard_storage(
        target.parent,
        dataset_limit_bytes=dataset_limit_bytes,
        linux_free_floor_bytes=linux_free_floor_bytes,
        c_free_floor_bytes=c_free_floor_bytes,
    )
    shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    buffers: dict[str, list[Mapping[str, Any]]] = {"train": [], "validation": []}
    counts = {"train": 0, "validation": 0}
    eligible = {"train": 0, "validation": 0}
    card_data_path = Path(__file__).parents[3] / "data" / "official" / "EN_Card_Data.csv"
    try:
        rows = _iter_raw_rows(raw_root, raw_reference)
        groups = _record_groups(rows)
        compiled_groups: Iterable[list[dict[str, Any]]]
        if source_workers == 1:
            compiled_groups = (_compile_group_local(group, registry) for group in groups)
        else:
            compiled_groups = _parallel_groups(
                groups,
                workers=source_workers,
                card_data_path=card_data_path,
                ontology_sha256=registry.sha256,
            )
        for group in compiled_groups:
            for record in group:
                split = record["split"]
                counts[split] += 1
                eligible[split] += int(record["a0_eligible"])
                buffers[split].append(record)
                if len(buffers[split]) == shard_records:
                    shards[split].append(
                        _write_shard(staging, split, len(shards[split]), buffers[split])
                    )
                    buffers[split].clear()
                    guard_storage(
                        target.parent,
                        dataset_limit_bytes=dataset_limit_bytes,
                        linux_free_floor_bytes=linux_free_floor_bytes,
                        c_free_floor_bytes=c_free_floor_bytes,
                    )
        for split in counts:
            if buffers[split]:
                shards[split].append(
                    _write_shard(staging, split, len(shards[split]), buffers[split])
                )
                buffers[split].clear()
                guard_storage(
                    target.parent,
                    dataset_limit_bytes=dataset_limit_bytes,
                    linux_free_floor_bytes=linux_free_floor_bytes,
                    c_free_floor_bytes=c_free_floor_bytes,
                )
        expected_counts = {
            split: sum(item["count"] for item in raw_reference["shards"][split])
            for split in counts
        }
        if counts != expected_counts:
            raise ValueError("model-ready counts disagree with raw dataset")
        ontology_path = staging / "card_ontology.json"
        ontology_path.write_bytes(_canonical(_ontology_payload(registry)))
        reference: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "raw_dataset_path": str(raw_root),
            "raw_dataset_content_sha256": raw_reference["content_sha256"],
            "raw_record_schema_version": "0019_universal_winner_decision_v1",
            "action_contract_version": "ordered_full_action_v1",
            "trajectory_index_sha256": raw_reference["trajectory_index_sha256"],
            "source_vocabulary_sha256": raw_reference["source_vocabulary_sha256"],
            "sources": raw_reference["sources"],
            "raw_card_action_diagnostics": raw_reference["card_action_diagnostics"],
            "raw_maximum_action_length": raw_reference["maximum_action_length"],
            "feature_compiler_version": COMPILER_VERSION,
            "feature_compiler_sha256": compiler_commitment,
            "materializer_sha256": materializer_commitment,
            "ontology_sha256": registry.sha256,
            "ontology_path": ontology_path.name,
            "ontology_file_sha256": _sha256(ontology_path),
            "counts": counts,
            "a0_eligible_counts": eligible,
            "shard_records": shard_records,
            "source_workers": source_workers,
            "storage_guard": {
                **initial_storage,
                "check_frequency": "before_write_and_after_each_model_ready_shard",
            },
            "fields": sorted(CACHE_FIELDS),
            "legacy_numeric_dtype": "float32",
            "auxiliary_numeric_dtype": "float16",
            "shards": shards,
        }
        reference["content_sha256"] = hashlib.sha256(_canonical(reference)).hexdigest()
        (staging / "feature_dataset_reference.json").write_bytes(_canonical(reference))
        if (
            compiler_sha256() != compiler_commitment
            or _sha256(Path(__file__)) != materializer_commitment
        ):
            raise RuntimeError("materialization sources changed during build")
        staging.replace(target)
        return reference
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_shard(path: Path, expected: Mapping[str, Any]) -> dict[str, Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict) or set(value) != CACHE_FIELDS:
        raise ValueError(f"invalid model-ready shard: {path}")
    tensors = {name: tensor for name, tensor in value.items() if isinstance(tensor, Tensor)}
    if len(tensors) != len(CACHE_FIELDS) or any(
        tensor.size(0) != expected["count"] for tensor in tensors.values()
    ):
        raise ValueError(f"invalid leading dimension: {path}")
    return tensors


def validate_materialized_dataset(root: Path | str, *, registry: Any) -> dict[str, Any]:
    base = Path(root)
    reference = json.loads((base / "feature_dataset_reference.json").read_text())
    commitment = reference.pop("content_sha256")
    if hashlib.sha256(_canonical(reference)).hexdigest() != commitment:
        raise ValueError("model-ready reference commitment mismatch")
    reference["content_sha256"] = commitment
    if reference["schema_version"] != SCHEMA_VERSION:
        raise ValueError("model-ready schema mismatch")
    if reference["feature_compiler_sha256"] not in {
        compiler_sha256(),
        *SEMANTICALLY_EQUIVALENT_COMPILER_SHA256S,
    }:
        raise ValueError("feature compiler commitment mismatch")
    if reference["materializer_sha256"] not in {
        _sha256(Path(__file__)),
        *V1_COMPATIBLE_MATERIALIZER_SHA256S,
    }:
        raise ValueError("materializer commitment mismatch")
    if reference["ontology_sha256"] != registry.sha256:
        raise ValueError("card ontology mismatch")
    ontology = base / reference["ontology_path"]
    if _sha256(ontology) != reference["ontology_file_sha256"]:
        raise ValueError("card ontology file commitment mismatch")
    counts = {"train": 0, "validation": 0}
    eligible = {"train": 0, "validation": 0}
    for split in counts:
        for expected in reference["shards"][split]:
            path = base / expected["path"]
            if _sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"shard commitment mismatch: {path}")
            tensors = _load_shard(path, expected)
            counts[split] += expected["count"]
            eligible[split] += int(tensors["a0_eligible"].sum())
    if counts != reference["counts"] or eligible != reference["a0_eligible_counts"]:
        raise ValueError("model-ready aggregate counts mismatch")
    return reference


def select_a0_batch(tensors: Mapping[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    """Copy only faithful fields, then dynamically trim and construct legacy STOP labels."""
    selected = {
        name: tensors[name].index_select(0, indices)
        for name in LEGACY_FIELDS
        if name != "a0_eligible"
    }
    entity_width = max(1, int(selected["legacy_entity_mask"].sum(1).max()))
    option_width = int(selected["legacy_option_mask"].sum(1).max())
    action_width = int(selected["action_mask"].sum(1).max())
    batch_size = indices.numel()
    targets = torch.full((batch_size, action_width + 1), -100, dtype=torch.long)
    actions = selected.pop("ordered_action")[:, :action_width].long()
    action_mask = selected.pop("action_mask")[:, :action_width]
    targets[:, :action_width][action_mask] = actions[action_mask]
    lengths = action_mask.sum(1).long()
    targets[torch.arange(batch_size), lengths] = option_width
    return {
        "global_cat": selected["legacy_global_cat"].long(),
        "global_num": selected["legacy_global_num"],
        "entity_cat": selected["legacy_entity_cat"][:, :entity_width].long(),
        "entity_num": selected["legacy_entity_num"][:, :entity_width],
        "entity_mask": selected["legacy_entity_mask"][:, :entity_width],
        "option_cat": selected["legacy_option_cat"][:, :option_width].long(),
        "option_mask": selected["legacy_option_mask"][:, :option_width],
        "targets": targets,
        "min_count": selected["min_count"].long(),
        "max_count": selected["max_count"].long(),
    }


def iter_a0_batches(
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
        tensors = _load_shard(base / expected["path"], expected)
        eligible = tensors["a0_eligible"].nonzero(as_tuple=False).flatten()
        if shuffle:
            permutation = torch.randperm(
                eligible.numel(),
                generator=torch.Generator().manual_seed(rng.randrange(2**63)),
            )
            eligible = eligible[permutation]
        for start in range(0, eligible.numel(), batch_size):
            yield select_a0_batch(tensors, eligible[start : start + batch_size])


__all__ = [
    "CACHE_FIELDS",
    "SCHEMA_VERSION",
    "build_materialized_dataset",
    "iter_a0_batches",
    "select_a0_batch",
    "validate_materialized_dataset",
]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--raw", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--shard-records", type=int, default=1024)
    validate = subparsers.add_parser("validate")
    validate.add_argument("dataset", type=Path)
    arguments = parser.parse_args()
    registry = CardSemanticRegistry.from_official_csv(
        Path(__file__).parents[3] / "data" / "official" / "EN_Card_Data.csv"
    )
    if arguments.command == "build":
        result = build_materialized_dataset(
            arguments.raw,
            arguments.output,
            registry=registry,
            shard_records=arguments.shard_records,
            source_workers=arguments.workers,
        )
    else:
        result = validate_materialized_dataset(arguments.dataset, registry=registry)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
