"""Compile one shared R15 feature cache with auditable 0015 row metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch

_m = importlib.import_module(
    "train.0014_faithful_board_causal_features.training.materialized"
)
_semantics = importlib.import_module("train.0014_faithful_board_causal_features.card_semantics")

BUILD_IDS = {
    "dragapult_dusknoir": 1,
    "pure_dragapult": 2,
    "starmie_dusknoir": 3,
    "marnie_munkidori": 4,
}
OUTCOME_IDS = {"loss": -1, "draw": 0, "win": 1}
METADATA_TENSORS = {
    "source_id",
    "build_id",
    "outcome_id",
    "episode_id",
    "player_index",
    "first_player",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_raw(root: Path, reference: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for split in ("train", "validation"):
        for item in reference["shards"][split]:
            path = root / item["path"]
            if _sha256(path) != item["sha256"]:
                raise ValueError(f"raw shard hash mismatch: {path}")
            count = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["split"] != split:
                        raise ValueError(f"raw split mismatch: {path}")
                    yield row
                    count += 1
            if count != item["count"]:
                raise ValueError(f"raw shard count mismatch: {path}")


def _compact(compiled: list[dict[str, Any]], raw: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if len(compiled) != len(raw):
        raise ValueError("compiled/raw group length mismatch")
    tensors = _m._compact_records(compiled)
    tensors.update(
        {
            "source_id": torch.tensor([row["source_id"] for row in raw], dtype=torch.int16),
            "build_id": torch.tensor([BUILD_IDS[row["build"]] for row in raw], dtype=torch.int8),
            "outcome_id": torch.tensor(
                [OUTCOME_IDS[row["terminal_outcome"]] for row in raw], dtype=torch.int8
            ),
            "episode_id": torch.tensor(
                [row["identity"]["episode_id"] for row in raw], dtype=torch.int64
            ),
            "player_index": torch.tensor(
                [row["identity"]["player_index"] for row in raw], dtype=torch.int8
            ),
            "first_player": torch.tensor([row["first_player"] for row in raw], dtype=torch.int8),
        }
    )
    return {name: value.contiguous() for name, value in tensors.items()}


def _write_shard(
    stage: Path,
    split: str,
    ordinal: int,
    compiled: list[dict[str, Any]],
    raw: list[dict[str, Any]],
) -> dict[str, Any]:
    path = stage / f"{split}-{ordinal:05d}.pt"
    partial = path.with_suffix(".pt.partial")
    tensors = _compact(compiled, raw)
    torch.save(tensors, partial)
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    partial.replace(path)
    eligible = tensors["a0_eligible"]
    build_counts = Counter(tensors["build_id"][eligible].tolist())
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "count": len(raw),
        "r15_eligible": int(eligible.sum()),
        "eligible_by_build_id": {str(key): value for key, value in sorted(build_counts.items())},
        "shapes": {name: list(value.shape[1:]) for name, value in sorted(tensors.items())},
    }


def build_materialized(raw_root: Path, output: Path, shard_size: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    raw_reference_path = raw_root / "dataset_reference.json"
    raw_reference = json.loads(raw_reference_path.read_text(encoding="utf-8"))
    registry = _semantics.CardSemanticRegistry.from_official_csv(
        Path("data/official/EN_Card_Data.csv")
    )
    stage = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    buffers: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {
        "train": ([], []),
        "validation": ([], []),
    }
    decision_counts = Counter()

    def flush(split: str) -> None:
        raw_buffer, compiled_buffer = buffers[split]
        if not raw_buffer:
            return
        shards[split].append(
            _write_shard(
                stage,
                split,
                len(shards[split]),
                compiled_buffer,
                raw_buffer,
            )
        )
        raw_buffer.clear()
        compiled_buffer.clear()

    try:
        groups = _m._record_groups(_iter_raw(raw_root, raw_reference))
        for raw_group in groups:
            split = raw_group[0]["split"]
            compiled_group = _m._compile_group_local(raw_group, registry)
            raw_buffer, compiled_buffer = buffers[split]
            if raw_buffer and len(raw_buffer) + len(raw_group) > shard_size:
                flush(split)
            raw_buffer.extend(raw_group)
            compiled_buffer.extend(compiled_group)
            for row, compiled in zip(raw_group, compiled_group, strict=True):
                decision_counts[(split, row["build"], int(compiled["a0_eligible"]))] += 1
        for split in ("train", "validation"):
            flush(split)
        ontology_path = stage / "card_ontology.json"
        ontology_path.write_text(
            json.dumps(_m._ontology_payload(registry), sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        reference = {
            "schema_version": "0015_r15_feature_cache_v1",
            "raw_dataset": str(raw_root),
            "raw_reference_sha256": _sha256(raw_reference_path),
            "raw_content_sha256": raw_reference["content_sha256"],
            "source_vocabulary": raw_reference["source_vocabulary"],
            "build_ids": BUILD_IDS,
            "outcome_ids": OUTCOME_IDS,
            "compiler_version": _m.COMPILER_VERSION,
            "compiler_sha256": _m.compiler_sha256(),
            "materializer_sha256": _sha256(Path(__file__)),
            "ontology_sha256": registry.sha256,
            "ontology_file_sha256": _sha256(ontology_path),
            "metadata_tensors": sorted(METADATA_TENSORS),
            "shards": shards,
            "decision_counts": [
                {
                    "split": key[0],
                    "build": key[1],
                    "r15_eligible": bool(key[2]),
                    "decisions": value,
                }
                for key, value in sorted(decision_counts.items())
            ],
        }
        reference["content_sha256"] = hashlib.sha256(
            (json.dumps(reference, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        (stage / "feature_dataset_reference.json").write_text(
            json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output)
        return reference
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=10_000)
    args = parser.parse_args()
    result = build_materialized(args.raw_dataset, args.output, args.shard_size)
    print(json.dumps(result["decision_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
