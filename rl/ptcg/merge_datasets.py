"""Merge compatible versioned BC/DAgger JSONL datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from .dataset import load_behavior_cloning_dataset


def merge(
    inputs: list[Path],
    output: Path,
    *,
    storage_path: Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, int | str | float]:
    storage = assert_storage_safe(storage_path, min_free_gib)
    records: list[dict] = []
    schemas: set[str] = set()
    for path in inputs:
        loaded = load_behavior_cloning_dataset(path)
        records.extend(loaded)
        schemas.update(str(record["feature_schema_version"]) for record in loaded)
    if len(schemas) != 1:
        raise ValueError(f"all datasets must use one feature schema: {sorted(schemas)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    return {
        "datasets": len(inputs),
        "records": len(records),
        "feature_schema_version": next(iter(schemas)),
        "output": str(output),
        "storage_path": storage.path,
        "storage_free_gib": round(storage.free_gib, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    print(
        json.dumps(
            merge(
                args.inputs,
                args.output,
                storage_path=args.storage_path,
                min_free_gib=args.min_free_gib,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
