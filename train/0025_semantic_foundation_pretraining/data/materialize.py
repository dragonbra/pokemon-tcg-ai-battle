"""Single-pass, bounded-memory materializer for 0019-contract raw decisions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..features.compiler import SCHEMA_VERSION, actor_payload, compile_row
from ..features.prototypes import PrototypeIndex
from ..knowledge.state import CausalKnowledge


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"


def _deck(row: Mapping[str, Any]) -> list[int]:
    output: list[int] = []
    for card_id, count in row["deck_manifest"]["counts"]:
        output.extend([int(card_id)] * int(count))
    if len(output) != 60:
        raise ValueError("registered deck is not exactly 60 cards")
    return output


def iter_rows(raw_root: Path, *, start_date: str, end_date: str) -> Iterator[Mapping[str, Any]]:
    reference = json.loads((raw_root / "dataset_reference.json").read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        for item in reference["shards"][split]:
            path = raw_root / item["path"]
            if _sha256(path) != item["sha256"]:
                raise ValueError(f"raw shard commitment mismatch: {path}")
            count = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    count += 1
                    date = row["identity"]["date"]
                    if start_date <= date <= end_date:
                        yield row
            if count != item["count"]:
                raise ValueError(f"raw shard row count mismatch: {path}")


class ShardWriter:
    def __init__(self, root: Path, records_per_shard: int):
        self.root = root
        self.records_per_shard = records_per_shard
        self.rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        self.shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}

    def add(self, split: str, row: dict[str, Any]) -> None:
        if split not in self.rows:
            raise ValueError(f"unknown dataset split: {split}")
        self.rows[split].append(row)
        if len(self.rows[split]) >= self.records_per_shard:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.rows[split]
        if not rows:
            return
        name = f"{split}-{len(self.shards[split]):05d}.jsonl.gz"
        final = self.root / name
        temporary = self.root / f".{name}.{uuid.uuid4().hex}.tmp"
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            for row in rows:
                handle.write(_canonical(row))
        os.replace(temporary, final)
        self.shards[split].append(
            {
                "path": name,
                "count": len(rows),
                "bytes": final.stat().st_size,
                "sha256": _sha256(final),
            }
        )
        rows.clear()

    def flush_all(self) -> None:
        for split in self.rows:
            self.flush(split)


def materialize(
    raw_root: Path,
    output: Path,
    prototype_path: Path,
    *,
    start_date: str,
    end_date: str,
    records_per_shard: int = 4096,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite materialized dataset: {output}")
    output.mkdir(parents=True)
    prototypes = PrototypeIndex.load(prototype_path)
    shutil.copyfile(prototype_path, output / "prototypes.json")
    full_engine_source = prototype_path.with_name("official_full_engine_prototypes_v1.json")
    shutil.copyfile(full_engine_source, output / "official_full_engine_prototypes_v1.json")
    writer = ShardWriter(output, records_per_shard)
    knowledge: CausalKnowledge | None = None
    group_key: tuple[Any, ...] | None = None
    count = 0
    try:
        for raw in iter_rows(raw_root, start_date=start_date, end_date=end_date):
            identity = raw["identity"]
            key = (identity["date"], identity["episode_id"], identity["player_index"])
            if key != group_key:
                group_key = key
                knowledge = CausalKnowledge(int(identity["player_index"]), _deck(raw))
            assert knowledge is not None
            snapshot = knowledge.consume(raw["actor_observation"], raw["event_cursor"])
            compiled = compile_row(raw, snapshot, prototypes)
            split = str(raw["split"])
            writer.add(split, {
                "actor": actor_payload(compiled),
                "target": {"ordered_action": compiled["legacy"]["action"], "termination": compiled["action_termination"]},
                "audit": {
                    "identity": identity, "split": split,
                    "source_id": raw.get("source_id"),
                    "source_payload_sha256": raw.get("source_payload_sha256"),
                },
            })
            count += 1
            if max_decisions is not None and count >= max_decisions:
                break
        writer.flush_all()
        split_counts = {
            split: sum(item["count"] for item in shards)
            for split, shards in writer.shards.items()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "date_start": start_date,
            "date_end": end_date,
            "decisions": count,
            "records_per_shard": records_per_shard,
            "split_counts": split_counts,
            "prototype_sha256": _sha256(output / "prototypes.json"),
            "full_engine_prototype_sha256": _sha256(output / "official_full_engine_prototypes_v1.json"),
            "shards": writer.shards,
            "actor_forward_excludes": ["source_id", "team_name", "source_payload_sha256"],
        }
        (output / "manifest.json").write_text(_canonical(manifest), encoding="utf-8")
        return manifest
    except Exception:
        (output / "FAILED").write_text("materialization failed; directory is not resumable\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--start-date", default="2026-07-10")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--records-per-shard", type=int, default=4096)
    parser.add_argument("--max-decisions", type=int)
    args = parser.parse_args()
    result = materialize(
        args.raw_root, args.output, args.prototypes, start_date=args.start_date,
        end_date=args.end_date, records_per_shard=args.records_per_shard,
        max_decisions=args.max_decisions,
    )
    print(_canonical(result), end="")


if __name__ == "__main__":
    main()
