"""Materialize the canonical semantic dataset directly from audited raw decisions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..features.canonical.compiler import compile_canonical_row
from ..features.canonical.schema import SCHEMA_VERSION
from ..features.prototypes import PrototypeIndex
from ..knowledge.state import CausalKnowledge
from ..storage import guard_storage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"


def _deck(row: Mapping[str, Any]) -> list[int]:
    deck = [
        int(identity)
        for identity, count in row["deck_manifest"]["counts"]
        for _ in range(int(count))
    ]
    if len(deck) != 60:
        raise ValueError("canonical source deck is not exactly 60 cards")
    return deck


def _deck_sha256(deck: list[int]) -> str:
    canonical = ",".join(str(identity) for identity in sorted(deck)).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _iter_raw(raw_root: Path) -> Iterator[Mapping[str, Any]]:
    reference = json.loads((raw_root / "dataset_reference.json").read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        for item in reference["shards"][split]:
            path = raw_root / item["path"]
            if _sha256(path) != item["sha256"]:
                raise ValueError(f"canonical raw shard commitment mismatch: {path}")
            count = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("split") != split:
                        raise ValueError("canonical raw row split mismatch")
                    count += 1
                    yield row
            if count != int(item["count"]):
                raise ValueError(f"canonical raw shard count mismatch: {path}")


class _ShardWriter:
    def __init__(self, root: Path, records_per_shard: int):
        self.root = root
        self.records_per_shard = records_per_shard
        self.pending: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        self.shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}

    def add(self, split: str, row: dict[str, Any]) -> None:
        self.pending[split].append(row)
        if len(self.pending[split]) >= self.records_per_shard:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.pending[split]
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
        guard_storage(self.root)

    def close(self) -> None:
        for split in self.pending:
            self.flush(split)


def materialize_canonical(
    raw_root: Path,
    output: Path,
    prototype_path: Path,
    *,
    records_per_shard: int = 4096,
    max_decisions_per_split: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite canonical dataset: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    guard_storage(output)
    raw_reference = json.loads(
        (raw_root / "dataset_reference.json").read_text(encoding="utf-8")
    )
    prototypes = PrototypeIndex.load(prototype_path)
    shutil.copy2(prototype_path, output / "prototypes.json")
    shutil.copy2(
        prototype_path.with_name("official_full_engine_prototypes_v1.json"),
        output / "official_full_engine_prototypes_v1.json",
    )
    writer = _ShardWriter(output, records_per_shard)
    knowledge: CausalKnowledge | None = None
    group_key: tuple[Any, ...] | None = None
    counts: Counter[str] = Counter()
    maxima: Counter[str] = Counter()
    deck_hashes: set[str] = set()
    episodes: dict[str, set[tuple[Any, ...]]] = {
        "train": set(),
        "validation": set(),
    }
    try:
        for raw in _iter_raw(raw_root):
            split = str(raw["split"])
            if (
                max_decisions_per_split is not None
                and counts[split] >= max_decisions_per_split
            ):
                continue
            identity = raw["identity"]
            key = (identity["date"], identity["episode_id"], identity["player_index"])
            deck_hashes.add(_deck_sha256(_deck(raw)))
            if len(deck_hashes) != 1:
                raise ValueError("canonical materialization mixed multiple exact decks")
            episodes[split].add(key)
            if key != group_key:
                group_key = key
                knowledge = CausalKnowledge(int(identity["player_index"]), _deck(raw))
            assert knowledge is not None
            snapshot = knowledge.consume(raw["actor_observation"], raw["event_cursor"])
            compiled = compile_canonical_row(raw, snapshot, prototypes)
            writer.add(
                split,
                {
                    "schema_version": SCHEMA_VERSION,
                    "actor": compiled["actor"],
                    "target": compiled["target"],
                    "audit": {
                        "identity": identity,
                        "split": split,
                        "source_id": raw.get("source_id"),
                        "source_payload_sha256": raw.get("source_payload_sha256"),
                    },
                },
            )
            counts[split] += 1
            for name in (
                "card_cat",
                "resource_cat",
                "event_cat",
                "option_cat",
                "option_skill_id",
                "option_effect_id",
            ):
                maxima[name] = max(maxima[name], len(compiled["actor"][name]))
        writer.close()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "source_raw_reference_sha256": _sha256(raw_root / "dataset_reference.json"),
            "prototype_sha256": _sha256(output / "prototypes.json"),
            "full_engine_prototype_sha256": _sha256(output / "official_full_engine_prototypes_v1.json"),
            "decisions": sum(counts.values()),
            "split_counts": {split: counts[split] for split in ("train", "validation")},
            "episode_split_counts": {
                split: len(episodes[split]) for split in ("train", "validation")
            },
            "exact_deck_sha256": next(iter(deck_hashes), None),
            "sources": raw_reference.get("sources", []),
            "records_per_shard": records_per_shard,
            "shards": writer.shards,
            "maximum_lengths": dict(maxima),
            "actor_forward_excludes": [
                "legacy",
                "ordered_action",
                "source_id",
                "source_team_name",
                "source_payload_sha256",
            ],
            "source_identity_actor_visible": False,
            "materialization_seconds": time.perf_counter() - started,
            "dataset_payload_bytes": sum(
                int(item["bytes"])
                for items in writer.shards.values()
                for item in items
            )
            + (output / "prototypes.json").stat().st_size
            + (output / "official_full_engine_prototypes_v1.json").stat().st_size,
        }
        (output / "manifest.json").write_text(_canonical(manifest), encoding="utf-8")
        return manifest
    except BaseException:
        (output / "FAILED").write_text(
            "canonical materialization failed; this directory is not resumable\n",
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--records-per-shard", type=int, default=4096)
    parser.add_argument(
        "--max-decisions-per-split",
        type=int,
        help="Smoke-only cap applied independently to train and validation.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            materialize_canonical(
                args.raw_root,
                args.output,
                args.prototypes,
                records_per_shard=args.records_per_shard,
                max_decisions_per_split=args.max_decisions_per_split,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
