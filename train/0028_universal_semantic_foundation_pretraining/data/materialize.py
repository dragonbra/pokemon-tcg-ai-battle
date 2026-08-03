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
from collections import deque
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ..features.compiler import compile_canonical_row
from ..contracts.fields import SCHEMA_VERSION
from ..domain.prototypes import PrototypeIndex
from ..knowledge.state import CausalKnowledge
from ..storage import guard_storage


_WORKER_PROTOTYPES: PrototypeIndex | None = None


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


def _iter_raw(
    raw_roots: list[Path],
    catalog_episodes: Mapping[int, Mapping[str, Any]],
    *,
    require_complete_catalog: bool,
) -> Iterator[Mapping[str, Any]]:
    seen_episodes: set[int] = set()
    for raw_root in raw_roots:
        reference = json.loads(
            (raw_root / "dataset_reference.json").read_text(encoding="utf-8")
        )
        for source_split in ("train", "validation"):
            for item in reference["shards"][source_split]:
                path = raw_root / item["path"]
                if _sha256(path) != item["sha256"]:
                    raise ValueError(f"canonical raw shard commitment mismatch: {path}")
                count = 0
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        source = json.loads(line)
                        if source.get("split") != source_split:
                            raise ValueError("canonical raw row split mismatch")
                        identity = source["identity"]
                        episode_id = int(identity["episode_id"])
                        catalog = catalog_episodes.get(episode_id)
                        if catalog is None:
                            raise ValueError(f"raw Episode is absent from 0028 catalog: {episode_id}")
                        if int(identity["player_index"]) != int(catalog["player_index"]):
                            raise ValueError(f"winner perspective drift: {episode_id}")
                        if source.get("source_payload_sha256") != catalog.get("payload_sha256"):
                            raise ValueError(f"raw payload commitment drift: {episode_id}")
                        if source["deck_manifest"].get("sha256") != catalog.get("deck_sha256"):
                            raise ValueError(f"registered deck commitment drift: {episode_id}")
                        row = dict(source)
                        row["split"] = catalog["split"]
                        row["source_id"] = catalog["source_id"]
                        row["source_team_name"] = catalog["team_name"]
                        seen_episodes.add(episode_id)
                        count += 1
                        yield row
                if count != int(item["count"]):
                    raise ValueError(f"canonical raw shard count mismatch: {path}")
    missing = set(catalog_episodes) - seen_episodes
    if require_complete_catalog and missing:
        raise ValueError(f"catalog Episodes missing from raw inputs: {len(missing)}")


def _group_key(raw: Mapping[str, Any]) -> tuple[Any, ...]:
    identity = raw["identity"]
    return (identity["date"], identity["episode_id"], identity["player_index"])


def _iter_episode_groups(
    raw_roots: list[Path],
    catalog_episodes: Mapping[int, Mapping[str, Any]],
    *,
    require_complete_catalog: bool,
) -> Iterator[list[Mapping[str, Any]]]:
    pending: list[Mapping[str, Any]] = []
    key: tuple[Any, ...] | None = None
    for raw in _iter_raw(
        raw_roots,
        catalog_episodes,
        require_complete_catalog=require_complete_catalog,
    ):
        current = _group_key(raw)
        if pending and current != key:
            yield pending
            pending = []
        key = current
        pending.append(raw)
    if pending:
        yield pending


def _init_compile_worker(prototype_path: str) -> None:
    global _WORKER_PROTOTYPES
    _WORKER_PROTOTYPES = PrototypeIndex.load(prototype_path)


def _compile_episode(
    rows: list[Mapping[str, Any]],
    prototypes: PrototypeIndex | None = None,
) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    index = prototypes or _WORKER_PROTOTYPES
    if index is None or not rows:
        raise RuntimeError("canonical compile worker is not initialized")
    first = rows[0]
    knowledge = CausalKnowledge(
        int(first["identity"]["player_index"]),
        _deck(first),
    )
    output = []
    expected_key = _group_key(first)
    for raw in rows:
        if _group_key(raw) != expected_key:
            raise ValueError("canonical compile group crossed Episode boundary")
        snapshot = knowledge.consume(raw["actor_observation"], raw["event_cursor"])
        output.append((raw, compile_canonical_row(raw, snapshot, index)))
    return output


def _compiled_groups(
    raw_roots: list[Path],
    catalog_episodes: Mapping[int, Mapping[str, Any]],
    prototype_path: Path,
    prototypes: PrototypeIndex,
    workers: int,
    require_complete_catalog: bool,
) -> Iterator[list[tuple[Mapping[str, Any], dict[str, Any]]]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    groups = _iter_episode_groups(
        raw_roots,
        catalog_episodes,
        require_complete_catalog=require_complete_catalog,
    )
    if workers == 1:
        for group in groups:
            yield _compile_episode(group, prototypes)
        return
    pending: deque[Future] = deque()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_compile_worker,
        initargs=(str(prototype_path),),
    ) as pool:
        for _ in range(workers * 2):
            try:
                pending.append(pool.submit(_compile_episode, next(groups)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(pool.submit(_compile_episode, next(groups)))
            except StopIteration:
                pass


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
    raw_roots: Path | list[Path],
    output: Path,
    prototype_path: Path,
    catalog_path: Path,
    *,
    records_per_shard: int = 4096,
    max_decisions_per_split: int | None = None,
    workers: int = 1,
    require_complete_catalog: bool = True,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite canonical dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir()
    started = time.perf_counter()
    guard_storage(output.parent)
    raw_roots = [raw_roots] if isinstance(raw_roots, Path) else list(raw_roots)
    if not raw_roots:
        raise ValueError("at least one immutable raw root is required")
    raw_references = [
        json.loads((root / "dataset_reference.json").read_text(encoding="utf-8"))
        for root in raw_roots
    ]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_episodes = {int(row["episode_id"]): row for row in catalog["episodes"]}
    if len(catalog_episodes) != len(catalog["episodes"]):
        raise ValueError("0028 catalog contains duplicate Episode IDs")
    prototypes = PrototypeIndex.load(prototype_path)
    shutil.copy2(prototype_path, stage / "prototypes.json")
    shutil.copy2(
        prototype_path.with_name("official_full_engine_prototypes_v1.json"),
        stage / "official_full_engine_prototypes_v1.json",
    )
    writer = _ShardWriter(stage, records_per_shard)
    counts: Counter[str] = Counter()
    maxima: Counter[str] = Counter()
    deck_hashes: set[str] = set()
    episodes: dict[str, set[tuple[Any, ...]]] = {
        "train": set(),
        "validation": set(),
    }
    processed_groups = 0
    try:
        for group in _compiled_groups(
            raw_roots,
            catalog_episodes,
            prototype_path,
            prototypes,
            workers,
            require_complete_catalog,
        ):
            processed_groups += 1
            for raw, compiled in group:
                split = str(raw["split"])
                if (
                    max_decisions_per_split is not None
                    and counts[split] >= max_decisions_per_split
                ):
                    continue
                identity = raw["identity"]
                key = _group_key(raw)
                deck_hashes.add(_deck_sha256(_deck(raw)))
                episodes[split].add(key)
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
                            "source_team_name": raw.get("source_team_name"),
                            "deck_sha256": raw["deck_manifest"].get("sha256"),
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
            if processed_groups % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "0028_canonical_progress",
                            "episodes": processed_groups,
                            "decisions": sum(counts.values()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        writer.close()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "source_raw_references": [
                {
                    "path": str(root),
                    "sha256": _sha256(root / "dataset_reference.json"),
                    "schema_version": reference.get("schema_version"),
                }
                for root, reference in zip(raw_roots, raw_references, strict=True)
            ],
            "winner_catalog_path": str(catalog_path),
            "winner_catalog_file_sha256": _sha256(catalog_path),
            "winner_catalog_sha256": catalog.get("catalog_sha256"),
            "complete_catalog_coverage_required": require_complete_catalog,
            "prototype_sha256": _sha256(stage / "prototypes.json"),
            "full_engine_prototype_sha256": _sha256(stage / "official_full_engine_prototypes_v1.json"),
            "decisions": sum(counts.values()),
            "split_counts": {split: counts[split] for split in ("train", "validation")},
            "episode_split_counts": {
                split: len(episodes[split]) for split in ("train", "validation")
            },
            "exact_deck_count": len(deck_hashes),
            "exact_deck_conditioning": "registered_60_card_multiset",
            "sources": catalog.get("sources", []),
            "records_per_shard": records_per_shard,
            "materialization_workers": workers,
            "processed_episode_groups": processed_groups,
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
            + (stage / "prototypes.json").stat().st_size
            + (stage / "official_full_engine_prototypes_v1.json").stat().st_size,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(_canonical(manifest), encoding="utf-8")
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        stage.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--records-per-shard", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--allow-catalog-subset", action="store_true")
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
                args.catalog,
                records_per_shard=args.records_per_shard,
                max_decisions_per_split=args.max_decisions_per_split,
                workers=args.workers,
                require_complete_catalog=not args.allow_catalog_subset,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
