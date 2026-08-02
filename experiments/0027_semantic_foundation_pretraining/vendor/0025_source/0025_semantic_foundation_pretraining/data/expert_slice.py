"""Create a fail-closed exact-expert, exact-deck slice of a replay catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0025_exact_expert_deck_catalog_v1"
AUDIT_SCHEMA_VERSION = "0025_exact_expert_deck_audit_v1"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value


def _validate_deck(row: Mapping[str, Any], expected_hash: str) -> None:
    if row.get("deck_sha256") != expected_hash:
        raise ValueError(f"accepted Episode has wrong deck: {row.get('episode_id')}")
    counts = row.get("deck_counts")
    if not isinstance(counts, list):
        raise ValueError(f"accepted Episode has no deck counts: {row.get('episode_id')}")
    total = 0
    seen: set[int] = set()
    for item in counts:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or isinstance(item[0], bool)
            or not isinstance(item[0], int)
            or item[0] < 0
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] < 1
            or item[0] in seen
        ):
            raise ValueError(f"accepted Episode has invalid deck counts: {row.get('episode_id')}")
        seen.add(item[0])
        total += item[1]
    if total != 60:
        raise ValueError(
            f"accepted Episode deck must contain exactly 60 cards: {row.get('episode_id')}"
        )


def slice_catalog(
    catalog: Mapping[str, Any],
    *,
    team_names: Sequence[str],
    deck_sha256: str,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a raw-builder catalog and an independent audit for one expert deck."""
    names = tuple(team_names)
    if not names or len(names) != len(set(names)) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("team names must be unique non-empty exact strings")
    if start_date > end_date:
        raise ValueError("start date is after end date")
    expected_deck = _validate_hash(deck_sha256, "deck_sha256")
    episodes = catalog.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("catalog episodes are absent")

    accepted = [
        dict(row)
        for row in episodes
        if isinstance(row, Mapping)
        and row.get("team_name") in names
        and row.get("deck_sha256") == expected_deck
        and start_date <= str(row.get("episode_date", "")) <= end_date
    ]
    accepted.sort(key=lambda row: int(row["episode_id"]))
    identities: set[int] = set()
    source_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    seat_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_split_counts: Counter[tuple[int, str]] = Counter()
    archive_paths: set[str] = set()
    for row in accepted:
        episode_id = row.get("episode_id")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int):
            raise ValueError("accepted Episode has invalid ID")
        if episode_id in identities:
            raise ValueError(f"duplicate Episode ID in expert slice: {episode_id}")
        identities.add(episode_id)
        if row.get("terminal_outcome") != "win":
            raise ValueError(f"accepted Episode is not a winning view: {episode_id}")
        split = row.get("split")
        if split not in {"train", "validation"}:
            raise ValueError(f"accepted Episode has invalid split: {episode_id}")
        _validate_hash(row.get("payload_sha256"), "payload_sha256")
        _validate_deck(row, expected_deck)
        team_name = str(row["team_name"])
        source_id = row.get("source_id")
        if isinstance(source_id, bool) or not isinstance(source_id, int) or source_id < 1:
            raise ValueError(f"accepted Episode has invalid source ID: {episode_id}")
        source_counts[team_name] += 1
        date_counts[str(row["episode_date"])] += 1
        seat_counts[str(row.get("seat"))] += 1
        split_counts[str(split)] += 1
        source_split_counts[(source_id, str(split))] += 1
        locator = row.get("locator")
        if isinstance(locator, Mapping) and locator.get("kind") == "zip_member":
            archive_paths.add(str(locator.get("archive")))

    missing = [name for name in names if source_counts[name] == 0]
    if missing:
        raise ValueError(f"requested team names have no accepted Episodes: {missing}")
    if not accepted:
        raise ValueError("expert slice is empty")

    source_by_name = {
        item.get("team_name"): dict(item)
        for item in catalog.get("sources", [])
        if isinstance(item, Mapping)
    }
    sources: list[dict[str, Any]] = []
    for name in names:
        source = source_by_name.get(name)
        if source is None:
            raise ValueError(f"accepted source is absent from vocabulary: {name}")
        expected_source_id = int(source["source_id"])
        if any(
            int(row["source_id"]) != expected_source_id
            for row in accepted
            if row["team_name"] == name
        ):
            raise ValueError(f"source ID drift for team: {name}")
        sources.append(source)

    archives = [
        dict(item)
        for item in catalog.get("archives", [])
        if isinstance(item, Mapping) and str(item.get("path")) in archive_paths
    ]
    parent_split = dict(catalog.get("split", {}))
    split = {
        **parent_split,
        "counts": dict(sorted(split_counts.items())),
        "group_count": len(source_split_counts),
        "parent_counts": parent_split.get("counts", {}),
        "preserved_from_parent": True,
    }
    totals = {
        "archive_count": len(archives),
        "unique_winning_episodes": len(accepted),
        "source_count": len(sources),
        "deck_count": 1,
    }
    parent_catalog_sha256 = _validate_hash(
        catalog.get("catalog_sha256"), "parent catalog_sha256"
    )
    sliced: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "parent_schema_version": catalog.get("schema_version"),
        "parent_catalog_sha256": parent_catalog_sha256,
        "contracts": {
            **dict(catalog.get("contracts", {})),
            "expert_selection": "exact_team_name_allowlist",
            "deck_selection": "exact_registered_60_card_sha256",
            "source_identity_actor_visible": False,
            "split_assignment": "preserved_from_parent_catalog",
        },
        "selection": {
            "team_names": list(names),
            "deck_sha256": expected_deck,
            "start_date": start_date,
            "end_date": end_date,
        },
        "archives": archives,
        "exclusions": [],
        "sources": sources,
        "source_vocabulary_sha256": _canonical_hash(sources),
        "split": split,
        "totals": totals,
        "episodes": accepted,
    }
    sliced["catalog_sha256"] = _canonical_hash(
        {
            "parent_catalog_sha256": parent_catalog_sha256,
            "selection": sliced["selection"],
            "sources": sources,
            "split": split,
            "totals": totals,
            "episodes": accepted,
        }
    )
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "catalog_sha256": sliced["catalog_sha256"],
        "parent_catalog_sha256": parent_catalog_sha256,
        "selection": sliced["selection"],
        "totals": {"episodes": len(accepted), "source_count": len(sources), "deck_count": 1},
        "source_counts": dict(source_counts),
        "date_counts": dict(sorted(date_counts.items())),
        "seat_counts": dict(sorted(seat_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_split_counts": [
            {
                "source_id": source_id,
                "team_name": next(
                    source["team_name"] for source in sources if source["source_id"] == source_id
                ),
                "split": split_name,
                "episodes": count,
            }
            for (source_id, split_name), count in sorted(source_split_counts.items())
        ],
        "episode_commitments": [
            {
                "episode_id": row["episode_id"],
                "episode_date": row["episode_date"],
                "team_name": row["team_name"],
                "source_id": row["source_id"],
                "seat": row["seat"],
                "split": row["split"],
                "payload_sha256": row["payload_sha256"],
                "deck_sha256": row["deck_sha256"],
            }
            for row in accepted
        ],
    }
    audit["audit_sha256"] = _canonical_hash(audit)
    return sliced, audit


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            payload = _canonical_bytes(value)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--team-name", action="append", required=True)
    parser.add_argument("--deck-sha256", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    sliced, audit = slice_catalog(
        catalog,
        team_names=tuple(args.team_name),
        deck_sha256=args.deck_sha256,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    _write_new(args.output_catalog, sliced)
    _write_new(args.output_audit, audit)
    print(
        json.dumps(
            {
                "catalog_sha256": sliced["catalog_sha256"],
                "episodes": audit["totals"]["episodes"],
                "source_counts": audit["source_counts"],
                "split_counts": audit["split_counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
