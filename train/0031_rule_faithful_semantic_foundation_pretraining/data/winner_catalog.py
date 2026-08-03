"""Build the persona-free universal winner catalog from official Episode archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date as calendar_date
from pathlib import Path
from typing import Any

from .replay_contract import DeckManifest, registration_decks


SCHEMA_VERSION = "0031_rule_faithful_winner_catalog_v1"
SPLIT_VERSION = "0031_episode_split_v1"
SPLIT_SEED = 20260802
_ARCHIVE_DATE_RE = re.compile(r".*?(\d{4}-\d{2}-\d{2})\.zip")
_MEMBER_RE = re.compile(r"(?:episode-)?(\d+)(?:-replay)?\.json")


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


def archive_date(path: Path) -> str:
    match = _ARCHIVE_DATE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"archive filename has no date: {path.name}")
    return calendar_date.fromisoformat(match.group(1)).isoformat()


def _path_date(path: Path) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(path))
    if match is None:
        raise ValueError(f"path has no date: {path}")
    return calendar_date.fromisoformat(match.group(0)).isoformat()


def _episode_id(payload: dict[str, Any], member_name: str) -> int:
    info = payload.get("info")
    identity = info.get("EpisodeId") if isinstance(info, dict) else None
    if isinstance(identity, bool) or not isinstance(identity, int):
        raise ValueError("EpisodeId is absent")
    match = _MEMBER_RE.fullmatch(Path(member_name).name)
    if match is None or int(match.group(1)) != identity:
        raise ValueError(f"member/EpisodeId mismatch: {member_name} != {identity}")
    return identity


def _winner_index(payload: dict[str, Any]) -> int:
    rewards = payload.get("rewards")
    statuses = payload.get("statuses")
    if not isinstance(rewards, list) or not isinstance(statuses, list):
        raise ValueError("terminal rewards/statuses are absent")
    if len(rewards) != 2 or len(statuses) != 2:
        raise ValueError("Episode is not a terminal two-player game")
    numeric = [
        (index, float(value))
        for index, value in enumerate(rewards)
        if not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ]
    if not numeric:
        raise ValueError("Episode has no finite reward")
    maximum = max(value for _, value in numeric)
    winners = [index for index, value in numeric if value == maximum]
    if len(winners) != 1 or maximum <= 0:
        raise ValueError("Episode has no unique positive winner")
    winner = winners[0]
    if statuses[winner] != "DONE":
        raise ValueError("positive winner is not DONE")
    loser_status = statuses[1 - winner]
    if not isinstance(loser_status, str) or loser_status in {"", "ACTIVE"}:
        raise ValueError("opponent status is not terminal")
    return winner


def _first_player(payload: dict[str, Any]) -> int:
    evidence: set[int] = set()
    for step in payload.get("steps", []):
        if not isinstance(step, list):
            continue
        for member in step:
            visualize = member.get("visualize") if isinstance(member, dict) else None
            if not isinstance(visualize, list):
                continue
            for frame in visualize:
                current = frame.get("current") if isinstance(frame, dict) else None
                value = current.get("firstPlayer") if isinstance(current, dict) else None
                if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
                    evidence.add(value)
        if len(evidence) > 1:
            break
    if len(evidence) != 1:
        raise ValueError(f"first-player evidence is ambiguous: {sorted(evidence)}")
    return evidence.pop()


def _source_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("team name is not a string")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or len(normalized.encode("utf-8")) > 512:
        raise ValueError("team name is empty or unreasonably large")
    return normalized


def assign_split(*, episode_id: int, deck_sha256: str, seat: str) -> tuple[str, str]:
    key = "|".join(
        (SPLIT_VERSION, str(SPLIT_SEED), deck_sha256, seat, str(episode_id))
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    split = "validation" if int(digest[:8], 16) % 10 == 0 else "train"
    return split, digest


def _record(
    raw: bytes,
    archive: Path,
    member_name: str,
    *,
    episode_date: str | None = None,
    locator: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Episode payload is not an object")
    identity = _episode_id(payload, member_name)
    winner = _winner_index(payload)
    info = payload.get("info")
    teams = info.get("TeamNames") if isinstance(info, dict) else None
    if not isinstance(teams, list) or len(teams) != 2:
        raise ValueError("TeamNames is absent")
    team_name = _source_name(teams[winner])
    decks = registration_decks(payload)
    deck = decks[winner]
    deck_manifest = DeckManifest.from_card_ids(deck)
    first_player = _first_player(payload)
    seat = "first" if winner == first_player else "second"
    split, split_digest = assign_split(
        episode_id=identity,
        deck_sha256=deck_manifest.sha256,
        seat=seat,
    )
    return {
        "episode_id": identity,
        "episode_date": episode_date or archive_date(archive),
        "team_name": team_name,
        "player_index": winner,
        "first_player": first_player,
        "seat": seat,
        "terminal_outcome": "win",
        "opponent_terminal_status": payload["statuses"][1 - winner],
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "deck_sha256": deck_manifest.sha256,
        "deck_counts": [list(item) for item in deck_manifest.counts],
        "split": split,
        "split_rank_sha256": split_digest,
        "locator": locator
        or {"kind": "zip_member", "archive": str(archive.resolve()), "member": member_name},
    }


def _scan_archive(
    archive: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    records = []
    exclusions: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as bundle:
        members = [
            item
            for item in bundle.infolist()
            if not item.is_dir() and _MEMBER_RE.fullmatch(Path(item.filename).name)
        ]
        audit = {
            "path": str(archive),
            "date": archive_date(archive),
            "compressed_file_bytes": archive.stat().st_size,
            "episode_members": len(members),
            "declared_episode_bytes": sum(item.file_size for item in members),
        }
        for member in members:
            raw = bundle.read(member)
            try:
                records.append(_record(raw, archive, member.filename))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                exclusions[f"{type(error).__name__}:{error}"] += 1
    return audit, records, exclusions


def scan_archives(
    archive_paths: list[Path],
    patch_paths: list[Path] | None = None,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    if not archive_paths:
        raise ValueError("at least one archive is required")
    if workers < 1:
        raise ValueError("workers must be positive")
    records_by_id: dict[int, dict[str, Any]] = {}
    archives = []
    exclusions: Counter[str] = Counter()
    duplicate_count = 0

    def merge(record: dict[str, Any]) -> None:
        nonlocal duplicate_count
        existing = records_by_id.get(record["episode_id"])
        if existing is not None:
            duplicate_count += 1
            if existing["payload_sha256"] != record["payload_sha256"]:
                raise ValueError(f"duplicate Episode payload drift: {record['episode_id']}")
            return
        records_by_id[record["episode_id"]] = record

    ordered = sorted(archive_paths)
    if workers == 1:
        scanned = map(_scan_archive, ordered)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        scanned = executor.map(_scan_archive, ordered)
    try:
        for audit, records, archive_exclusions in scanned:
            archives.append(audit)
            exclusions.update(archive_exclusions)
            for record in records:
                merge(record)
    finally:
        if executor is not None:
            executor.shutdown()

    accepted_patches = 0
    for patch in sorted(patch_paths or []):
        patch = patch.resolve()
        raw = patch.read_bytes()
        try:
            record = _record(
                raw,
                patch,
                patch.name,
                episode_date=_path_date(patch),
                locator={"kind": "file", "archive": "", "member": str(patch)},
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            exclusions[f"patch:{type(error).__name__}:{error}"] += 1
            continue
        merge(record)
        accepted_patches += 1

    records = [records_by_id[key] for key in sorted(records_by_id)]
    names = sorted({record["team_name"] for record in records})
    source_ids = {name: index for index, name in enumerate(names, start=1)}
    for record in records:
        record["source_id"] = source_ids[record["team_name"]]
    sources = [{"source_id": source_ids[name], "team_name": name} for name in names]
    split_counts = Counter(record["split"] for record in records)
    totals = {
        "archive_count": len(archives),
        "archive_episode_members": sum(item["episode_members"] for item in archives),
        "unique_winning_episodes": len(records),
        "duplicate_episode_members": duplicate_count,
        "accepted_gap_patches": accepted_patches,
        "excluded_episode_members": sum(exclusions.values()),
        "source_count": len(sources),
        "deck_count": len({record["deck_sha256"] for record in records}),
        "compressed_archive_bytes": sum(item["compressed_file_bytes"] for item in archives),
        "declared_episode_bytes": sum(item["declared_episode_bytes"] for item in archives),
    }
    split = {
        "algorithm": "deck_seat_episode_sha256_mod10_v1",
        "seed": SPLIT_SEED,
        "group_unit": "whole_episode",
        "digest_input": (
            f"{SPLIT_VERSION}|{SPLIT_SEED}|<deck_sha256>|<seat>|<episode_id>"
        ),
        "counts": dict(sorted(split_counts.items())),
    }
    catalog: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contracts": {
            "selection": "unique_positive_terminal_winner_perspective",
            "deduplication_key": "episode_id",
            "source_identity": "provenance_only",
            "actor_source_identity_visible": False,
            "deck_identity": "exact_registered_60_card_multiset",
        },
        "archives": archives,
        "exclusions": [
            {"reason": reason, "count": count}
            for reason, count in sorted(exclusions.items())
        ],
        "sources": sources,
        "source_vocabulary_sha256": _canonical_hash(sources),
        "split": split,
        "totals": totals,
        "episodes": records,
    }
    catalog["catalog_sha256"] = _canonical_hash(
        {
            "sources": sources,
            "split": split,
            "totals": totals,
            "episodes": records,
        }
    )
    return catalog


def merge_catalogs(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """Merge immutable catalog evidence and recompute the 0031 split/source vocabulary."""
    records_by_id: dict[int, dict[str, Any]] = {}
    duplicate_count = 0
    for source in (base, delta):
        episodes = source.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("catalog merge source has no Episode records")
        for raw in episodes:
            record = dict(raw)
            identity = int(record["episode_id"])
            deck_cards = [
                int(card_id)
                for card_id, count in record["deck_counts"]
                for _ in range(int(count))
            ]
            record["deck_sha256"] = DeckManifest.from_card_ids(deck_cards).sha256
            existing = records_by_id.get(identity)
            if existing is not None:
                duplicate_count += 1
                if existing.get("payload_sha256") != record.get("payload_sha256"):
                    raise ValueError(f"duplicate Episode payload drift: {identity}")
                continue
            split, digest = assign_split(
                episode_id=identity,
                deck_sha256=str(record["deck_sha256"]),
                seat=str(record["seat"]),
            )
            record["split"] = split
            record["split_rank_sha256"] = digest
            record.pop("source_id", None)
            records_by_id[identity] = record

    records = [records_by_id[key] for key in sorted(records_by_id)]
    names = sorted({str(record["team_name"]) for record in records})
    source_ids = {name: index for index, name in enumerate(names, start=1)}
    for record in records:
        record["source_id"] = source_ids[str(record["team_name"])]
    sources = [{"source_id": source_ids[name], "team_name": name} for name in names]
    archives = [*base.get("archives", []), *delta.get("archives", [])]
    exclusions = Counter()
    for source in (base, delta):
        for item in source.get("exclusions", []):
            exclusions[str(item["reason"])] += int(item["count"])
    split_counts = Counter(str(record["split"]) for record in records)
    totals = {
        "archive_count": len(archives),
        "archive_episode_members": sum(int(item["episode_members"]) for item in archives),
        "unique_winning_episodes": len(records),
        "duplicate_episode_members": (
            int(base.get("totals", {}).get("duplicate_episode_members", 0))
            + int(delta.get("totals", {}).get("duplicate_episode_members", 0))
            + duplicate_count
        ),
        "accepted_gap_patches": (
            int(base.get("totals", {}).get("accepted_gap_patches", 0))
            + int(delta.get("totals", {}).get("accepted_gap_patches", 0))
        ),
        "excluded_episode_members": sum(exclusions.values()),
        "source_count": len(sources),
        "deck_count": len({record["deck_sha256"] for record in records}),
        "compressed_archive_bytes": sum(int(item["compressed_file_bytes"]) for item in archives),
        "declared_episode_bytes": sum(int(item["declared_episode_bytes"]) for item in archives),
    }
    split = {
        "algorithm": "deck_seat_episode_sha256_mod10_v1",
        "seed": SPLIT_SEED,
        "group_unit": "whole_episode",
        "digest_input": f"{SPLIT_VERSION}|{SPLIT_SEED}|<deck_sha256>|<seat>|<episode_id>",
        "counts": dict(sorted(split_counts.items())),
    }
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "contracts": {
            "selection": "unique_positive_terminal_winner_perspective",
            "deduplication_key": "episode_id",
            "source_identity": "provenance_only",
            "actor_source_identity_visible": False,
            "deck_identity": "exact_registered_60_card_multiset",
        },
        "archives": archives,
        "exclusions": [
            {"reason": reason, "count": count}
            for reason, count in sorted(exclusions.items())
        ],
        "sources": sources,
        "source_vocabulary_sha256": _canonical_hash(sources),
        "split": split,
        "totals": totals,
        "episodes": records,
        "immutable_base_catalog_sha256": base.get("catalog_sha256"),
        "delta_catalog_sha256": delta.get("catalog_sha256"),
        "catalog_lineage": {
            "normalized_from_catalog_sha256": base.get("catalog_sha256"),
            "immutable_prefix_catalog_sha256": (
                base.get("catalog_lineage", {}).get("immutable_prefix_catalog_sha256")
                or base.get("immutable_base_catalog_sha256")
                or base.get("catalog_sha256")
            ),
            "delta_catalog_sha256": (
                base.get("catalog_lineage", {}).get("delta_catalog_sha256")
                or base.get("delta_catalog_sha256")
                or delta.get("catalog_sha256")
            ),
        },
    }
    catalog["catalog_sha256"] = _canonical_hash(
        {"sources": sources, "split": split, "totals": totals, "episodes": records}
    )
    return catalog


def write_catalog(catalog: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(_canonical_bytes(catalog))
    partial.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", type=Path)
    parser.add_argument("--start-date", default="2026-07-10")
    parser.add_argument("--end-date", default="2026-08-01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--patches-dir", type=Path)
    parser.add_argument(
        "--immutable-base-catalog",
        type=Path,
        help="Merge an already audited immutable prefix and recompute 0031 splits.",
    )
    parser.add_argument(
        "--normalize-existing-catalog",
        type=Path,
        help="Recommit an existing merged catalog under the current 0031 contract.",
    )
    arguments = parser.parse_args()
    if arguments.normalize_existing_catalog is not None:
        source = json.loads(arguments.normalize_existing_catalog.read_text(encoding="utf-8"))
        empty = {
            "episodes": [],
            "archives": [],
            "exclusions": [],
            "totals": {},
            "catalog_sha256": None,
        }
        catalog = merge_catalogs(source, empty)
        write_catalog(catalog, arguments.output)
        print(json.dumps(catalog["totals"], ensure_ascii=False, sort_keys=True))
        return
    if arguments.archives_dir is None:
        parser.error("--archives-dir is required unless normalizing an existing catalog")
    archives = [
        path
        for path in sorted(arguments.archives_dir.glob("*.zip"))
        if arguments.start_date <= archive_date(path) <= arguments.end_date
    ]
    patches = (
        [
            path
            for path in sorted(arguments.patches_dir.rglob("*.json"))
            if arguments.start_date <= _path_date(path) <= arguments.end_date
        ]
        if arguments.patches_dir is not None and arguments.patches_dir.is_dir()
        else []
    )
    catalog = scan_archives(archives, patches, workers=arguments.workers)
    if arguments.immutable_base_catalog is not None:
        base = json.loads(arguments.immutable_base_catalog.read_text(encoding="utf-8"))
        catalog = merge_catalogs(base, catalog)
    write_catalog(catalog, arguments.output)
    print(json.dumps(catalog["totals"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA_VERSION",
    "SPLIT_SEED",
    "archive_date",
    "assign_split",
    "scan_archives",
    "merge_catalogs",
    "write_catalog",
]
