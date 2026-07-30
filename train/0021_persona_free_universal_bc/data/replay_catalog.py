"""Build the universal winner-perspective catalog from official Episode ZIPs."""

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

from .replay_contract import registration_decks


SCHEMA_VERSION = "0019_universal_winner_catalog_v1"
SPLIT_VERSION = "0019_source_deck_seat_episode_split_v1"
SPLIT_SEED = 20260729
_ARCHIVE_DATE_RE = re.compile(r".*?(\d{4}-\d{2}-\d{2})\.zip")
_MEMBER_RE = re.compile(r"(?:episode-)?(\d+)(?:-replay)?\.json")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _archive_date(path: Path) -> str:
    match = _ARCHIVE_DATE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"archive filename has no date: {path.name}")
    return calendar_date.fromisoformat(match.group(1)).isoformat()


def _path_date(path: Path) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(path))
    if match is None:
        raise ValueError(f"path has no date: {path}")
    return calendar_date.fromisoformat(match.group(0)).isoformat()


def _deck_hash(cards: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(card_id) for card_id in sorted(cards)).encode("ascii")
    ).hexdigest()


def _source_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("team name is not a string")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or len(normalized.encode("utf-8")) > 512:
        raise ValueError("team name is empty or unreasonably large")
    return normalized


def _episode_id(payload: dict[str, Any], member_name: str) -> int:
    info = payload.get("info")
    value = info.get("EpisodeId") if isinstance(info, dict) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("EpisodeId is absent")
    match = _MEMBER_RE.fullmatch(Path(member_name).name)
    if match is None or int(match.group(1)) != value:
        raise ValueError(f"member/EpisodeId mismatch: {member_name} != {value}")
    return value


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
    loser = 1 - winner
    if statuses[winner] != "DONE":
        raise ValueError("positive winner is not DONE")
    if not isinstance(statuses[loser], str) or statuses[loser] in {"", "ACTIVE"}:
        raise ValueError("opponent status is not terminal")
    return winner


def _first_player(payload: dict[str, Any]) -> int:
    evidence: set[int] = set()
    for step in payload.get("steps", []):
        if not isinstance(step, list):
            continue
        for agent in step:
            visualize = agent.get("visualize") if isinstance(agent, dict) else None
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


def _record(
    raw: bytes,
    archive: Path,
    member: str,
    *,
    episode_date: str | None = None,
    locator: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Episode payload is not an object")
    episode_id = _episode_id(payload, member)
    winner = _winner_index(payload)
    info = payload.get("info")
    teams = info.get("TeamNames") if isinstance(info, dict) else None
    if not isinstance(teams, list) or len(teams) != 2:
        raise ValueError("TeamNames is absent")
    team_name = _source_name(teams[winner])
    decks = registration_decks(payload)
    if len(decks) != 2:
        raise ValueError("registration player count is not two")
    deck = decks[winner]
    first_player = _first_player(payload)
    counts = Counter(deck)
    return {
        "episode_id": episode_id,
        "episode_date": episode_date or _archive_date(archive),
        "team_name": team_name,
        "player_index": winner,
        "first_player": first_player,
        "seat": "first" if winner == first_player else "second",
        "terminal_outcome": "win",
        "opponent_terminal_status": payload["statuses"][1 - winner],
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "deck_sha256": _deck_hash(deck),
        "deck_counts": [list(item) for item in sorted(counts.items())],
        "locator": locator
        or {"kind": "zip_member", "archive": str(archive), "member": member},
    }


def _split_digest(record: dict[str, Any]) -> str:
    key = "|".join(
        (
            SPLIT_VERSION,
            str(SPLIT_SEED),
            record["team_name"],
            record["deck_sha256"],
            record["seat"],
            str(record["episode_id"]),
        )
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _assign_sources_and_splits(records: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    team_names = sorted({record["team_name"] for record in records})
    source_ids = {name: index for index, name in enumerate(team_names, start=1)}
    sources = [
        {"source_id": source_ids[name], "team_name": name}
        for name in team_names
    ]
    split_counts: Counter[str] = Counter()
    group_counts: Counter[tuple[str, str, str, str]] = Counter()
    for record in records:
        record["source_id"] = source_ids[record["team_name"]]
        digest = _split_digest(record)
        record["split_rank_sha256"] = digest
        record["split"] = "validation" if int(digest[:8], 16) % 10 == 0 else "train"
        split_counts[record["split"]] += 1
        group_counts[
            (record["split"], record["team_name"], record["deck_sha256"], record["seat"])
        ] += 1
    split = {
        "algorithm": "source_deck_seat_episode_sha256_mod10_v1",
        "seed": SPLIT_SEED,
        "group_unit": "whole_episode",
        "digest_input": (
            f"{SPLIT_VERSION}|{SPLIT_SEED}|<team>|<deck_sha256>|<seat>|<episode_id>"
        ),
        "counts": dict(sorted(split_counts.items())),
        "group_count": len(group_counts),
    }
    return sources, split


def _scan_archive(
    archive: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as bundle:
        infos = [
            item
            for item in bundle.infolist()
            if not item.is_dir() and _MEMBER_RE.fullmatch(Path(item.filename).name)
        ]
        audit = {
            "path": str(archive),
            "date": _archive_date(archive),
            "compressed_file_bytes": archive.stat().st_size,
            "episode_members": len(infos),
            "declared_episode_bytes": sum(item.file_size for item in infos),
        }
        for item in infos:
            raw = bundle.read(item)
            try:
                records.append(_record(raw, archive, item.filename))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                exclusions[type(error).__name__ + ":" + str(error)] += 1
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
    archive_audit: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    duplicates = 0

    def merge_record(record: dict[str, Any]) -> None:
        nonlocal duplicates
        existing = records_by_id.get(record["episode_id"])
        if existing is not None:
            duplicates += 1
            if existing["payload_sha256"] != record["payload_sha256"]:
                raise ValueError(f"duplicate Episode payload drift: {record['episode_id']}")
            return
        records_by_id[record["episode_id"]] = record

    ordered_archives = sorted(archive_paths)
    if workers == 1:
        scanned = map(_scan_archive, ordered_archives)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        scanned = executor.map(_scan_archive, ordered_archives)
    try:
        for audit, archive_records, archive_exclusions in scanned:
            archive_audit.append(audit)
            exclusions.update(archive_exclusions)
            for record in archive_records:
                merge_record(record)
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
            exclusions["patch:" + type(error).__name__ + ":" + str(error)] += 1
            continue
        merge_record(record)
        accepted_patches += 1
    records = [records_by_id[key] for key in sorted(records_by_id)]
    sources, split = _assign_sources_and_splits(records)
    totals = {
        "archive_count": len(archive_audit),
        "archive_episode_members": sum(item["episode_members"] for item in archive_audit),
        "unique_winning_episodes": len(records),
        "duplicate_episode_members": duplicates,
        "accepted_gap_patches": accepted_patches,
        "excluded_episode_members": sum(exclusions.values()),
        "source_count": len(sources),
        "deck_count": len({record["deck_sha256"] for record in records}),
        "compressed_archive_bytes": sum(item["compressed_file_bytes"] for item in archive_audit),
        "declared_episode_bytes": sum(item["declared_episode_bytes"] for item in archive_audit),
    }
    catalog: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contracts": {
            "selection": "unique_completed_decisive_positive_winner_view",
            "deduplication_key": "episode_id",
            "source_identity": "unicode_nfc_exact_team_name",
            "source_id_zero": "reserved_for_neutral_inference",
            "deck_hash": "sha256(sorted_card_ids_joined_by_comma)",
        },
        "archives": archive_audit,
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


def write_catalog(catalog: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(_canonical_bytes(catalog))
    partial.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument("--start-date", default="2026-07-10")
    parser.add_argument("--end-date", default="2026-07-28")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--patches-dir", type=Path, default=Path("data/raw/episodes/patches"))
    arguments = parser.parse_args()
    archives = [
        path
        for path in sorted(arguments.archives_dir.glob("*.zip"))
        if arguments.start_date <= _archive_date(path) <= arguments.end_date
    ]
    patches = [
        path
        for path in sorted(arguments.patches_dir.rglob("*.json"))
        if arguments.start_date <= _path_date(path) <= arguments.end_date
    ] if arguments.patches_dir.is_dir() else []
    catalog = scan_archives(archives, patches, workers=arguments.workers)
    write_catalog(catalog, arguments.output)
    print(json.dumps(catalog["totals"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
