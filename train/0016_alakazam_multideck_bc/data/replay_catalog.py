"""Build the immutable 0016 replay catalog without materializing decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date as calendar_date
from pathlib import Path
from typing import Any


BATTLE_CAGE_ID = 1264
WONDROUS_PATCH_ID = 1146
_MEMBER_RE = re.compile(r"(?:episode-)?(\d+)(?:-replay)?\.json")
_ARCHIVE_DATE_RE = re.compile(r".*?(\d{4}-\d{2}-\d{2})\.zip")
_SPLIT_VERSION = "0016_source_seat_split_v1"
_SPLIT_SEED = 20260727


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _deck_hash(cards: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(card_id) for card_id in sorted(cards)).encode("ascii")
    ).hexdigest()


def _registration_decks(payload: dict[str, Any]) -> list[list[int]]:
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("replay steps are absent")

    decks: object | None = None
    first_step = steps[0]
    if isinstance(first_step, list):
        for agent in first_step:
            visualize = agent.get("visualize") if isinstance(agent, dict) else None
            if not isinstance(visualize, list):
                continue
            for frame in visualize:
                action = frame.get("action") if isinstance(frame, dict) else None
                if isinstance(action, list):
                    decks = action
                    break
            if decks is not None:
                break
    if decks is None and len(steps) > 1 and isinstance(steps[1], list):
        decks = [
            agent.get("action") if isinstance(agent, dict) else None
            for agent in steps[1]
        ]
    if not isinstance(decks, list):
        raise ValueError("registration decks are absent")

    result: list[list[int]] = []
    for deck in decks:
        if (
            not isinstance(deck, list)
            or len(deck) != 60
            or any(isinstance(card, bool) or not isinstance(card, int) for card in deck)
        ):
            raise ValueError("registration deck must contain exactly 60 integer IDs")
        result.append(deck)
    return result


def _outcomes(rewards: object) -> list[str]:
    if not isinstance(rewards, list) or not rewards:
        raise ValueError("replay rewards are absent")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in rewards
    ):
        raise ValueError("replay rewards must be finite numbers")
    maximum = max(float(value) for value in rewards)
    if sum(float(value) == maximum for value in rewards) != 1:
        return ["draw"] * len(rewards)
    return ["win" if float(value) == maximum else "loss" for value in rewards]


def _episode_id(payload: dict[str, Any]) -> int:
    info = payload.get("info")
    if not isinstance(info, dict):
        raise ValueError("replay info is absent")
    value = info.get("EpisodeId")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("replay EpisodeId is absent")
    return value


def _first_player(payload: dict[str, Any], player_count: int) -> int:
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
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 <= value < player_count
                ):
                    evidence.add(value)
    if len(evidence) != 1:
        raise ValueError(f"first-player evidence is ambiguous: {sorted(evidence)}")
    return evidence.pop()


def _iso_date(value: str, *, label: str) -> str:
    candidate = value[:10]
    try:
        parsed = calendar_date.fromisoformat(candidate)
    except ValueError as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    return parsed.isoformat()


def _archive_date(archive_name: str) -> str:
    match = _ARCHIVE_DATE_RE.fullmatch(Path(archive_name).name)
    if match is None:
        raise ValueError(f"archive filename has no date: {archive_name}")
    return _iso_date(match.group(1), label="archive date")


def _locator_date(path: Path) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(path))
    if match is None:
        raise ValueError(f"replay locator has no date: {path}")
    return _iso_date(match.group(0), label="locator date")


def _episode_record(
    payload: dict[str, Any],
    *,
    expected_id: int,
    expected_team: str,
    player_index: int,
    episode_date: str,
    payload_sha256: str,
    locator: dict[str, str],
) -> dict[str, Any]:
    if _episode_id(payload) != expected_id:
        raise ValueError(f"episode ID mismatch for {expected_id}")
    info = payload["info"]
    teams = info.get("TeamNames")
    rewards = payload.get("rewards")
    statuses = payload.get("statuses")
    decks = _registration_decks(payload)
    if not (
        isinstance(teams, list)
        and isinstance(rewards, list)
        and isinstance(statuses, list)
        and len(teams) == len(rewards) == len(statuses) == len(decks)
    ):
        raise ValueError(f"player dimensions disagree for episode {expected_id}")
    if player_index < 0 or player_index >= len(teams):
        raise ValueError(f"player index is invalid for episode {expected_id}")
    if statuses[player_index] != "DONE":
        raise ValueError(f"selected player is not DONE in episode {expected_id}")
    team_name = teams[player_index]
    if not isinstance(team_name, str) or team_name != expected_team:
        raise ValueError(f"exact team name mismatch for episode {expected_id}")
    reward = rewards[player_index]
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or reward <= 0
    ):
        raise ValueError(f"selected player did not win episode {expected_id}")
    outcome = "win"
    deck = decks[player_index]
    counts = Counter(deck)
    first_player = _first_player(payload, len(teams))
    return {
        "episode_id": expected_id,
        "episode_date": _iso_date(episode_date, label="episode date"),
        "team_name": team_name,
        "player_index": player_index,
        "first_player": first_player,
        "seat": "first" if player_index == first_player else "second",
        "outcome": outcome,
        "payload_sha256": payload_sha256,
        "deck_sha256": _deck_hash(deck),
        "deck_counts": [list(item) for item in sorted(counts.items())],
        "battle_cage_count": counts[BATTLE_CAGE_ID],
        "wondrous_patch_count": counts[WONDROUS_PATCH_ID],
        "locator": locator,
    }


def _member_index(bundle: zipfile.ZipFile) -> dict[int, str]:
    result: dict[int, str] = {}
    for item in bundle.infolist():
        if item.is_dir():
            continue
        match = _MEMBER_RE.fullmatch(Path(item.filename).name)
        if match is None:
            continue
        episode_id = int(match.group(1))
        if episode_id in result:
            raise ValueError(f"duplicate episode member {episode_id} in archive")
        result[episode_id] = item.filename
    return result


def _validate_audit(audit: object) -> list[dict[str, Any]]:
    if not isinstance(audit, dict) or not isinstance(audit.get("episodes"), list):
        raise ValueError("archive audit has no episodes")
    if audit.get("errors"):
        raise ValueError("archive audit contains errors")
    episodes: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in audit["episodes"]:
        if not isinstance(item, dict):
            raise ValueError("archive audit episode is not an object")
        episode_id = item.get("episode_id")
        team_name = item.get("team_name")
        player_index = item.get("player_index")
        archives = item.get("archives")
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or not isinstance(team_name, str)
            or isinstance(player_index, bool)
            or not isinstance(player_index, int)
            or not isinstance(archives, list)
            or not archives
            or any(not isinstance(name, str) for name in archives)
        ):
            raise ValueError("archive audit episode has invalid fields")
        if episode_id in seen:
            raise ValueError(f"archive audit repeats episode {episode_id}")
        seen.add(episode_id)
        episodes.append(item)
    return episodes


def _load_archive_group(
    arguments: tuple[str, list[dict[str, Any]]]
) -> dict[int, dict[str, Any]]:
    archive_name, items = arguments
    archive_path = Path(archive_name)
    result: dict[int, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as bundle:
        members = _member_index(bundle)
        for item in sorted(items, key=lambda row: row["episode_id"]):
            episode_id = item["episode_id"]
            member = members.get(episode_id)
            if member is None:
                raise ValueError(f"episode {episode_id} is absent from {archive_path.name}")
            raw = bundle.read(member)
            record = _episode_record(
                json.loads(raw),
                expected_id=episode_id,
                expected_team=item["team_name"],
                player_index=item["player_index"],
                episode_date=_archive_date(archive_path.name),
                payload_sha256=hashlib.sha256(raw).hexdigest(),
                locator={
                    "kind": "zip_member",
                    "archive": str(archive_path),
                    "member": member,
                },
            )
            for field in ("battle_cage_count", "wondrous_patch_count"):
                expected = item.get(field)
                if expected is not None and record[field] != expected:
                    raise ValueError(
                        f"{field} disagrees for episode {episode_id}: "
                        f"{record[field]} != {expected}"
                    )
            result[episode_id] = record
    return result


def _load_archive_records(
    audit_episodes: list[dict[str, Any]], archives_dir: Path
) -> dict[int, dict[str, Any]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for item in audit_episodes:
        archive_path = archives_dir / min(item["archives"])
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        selected.setdefault(str(archive_path), []).append(item)
    arguments = sorted(selected.items())
    if len(arguments) == 1:
        groups = [_load_archive_group(arguments[0])]
    else:
        with ProcessPoolExecutor(max_workers=min(8, len(arguments))) as pool:
            groups = list(pool.map(_load_archive_group, arguments))
    result: dict[int, dict[str, Any]] = {}
    for group in groups:
        overlap = set(result).intersection(group)
        if overlap:
            raise ValueError(f"archive groups repeat Episodes: {sorted(overlap)[:3]}")
        result.update(group)
    return result


def _load_goonew_records(manifest_path: Path) -> dict[int, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list):
        raise ValueError("goonew manifest has no episodes")
    if manifest.get("errors"):
        raise ValueError("goonew manifest contains errors")
    team_name = manifest.get("team_name")
    if not isinstance(team_name, str):
        raise ValueError("goonew manifest has no exact team name")
    manifest_date_value = manifest.get("captured_at_utc") or manifest.get(
        "selection_snapshot"
    )

    result: dict[int, dict[str, Any]] = {}
    for item in manifest["episodes"]:
        if not isinstance(item, dict):
            raise ValueError("goonew manifest episode is not an object")
        episode_id = item.get("episode_id")
        player_index = item.get("player_index")
        filename = item.get("file")
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or isinstance(player_index, bool)
            or not isinstance(player_index, int)
            or not isinstance(filename, str)
        ):
            raise ValueError("goonew manifest episode has invalid fields")
        if episode_id in result:
            raise ValueError(f"goonew manifest repeats episode {episode_id}")
        replay_path = manifest_path.parent / filename
        raw = replay_path.read_bytes()
        declared_payload_hash = item.get("sha256")
        if declared_payload_hash and hashlib.sha256(raw).hexdigest() != declared_payload_hash:
            raise ValueError(f"payload hash disagrees for episode {episode_id}")
        date_value = item.get("create_time") or manifest_date_value
        if not isinstance(date_value, str):
            date_value = _locator_date(replay_path)
        record = _episode_record(
            json.loads(raw),
            expected_id=episode_id,
            expected_team=team_name,
            player_index=player_index,
            episode_date=_iso_date(date_value, label="goonew episode date"),
            payload_sha256=hashlib.sha256(raw).hexdigest(),
            locator={"kind": "file", "archive": "", "member": str(replay_path)},
        )
        expected_fields = {
            "outcome": item.get("outcome"),
            "deck_sha256": item.get("deck_sha256"),
            "battle_cage_count": item.get("battle_cage_count"),
            "wondrous_patch_count": item.get("wondrous_patch_count"),
        }
        for field, expected in expected_fields.items():
            if expected is not None and record[field] != expected:
                raise ValueError(
                    f"{field} disagrees for episode {episode_id}: "
                    f"{record[field]} != {expected}"
                )
        result[episode_id] = record
    return result


def _merge_records(
    archive_records: dict[int, dict[str, Any]],
    goonew_records: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = dict(archive_records)
    identity_fields = (
        "team_name",
        "player_index",
        "outcome",
        "first_player",
        "seat",
        "deck_sha256",
        "deck_counts",
        "battle_cage_count",
        "wondrous_patch_count",
    )
    for episode_id, direct in goonew_records.items():
        existing = merged.get(episode_id)
        if existing is None:
            merged[episode_id] = direct
            continue
        if any(existing[field] != direct[field] for field in identity_fields):
            raise ValueError(f"duplicate sources disagree for episode {episode_id}")
        existing["alternate_locators"] = [direct["locator"]]
    return [merged[episode_id] for episode_id in sorted(merged)]


def _split_digest(record: dict[str, Any]) -> str:
    value = "|".join(
        (
            _SPLIT_VERSION,
            str(_SPLIT_SEED),
            record["team_name"],
            record["seat"],
            str(record["episode_id"]),
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validation_count(size: int) -> int:
    if size < 1:
        raise ValueError("split stratum must not be empty")
    if size == 1:
        return 0
    half_up_tenth = (size + 5) // 10
    return max(1, min(size - 1, half_up_tenth))


def _assign_splits(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in episodes:
        strata.setdefault((record["team_name"], record["seat"]), []).append(record)

    stratum_audit = []
    for (team_name, seat), members in sorted(strata.items()):
        for record in members:
            record["split_rank_sha256"] = _split_digest(record)
        members.sort(key=lambda row: (row["split_rank_sha256"], row["episode_id"]))
        validation_count = _validation_count(len(members))
        for index, record in enumerate(members):
            record["split"] = "validation" if index < validation_count else "train"
        stratum_audit.append(
            {
                "source_id": members[0]["source_id"],
                "team_name": team_name,
                "seat": seat,
                "episodes": len(members),
                "train": len(members) - validation_count,
                "validation": validation_count,
            }
        )
    episodes.sort(key=lambda row: row["episode_id"])
    validation = sum(record["split"] == "validation" for record in episodes)
    return {
        "algorithm": "source_seat_stratified_sha256_rank_half_up_tenth_v1",
        "seed": _SPLIT_SEED,
        "group_unit": "whole_episode",
        "stratum": "exact_utf8_team_name_x_first_or_second_seat",
        "digest_input": (
            "0016_source_seat_split_v1|20260727|<exact_team_name>|<seat>|<episode_id>"
        ),
        "singleton_rule": "train",
        "counts": {"train": len(episodes) - validation, "validation": validation},
        "strata": stratum_audit,
    }


def build_replay_catalog(
    audit_path: Path | str,
    archives_dir: Path | str,
    goonew_manifest_path: Path | str,
) -> dict[str, Any]:
    """Validate both inventories and return a deterministic episode catalog."""

    audit_file = Path(audit_path)
    archive_root = Path(archives_dir)
    goonew_file = Path(goonew_manifest_path)
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    audit_episodes = _validate_audit(audit)
    archive_records = _load_archive_records(audit_episodes, archive_root)
    goonew_records = _load_goonew_records(goonew_file)
    episodes = _merge_records(archive_records, goonew_records)

    team_names = sorted({record["team_name"] for record in episodes})
    source_ids = {team_name: index for index, team_name in enumerate(team_names, start=1)}
    sources = [
        {"source_id": source_ids[team_name], "team_name": team_name}
        for team_name in team_names
    ]
    for record in episodes:
        record["source_id"] = source_ids[record["team_name"]]
    split = _assign_splits(episodes)

    totals = {
        "archive_episode_count": len(archive_records),
        "goonew_episode_count": len(goonew_records),
        "overlap_count": len(set(archive_records) & set(goonew_records)),
        "episode_count": len(episodes),
        "source_count": len(sources),
        "battle_cage_episode_count": sum(
            record["battle_cage_count"] > 0 for record in episodes
        ),
        "wondrous_patch_episode_count": sum(
            record["wondrous_patch_count"] > 0 for record in episodes
        ),
    }
    vocabulary_sha256 = _canonical_hash(sources)
    return {
        "schema_version": "0016_replay_catalog_v1",
        "contracts": {
            "deduplication_key": "episode_id",
            "source_identity": "exact_utf8_team_name",
            "source_id_zero": "reserved_for_neutral_inference",
            "deck_hash": "sha256(sorted_card_ids_joined_by_comma)",
            "selection": "completed_winning_alakazam_player_views",
        },
        "inputs": {
            "archive_audit": str(audit_file),
            "archives_dir": str(archive_root),
            "goonew_manifest": str(goonew_file),
        },
        "source_vocabulary_sha256": vocabulary_sha256,
        "sources": sources,
        "split": split,
        "totals": totals,
        "episodes": episodes,
        "catalog_sha256": _canonical_hash(
            {
                "sources": sources,
                "split": split,
                "totals": totals,
                "episodes": episodes,
            }
        ),
    }


def write_replay_catalog(catalog: dict[str, Any], output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument("--goonew-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = build_replay_catalog(
        args.audit, args.archives_dir, args.goonew_manifest
    )
    write_replay_catalog(catalog, args.output)
    print(json.dumps({"output": str(args.output), **catalog["totals"]}))


if __name__ == "__main__":
    main()
