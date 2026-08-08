"""Build the immutable recent 10k dual-perspective Value Episode catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import uuid
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .archetypes import ArchetypeContract, DEFAULT_CONTRACT
from .replay_contract import DeckManifest, registration_decks


SCHEMA_VERSION = "0036_value_episode_catalog_v1"
DEFAULT_DATE_QUOTAS = {
    "2026-08-06": 2_500,
    "2026-08-05": 2_500,
    "2026-08-04": 2_500,
    "2026-08-03": 2_500,
}
_DATE = re.compile(r"(20\d\d-\d\d-\d\d)")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_rank(*values: object) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _exact_deck_hash(deck: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(card) for card in sorted(deck)).encode("ascii")
    ).hexdigest()


def _episode_id(payload: Mapping[str, Any]) -> int:
    info = payload.get("info")
    value = info.get("EpisodeId") if isinstance(info, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("EpisodeId is absent")
    return value


def _winner(payload: Mapping[str, Any]) -> int:
    rewards, statuses = payload.get("rewards"), payload.get("statuses")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 2
        or not isinstance(statuses, list)
        or statuses != ["DONE", "DONE"]
    ):
        raise ValueError("Episode is not a completed two-player terminal")
    values = []
    for reward in rewards:
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("Episode reward is invalid")
        values.append(float(reward))
    winners = [index for index, reward in enumerate(values) if reward > 0]
    losers = [index for index, reward in enumerate(values) if reward < 0]
    if len(winners) != 1 or len(losers) != 1:
        raise ValueError("0036 small dataset requires one winner and one loser")
    return winners[0]


def _first_player(payload: Mapping[str, Any]) -> int:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Episode steps are absent")
    for step in steps:
        if not isinstance(step, list):
            continue
        for member in step:
            observation = member.get("observation") if isinstance(member, Mapping) else None
            current = observation.get("current") if isinstance(observation, Mapping) else None
            value = current.get("firstPlayer") if isinstance(current, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
                return value
    raise ValueError("first player is absent")


def _terminal_prize_counts(payload: Mapping[str, Any]) -> tuple[int, int]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Episode steps are absent")
    found: dict[int, tuple[int, int]] = {}
    for step in reversed(steps):
        if not isinstance(step, list):
            continue
        for member in step:
            observation = member.get("observation") if isinstance(member, Mapping) else None
            current = observation.get("current") if isinstance(observation, Mapping) else None
            actor = current.get("yourIndex") if isinstance(current, Mapping) else None
            players = current.get("players") if isinstance(current, Mapping) else None
            if actor not in (0, 1) or not isinstance(players, list) or len(players) != 2:
                continue
            counts = []
            for player in players:
                prize = player.get("prize") if isinstance(player, Mapping) else None
                if not isinstance(prize, list) or not 0 <= len(prize) <= 6:
                    raise ValueError("terminal Prize count is invalid")
                counts.append(len(prize))
            found[int(actor)] = (counts[0], counts[1])
        if len(found) == 2:
            break
    if not found:
        raise ValueError("terminal Prize observations are absent")
    values = set(found.values())
    if len(values) != 1:
        raise ValueError("terminal player perspectives disagree on Prize counts")
    return next(iter(values))


def inspect_episode(
    raw: bytes,
    *,
    date: str,
    archive: Path,
    member: str,
    contract: ArchetypeContract,
) -> dict[str, Any]:
    payload = json.loads(raw)
    episode_id = _episode_id(payload)
    winner = _winner(payload)
    decks = registration_decks(payload)
    manifests = [DeckManifest.from_card_ids(deck) for deck in decks]
    terminal_prizes = _terminal_prize_counts(payload)
    first_player = _first_player(payload)
    info = payload.get("info")
    teams = info.get("TeamNames") if isinstance(info, Mapping) else None
    if not isinstance(teams, list) or len(teams) != 2:
        teams = [None, None]
    classes = [contract.classify(deck) for deck in decks]
    trajectories = []
    for actor in (0, 1):
        won = actor == winner
        final_diff = terminal_prizes[1 - actor] if won else -terminal_prizes[actor]
        if not -6 <= final_diff <= 6:
            raise ValueError("signed terminal Prize difference is outside [-6,6]")
        trajectories.append(
            {
                "player_index": actor,
                "terminal_outcome": "win" if won else "loss",
                "value_target": 1 if won else 0,
                "final_diff": final_diff,
                "final_diff_target": final_diff + 6,
                "opponent_archetype_target": classes[1 - actor],
                "opponent_archetype_name": contract.name(classes[1 - actor]),
                "own_deck_sha256": manifests[actor].sha256,
                "opponent_deck_sha256": manifests[1 - actor].sha256,
                "team_name": teams[actor],
                "opponent_team_name": teams[1 - actor],
            }
        )
    return {
        "episode_id": episode_id,
        "episode_date": date,
        "first_player": first_player,
        "winner": winner,
        "terminal_prize_counts": list(terminal_prizes),
        "deck_counts": [
            [list(item) for item in manifest.counts] for manifest in manifests
        ],
        "exact_deck_sha256": [_exact_deck_hash(deck) for deck in decks],
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "locator": {"kind": "zip_member", "archive": str(archive), "member": member},
        "trajectories": trajectories,
    }


def _scan_archive(task: tuple[str, str, str]) -> tuple[str, list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    date, archive_text, contract_text = task
    archive = Path(archive_text)
    contract = ArchetypeContract.load(Path(contract_text))
    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not member.endswith(".json"):
                continue
            raw = bundle.read(member)
            try:
                rows.append(inspect_episode(raw, date=date, archive=archive, member=member, contract=contract))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                exclusions[type(exc).__name__ + ":" + str(exc)] += 1
    commitment = {
        "date": date,
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
        "qualified": len(rows),
    }
    return date, rows, dict(exclusions), commitment


def _assign_splits(rows: list[dict[str, Any]], validation_count: int) -> None:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair = sorted(
            trajectory["opponent_archetype_target"] for trajectory in row["trajectories"]
        )
        groups[(row["episode_date"], pair[0], pair[1])].append(row)
    base = {key: len(values) * validation_count // len(rows) for key, values in groups.items()}
    remaining = validation_count - sum(base.values())
    order = sorted(
        groups,
        key=lambda key: (-(len(groups[key]) * validation_count % len(rows)), key),
    )
    for key in order[:remaining]:
        base[key] += 1
    for key, values in groups.items():
        ranked = sorted(
            values,
            key=lambda row: _hash_rank("0036_split_v1", row["episode_id"], row["payload_sha256"]),
        )
        for index, row in enumerate(ranked):
            row["split"] = "validation" if index < base[key] else "train"
    _ensure_validation_class_coverage(rows)


def _row_classes(row: Mapping[str, Any]) -> set[int]:
    return {int(item["opponent_archetype_target"]) for item in row["trajectories"]}


def _ensure_validation_class_coverage(rows: list[dict[str, Any]]) -> None:
    present = set().union(*(_row_classes(row) for row in rows))
    validation_counts: Counter[int] = Counter(
        class_id
        for row in rows
        if row["split"] == "validation"
        for class_id in _row_classes(row)
    )
    for missing in sorted(present - set(validation_counts)):
        candidates = sorted(
            (row for row in rows if row["split"] == "train" and missing in _row_classes(row)),
            key=lambda row: _hash_rank("0036_class_coverage_v2", missing, row["episode_id"], row["payload_sha256"]),
        )
        if not candidates:
            raise ValueError(f"0036 cannot place class {missing} into validation")
        incoming = candidates[0]
        donors = []
        for row in rows:
            if row["split"] != "validation" or row["episode_date"] != incoming["episode_date"]:
                continue
            classes = _row_classes(row)
            if all(validation_counts[class_id] > 1 for class_id in classes):
                donors.append(row)
        if not donors:
            raise ValueError(f"0036 cannot preserve validation coverage while swapping class {missing}")
        outgoing = min(
            donors,
            key=lambda row: _hash_rank("0036_class_coverage_donor_v2", missing, row["episode_id"], row["payload_sha256"]),
        )
        incoming["split"] = "validation"
        outgoing["split"] = "train"
        for class_id in _row_classes(incoming):
            validation_counts[class_id] += 1
        for class_id in _row_classes(outgoing):
            validation_counts[class_id] -= 1


def build_catalog(
    archives_dir: Path,
    output: Path,
    *,
    date_quotas: Mapping[str, int] = DEFAULT_DATE_QUOTAS,
    validation_count: int = 1_000,
    archetype_contract: Path = DEFAULT_CONTRACT,
    archive_workers: int = 4,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite catalog: {output}")
    contract = ArchetypeContract.load(archetype_contract)
    candidates: dict[str, list[dict[str, Any]]] = {}
    exclusions: Counter[str] = Counter()
    archive_commitments = []
    tasks = []
    for date in sorted(date_quotas, reverse=True):
        archive = archives_dir / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"required recent Episode archive is missing: {archive}")
        tasks.append((date, str(archive), str(archetype_contract)))
    if archive_workers < 1:
        raise ValueError("archive_workers must be positive")
    with ProcessPoolExecutor(max_workers=min(archive_workers, len(tasks))) as pool:
        for date, rows, local_exclusions, commitment in pool.map(_scan_archive, tasks):
            candidates[date] = rows
            exclusions.update(local_exclusions)
            archive_commitments.append(commitment)
    archive_commitments.sort(key=lambda row: row["date"], reverse=True)
    selected = []
    for date, quota in date_quotas.items():
        values = candidates[date]
        if len(values) < quota:
            raise ValueError(f"{date} has {len(values)} qualified Episodes, below quota {quota}")
        values.sort(key=lambda row: _hash_rank("0036_select_v1", date, row["episode_id"], row["payload_sha256"]))
        selected.extend(values[:quota])
    if len({row["episode_id"] for row in selected}) != len(selected):
        raise ValueError("selected catalog contains duplicate Episode IDs")
    if not 0 < validation_count < len(selected):
        raise ValueError("validation_count must be inside selected Episode count")
    _assign_splits(selected, validation_count)
    selected.sort(key=lambda row: (row["episode_date"], row["episode_id"]))
    split_counts = Counter(row["split"] for row in selected)
    trajectory_classes = Counter()
    for row in selected:
        for trajectory in row["trajectories"]:
            trajectory_classes[(row["split"], trajectory["opponent_archetype_target"])] += 1
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "selection": {
            "algorithm": "recent_date_quota_then_sha256_rank_v1",
            "date_quotas": dict(date_quotas),
            "selected_episodes": len(selected),
            "player_trajectories": 2 * len(selected),
        },
        "split": {
            "algorithm": "date_and_unordered_archetype_pair_stratified_sha256_class_coverage_v2",
            "group_unit": "whole_episode",
            "train_episodes": split_counts["train"],
            "validation_episodes": split_counts["validation"],
        },
        "archetype_contract": {
            "path": str(archetype_contract),
            "sha256": _sha256(archetype_contract),
            "schema_version": contract.schema_version,
        },
        "archives": archive_commitments,
        "exclusions": [{"reason": key, "episodes": value} for key, value in sorted(exclusions.items())],
        "trajectory_class_counts": [
            {"split": split, "class_id": class_id, "trajectories": count}
            for (split, class_id), count in sorted(trajectory_classes.items())
        ],
        "episodes": selected,
    }
    result["catalog_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(_canonical(result))
    os.replace(temporary, output)
    return result


def rewrite_catalog_splits(source: Path, output: Path, validation_count: int = 1_000) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite catalog: {output}")
    result = json.loads(source.read_text(encoding="utf-8"))
    verify_catalog(source)
    episodes = result["episodes"]
    _assign_splits(episodes, validation_count)
    split_counts = Counter(row["split"] for row in episodes)
    trajectory_classes = Counter(
        (row["split"], trajectory["opponent_archetype_target"])
        for row in episodes
        for trajectory in row["trajectories"]
    )
    result["split"] = {
        "algorithm": "date_and_unordered_archetype_pair_stratified_sha256_class_coverage_v2",
        "group_unit": "whole_episode",
        "train_episodes": split_counts["train"],
        "validation_episodes": split_counts["validation"],
    }
    result["trajectory_class_counts"] = [
        {"split": split, "class_id": class_id, "trajectories": count}
        for (split, class_id), count in sorted(trajectory_classes.items())
    ]
    result.pop("catalog_sha256", None)
    result["catalog_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(_canonical(result))
    os.replace(temporary, output)
    return result


def verify_catalog(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != "complete":
        raise ValueError("0036 catalog schema/status mismatch")
    expected_digest = result.get("catalog_sha256")
    commitment = dict(result)
    commitment.pop("catalog_sha256", None)
    if hashlib.sha256(_canonical(commitment)).hexdigest() != expected_digest:
        raise ValueError("0036 catalog commitment mismatch")
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != result["selection"]["selected_episodes"]:
        raise ValueError("0036 catalog Episode count mismatch")
    if len({row["episode_id"] for row in episodes}) != len(episodes):
        raise ValueError("0036 catalog Episode IDs are not unique")
    split_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    for row in episodes:
        split_counts[row["split"]] += 1
        date_counts[row["episode_date"]] += 1
        trajectories = row.get("trajectories")
        if not isinstance(trajectories, list) or len(trajectories) != 2:
            raise ValueError("0036 catalog must retain exactly two perspectives")
        if {item["player_index"] for item in trajectories} != {0, 1}:
            raise ValueError("0036 catalog player perspectives are invalid")
        if {item["value_target"] for item in trajectories} != {0, 1}:
            raise ValueError("0036 catalog outcome targets are not balanced by Episode")
        if trajectories[0]["final_diff"] != -trajectories[1]["final_diff"]:
            raise ValueError("0036 catalog signed final diff is not perspective symmetric")
        for item in trajectories:
            if not 0 <= item["opponent_archetype_target"] < 15:
                raise ValueError("0036 catalog archetype target is outside [0,14]")
            if item["final_diff_target"] != item["final_diff"] + 6:
                raise ValueError("0036 catalog final diff encoding mismatch")
            outcome_counts[item["terminal_outcome"]] += 1
    expected_dates = Counter(result["selection"]["date_quotas"])
    if date_counts != expected_dates:
        raise ValueError(f"0036 catalog date quotas mismatch: {dict(date_counts)}")
    if split_counts != Counter({
        "train": result["split"]["train_episodes"],
        "validation": result["split"]["validation_episodes"],
    }):
        raise ValueError("0036 catalog split counts mismatch")
    if outcome_counts["win"] != len(episodes) or outcome_counts["loss"] != len(episodes):
        raise ValueError("0036 catalog win/loss perspective counts mismatch")
    return {
        "status": "verified",
        "catalog_sha256": expected_digest,
        "episodes": len(episodes),
        "trajectories": 2 * len(episodes),
        "date_counts": dict(sorted(date_counts.items())),
        "split_counts": dict(split_counts),
        "outcome_counts": dict(outcome_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", type=Path, default=Path("data/raw/episodes/archives"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archetype-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--validation-count", type=int, default=1_000)
    parser.add_argument("--archive-workers", type=int, default=4)
    args = parser.parse_args()
    result = build_catalog(
        args.archives_dir,
        args.output,
        validation_count=args.validation_count,
        archetype_contract=args.archetype_contract,
        archive_workers=args.archive_workers,
    )
    print(json.dumps({
        "status": result["status"],
        "catalog_sha256": result["catalog_sha256"],
        "selection": result["selection"],
        "split": result["split"],
        "archives": result["archives"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_DATE_QUOTAS", "SCHEMA_VERSION", "build_catalog", "inspect_episode", "rewrite_catalog_splits", "verify_catalog"]
