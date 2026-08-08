"""Recency-weighted focal catalog for players whose deck contains Dragapult ex."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import uuid
import zipfile
from collections import Counter
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import date as calendar_date
from functools import lru_cache
from pathlib import Path
from typing import Any

from .archetypes import ArchetypeContract
from .catalog import _exact_deck_hash
from .replay_contract import DeckManifest


SCHEMA_VERSION = "0036_dragapult_focal_catalog_v2"
DRAGAPULT_EX_CARD_ID = 121
EXACT_007_SHA256 = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
DEFAULT_EXACT_007 = (
    Path(__file__).resolve().parents[3]
    / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks"
    / "dragapult_ex_07bedfffbfad/deck.csv"
)


def _json_array_after(raw: bytes, key: bytes, *, start: int = 0) -> tuple[Any, int]:
    position = raw.find(key, start)
    if position < 0:
        raise ValueError(f"missing replay field {key.decode(errors='replace')}")
    position = raw.find(b":", position + len(key)) + 1
    while position < len(raw) and raw[position] in b" \r\n\t":
        position += 1
    if position >= len(raw) or raw[position] != ord("["):
        raise ValueError(f"replay field {key.decode(errors='replace')} is not an array")
    depth = 0
    in_string = False
    escaped = False
    for index in range(position, len(raw)):
        value = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                in_string = False
            continue
        if value == ord('"'):
            in_string = True
        elif value == ord("["):
            depth += 1
        elif value == ord("]"):
            depth -= 1
            if depth == 0:
                return json.loads(raw[position:index + 1]), index + 1
    raise ValueError(f"unterminated replay array {key.decode(errors='replace')}")


def _positive_int_after(raw: bytes, key: bytes) -> int:
    match = re.search(re.escape(key) + rb"\s*:\s*([1-9][0-9]*)", raw)
    if match is None:
        raise ValueError(f"missing positive replay integer {key.decode(errors='replace')}")
    return int(match.group(1))


def _first_player(raw: bytes) -> int | None:
    match = re.search(rb'"firstPlayer"\s*:\s*([01])', raw)
    return int(match.group(1)) if match is not None else None


def _deck(path: Path) -> list[int]:
    cards = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    DeckManifest.from_card_ids(cards)
    if _exact_deck_hash(cards) != EXACT_007_SHA256:
        raise ValueError("0036 exact-007 deck commitment mismatch")
    return cards


@lru_cache(maxsize=4)
def _exact_counter(path: str) -> Counter[int]:
    return Counter(_deck(Path(path)))


def inspect_registration_header(
    raw: bytes,
    *,
    date: str,
    archive: Path,
    member: str,
    contract: ArchetypeContract,
    exact_007_path: Path = DEFAULT_EXACT_007,
) -> dict[str, Any]:
    """Inspect terminal/header/registration facts without decoding multi-megabyte steps."""
    episode_id = _positive_int_after(raw, b'"EpisodeId"')
    rewards, _ = _json_array_after(raw, b'"rewards"')
    statuses, _ = _json_array_after(raw, b'"statuses"')
    visualize = raw.find(b'"visualize"')
    if visualize < 0:
        raise ValueError("registration visualize frame is absent")
    decks, _ = _json_array_after(raw, b'"action"', start=visualize)
    if statuses != ["DONE", "DONE"] or not isinstance(rewards, list) or len(rewards) != 2:
        raise ValueError("Episode is not a completed two-player terminal")
    winners = [actor for actor, reward in enumerate(rewards) if reward == 1]
    losers = [actor for actor, reward in enumerate(rewards) if reward == -1]
    if len(winners) != 1 or len(losers) != 1:
        raise ValueError("0036 focal corpus requires one winner and one loser")
    if not isinstance(decks, list) or len(decks) != 2:
        raise ValueError("registration decks must contain two players")
    manifests = [DeckManifest.from_card_ids(deck) for deck in decks]
    exact_cards = _exact_counter(str(exact_007_path.resolve()))
    classes = [contract.classify(deck) for deck in decks]
    trajectories = []
    for actor, deck in enumerate(decks):
        if DRAGAPULT_EX_CARD_ID not in deck:
            continue
        won = actor == winners[0]
        trajectories.append({
            "player_index": actor,
            "terminal_outcome": "win" if won else "loss",
            "value_target": int(won),
            "is_exact_007": Counter(deck) == exact_cards,
            "own_exact_deck_sha256": _exact_deck_hash(deck),
            "own_deck_sha256": manifests[actor].sha256,
            "opponent_deck_sha256": manifests[1 - actor].sha256,
            "opponent_archetype_target": classes[1 - actor],
            "opponent_archetype_name": contract.name(classes[1 - actor]),
        })
    if not trajectories:
        raise ValueError("Episode has no Dragapult focal actor")
    return {
        "episode_id": episode_id,
        "episode_date": date,
        "first_player": _first_player(raw),
        "winner": winners[0],
        "deck_counts": [[list(item) for item in manifest.counts] for manifest in manifests],
        "exact_deck_sha256": [_exact_deck_hash(deck) for deck in decks],
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "locator": {"kind": "zip_member", "archive": str(archive), "member": member},
        "trajectories": trajectories,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_rank(*values: object) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode()).hexdigest()


def _scan_archive(task: tuple[str, str, str, str]) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    date, archive_text, contract_text, exact_text = task
    archive = Path(archive_text)
    contract = ArchetypeContract.load(Path(contract_text))
    rows = []
    exclusions: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.endswith(".json")]
        for member in members:
            raw = bundle.read(member)
            try:
                rows.append(inspect_registration_header(
                    raw, date=date, archive=archive, member=member,
                    contract=contract, exact_007_path=Path(exact_text),
                ))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                reason = str(exc)
                key = "no_dragapult_focal_actor" if reason == "Episode has no Dragapult focal actor" else type(exc).__name__ + ":" + reason
                exclusions[key] += 1
    return rows, dict(exclusions), {
        "date": date,
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
        "episode_json_members": len(members),
        "dragapult_episodes": len(rows),
        "dragapult_focal_trajectories": sum(len(row["trajectories"]) for row in rows),
    }


def _select_weighted_dates(
    rows: list[dict[str, Any]],
    target_episodes: int,
    *,
    uniform_fraction: float = 0.25,
    recency_half_life_days: float = 7.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select unique Episodes across every date, with a deterministic recent bias."""
    if target_episodes < 1:
        raise ValueError("0036 focal Episode target must be positive")
    if not 0 <= uniform_fraction <= 1 or recency_half_life_days <= 0:
        raise ValueError("0036 focal date-weight contract is invalid")
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["episode_date"]].append(row)
    dates = sorted(by_date)
    selected_target = min(target_episodes, len(rows))
    if selected_target < len(dates):
        raise ValueError("0036 focal target is too small to cover every available date")

    newest = calendar_date.fromisoformat(dates[-1])
    recent_raw = {
        value: math.exp(
            -math.log(2.0)
            * (newest - calendar_date.fromisoformat(value)).days
            / recency_half_life_days
        )
        for value in dates
    }
    recent_total = sum(recent_raw.values())
    weights = {
        value: uniform_fraction / len(dates)
        + (1.0 - uniform_fraction) * recent_raw[value] / recent_total
        for value in dates
    }

    # One Episode per available date is mandatory. Remaining capacity is assigned
    # by capped weighted largest-remainder allocation, so sparse dates donate quota.
    allocations = {value: 1 for value in dates}
    remaining = selected_target - len(dates)
    while remaining:
        active = [value for value in dates if allocations[value] < len(by_date[value])]
        if not active:
            raise AssertionError("0036 focal allocation exhausted before its target")
        active_weight = sum(weights[value] for value in active)
        raw = {value: remaining * weights[value] / active_weight for value in active}
        grants = {
            value: min(
                len(by_date[value]) - allocations[value],
                math.floor(raw[value]),
            )
            for value in active
        }
        granted = sum(grants.values())
        if granted:
            for value, amount in grants.items():
                allocations[value] += amount
            remaining -= granted
            continue
        chosen = max(
            active,
            key=lambda value: (raw[value] - math.floor(raw[value]), weights[value], value),
        )
        allocations[chosen] += 1
        remaining -= 1

    selected = []
    allocation_audit = []
    for value in dates:
        ranked = sorted(
            by_date[value],
            key=lambda row: _hash_rank("0036_dragapult_select_v1", row["episode_id"], row["payload_sha256"]),
        )
        selected.extend(ranked[:allocations[value]])
        allocation_audit.append({
            "date": value,
            "available_episodes": len(ranked),
            "selected_episodes": allocations[value],
            "normalized_weight": weights[value],
            "age_days": (newest - calendar_date.fromisoformat(value)).days,
        })
    if len(selected) != selected_target or len({row["episode_id"] for row in selected}) != len(selected):
        raise AssertionError("0036 focal weighted selection is not unique or exact")
    return selected, allocation_audit


def _terminal_own_prize_counts(payload: Mapping[str, Any]) -> tuple[int, int]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Episode steps are absent")
    found: dict[int, int] = {}
    for step in reversed(steps):
        if not isinstance(step, list):
            continue
        for member in step:
            observation = member.get("observation") if isinstance(member, Mapping) else None
            current = observation.get("current") if isinstance(observation, Mapping) else None
            actor = current.get("yourIndex") if isinstance(current, Mapping) else None
            players = current.get("players") if isinstance(current, Mapping) else None
            if actor not in (0, 1) or actor in found or not isinstance(players, list) or len(players) != 2:
                continue
            own = players[actor]
            prize = own.get("prize") if isinstance(own, Mapping) else None
            if not isinstance(prize, list) or not 0 <= len(prize) <= 6:
                raise ValueError("terminal own Prize count is invalid")
            found[int(actor)] = len(prize)
        if len(found) == 2:
            break
    if set(found) != {0, 1}:
        raise ValueError("terminal own Prize observations are absent")
    return found[0], found[1]


def _hydrate_archive(task: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    archive_text, rows = task
    with zipfile.ZipFile(archive_text) as bundle:
        for row in rows:
            raw = bundle.read(row["locator"]["member"])
            if hashlib.sha256(raw).hexdigest() != row["payload_sha256"]:
                raise ValueError(f"0036 focal payload hash mismatch: {row['episode_id']}")
            own_prizes = _terminal_own_prize_counts(json.loads(raw))
            row["terminal_own_prize_counts"] = list(own_prizes)
            for trajectory in row["trajectories"]:
                actor = trajectory["player_index"]
                won = bool(trajectory["value_target"])
                diff = own_prizes[1 - actor] if won else -own_prizes[actor]
                trajectory["final_diff"] = diff
                trajectory["final_diff_target"] = diff + 6
    return rows


def _assign_splits(rows: list[dict[str, Any]], validation_count: int) -> None:
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        outcomes = tuple(sorted(item["value_target"] for item in row["trajectories"]))
        exact = tuple(sorted(item["is_exact_007"] for item in row["trajectories"]))
        opponents = tuple(sorted(item["opponent_archetype_target"] for item in row["trajectories"]))
        groups[(row["episode_date"], outcomes, exact, opponents)].append(row)
    base = {key: len(values) * validation_count // len(rows) for key, values in groups.items()}
    remaining = validation_count - sum(base.values())
    order = sorted(groups, key=lambda key: (-(len(groups[key]) * validation_count % len(rows)), str(key)))
    for key in order[:remaining]:
        base[key] += 1
    for key, values in groups.items():
        ranked = sorted(values, key=lambda row: _hash_rank("0036_dragapult_split_v1", row["episode_id"], row["payload_sha256"]))
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
            key=lambda row: _hash_rank(
                "0036_dragapult_class_coverage_v2", missing,
                row["episode_id"], row["payload_sha256"],
            ),
        )
        if not candidates:
            raise ValueError(f"0036 focal cannot place class {missing} into validation")
        incoming = candidates[0]
        donors = []
        for row in rows:
            if row["split"] != "validation" or row["episode_date"] != incoming["episode_date"]:
                continue
            classes = _row_classes(row)
            if all(validation_counts[class_id] > 1 for class_id in classes):
                donors.append(row)
        if not donors:
            raise ValueError(
                f"0036 focal cannot preserve validation coverage while swapping class {missing}"
            )
        outgoing = min(
            donors,
            key=lambda row: _hash_rank(
                "0036_dragapult_class_coverage_donor_v2", missing,
                row["episode_id"], row["payload_sha256"],
            ),
        )
        incoming["split"] = "validation"
        outgoing["split"] = "train"
        for class_id in _row_classes(incoming):
            validation_counts[class_id] += 1
        for class_id in _row_classes(outgoing):
            validation_counts[class_id] -= 1


def build_dragapult_catalog(
    archives_dir: Path,
    output: Path,
    *,
    target_episodes: int = 10_000,
    validation_fraction: float = 0.1,
    uniform_date_fraction: float = 0.25,
    recency_half_life_days: float = 7.0,
    workers: int = 8,
    archetype_contract: Path | None = None,
    exact_007_path: Path = DEFAULT_EXACT_007,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite 0036 focal catalog: {output}")
    if not 0 < validation_fraction < 1 or workers < 1:
        raise ValueError("0036 focal validation/workers contract is invalid")
    contract_path = archetype_contract or (Path(__file__).resolve().parents[1] / "assets/archetypes_v1.json")
    archives = []
    for archive in archives_dir.glob("pokemon-tcg-ai-battle-episodes-20??-??-??.zip"):
        match = re.search(r"(20\d\d-\d\d-\d\d)", archive.name)
        if match and match.group(1) <= "2026-08-06":
            archives.append((match.group(1), archive))
    if not archives:
        raise FileNotFoundError("no official Episode archives found for 0036 focal scan")
    tasks = [(date, str(path), str(contract_path), str(exact_007_path)) for date, path in sorted(archives, reverse=True)]
    scanned_rows = []
    exclusions: Counter[str] = Counter()
    commitments = []
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        for rows, local_exclusions, commitment in pool.map(_scan_archive, tasks):
            scanned_rows.extend(rows)
            exclusions.update(local_exclusions)
            commitments.append(commitment)
    if len({row["episode_id"] for row in scanned_rows}) != len(scanned_rows):
        raise ValueError("0036 focal archive scan contains duplicate Episode IDs")
    selected, date_allocation = _select_weighted_dates(
        scanned_rows,
        target_episodes,
        uniform_fraction=uniform_date_fraction,
        recency_half_life_days=recency_half_life_days,
    )
    by_archive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_archive[row["locator"]["archive"]].append(row)
    with ProcessPoolExecutor(max_workers=min(workers, len(by_archive))) as pool:
        hydrated_groups = list(pool.map(_hydrate_archive, sorted(by_archive.items())))
    selected = [row for group in hydrated_groups for row in group]
    selected.sort(key=lambda row: (row["episode_date"], row["episode_id"]))
    validation_count = max(1, min(len(selected) - 1, round(len(selected) * validation_fraction)))
    _assign_splits(selected, validation_count)
    focal_count = sum(len(row["trajectories"]) for row in selected)
    exact_count = sum(item["is_exact_007"] for row in selected for item in row["trajectories"])
    validation_exact = sum(
        item["is_exact_007"]
        for row in selected if row["split"] == "validation"
        for item in row["trajectories"]
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "selection": {
            "algorithm": "all_dates_uniform25_recency75_half_life7d_capacity_redistribution_v1",
            "unit": "unique_episode",
            "requested_episodes": target_episodes,
            "selected_episodes": len(selected),
            "episode_shortfall": max(0, target_episodes - len(selected)),
            "uniform_date_fraction": uniform_date_fraction,
            "recency_fraction": 1.0 - uniform_date_fraction,
            "recency_half_life_days": recency_half_life_days,
            "date_allocation": date_allocation,
            "selected_focal_trajectories": focal_count,
            "focal_card_id": DRAGAPULT_EX_CARD_ID,
            "exact_007_sha256": EXACT_007_SHA256,
            "exact_007_focal_trajectories": exact_count,
        },
        "split": {
            "algorithm": "episode_date_outcome_exact007_opponent_stratified_class_coverage_v2",
            "group_unit": "whole_episode",
            "train_episodes": len(selected) - validation_count,
            "validation_episodes": validation_count,
            "validation_exact_007_focal_trajectories": validation_exact,
        },
        "archives": sorted(commitments, key=lambda row: row["date"], reverse=True),
        "exclusions": [{"reason": key, "episodes": value} for key, value in sorted(exclusions.items())],
        "episodes": selected,
    }
    encoded = json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    result["catalog_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return result


def rewrite_dragapult_catalog_splits(
    source: Path,
    output: Path,
    *,
    validation_fraction: float = 0.1,
) -> dict[str, Any]:
    """Rewrite only whole-Episode split labels and preserve selected Episode identities."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite 0036 focal catalog: {output}")
    result = json.loads(source.read_text(encoding="utf-8"))
    verify_dragapult_catalog(source)
    episodes = result["episodes"]
    validation_count = max(1, min(len(episodes) - 1, round(len(episodes) * validation_fraction)))
    _assign_splits(episodes, validation_count)
    validation_exact = sum(
        item["is_exact_007"]
        for row in episodes if row["split"] == "validation"
        for item in row["trajectories"]
    )
    result["split"] = {
        "algorithm": "episode_date_outcome_exact007_opponent_stratified_class_coverage_v2",
        "group_unit": "whole_episode",
        "train_episodes": len(episodes) - validation_count,
        "validation_episodes": validation_count,
        "validation_exact_007_focal_trajectories": validation_exact,
    }
    result.pop("catalog_sha256", None)
    encoded = json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    result["catalog_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return result


def verify_dragapult_catalog(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != "complete":
        raise ValueError("0036 focal catalog schema/status mismatch")
    commitment = dict(result)
    expected = commitment.pop("catalog_sha256")
    encoded = json.dumps(commitment, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    if hashlib.sha256(encoded.encode()).hexdigest() != expected:
        raise ValueError("0036 focal catalog commitment mismatch")
    identities = set()
    focal = exact = 0
    split_episodes: Counter[str] = Counter()
    for row in result["episodes"]:
        split_episodes[row["split"]] += 1
        for item in row["trajectories"]:
            identity = (row["episode_id"], item["player_index"])
            if identity in identities:
                raise ValueError("0036 focal trajectory identity is duplicated")
            identities.add(identity)
            deck = Counter({int(card): int(count) for card, count in row["deck_counts"][item["player_index"]]})
            if deck[DRAGAPULT_EX_CARD_ID] <= 0:
                raise ValueError("0036 focal catalog contains an opponent-only perspective")
            is_exact = _exact_deck_hash(list(deck.elements())) == EXACT_007_SHA256
            if bool(item["is_exact_007"]) != is_exact:
                raise ValueError("0036 focal exact-007 audit flag mismatch")
            if item["final_diff_target"] != item["final_diff"] + 6:
                raise ValueError("0036 focal final-diff encoding mismatch")
            focal += 1
            exact += int(is_exact)
    if focal != result["selection"]["selected_focal_trajectories"]:
        raise ValueError("0036 focal trajectory count mismatch")
    if len(result["episodes"]) != result["selection"]["selected_episodes"]:
        raise ValueError("0036 focal Episode count mismatch")
    selected_dates = Counter(row["episode_date"] for row in result["episodes"])
    allocation_dates = {
        row["date"]: row["selected_episodes"]
        for row in result["selection"]["date_allocation"]
    }
    if dict(selected_dates) != allocation_dates or any(value < 1 for value in allocation_dates.values()):
        raise ValueError("0036 focal all-date allocation mismatch")
    return {
        "status": "verified",
        "catalog_sha256": expected,
        "episodes": len(result["episodes"]),
        "focal_trajectories": focal,
        "exact_007_focal_trajectories": exact,
        "split_episode_counts": dict(split_episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--archives-dir", type=Path, default=Path("data/raw/episodes/archives"))
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--target-episodes", type=int, default=10_000)
    build.add_argument("--workers", type=int, default=8)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--catalog", type=Path, required=True)
    rewrite = subparsers.add_parser("rewrite-splits")
    rewrite.add_argument("--source", type=Path, required=True)
    rewrite.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_dragapult_catalog(
            args.archives_dir, args.output,
            target_episodes=args.target_episodes, workers=args.workers,
        )
    elif args.command == "verify":
        result = verify_dragapult_catalog(args.catalog)
    else:
        result = rewrite_dragapult_catalog_splits(args.source, args.output)
    summary = {key: result[key] for key in result if key not in {"episodes", "archives", "exclusions"}}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_EXACT_007",
    "DRAGAPULT_EX_CARD_ID",
    "EXACT_007_SHA256",
    "SCHEMA_VERSION",
    "build_dragapult_catalog",
    "inspect_registration_header",
    "rewrite_dragapult_catalog_splits",
    "_select_weighted_dates",
    "verify_dragapult_catalog",
]
