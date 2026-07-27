"""Materialize actor-only causal decision rows from the frozen 0016 replay catalog."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import uuid
import zipfile
from collections import Counter
from collections import deque
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .replay_contract import (
    DeckManifest,
    decision_frames,
    project_observation,
    registration_decks,
    validate_ordered_action,
)


BATTLE_CAGE_ID = 1264
WONDROUS_PATCH_ID = 1146
SCHEMA_VERSION = "0016_causal_decision_v1"
ACTION_CONTRACT_VERSION = "ordered_full_action_v1"
_ARCHIVE_DATE = re.compile(r"(20\d\d-\d\d-\d\d)")
_WORKER_ARCHIVES: dict[str, zipfile.ZipFile] = {}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload(row: Mapping[str, Any], archives: dict[str, zipfile.ZipFile]) -> bytes:
    locator = row["locator"]
    kind = locator["kind"]
    if kind == "zip_member":
        archive_path = str(locator["archive"])
        bundle = archives.get(archive_path)
        if bundle is None:
            bundle = zipfile.ZipFile(archive_path)
            archives[archive_path] = bundle
        raw = bundle.read(locator["member"])
    elif kind == "file":
        raw = Path(locator["member"]).read_bytes()
    else:
        raise ValueError(f"unknown replay locator kind: {kind}")
    expected = row.get("payload_sha256")
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None and expected != actual:
        raise ValueError(f"payload hash mismatch for Episode {row['episode_id']}")
    return raw


def _episode_date(row: Mapping[str, Any]) -> str:
    value = row.get("episode_date")
    if isinstance(value, str) and re.fullmatch(r"20\d\d-\d\d-\d\d", value):
        return value
    locator = row["locator"]
    match = _ARCHIVE_DATE.search(str(locator.get("archive") or locator.get("member")))
    if match is not None:
        return match.group(1)
    return "2026-07-27"


def _option_card_id(observation: Mapping[str, Any], option: Mapping[str, Any]) -> int:
    direct = option.get("cardId")
    if isinstance(direct, int) and not isinstance(direct, bool) and direct > 0:
        return direct
    current = observation["current"]
    select = observation["select"]
    actor = current["yourIndex"]
    player_index = option.get("playerIndex", actor)
    area = option.get("area", -1)
    if option.get("type") == 7 and area == -1:
        area = 2
    index = option.get("index", -1)
    values: object = []
    if isinstance(player_index, int) and player_index in (0, 1):
        player = current["players"][player_index]
        zone = {2: "hand", 3: "discard", 4: "active", 5: "bench"}.get(area)
        if zone is not None:
            values = player.get(zone, [])
    if area == 7:
        values = current.get("stadium", [])
    elif area == 12:
        values = current.get("looking", [])
    elif area == 1:
        values = select.get("deck", [])
    if (
        isinstance(values, list)
        and isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < len(values)
        and isinstance(values[index], Mapping)
    ):
        value = values[index].get("id")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def build_trajectory_index(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze replay hashes, seat evidence, and grouped split membership."""
    rows: list[dict[str, Any]] = []
    for item in catalog["episodes"]:
        row = dict(item)
        row["episode_date"] = _episode_date(row)
        if row.get("split") not in {"train", "validation"}:
            raise ValueError(f"catalog split is absent: {row['episode_id']}")
        cards: list[int] = []
        for card_id, count in row.get("deck_counts", []):
            if (
                isinstance(card_id, bool)
                or not isinstance(card_id, int)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
            ):
                raise ValueError(f"invalid catalog deck counts: {row['episode_id']}")
            cards.extend([card_id] * count)
        row["deck_manifest"] = DeckManifest.from_card_ids(cards).as_dict()
        if row.get("deck_sha256") is None or row.get("payload_sha256") is None:
            raise ValueError(f"catalog commitments are absent: {row['episode_id']}")
        rows.append(row)
    identities = [(row["episode_id"], row["player_index"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("trajectory index contains duplicate Episode/player identities")
    source_counts = Counter((row["source_id"], row["split"]) for row in rows)
    index: dict[str, Any] = {
        "schema_version": "0016_trajectory_source_index_v1",
        "catalog_sha256": catalog["catalog_sha256"],
        "source_vocabulary_sha256": catalog["source_vocabulary_sha256"],
        "sources": catalog["sources"],
        "split_contract": catalog["split"],
        "counts": [
            {"source_id": key[0], "split": key[1], "episodes": value}
            for key, value in sorted(source_counts.items())
        ],
        "trajectories": sorted(rows, key=lambda row: (row["episode_id"], row["player_index"])),
    }
    index["content_sha256"] = _canonical_hash(index)
    return index


def _decision_rows(
    trajectory: Mapping[str, Any], raw: bytes
) -> Iterator[tuple[dict[str, Any], dict[str, int]]]:
    payload = json.loads(raw)
    episode_id = int(trajectory["episode_id"])
    info = payload.get("info")
    if not isinstance(info, Mapping) or info.get("EpisodeId") != episode_id:
        raise ValueError(f"Episode identity mismatch: {episode_id}")
    rewards = payload.get("rewards")
    statuses = payload.get("statuses")
    actor = int(trajectory["player_index"])
    if not isinstance(rewards, list) or not isinstance(statuses, list):
        raise ValueError(f"Episode terminal fields are absent: {episode_id}")
    reward = rewards[actor]
    if (
        actor >= len(statuses)
        or statuses[actor] != "DONE"
        or isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or reward <= 0
    ):
        raise ValueError(f"trajectory actor is not a completed winner: {episode_id}")
    deck = DeckManifest.from_card_ids(registration_decks(payload)[actor])
    if deck.as_dict() != trajectory["deck_manifest"]:
        raise ValueError(f"registered deck drift: {episode_id}")
    decision_index = 0
    episode_duration = payload.get("duration")
    if not isinstance(episode_duration, (int, float)) or isinstance(episode_duration, bool):
        episode_duration = 0.0
    for frame_index, raw_observation, target, decision_duration in decision_frames(payload, actor):
        observation = project_observation(raw_observation)
        select = observation["select"]
        options = select["option"]
        ordered, termination = validate_ordered_action(
            target,
            option_count=len(options),
            min_count=select["minCount"],
            max_count=select["maxCount"],
            capacity=64,
        )
        selected = [options[index] for index in ordered]
        option_card_ids = [_option_card_id(observation, option) for option in options]
        selected_card_ids = [option_card_ids[index] for index in ordered]
        context = select.get("contextCard")
        context_card_id = context.get("id") if isinstance(context, Mapping) else None
        effect = select.get("effect")
        effect_card_id = effect.get("id") if isinstance(effect, Mapping) else None
        patch_context = WONDROUS_PATCH_ID in {context_card_id, effect_card_id}
        diagnostics = {
            "battle_cage_legal": int(
                BATTLE_CAGE_ID in option_card_ids
            ),
            "battle_cage_selected": int(BATTLE_CAGE_ID in selected_card_ids),
            "wondrous_patch_legal": int(WONDROUS_PATCH_ID in option_card_ids),
            "wondrous_patch_selected": int(WONDROUS_PATCH_ID in selected_card_ids),
            "wondrous_patch_target_decision": int(patch_context),
            "wondrous_patch_targets": len(ordered) if patch_context else 0,
            "action_over_16": int(len(ordered) > 16),
            "action_length": len(ordered),
        }
        row = {
            "identity": {
                "date": trajectory["episode_date"],
                "episode_id": episode_id,
                "player_index": actor,
                "submission_id_unavailable": True,
                "episode_step": frame_index,
            },
            "split": trajectory["split"],
            "deck_manifest": deck.as_dict(),
            "actor_observation": observation,
            "legal_options": options,
            "ordered_action": list(ordered),
            "action_termination": termination,
            "event_cursor": {
                "visual_frame_index": frame_index,
                "actor_decision_index": decision_index,
                "incoming_log_count": len(observation.get("logs", [])),
            },
            "terminal_outcome": "win",
            "decision_duration_ms": decision_duration,
            "episode_duration_ms": float(episode_duration),
            "source_payload_sha256": trajectory["payload_sha256"],
            "source_id": trajectory["source_id"],
            "source_team_name": trajectory["team_name"],
            "first_player": trajectory["first_player"],
            "schema_version": SCHEMA_VERSION,
            "action_contract_version": ACTION_CONTRACT_VERSION,
        }
        yield row, diagnostics
        decision_index += 1


def _process_trajectory(
    trajectory: dict[str, Any], raw: bytes | None = None
) -> tuple[dict[str, Any], list[tuple[str, dict[str, int]]]]:
    payload = raw if raw is not None else _payload(trajectory, _WORKER_ARCHIVES)
    rows = [
        (_canonical_bytes(row).decode("utf-8"), diagnostics)
        for row, diagnostics in _decision_rows(trajectory, payload)
    ]
    return trajectory, rows


def _parallel_trajectory_rows(
    trajectories: list[dict[str, Any]], workers: int
) -> Iterator[tuple[dict[str, Any], list[tuple[str, dict[str, int]]]]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    archives: dict[str, zipfile.ZipFile] = {}
    if workers == 1:
        try:
            for trajectory in trajectories:
                yield _process_trajectory(trajectory, _payload(trajectory, archives))
        finally:
            for bundle in archives.values():
                bundle.close()
        return

    pending: deque[Future[tuple[dict[str, Any], list[tuple[str, dict[str, int]]]]]] = (
        deque()
    )
    values = iter(trajectories)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for _ in range(workers):
                try:
                    trajectory = next(values)
                except StopIteration:
                    break
                pending.append(
                    pool.submit(_process_trajectory, trajectory)
                )
            while pending:
                yield pending.popleft().result()
                try:
                    trajectory = next(values)
                except StopIteration:
                    continue
                pending.append(pool.submit(_process_trajectory, trajectory))
    finally:
        for bundle in archives.values():
            bundle.close()


def build_raw_dataset(
    catalog_path: Path,
    output: Path,
    *,
    shard_size: int = 10_000,
    workers: int = 8,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if shard_size < 1:
        raise ValueError("shard size must be positive")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    trajectory_index = build_trajectory_index(catalog)
    stage = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    handles: dict[str, io.TextIOWrapper] = {}
    raw_handles: dict[str, Any] = {}
    shard_counts: Counter[str] = Counter()
    shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    decision_counts: Counter[tuple[int, str]] = Counter()
    diagnostics: Counter[tuple[int, str, str]] = Counter()
    maximum_action_length = 0

    def close_shard(split: str) -> None:
        handle = handles.pop(split, None)
        raw_handle = raw_handles.pop(split, None)
        if handle is None or raw_handle is None:
            return
        handle.flush()
        handle.detach().close()
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
        raw_handle.close()
        partial = stage / f"{split}-{len(shards[split]):05d}.jsonl.gz.partial"
        final = partial.with_suffix("")
        partial.replace(final)
        shards[split].append(
            {"path": final.name, "sha256": _sha256(final), "count": shard_counts[split]}
        )
        shard_counts[split] = 0

    def write(split: str, serialized: str) -> None:
        if shard_counts[split] == 0:
            partial = stage / f"{split}-{len(shards[split]):05d}.jsonl.gz.partial"
            raw_handle = partial.open("wb")
            compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0)
            handles[split] = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            raw_handles[split] = raw_handle
        handles[split].write(serialized)
        shard_counts[split] += 1
        if shard_counts[split] >= shard_size:
            close_shard(split)

    try:
        for trajectory, decision_rows in _parallel_trajectory_rows(
            trajectory_index["trajectories"], workers
        ):
            split = str(trajectory["split"])
            for serialized, row_diagnostics in decision_rows:
                write(split, serialized)
                source_id = int(trajectory["source_id"])
                decision_counts[(source_id, split)] += 1
                for name, value in row_diagnostics.items():
                    if name == "action_length":
                        maximum_action_length = max(maximum_action_length, value)
                    else:
                        diagnostics[(source_id, split, name)] += value
            if not decision_rows:
                raise ValueError(f"trajectory has no actor decisions: {trajectory['episode_id']}")
        for split in ("train", "validation"):
            close_shard(split)
        (stage / "trajectory_index.json").write_bytes(_canonical_bytes(trajectory_index))
        source_names = {
            int(source["source_id"]): source["team_name"] for source in catalog["sources"]
        }
        reference: dict[str, Any] = {
            "schema_version": "0016_raw_dataset_reference_v1",
            "catalog_path": str(catalog_path),
            "catalog_file_sha256": _sha256(catalog_path),
            "catalog_sha256": catalog["catalog_sha256"],
            "trajectory_index_sha256": trajectory_index["content_sha256"],
            "source_vocabulary_sha256": catalog["source_vocabulary_sha256"],
            "sources": catalog["sources"],
            "shards": shards,
            "decision_counts": [
                {
                    "source_id": key[0],
                    "team_name": source_names[key[0]],
                    "split": key[1],
                    "decisions": value,
                }
                for key, value in sorted(decision_counts.items())
            ],
            "card_action_diagnostics": [
                {
                    "source_id": key[0],
                    "team_name": source_names[key[0]],
                    "split": key[1],
                    "metric": key[2],
                    "count": value,
                }
                for key, value in sorted(diagnostics.items())
            ],
            "maximum_action_length": maximum_action_length,
            "training_action_length_limit": 64,
            "long_action_policy": "all_validated_actions_are_retained_and_trained",
            "materialization_workers": workers,
        }
        reference["content_sha256"] = _canonical_hash(reference)
        (stage / "dataset_reference.json").write_bytes(_canonical_bytes(reference))
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output)
        return reference
    except BaseException:
        for handle in handles.values():
            try:
                handle.close()
            except OSError:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    reference = build_raw_dataset(
        args.catalog,
        args.output,
        shard_size=args.shard_size,
        workers=args.workers,
    )
    print(json.dumps(reference["decision_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
