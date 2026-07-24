from __future__ import annotations

import gzip
import hashlib
import json
import random
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Any

from torch.utils.data import IterableDataset

from rl_environment.streaming import iter_jsonl

from .model import IDOnlyCodec, IDOnlyConfig


SPLITS = ("train", "validation", "test")
DATASET_SCHEMA = "alakazam_sota_id_only_dataset_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partial(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial")


def dataset_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {split: base / f"{split}.jsonl.gz" for split in SPLITS}


def _visual_frames(payload: dict[str, Any]) -> list[dict[str, Any]]:
    traces: list[list[Any]] = []
    singletons: list[Any] = []
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for row in step:
            if not isinstance(row, dict):
                continue
            raw = row.get("visualize", row.get("visual"))
            if isinstance(raw, list) and raw:
                traces.append(raw)
            elif isinstance(raw, dict):
                singletons.append(raw)
    selected = max(traces, key=len) if traces else singletons
    return [frame for frame in selected if isinstance(frame, dict)]


class ReplayArchiveResolver:
    """Read selected replay JSON directly from frozen daily ZIP archives."""

    def __init__(self, archive_root: Path) -> None:
        self._stack = ExitStack()
        self._members: dict[str, tuple[zipfile.ZipFile, str, Path]] = {}
        self.archives: list[dict[str, Any]] = []
        for archive in sorted(archive_root.rglob("*.zip")):
            handle = self._stack.enter_context(zipfile.ZipFile(archive))
            json_members = 0
            for member in handle.namelist():
                name = Path(member).name
                if not name.endswith(".json"):
                    continue
                json_members += 1
                previous = self._members.get(name)
                if previous is not None:
                    raise ValueError(
                        f"duplicate replay {name} in {previous[2]} and {archive}"
                    )
                self._members[name] = (handle, member, archive)
            self.archives.append(
                {
                    "path": str(archive.resolve()),
                    "bytes": archive.stat().st_size,
                    "json_members": json_members,
                }
            )
        if not self._members:
            raise ValueError(f"archive root contains no replay JSON: {archive_root}")

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> "ReplayArchiveResolver":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read(self, source: str) -> tuple[dict[str, Any], str]:
        name = Path(source).name
        resolved = self._members.get(name)
        if resolved is None:
            raise FileNotFoundError(f"replay archive does not contain {name}")
        handle, member, archive = resolved
        payload = json.loads(handle.read(member))
        if not isinstance(payload, dict):
            raise ValueError(f"replay is not a JSON object: {archive}::{member}")
        return payload, f"{archive.resolve()}::{member}"


def _record_identity(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["episode_id"]), int(record["player_index"])


def _frame_observation(frame: dict[str, Any]) -> dict[str, Any]:
    observation = frame.get("obs", frame.get("observation"))
    return observation if isinstance(observation, dict) else {}


def build_id_only_dataset(
    source_dataset: Path,
    archive_root: Path,
    output_root: Path,
    *,
    model_config: IDOnlyConfig,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-encode the frozen 0009 identities with the reference Notebook codec."""

    outputs = dataset_paths(output_root)
    audit_path = output_root / "dataset_audit.json"
    occupied = [
        path
        for path in (*outputs.values(), audit_path)
        if path.exists() or _partial(path).exists()
    ]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite dataset outputs: " + ", ".join(map(str, occupied))
        )
    output_root.mkdir(parents=True, exist_ok=True)
    codec = IDOnlyCodec(model_config)
    records_by_split = {split: 0 for split in SPLITS}
    source_records_by_split = {split: 0 for split in SPLITS}
    episodes_by_split: dict[str, set[int]] = {split: set() for split in SPLITS}
    actor_mismatches = 0
    action_order_differences = 0
    encoded_groups = 0
    seen_groups: set[tuple[int, int]] = set()

    with ReplayArchiveResolver(archive_root) as resolver, ExitStack() as stack:
        handles = {
            split: stack.enter_context(
                gzip.open(_partial(path), "wt", encoding="utf-8", compresslevel=6)
            )
            for split, path in outputs.items()
        }
        records = iter_jsonl(source_dataset)
        for group, grouped in groupby(records, key=_record_identity):
            if group in seen_groups:
                raise ValueError(f"source records are not contiguous for episode/player {group}")
            seen_groups.add(group)
            rows = list(grouped)
            episode_id, player_index = group
            payload, replay_source = resolver.read(str(rows[0].get("source", "")))
            actual_episode = int((payload.get("info") or {}).get("EpisodeId", episode_id))
            if actual_episode != episode_id:
                raise ValueError(f"episode ID mismatch: {actual_episode} != {episode_id}")
            frames = _visual_frames(payload)
            if not frames:
                raise ValueError(f"episode {episode_id} has no visual decision frames")
            for source_record in rows:
                split = str(source_record.get("split", "train"))
                if split not in handles:
                    raise ValueError(f"unknown source split: {split}")
                source_records_by_split[split] += 1
                step = int(source_record["episode_step"])
                if not 0 <= step < len(frames):
                    raise ValueError(
                        f"episode {episode_id} step {step} exceeds {len(frames)} visual frames"
                    )
                frame = frames[step]
                observation = _frame_observation(frame)
                current = observation.get("current") if isinstance(observation, dict) else None
                actor = int(current.get("yourIndex", -1)) if isinstance(current, dict) else -1
                if actor != player_index:
                    actor_mismatches += 1
                    raise ValueError(
                        f"actor mismatch at episode {episode_id} step {step}: "
                        f"dataset={player_index}, replay={actor}"
                    )
                frame_action = frame.get("selected")
                if not isinstance(frame_action, list) or not all(
                    isinstance(value, int) for value in frame_action
                ):
                    raise ValueError(f"invalid selected action at episode {episode_id} step {step}")
                source_action = list(source_record.get("targets") or [])
                if sorted(frame_action) != sorted(source_action):
                    raise ValueError(
                        f"action mismatch at episode {episode_id} step {step}: "
                        f"dataset={source_action}, replay={frame_action}"
                    )
                if frame_action != source_action:
                    action_order_differences += 1
                encoded = codec.encode(observation, frame_action)
                if encoded is None:
                    raise ValueError(
                        f"reference codec rejected frozen record {episode_id}/{player_index}/{step}"
                    )
                encoded.update(
                    {
                        "dataset_schema_version": DATASET_SCHEMA,
                        "episode_id": episode_id,
                        "episode_step": step,
                        "player_index": player_index,
                        "replay_source": replay_source,
                        "source_dataset_split": split,
                    }
                )
                handles[split].write(
                    json.dumps(encoded, ensure_ascii=True, separators=(",", ":")) + "\n"
                )
                records_by_split[split] += 1
                episodes_by_split[split].add(episode_id)
            encoded_groups += 1
            if encoded_groups == 1 or encoded_groups % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "id_only_dataset_progress",
                            "episode_players": encoded_groups,
                            "records": sum(records_by_split.values()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        archive_rows = resolver.archives

    if records_by_split != source_records_by_split:
        raise ValueError(
            f"encoded/source count mismatch: {records_by_split} != {source_records_by_split}"
        )
    for path in outputs.values():
        _partial(path).replace(path)
    source_sha256 = _sha256(source_dataset)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source dataset hash mismatch: {source_sha256} != {expected_source_sha256}"
        )
    audit = {
        "schema_version": DATASET_SCHEMA,
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": source_sha256,
        "archive_root": str(archive_root.resolve()),
        "archives": archive_rows,
        "model_config": model_config.to_dict(),
        "records_by_split": records_by_split,
        "episodes_by_split": {
            split: len(episode_ids) for split, episode_ids in episodes_by_split.items()
        },
        "episode_player_groups": encoded_groups,
        "actor_mismatches": actor_mismatches,
        "action_order_differences": action_order_differences,
        "outputs": {
            split: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for split, path in outputs.items()
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


class JsonlShardDataset(IterableDataset[dict[str, Any]]):
    """Notebook-compatible bounded-buffer streaming over local gzip shards."""

    def __init__(
        self,
        paths: list[Path],
        *,
        shuffle: bool,
        seed: int,
        buffer_size: int = 4096,
    ) -> None:
        self.paths = paths
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.seed + self.epoch)
        paths = list(self.paths)
        if self.shuffle:
            rng.shuffle(paths)
        buffer: list[dict[str, Any]] = []
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not self.shuffle:
                        yield row
                        continue
                    buffer.append(row)
                    if len(buffer) >= self.buffer_size:
                        yield buffer.pop(rng.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))


def load_dataset_audit(root: str | Path) -> dict[str, Any]:
    value = json.loads((Path(root) / "dataset_audit.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("invalid Alakazam SOTA dataset audit")
    return value
