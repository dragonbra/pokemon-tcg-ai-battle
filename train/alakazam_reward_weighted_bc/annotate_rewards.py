from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Any

from .data import iter_full_action_records
from .reward_annotations import annotate_episode_records, reward_metric_counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["episode_id"]),
        int(record["player_index"]),
        int(record["episode_step"]),
    )


class ReplayResolver:
    """Resolve exact source replays from extracted files or frozen zip archives."""

    def __init__(
        self,
        *,
        replay_root: Path | None = None,
        archive_root: Path | None = None,
    ) -> None:
        self.replay_root = replay_root
        self._stack = ExitStack()
        self._members: dict[str, tuple[zipfile.ZipFile, str, Path]] = {}
        if archive_root is not None:
            for archive in sorted(archive_root.rglob("*.zip")):
                handle = self._stack.enter_context(zipfile.ZipFile(archive))
                for member in handle.namelist():
                    name = Path(member).name
                    if not name.endswith(".json"):
                        continue
                    previous = self._members.get(name)
                    if previous is not None:
                        raise ValueError(
                            f"duplicate replay {name} in {previous[2]} and {archive}"
                        )
                    self._members[name] = (handle, member, archive)

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> "ReplayResolver":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read(self, record: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        source = Path(str(record.get("source", "")))
        candidates = [source]
        if self.replay_root is not None:
            candidates.extend(
                (
                    self.replay_root / source.name,
                    self.replay_root / f"episode-{int(record['episode_id'])}-replay.json",
                    self.replay_root / f"{int(record['episode_id'])}.json",
                )
            )
        for candidate in candidates:
            if candidate.is_file():
                raw = candidate.read_bytes()
                return json.loads(raw), str(candidate.resolve()), _bytes_sha256(raw)
        archived = self._members.get(source.name)
        if archived is not None:
            handle, member, archive = archived
            raw = handle.read(member)
            return json.loads(raw), f"{archive.resolve()}::{member}", _bytes_sha256(raw)
        raise FileNotFoundError(f"cannot resolve replay for episode {record.get('episode_id')}")


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial")


def _split_outputs(output: Path) -> dict[str, Path]:
    return {
        split: output.with_name(f"{output.stem}.{split}{output.suffix}")
        for split in ("train", "validation", "test")
    }


def annotate_dataset(
    dataset: Path,
    output: Path,
    *,
    replay_root: Path | None = None,
    archive_root: Path | None = None,
) -> dict[str, Any]:
    split_outputs = _split_outputs(output)
    targets = [output, *split_outputs.values()]
    occupied = [path for path in targets if path.exists() or _temporary(path).exists()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite reward sidecar: " + ", ".join(map(str, occupied))
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    records_by_split = {"train": 0, "validation": 0, "test": 0}
    replay_sources: dict[str, dict[str, str]] = {}
    episodes = 0
    seen_groups: set[tuple[int, int]] = set()
    with ReplayResolver(replay_root=replay_root, archive_root=archive_root) as resolver:
        with ExitStack() as stack:
            handle = stack.enter_context(_temporary(output).open("w", encoding="utf-8"))
            split_handles = {
                split: stack.enter_context(_temporary(path).open("w", encoding="utf-8"))
                for split, path in split_outputs.items()
            }
            records = iter_full_action_records(dataset)
            for group, grouped in groupby(
                records,
                key=lambda record: (
                    int(record["episode_id"]), int(record["player_index"])
                ),
            ):
                if group in seen_groups:
                    raise ValueError(f"episode/player records are not contiguous: {group}")
                seen_groups.add(group)
                rows = list(grouped)
                episode_id, _ = group
                payload, source, replay_sha256 = resolver.read(rows[0])
                actual_episode = int((payload.get("info") or {}).get("EpisodeId", episode_id))
                if actual_episode != episode_id:
                    raise ValueError(f"episode ID mismatch: {actual_episode} != {episode_id}")
                replay_sources[str(episode_id)] = {
                    "source": source,
                    "sha256": replay_sha256,
                }
                annotated = annotate_episode_records(rows, payload)
                episode_counts = reward_metric_counts(annotated)
                for name, value in episode_counts.items():
                    counts[name] = counts.get(name, 0) + value
                for record in annotated:
                    split = str(record.get("split", "train"))
                    if split not in split_handles:
                        raise ValueError(f"unknown dataset split: {split}")
                    sidecar = {
                        "episode_id": int(record["episode_id"]),
                        "player_index": int(record["player_index"]),
                        "episode_step": int(record["episode_step"]),
                        "reward_schema_version": record["reward_schema_version"],
                        "reward_metrics": record["reward_metrics"],
                    }
                    line = json.dumps(sidecar, ensure_ascii=True, sort_keys=True) + "\n"
                    handle.write(line)
                    split_handles[split].write(line)
                    records_by_split[split] += 1
                episodes += 1
                if episodes == 1 or episodes % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "reward_annotation_progress",
                                "episodes": episodes,
                                "records": sum(records_by_split.values()),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    for path in targets:
        _temporary(path).replace(path)
    summary = {
        "schema_version": "alakazam_reward_annotation_summary_v3",
        "reward_schema_version": "alakazam_offline_reward_metrics_v2",
        "format": "identity-keyed reward sidecar; encoded observations remain in base dataset",
        "input": str(dataset.resolve()),
        "input_sha256": _sha256(dataset),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "split_outputs": {
            split: {
                "path": str(path.resolve()),
                "records": records_by_split[split],
                "sha256": _sha256(path),
            }
            for split, path in split_outputs.items()
        },
        "episodes": episodes,
        "metrics": counts,
        "replays": replay_sources,
    }
    summary_path = output.with_suffix(output.suffix + ".reward_audit.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build a streaming reward sidecar")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            annotate_dataset(
                args.dataset,
                args.output,
                replay_root=args.replay_root,
                archive_root=args.archive_root,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
