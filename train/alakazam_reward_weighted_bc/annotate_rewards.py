from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import load_full_action_dataset
from .reward_annotations import annotate_episode_records, reward_metric_counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(record: dict[str, Any], replay_root: Path | None) -> Path:
    source = Path(str(record.get("source", "")))
    if source.is_file():
        return source
    if replay_root is not None:
        candidates = (
            replay_root / source.name,
            replay_root / f"episode-{int(record['episode_id'])}-replay.json",
            replay_root / f"{int(record['episode_id'])}.json",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"cannot resolve replay for episode {record.get('episode_id')}")


def annotate_dataset(
    dataset: Path,
    output: Path,
    *,
    replay_root: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reward dataset: {output}")
    records = load_full_action_dataset(dataset)
    episodes: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episodes[(int(record["episode_id"]), int(record["player_index"]))].append(record)
    annotated_by_identity: dict[tuple[int, int, int], dict[str, Any]] = {}
    replay_sources: dict[str, str] = {}
    for (episode_id, player_index), rows in sorted(episodes.items()):
        source = _source(rows[0], replay_root)
        payload = json.loads(source.read_text(encoding="utf-8"))
        actual_episode = int((payload.get("info") or {}).get("EpisodeId", episode_id))
        if actual_episode != episode_id:
            raise ValueError(f"episode ID mismatch: {actual_episode} != {episode_id}")
        replay_sources[str(episode_id)] = _sha256(source)
        for record in annotate_episode_records(rows, payload):
            identity = (episode_id, player_index, int(record["episode_step"]))
            annotated_by_identity[identity] = record
    annotated = [
        annotated_by_identity[
            (int(record["episode_id"]), int(record["player_index"]), int(record["episode_step"]))
        ]
        for record in records
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in annotated:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    summary = {
        "schema_version": "alakazam_reward_annotation_summary_v1",
        "reward_schema_version": "alakazam_offline_reward_metrics_v1",
        "input": str(dataset.resolve()),
        "input_sha256": _sha256(dataset),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "episodes": len(episodes),
        "metrics": reward_metric_counts(annotated),
        "replay_sha256": replay_sources,
    }
    summary_path = output.with_suffix(output.suffix + ".reward_audit.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Annotate full-action BC JSONL with rewards")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            annotate_dataset(args.dataset, args.output, replay_root=args.replay_root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
