from __future__ import annotations

import gzip
import hashlib
import json
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Any

from rl_environment.streaming import iter_jsonl
from train.project_0010_alakazam_sota_model.dataset import (
    DATASET_SCHEMA as BASE_DATASET_SCHEMA,
    JsonlShardDataset,
    ReplayArchiveResolver,
    _frame_observation,
    _record_identity,
    _visual_frames,
    dataset_paths,
)

from .codec import FeatureEngineeringCodec
from .model import FeatureModelConfig
from .transitions import transition_label


FEATURE_DATASET_SCHEMA = "alakazam_sota_feature_engineering_dataset_v1"
TRANSITION_DATASET_SCHEMA = "alakazam_sota_feature_engineering_transition_dataset_v1"
HISTORY_DATASET_SCHEMA = "alakazam_sota_feature_engineering_history_dataset_v1"
SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partial(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial")


def load_dataset_audit(root: str | Path) -> dict[str, Any]:
    value = json.loads((Path(root) / "dataset_audit.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") not in {
        BASE_DATASET_SCHEMA,
        FEATURE_DATASET_SCHEMA,
        TRANSITION_DATASET_SCHEMA,
        HISTORY_DATASET_SCHEMA,
    }:
        raise ValueError("invalid 0010/0012 dataset audit")
    return value


def build_feature_dataset(
    source_dataset: Path,
    archive_root: Path,
    output_root: Path,
    *,
    model_config: FeatureModelConfig,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Re-encode exactly the frozen 0010 identities with E3 action fields."""

    if not (model_config.action_primitive_context or model_config.action_history):
        raise ValueError("feature dataset requires primitive/context or action history")
    source_sha256 = _sha256(source_dataset)
    if source_sha256 != expected_source_sha256:
        raise ValueError(f"source dataset hash mismatch: {source_sha256}")
    outputs = dataset_paths(output_root)
    audit_path = output_root / "dataset_audit.json"
    occupied = [
        path
        for path in (*outputs.values(), audit_path)
        if path.exists() or _partial(path).exists()
    ]
    if occupied:
        raise FileExistsError("refusing to overwrite: " + ", ".join(map(str, occupied)))
    output_root.mkdir(parents=True, exist_ok=True)
    codec = FeatureEngineeringCodec(model_config)
    counts = {split: 0 for split in SPLITS}
    source_counts = {split: 0 for split in SPLITS}
    episodes: dict[str, set[int]] = {split: set() for split in SPLITS}
    primitive_nonzero = {key: 0 for key in ("attack_id", "count", "energy_index", "tool_index")}
    context_nonzero = {key: 0 for key in ("effect_card", "context_card")}
    seen_groups: set[tuple[int, int]] = set()
    encoded_groups = 0

    with ReplayArchiveResolver(archive_root) as resolver, ExitStack() as stack:
        handles = {
            split: stack.enter_context(
                gzip.open(_partial(path), "wt", encoding="utf-8", compresslevel=6)
            )
            for split, path in outputs.items()
        }
        for group, grouped in groupby(iter_jsonl(source_dataset), key=_record_identity):
            if group in seen_groups:
                raise ValueError(f"non-contiguous source group: {group}")
            seen_groups.add(group)
            rows = list(grouped)
            episode_id, player_index = group
            payload, replay_source = resolver.read(str(rows[0].get("source", "")))
            actual_episode = int((payload.get("info") or {}).get("EpisodeId", episode_id))
            if actual_episode != episode_id:
                raise ValueError(f"episode mismatch: {actual_episode} != {episode_id}")
            frames = _visual_frames(payload)
            if not frames:
                raise ValueError(f"episode {episode_id} has no visual frames")
            history_events: list[list[int]] = []
            for source_record in rows:
                split = str(source_record.get("split", "train"))
                if split not in handles:
                    raise ValueError(f"unknown split: {split}")
                source_counts[split] += 1
                step = int(source_record["episode_step"])
                if not 0 <= step < len(frames):
                    raise ValueError(f"episode {episode_id} invalid step {step}")
                frame = frames[step]
                observation = _frame_observation(frame)
                current = observation.get("current") or {}
                if int(current.get("yourIndex", -1)) != player_index:
                    raise ValueError(f"actor mismatch at {episode_id}/{player_index}/{step}")
                frame_action = frame.get("selected")
                if not isinstance(frame_action, list) or not all(
                    isinstance(value, int) for value in frame_action
                ):
                    raise ValueError(f"invalid selected action at episode {episode_id} step {step}")
                source_action = list(source_record.get("targets") or [])
                if frame_action != source_action:
                    raise ValueError(
                        f"action/order mismatch at {episode_id}/{player_index}/{step}: "
                        f"{frame_action} != {source_action}"
                    )
                encoded = codec.encode(observation, frame_action)
                if encoded is None:
                    raise ValueError(f"codec rejected {episode_id}/{player_index}/{step}")
                if model_config.action_history:
                    selected_history = history_events[-model_config.max_action_history :]
                    encoded["action_history_cat"] = [
                        [*event, position]
                        for position, event in enumerate(selected_history, 1)
                    ]
                if model_config.action_transition_auxiliary:
                    next_observation = (
                        _frame_observation(frames[step + 1]) if step + 1 < len(frames) else None
                    )
                    encoded.update(transition_label(observation, next_observation, player_index))
                for primitive in encoded.get("option_primitive_cat", []):
                    for key, value in zip(primitive_nonzero, primitive):
                        primitive_nonzero[key] += int(value != 0)
                for key, value in zip(context_nonzero, encoded.get("action_context_cat", [])[:2]):
                    context_nonzero[key] += int(value != 0)
                encoded.update(
                    {
                        "dataset_schema_version": (
                            TRANSITION_DATASET_SCHEMA
                            if model_config.action_transition_auxiliary
                            else HISTORY_DATASET_SCHEMA
                            if model_config.action_history
                            else FEATURE_DATASET_SCHEMA
                        ),
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
                counts[split] += 1
                episodes[split].add(episode_id)
                if model_config.action_history:
                    for selected_index in frame_action:
                        option = encoded["option_cat"][selected_index]
                        history_events.append(
                            [
                                encoded["global_cat"][0],
                                encoded["global_cat"][1],
                                option[0],
                                option[4],
                                option[5],
                            ]
                        )
            encoded_groups += 1
            if encoded_groups == 1 or encoded_groups % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "0012_feature_dataset_progress",
                            "episode_players": encoded_groups,
                            "records": sum(counts.values()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        archives = resolver.archives

    if counts != source_counts:
        raise ValueError(f"encoded/source count mismatch: {counts} != {source_counts}")
    for path in outputs.values():
        _partial(path).replace(path)
    audit = {
        "schema_version": (
            TRANSITION_DATASET_SCHEMA
            if model_config.action_transition_auxiliary
            else HISTORY_DATASET_SCHEMA
            if model_config.action_history
            else FEATURE_DATASET_SCHEMA
        ),
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": source_sha256,
        "identity_contract": "exact frozen 0010 episode/player/step/action/order/split",
        "archive_root": str(archive_root.resolve()),
        "archives": archives,
        "model_config": model_config.to_dict(),
        "records_by_split": counts,
        "episodes_by_split": {split: len(values) for split, values in episodes.items()},
        "episode_player_groups": encoded_groups,
        "action_order_differences": 0,
        "primitive_nonzero": primitive_nonzero,
        "context_nonzero": context_nonzero,
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


__all__ = [
    "FEATURE_DATASET_SCHEMA",
    "TRANSITION_DATASET_SCHEMA",
    "HISTORY_DATASET_SCHEMA",
    "JsonlShardDataset",
    "build_feature_dataset",
    "dataset_paths",
    "load_dataset_audit",
]
