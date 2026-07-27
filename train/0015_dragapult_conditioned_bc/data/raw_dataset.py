"""Extract causal full-action decisions for the frozen 0015 trajectory index."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import io
import json
import os
import shutil
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

_legacy = importlib.import_module("train.0013_semantic_goal_policy.data.dataset")
_source = importlib.import_module("train.0013_semantic_goal_policy.data.source")
_records = importlib.import_module("train.0013_semantic_goal_policy.data.records")
_actions = importlib.import_module("train.0013_semantic_goal_policy.features.action_contract")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_raw(row: dict[str, Any], bundle: zipfile.ZipFile) -> bytes:
    origin = row["origin"]
    if origin["kind"] == "official_archive":
        raw = bundle.read(origin["member"])
    elif origin["kind"] == "targeted_overlay":
        raw = Path(origin["path"]).read_bytes()
    else:
        raise ValueError(f"unknown replay origin: {origin['kind']}")
    if hashlib.sha256(raw).hexdigest() != row["payload_sha256"]:
        raise ValueError(f"selected replay hash mismatch: {row['episode_id']}")
    return raw


def _episode(row: dict[str, Any], raw: bytes) -> Any:
    payload = json.loads(raw)
    info = payload.get("info")
    if not isinstance(info, dict) or int(info.get("EpisodeId", -1)) != row["episode_id"]:
        raise ValueError(f"selected replay Episode ID mismatch: {row['episode_id']}")
    source = _records.SourceIdentity(
        row["episode_date"], row["episode_id"], row["player_index"], True
    )
    return _source.CanonicalEpisode(
        source,
        payload,
        tuple(step[row["player_index"]] for step in payload["steps"]),
        row["payload_sha256"],
        row["origin"]["kind"],
    )


def _decision_rows(trajectory: dict[str, Any], raw: bytes) -> Iterator[dict[str, Any]]:
    episode = _episode(trajectory, raw)
    visual = _legacy._visual_frames(episode)
    deck = _legacy.DeckManifest.from_card_ids(_legacy._registration_decks(episode, visual))
    if trajectory.get("model_deck_sha256") and deck.sha256 != trajectory["model_deck_sha256"]:
        raise ValueError(f"model deck hash mismatch: {trajectory['episode_id']}")
    decision_index = 0
    for frame_index, raw_observation, target, decision_duration in _legacy._decision_frames(
        episode
    ):
        raw_select = _legacy._mapping(raw_observation.get("select"), "raw actor select")
        raw_options = raw_select.get("option")
        if not isinstance(raw_options, (tuple, list)):
            raise ValueError("raw select.option must be a sequence")
        unknown = _actions.unrecognized_option_fields(raw_options)
        if unknown:
            raise ValueError(f"unknown option fields: {sorted(unknown)}")
        observation = _legacy._project_observation(raw_observation)
        select = observation["select"]
        options = select["option"]
        ordered = _actions.validate_ordered_action(
            target,
            option_count=len(options),
            min_count=select["minCount"],
            max_count=select["maxCount"],
            decoder_capacity=64,
        )
        occurrences = _actions.build_occurrence_identities(options)
        action_types = sorted(
            {str(options[index].get("type", "missing")) for index in ordered.indices}
        )
        yield {
            "identity": {
                "date": trajectory["episode_date"],
                "episode_id": trajectory["episode_id"],
                "player_index": trajectory["player_index"],
                "episode_step": frame_index,
            },
            "split": trajectory["split"],
            "deck_manifest": deck.as_dict(),
            "actor_observation": _legacy._plain(observation),
            "legal_options": _legacy._plain(options),
            "option_alignment_audit": _legacy._plain(
                _actions.serialize_occurrence_identities(occurrences)
            ),
            "ordered_action": list(ordered.indices),
            "action_termination": ordered.termination.value,
            "event_cursor": {
                "visual_frame_index": frame_index,
                "actor_decision_index": decision_index,
                "incoming_log_count": len(observation.get("logs", [])),
            },
            "terminal_outcome": trajectory["outcome"],
            "decision_duration_ms": _legacy._duration(decision_duration),
            "episode_duration_ms": _legacy._duration(episode.payload.get("duration")),
            "source_payload_sha256": trajectory["payload_sha256"],
            "source_id": trajectory["source_id"],
            "source_key": trajectory["source_key"],
            "build": trajectory["build"],
            "first_player": trajectory["first_player"],
            "action_types": action_types,
            "schema_version": "0015_causal_decision_v1",
        }
        decision_index += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_raw_dataset(index_path: Path, output: Path, shard_size: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    stage = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    handles: dict[str, Any] = {}
    raw_handles: dict[str, Any] = {}
    shard_counts = Counter()
    decision_counts = Counter()
    shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}

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

    def write(row: dict[str, Any]) -> None:
        split = row["split"]
        if shard_counts[split] == 0:
            partial = stage / f"{split}-{len(shards[split]):05d}.jsonl.gz.partial"
            raw_handle = partial.open("wb")
            gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0)
            handles[split] = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            raw_handles[split] = raw_handle
        handles[split].write(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        shard_counts[split] += 1
        if shard_counts[split] >= shard_size:
            close_shard(split)

    try:
        with zipfile.ZipFile(index["official_archive"]) as bundle:
            for trajectory in index["core_trajectories"]:
                raw = _load_raw(trajectory, bundle)
                produced = 0
                for row in _decision_rows(trajectory, raw):
                    write(row)
                    decision_counts[
                        (
                            trajectory["build"],
                            trajectory["source_key"],
                            trajectory["split"],
                            trajectory["outcome"],
                        )
                    ] += 1
                    produced += 1
                if produced == 0:
                    episode_id = trajectory["episode_id"]
                    raise ValueError(f"trajectory has no actor decisions: {episode_id}")
        for split in ("train", "validation"):
            close_shard(split)
        reference = {
            "schema_version": "0015_raw_dataset_reference_v1",
            "source_index": str(index_path),
            "source_index_sha256": _sha256(index_path),
            "source_vocabulary": index["source_vocabulary"],
            "shards": shards,
            "decision_counts": [
                {
                    "build": key[0],
                    "source_key": key[1],
                    "split": key[2],
                    "outcome": key[3],
                    "decisions": value,
                }
                for key, value in sorted(decision_counts.items())
            ],
        }
        reference["content_sha256"] = _canonical_hash(reference)
        (stage / "dataset_reference.json").write_text(
            json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output)
        return reference
    except BaseException:
        for split in list(handles):
            try:
                handles[split].close()
            except OSError:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=10_000)
    args = parser.parse_args()
    result = build_raw_dataset(args.source_index, args.output, args.shard_size)
    print(json.dumps(result["decision_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
