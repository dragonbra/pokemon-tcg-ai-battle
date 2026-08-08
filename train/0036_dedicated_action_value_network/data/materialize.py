"""Compile the 0036 catalog directly into immutable action-decision Value shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import time
import uuid
import zipfile
from collections import Counter, deque
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ..contracts.fields import SCHEMA_VERSION as ACTOR_SCHEMA_VERSION
from ..domain import PrototypeIndex
from ..features.compiler import compile_canonical_row
from ..knowledge.state import CausalKnowledge
from .replay_contract import (
    DeckManifest,
    decision_frames,
    project_observation,
    validate_ordered_action,
)


DATASET_SCHEMA_VERSION = "0036_action_value_dataset_v1"
_WORKER_PROTOTYPES: PrototypeIndex | None = None
_WORKER_ARCHIVES: dict[str, zipfile.ZipFile] = {}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deck(counts: list[list[int]]) -> list[int]:
    cards = [int(card_id) for card_id, count in counts for _ in range(int(count))]
    DeckManifest.from_card_ids(cards)
    return cards


def _payload(episode: Mapping[str, Any]) -> bytes:
    locator = episode["locator"]
    archive_path = str(locator["archive"])
    archive = _WORKER_ARCHIVES.get(archive_path)
    if archive is None:
        archive = zipfile.ZipFile(archive_path)
        _WORKER_ARCHIVES[archive_path] = archive
    raw = archive.read(locator["member"])
    if hashlib.sha256(raw).hexdigest() != episode["payload_sha256"]:
        raise ValueError(f"0036 Episode payload hash mismatch: {episode['episode_id']}")
    return raw


def _init_worker(public_prototypes: str, engine_prototypes: str) -> None:
    global _WORKER_PROTOTYPES
    _WORKER_PROTOTYPES = PrototypeIndex.load(public_prototypes, engine_prototypes)


def _compile_trajectory(
    payload: Mapping[str, Any],
    episode: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    prototypes: PrototypeIndex,
) -> list[tuple[str, str, dict[str, int], tuple[int, int]]]:
    actor = int(trajectory["player_index"])
    deck = _deck(episode["deck_counts"][actor])
    frames = list(decision_frames(payload, actor))
    if not frames:
        raise ValueError(f"0036 trajectory has no action decisions: {episode['episode_id']}/{actor}")
    episode_weight = 1.0 / len(frames)
    knowledge = CausalKnowledge(actor, deck)
    output = []
    for decision_index, (frame_index, raw_observation, target, _duration) in enumerate(frames):
        observation = project_observation(raw_observation)
        select = observation["select"]
        ordered, termination = validate_ordered_action(
            target,
            option_count=len(select["option"]),
            min_count=int(select["minCount"]),
            max_count=int(select["maxCount"]),
            capacity=64,
        )
        identity = {
            "date": episode["episode_date"],
            "episode_id": int(episode["episode_id"]),
            "player_index": actor,
            "episode_step": frame_index,
            "actor_decision_index": decision_index,
        }
        source = {
            "identity": identity,
            "split": episode["split"],
            "deck_manifest": DeckManifest.from_card_ids(deck).as_dict(),
            "actor_observation": observation,
            "ordered_action": list(ordered),
            "action_termination": termination,
        }
        snapshot = knowledge.consume(
            observation,
            {
                "visual_frame_index": frame_index,
                "actor_decision_index": decision_index,
                "incoming_log_count": len(observation.get("logs", [])),
            },
        )
        compiled = compile_canonical_row(source, snapshot, prototypes)
        record = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "actor_schema_version": ACTOR_SCHEMA_VERSION,
            "actor": compiled["actor"],
            "target": compiled["target"],
            "labels": {
                "value_target": int(trajectory["value_target"]),
                "archetype_target": int(trajectory["opponent_archetype_target"]),
                "final_diff_target": int(trajectory["final_diff_target"]),
                "episode_weight": episode_weight,
                "is_exact_007": bool(trajectory.get("is_exact_007", False)),
            },
            "audit": {
                "identity": identity,
                "split": episode["split"],
                "terminal_outcome": trajectory["terminal_outcome"],
                "final_diff": int(trajectory["final_diff"]),
                "own_deck_sha256": trajectory["own_deck_sha256"],
                "opponent_deck_sha256": trajectory["opponent_deck_sha256"],
                "opponent_archetype_name": trajectory["opponent_archetype_name"],
                "is_exact_007": bool(trajectory.get("is_exact_007", False)),
                "source_payload_sha256": episode["payload_sha256"],
            },
        }
        lengths = {
            name: len(compiled["actor"][name])
            for name in ("card_cat", "resource_cat", "event_cat", "option_cat", "option_skill_id", "option_effect_id")
        }
        output.append((str(episode["split"]), _canonical(record), lengths, (actor, len(frames))))
    return output


def _compile_episode(episode: Mapping[str, Any]) -> list[tuple[str, str, dict[str, int], tuple[int, int]]]:
    if _WORKER_PROTOTYPES is None:
        raise RuntimeError("0036 materialize worker is not initialized")
    payload = json.loads(_payload(episode))
    output = []
    for trajectory in episode["trajectories"]:
        output.extend(_compile_trajectory(payload, episode, trajectory, _WORKER_PROTOTYPES))
    return output


def _compiled_episodes(
    episodes: list[Mapping[str, Any]], public_prototypes: Path, engine_prototypes: Path, workers: int
) -> Iterator[list[tuple[str, str, dict[str, int], tuple[int, int]]]]:
    if workers < 1:
        raise ValueError("0036 materialization workers must be positive")
    if workers == 1:
        _init_worker(str(public_prototypes), str(engine_prototypes))
        for episode in episodes:
            yield _compile_episode(episode)
        return
    pending: deque[Future] = deque()
    values = iter(episodes)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(public_prototypes), str(engine_prototypes)),
    ) as pool:
        for _ in range(workers * 2):
            try:
                pending.append(pool.submit(_compile_episode, next(values)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(pool.submit(_compile_episode, next(values)))
            except StopIteration:
                pass


class ShardWriter:
    def __init__(self, root: Path, records_per_shard: int, compression_level: int) -> None:
        self.root = root
        self.records_per_shard = records_per_shard
        self.compression_level = compression_level
        self.pending: dict[str, list[str]] = {"train": [], "validation": []}
        self.shards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}

    def add(self, split: str, serialized: str) -> None:
        self.pending[split].append(serialized)
        if len(self.pending[split]) >= self.records_per_shard:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.pending[split]
        if not rows:
            return
        name = f"{split}-{len(self.shards[split]):05d}.jsonl.gz"
        final = self.root / name
        temporary = self.root / f".{name}.{uuid.uuid4().hex}.tmp"
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=self.compression_level) as handle:
            handle.writelines(rows)
        os.replace(temporary, final)
        self.shards[split].append(
            {"path": name, "count": len(rows), "bytes": final.stat().st_size, "sha256": _sha256(final)}
        )
        rows.clear()

    def close(self) -> None:
        self.flush("train")
        self.flush("validation")


def materialize(
    catalog_path: Path,
    output: Path,
    public_prototypes: Path,
    engine_prototypes: Path,
    *,
    workers: int = 8,
    records_per_shard: int = 4096,
    compression_level: int = 3,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite 0036 dataset: {output}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    episodes = list(catalog["episodes"])
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir()
    shutil.copy2(public_prototypes, stage / public_prototypes.name)
    shutil.copy2(engine_prototypes, stage / engine_prototypes.name)
    writer = ShardWriter(stage, records_per_shard, compression_level)
    counts: Counter[str] = Counter()
    maxima: Counter[str] = Counter()
    started = time.perf_counter()
    try:
        for episode_index, group in enumerate(
            _compiled_episodes(episodes, public_prototypes, engine_prototypes, workers), start=1
        ):
            for split, serialized, lengths, _trajectory in group:
                writer.add(split, serialized)
                counts[split] += 1
                for name, length in lengths.items():
                    maxima[name] = max(maxima[name], length)
            if episode_index % 100 == 0:
                print(json.dumps({"event": "0036_materialize_progress", "episodes": episode_index, "decisions": sum(counts.values()), "seconds": time.perf_counter() - started}, sort_keys=True), flush=True)
        writer.close()
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "status": "complete",
            "catalog_path": str(catalog_path),
            "catalog_file_sha256": _sha256(catalog_path),
            "catalog_sha256": catalog["catalog_sha256"],
            "source_checkpoint_required_sha256": "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8",
            "actor_schema_version": ACTOR_SCHEMA_VERSION,
            "episodes": len(episodes),
            "split_episode_counts": dict(Counter(row["split"] for row in episodes)),
            "decisions": sum(counts.values()),
            "split_decision_counts": {key: counts[key] for key in ("train", "validation")},
            "records_per_shard": records_per_shard,
            "gzip_compression_level": compression_level,
            "materialization_workers": workers,
            "maximum_lengths": dict(maxima),
            "shards": writer.shards,
            "episode_weight_contract": "one_over_actor_decisions_in_episode",
            "label_contract": "pre_action_callback_terminal_win_archetype_signed_final_prize_diff",
            "source_identity_actor_visible": False,
            "materialization_seconds": time.perf_counter() - started,
        }
        (stage / "manifest.json").write_text(_canonical(manifest), encoding="utf-8")
        stage.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_dataset(root: Path, report_path: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    weights: Counter[tuple[int, int]] = Counter()
    labels: Counter[tuple[str, int]] = Counter()
    archetypes: Counter[tuple[str, int]] = Counter()
    final_diffs: Counter[tuple[str, int]] = Counter()
    for split in ("train", "validation"):
        for shard in manifest["shards"][split]:
            path = root / shard["path"]
            if _sha256(path) != shard["sha256"]:
                raise ValueError(f"0036 shard hash mismatch: {path}")
            local = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["schema_version"] != DATASET_SCHEMA_VERSION or row["audit"]["split"] != split:
                        raise ValueError("0036 row schema/split mismatch")
                    target = row["labels"]
                    if target["value_target"] not in (0, 1) or not 0 <= target["archetype_target"] < 15 or not 0 <= target["final_diff_target"] < 13:
                        raise ValueError("0036 label outside contract")
                    identity = row["audit"]["identity"]
                    weights[(int(identity["episode_id"]), int(identity["player_index"]))] += float(target["episode_weight"])
                    labels[(split, int(target["value_target"]))] += 1
                    archetypes[(split, int(target["archetype_target"]))] += 1
                    final_diffs[(split, int(target["final_diff_target"]) - 6)] += 1
                    local += 1
            if local != shard["count"]:
                raise ValueError(f"0036 shard row count mismatch: {path}")
            counts[split] += local
    bad_weights = {key: value for key, value in weights.items() if abs(value - 1.0) > 1e-5}
    if bad_weights:
        raise ValueError(f"0036 trajectory weights do not sum to one: {list(bad_weights.items())[:3]}")
    if {key: counts[key] for key in ("train", "validation")} != manifest["split_decision_counts"]:
        raise ValueError("0036 manifest decision counts mismatch")
    result = {
        "status": "verified",
        "dataset_schema_version": manifest["schema_version"],
        "catalog_sha256": manifest["catalog_sha256"],
        "decisions": sum(counts.values()),
        "trajectory_weights": len(weights),
        "label_counts": [
            {"split": key[0], "value_target": key[1], "decisions": value}
            for key, value in sorted(labels.items())
        ],
        "archetype_decision_counts": [
            {"split": key[0], "class_id": key[1], "decisions": value}
            for key, value in sorted(archetypes.items())
        ],
        "final_diff_decision_counts": [
            {"split": key[0], "final_diff": key[1], "decisions": value}
            for key, value in sorted(final_diffs.items())
        ],
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(_canonical(result), encoding="utf-8")
        os.replace(temporary, report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--public-prototypes", type=Path, default=Path(__file__).resolve().parents[1] / "assets/official_public_prototypes_v1.json")
    build.add_argument("--engine-prototypes", type=Path, default=Path(__file__).resolve().parents[1] / "assets/official_full_engine_prototypes_v2.json")
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--records-per-shard", type=int, default=4096)
    build.add_argument("--compression-level", type=int, default=3)
    build.add_argument("--max-episodes", type=int)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = materialize(args.catalog, args.output, args.public_prototypes, args.engine_prototypes, workers=args.workers, records_per_shard=args.records_per_shard, compression_level=args.compression_level, max_episodes=args.max_episodes)
    else:
        result = verify_dataset(args.dataset, args.report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["DATASET_SCHEMA_VERSION", "materialize", "verify_dataset"]
