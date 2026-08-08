"""Build a bounded PolicyCodecV1 diagnostic corpus from raw 0031 decisions."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CUDA_PYTHON = REPOSITORY_ROOT / "engine_cuda" / "python"
if str(CUDA_PYTHON) not in sys.path:
    sys.path.insert(0, str(CUDA_PYTHON))

from ptcg_cuda_engine.corpus import PolicyCodecV1CorpusWriter, sha256_file  # noqa: E402
from engine_cuda.tools.pure_policy_codec_v1 import PolicyCodecV1, compact_json  # noqa: E402


SCHEMA = "0034_pod_adaptation_corpus_v1"


@dataclass
class NumpyCodecBatch:
    global_cat: np.ndarray
    global_num: np.ndarray
    entity_cat: np.ndarray
    entity_num: np.ndarray
    entity_parent: np.ndarray
    option_cat: np.ndarray
    option_num: np.ndarray
    option_equiv: np.ndarray
    scalars: np.ndarray

    @classmethod
    def from_encoded(cls, encoded_rows: Sequence[Any]) -> "NumpyCodecBatch":
        if not encoded_rows:
            raise ValueError("encoded_rows must not be empty")
        batch_size = len(encoded_rows)
        entity_capacity = max(1, max(len(row.entity_cat) for row in encoded_rows))
        option_capacity = max(len(row.option_cat) for row in encoded_rows)
        entity_cat = np.zeros((batch_size, entity_capacity, 6), dtype=np.int64)
        entity_num = np.zeros((batch_size, entity_capacity, 10), dtype=np.float32)
        entity_parent = np.full((batch_size, entity_capacity), -1, dtype=np.int64)
        option_cat = np.zeros((batch_size, option_capacity, 12), dtype=np.int64)
        option_num = np.zeros((batch_size, option_capacity, 4), dtype=np.float32)
        option_equiv = np.full((batch_size, option_capacity), -1, dtype=np.int64)
        scalars = np.zeros((batch_size, 9), dtype=np.int64)
        for index, row in enumerate(encoded_rows):
            entities = len(row.entity_cat)
            options = len(row.option_cat)
            if entities:
                entity_cat[index, :entities] = row.entity_cat
                entity_num[index, :entities] = row.entity_num
                entity_parent[index, :entities] = row.entity_parent
            option_cat[index, :options] = row.option_cat
            option_num[index, :options] = row.option_num
            option_equiv[index, :options] = row.option_equiv
            scalars[index] = (
                entities,
                options,
                row.min_count,
                row.max_count,
                row.action_family,
                row.actor_index,
                row.turn,
                row.own_prizes,
                row.opp_prizes,
            )
        return cls(
            global_cat=np.asarray([row.global_cat for row in encoded_rows], dtype=np.int64),
            global_num=np.asarray([row.global_num for row in encoded_rows], dtype=np.float32),
            entity_cat=entity_cat,
            entity_num=entity_num,
            entity_parent=entity_parent,
            option_cat=option_cat,
            option_num=option_num,
            option_equiv=option_equiv,
            scalars=scalars,
        )


def iter_raw_rows(paths: Iterable[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for path in sorted(paths):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                yield path, line_number, json.loads(line)


def stable_state_digest(observation: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(compact_json(observation).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def build_split(
    rows: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    split: str,
    limit: int,
    shard_decisions: int,
    source_files: Sequence[Path] = (),
) -> dict[str, Any]:
    writer = PolicyCodecV1CorpusWriter(
        output_dir,
        shard_decisions=shard_decisions,
        entity_capacity=128,
        option_capacity=128,
        metadata={
            "project_id": "0038_action_boundary_rl",
            "purpose": "bounded_adapter_diagnostic",
            "split": split,
            "source_files": [str(path) for path in source_files],
        },
    )
    codec = PolicyCodecV1()
    selected = 0
    rejected = Counter()
    sources = Counter()
    decks = Counter()
    episodes: set[str] = set()
    source_ids: dict[str, int] = {}
    buffer: list[tuple[Any, dict[str, Any]]] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        encoded_rows = [item[0] for item in buffer]
        raw_rows = [item[1] for item in buffer]
        batch = NumpyCodecBatch.from_encoded(encoded_rows)
        actions = [[int(value) for value in row["ordered_action"]] for row in raw_rows]
        observations = [row["actor_observation"] for row in raw_rows]
        identities = [row.get("identity") or {} for row in raw_rows]
        selects = [(obs.get("select") or {}) for obs in observations]
        writer.append_batch(
            batch,
            state_digests=[stable_state_digest(obs) for obs in observations],
            game_indices=[int(identity.get("episode_id", 0)) for identity in identities],
            steps=[int(identity.get("episode_step", 0)) for identity in identities],
            select_types=[int(select.get("type", -1)) for select in selects],
            select_contexts=[int(select.get("context", -1)) for select in selects],
            select_players=[int(identity.get("player_index", 0)) for identity in identities],
            acting_agent_ids=[source_ids[str(row.get("source_id", "unknown"))] for row in raw_rows],
            actions=actions,
        )
        buffer = []

    for row in rows:
        if selected >= limit:
            break
        if str(row.get("split")) != split:
            rejected["other_split"] += 1
            continue
        observation = row.get("actor_observation")
        action = row.get("ordered_action")
        if not isinstance(observation, dict) or not isinstance(action, list):
            rejected["invalid_raw_contract"] += 1
            continue
        try:
            encoded = codec.encode(observation, selected_action=action)
        except (TypeError, ValueError):
            rejected["codec_error"] += 1
            continue
        if len(encoded.entity_cat) > 128:
            rejected["entity_capacity"] += 1
            continue
        if len(encoded.option_cat) > 128:
            rejected["option_capacity"] += 1
            continue
        if len(action) < encoded.min_count or len(action) > encoded.max_count:
            rejected["action_bounds"] += 1
            continue
        if len(set(action)) != len(action) or any(
            not isinstance(value, int) or not 0 <= value < len(encoded.option_cat)
            for value in action
        ):
            rejected["action_indices"] += 1
            continue
        source = str(row.get("source_id", "unknown"))
        source_ids.setdefault(source, len(source_ids))
        sources[source] += 1
        deck = str((row.get("deck_manifest") or {}).get("sha256", "unknown"))
        decks[deck] += 1
        identity = row.get("identity") or {}
        episodes.add(str(identity.get("episode_id", "unknown")))
        buffer.append((encoded, row))
        selected += 1
        if len(buffer) >= min(256, shard_decisions):
            flush()
    flush()
    if selected == 0:
        raise ValueError(f"no valid rows selected for split {split}")
    return writer.close(
        summary={
            "status": "pass",
            "selected_decisions": selected,
            "episode_count": len(episodes),
            "source_counts": dict(sorted(sources.items())),
            "source_id_map": dict(sorted(source_ids.items())),
            "deck_hash_counts": dict(sorted(decks.items())),
            "rejected_counts": dict(sorted(rejected.items())),
            "selection": "first valid rows in sorted immutable raw shard order",
        }
    )


def build_corpus(
    input_dir: Path,
    output_dir: Path,
    *,
    train_limit: int,
    validation_limit: int,
    shard_decisions: int = 1024,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    paths = sorted(input_dir.glob("*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no raw JSONL.GZ shards under {input_dir}")
    train_paths = [path for path in paths if path.name.startswith("train-")]
    validation_paths = [path for path in paths if path.name.startswith("validation-")]
    train = build_split(
        (row for _, _, row in iter_raw_rows(train_paths)),
        output_dir / "train",
        split="train",
        limit=train_limit,
        shard_decisions=shard_decisions,
        source_files=train_paths,
    )
    validation = build_split(
        (row for _, _, row in iter_raw_rows(validation_paths)),
        output_dir / "validation",
        split="validation",
        limit=validation_limit,
        shard_decisions=shard_decisions,
        source_files=validation_paths,
    )
    manifest = {
        "schema": SCHEMA,
        "project_id": "0038_action_boundary_rl",
        "codec_version": "pure_policy_codec_v1",
        "actor_visible_source_identity": False,
        "source_root": str(input_dir),
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in paths
        ],
        "splits": {
            "train": {
                "manifest": "train/manifest.json",
                "content_sha256": train["content_sha256"],
                "decisions": train["totals"]["decisions"],
            },
            "validation": {
                "manifest": "validation/manifest.json",
                "content_sha256": validation["content_sha256"],
                "decisions": validation["totals"]["decisions"],
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--shard-decisions", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_corpus(
        args.input,
        args.output,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        shard_decisions=args.shard_decisions,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
