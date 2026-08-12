from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


CORPUS_SCHEMA_VERSION = 1
POLICY_CODEC_V1 = "pure_policy_codec_v1"
GLOBAL_CAT_DIM = 8
GLOBAL_NUM_DIM = 16
ENTITY_CAT_DIM = 6
ENTITY_NUM_DIM = 10
OPTION_CAT_DIM = 12
OPTION_NUM_DIM = 4
SCALAR_DIM = 9

ARRAY_DTYPES = {
    "global_cat": np.dtype(np.int64),
    "global_num": np.dtype(np.float32),
    "entity_offsets": np.dtype(np.int64),
    "entity_cat": np.dtype(np.int64),
    "entity_num": np.dtype(np.float32),
    "entity_parent": np.dtype(np.int64),
    "option_offsets": np.dtype(np.int64),
    "option_cat": np.dtype(np.int64),
    "option_num": np.dtype(np.float32),
    "option_equiv": np.dtype(np.int64),
    "scalars": np.dtype(np.int64),
    "state_digest": np.dtype(np.uint64),
    "game_index": np.dtype(np.int32),
    "step": np.dtype(np.int32),
    "select_type": np.dtype(np.int16),
    "select_context": np.dtype(np.int16),
    "select_player": np.dtype(np.int8),
    "acting_agent_id": np.dtype(np.int16),
    "action_offsets": np.dtype(np.int64),
    "actions": np.dtype(np.int32),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("wb") as raw:
        with zipfile.ZipFile(
            raw,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.lib.format.write_array(
                    payload,
                    np.ascontiguousarray(arrays[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _require_shape(array: np.ndarray, shape: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(shape):
        raise ValueError(f"{name} rank {array.ndim} != {len(shape)}")
    for axis, expected in enumerate(shape):
        if expected is not None and array.shape[axis] != expected:
            raise ValueError(
                f"{name} shape {array.shape} has axis {axis} != {expected}"
            )


def _validate_offsets(
    offsets: np.ndarray,
    *,
    decisions: int,
    flat_count: int,
    name: str,
) -> None:
    _require_shape(offsets, (decisions + 1,), name)
    if offsets.dtype != np.int64:
        raise ValueError(f"{name} dtype {offsets.dtype} != int64")
    if int(offsets[0]) != 0 or int(offsets[-1]) != flat_count:
        raise ValueError(f"{name} endpoints do not cover the flattened values")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} is not monotonic")


def validate_policy_codec_v1_shard(
    arrays: Mapping[str, np.ndarray],
    *,
    entity_capacity: int,
    option_capacity: int,
) -> None:
    missing = set(ARRAY_DTYPES) - set(arrays)
    extra = set(arrays) - set(ARRAY_DTYPES)
    if missing or extra:
        raise ValueError(
            f"corpus shard fields differ: missing={sorted(missing)} extra={sorted(extra)}"
        )
    for name, dtype in ARRAY_DTYPES.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"{name} dtype {arrays[name].dtype} != {dtype}")

    decisions = int(arrays["global_cat"].shape[0])
    _require_shape(arrays["global_cat"], (decisions, GLOBAL_CAT_DIM), "global_cat")
    _require_shape(arrays["global_num"], (decisions, GLOBAL_NUM_DIM), "global_num")
    _require_shape(arrays["scalars"], (decisions, SCALAR_DIM), "scalars")
    for name in (
        "state_digest",
        "game_index",
        "step",
        "select_type",
        "select_context",
        "select_player",
        "acting_agent_id",
    ):
        _require_shape(arrays[name], (decisions,), name)

    entities = int(arrays["entity_cat"].shape[0])
    options = int(arrays["option_cat"].shape[0])
    actions = int(arrays["actions"].shape[0])
    _require_shape(arrays["entity_cat"], (entities, ENTITY_CAT_DIM), "entity_cat")
    _require_shape(arrays["entity_num"], (entities, ENTITY_NUM_DIM), "entity_num")
    _require_shape(arrays["entity_parent"], (entities,), "entity_parent")
    _require_shape(arrays["option_cat"], (options, OPTION_CAT_DIM), "option_cat")
    _require_shape(arrays["option_num"], (options, OPTION_NUM_DIM), "option_num")
    _require_shape(arrays["option_equiv"], (options,), "option_equiv")
    _require_shape(arrays["actions"], (actions,), "actions")
    _validate_offsets(
        arrays["entity_offsets"],
        decisions=decisions,
        flat_count=entities,
        name="entity_offsets",
    )
    _validate_offsets(
        arrays["option_offsets"],
        decisions=decisions,
        flat_count=options,
        name="option_offsets",
    )
    _validate_offsets(
        arrays["action_offsets"],
        decisions=decisions,
        flat_count=actions,
        name="action_offsets",
    )

    entity_counts = np.diff(arrays["entity_offsets"])
    option_counts = np.diff(arrays["option_offsets"])
    action_counts = np.diff(arrays["action_offsets"])
    if np.any(entity_counts > entity_capacity):
        raise ValueError("corpus shard exceeds entity capacity")
    if np.any(option_counts <= 0) or np.any(option_counts > option_capacity):
        raise ValueError("corpus shard has invalid option count")
    if not np.array_equal(entity_counts, arrays["scalars"][:, 0]):
        raise ValueError("entity offsets disagree with scalar counts")
    if not np.array_equal(option_counts, arrays["scalars"][:, 1]):
        raise ValueError("option offsets disagree with scalar counts")
    if np.any(arrays["scalars"][:, 2] < 0):
        raise ValueError("negative min_count")
    if np.any(arrays["scalars"][:, 3] < arrays["scalars"][:, 2]):
        raise ValueError("max_count is below min_count")
    if np.any(action_counts < arrays["scalars"][:, 2]):
        raise ValueError("captured action is shorter than min_count")
    if np.any(action_counts > arrays["scalars"][:, 3]):
        raise ValueError("captured action is longer than max_count")
    for index in range(decisions):
        begin = int(arrays["action_offsets"][index])
        end = int(arrays["action_offsets"][index + 1])
        row = arrays["actions"][begin:end]
        if np.any(row < 0) or np.any(row >= option_counts[index]):
            raise ValueError(f"captured action {index} has an out-of-range option")
        if len(np.unique(row)) != len(row):
            raise ValueError(f"captured action {index} contains duplicates")


@dataclass(frozen=True)
class CorpusShard:
    path: Path
    arrays: Mapping[str, np.ndarray]

    @property
    def decisions(self) -> int:
        return int(self.arrays["global_cat"].shape[0])


class PolicyCodecV1CorpusWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        shard_decisions: int = 4096,
        entity_capacity: int = 128,
        option_capacity: int = 80,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        if shard_decisions <= 0 or entity_capacity <= 0 or option_capacity <= 0:
            raise ValueError("corpus capacities must be positive")
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"corpus output is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_decisions = shard_decisions
        self.entity_capacity = entity_capacity
        self.option_capacity = option_capacity
        self.metadata = dict(metadata or {})
        self._chunks: dict[str, list[np.ndarray]] = {
            name: [] for name in ARRAY_DTYPES if name not in {"entity_offsets", "option_offsets", "action_offsets"}
        }
        self._entity_counts: list[np.ndarray] = []
        self._option_counts: list[np.ndarray] = []
        self._action_counts: list[np.ndarray] = []
        self._buffered_decisions = 0
        self._totals = {"decisions": 0, "entities": 0, "options": 0, "actions": 0}
        self._shards: list[dict[str, Any]] = []
        self._closed = False

    def append_batch(
        self,
        batch: Any,
        *,
        state_digests: Sequence[int],
        game_indices: Sequence[int],
        steps: Sequence[int],
        select_types: Sequence[int],
        select_contexts: Sequence[int],
        select_players: Sequence[int],
        acting_agent_ids: Sequence[int],
        actions: Sequence[Sequence[int]],
    ) -> None:
        if self._closed:
            raise RuntimeError("corpus writer is closed")
        scalars = np.asarray(batch.scalars, dtype=np.int64)
        _require_shape(scalars, (None, SCALAR_DIM), "scalars")
        count = int(scalars.shape[0])
        metadata_rows = (
            state_digests,
            game_indices,
            steps,
            select_types,
            select_contexts,
            select_players,
            acting_agent_ids,
            actions,
        )
        if any(len(values) != count for values in metadata_rows):
            raise ValueError("corpus metadata batch sizes differ")

        entity_counts = scalars[:, 0].astype(np.int64, copy=True)
        option_counts = scalars[:, 1].astype(np.int64, copy=True)
        if np.any(entity_counts < 0) or np.any(entity_counts > self.entity_capacity):
            raise ValueError("native batch exceeds corpus entity capacity")
        if np.any(option_counts <= 0) or np.any(option_counts > self.option_capacity):
            raise ValueError("native batch exceeds corpus option capacity")
        if batch.entity_cat.shape[0] != count or batch.option_cat.shape[0] != count:
            raise ValueError("native tensor batch sizes differ")

        self._chunks["global_cat"].append(np.asarray(batch.global_cat, dtype=np.int64).copy())
        self._chunks["global_num"].append(np.asarray(batch.global_num, dtype=np.float32).copy())
        self._chunks["scalars"].append(scalars.copy())
        self._chunks["state_digest"].append(np.asarray(state_digests, dtype=np.uint64))
        self._chunks["game_index"].append(np.asarray(game_indices, dtype=np.int32))
        self._chunks["step"].append(np.asarray(steps, dtype=np.int32))
        self._chunks["select_type"].append(np.asarray(select_types, dtype=np.int16))
        self._chunks["select_context"].append(np.asarray(select_contexts, dtype=np.int16))
        self._chunks["select_player"].append(np.asarray(select_players, dtype=np.int8))
        self._chunks["acting_agent_id"].append(np.asarray(acting_agent_ids, dtype=np.int16))
        self._entity_counts.append(entity_counts)
        self._option_counts.append(option_counts)

        entity_cat_rows: list[np.ndarray] = []
        entity_num_rows: list[np.ndarray] = []
        entity_parent_rows: list[np.ndarray] = []
        option_cat_rows: list[np.ndarray] = []
        option_num_rows: list[np.ndarray] = []
        option_equiv_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        action_counts = np.zeros(count, dtype=np.int64)
        for index in range(count):
            entity_count = int(entity_counts[index])
            option_count = int(option_counts[index])
            entity_cat_rows.append(
                np.asarray(batch.entity_cat[index, :entity_count], dtype=np.int64).copy()
            )
            entity_num_rows.append(
                np.asarray(batch.entity_num[index, :entity_count], dtype=np.float32).copy()
            )
            entity_parent_rows.append(
                np.asarray(batch.entity_parent[index, :entity_count], dtype=np.int64).copy()
            )
            option_cat_rows.append(
                np.asarray(batch.option_cat[index, :option_count], dtype=np.int64).copy()
            )
            option_num_rows.append(
                np.asarray(batch.option_num[index, :option_count], dtype=np.float32).copy()
            )
            option_equiv_rows.append(
                np.asarray(batch.option_equiv[index, :option_count], dtype=np.int64).copy()
            )
            action = np.asarray(actions[index], dtype=np.int32)
            action_counts[index] = len(action)
            action_rows.append(action)
        self._chunks["entity_cat"].append(np.concatenate(entity_cat_rows, axis=0))
        self._chunks["entity_num"].append(np.concatenate(entity_num_rows, axis=0))
        self._chunks["entity_parent"].append(np.concatenate(entity_parent_rows, axis=0))
        self._chunks["option_cat"].append(np.concatenate(option_cat_rows, axis=0))
        self._chunks["option_num"].append(np.concatenate(option_num_rows, axis=0))
        self._chunks["option_equiv"].append(np.concatenate(option_equiv_rows, axis=0))
        self._chunks["actions"].append(
            np.concatenate(action_rows) if action_rows else np.empty(0, dtype=np.int32)
        )
        self._action_counts.append(action_counts)
        self._buffered_decisions += count
        if self._buffered_decisions >= self.shard_decisions:
            self._flush()

    @staticmethod
    def _offsets(counts: np.ndarray) -> np.ndarray:
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return offsets

    def _flush(self) -> None:
        if not self._buffered_decisions:
            return
        arrays = {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in self._chunks.items()
        }
        entity_counts = np.concatenate(self._entity_counts)
        option_counts = np.concatenate(self._option_counts)
        action_counts = np.concatenate(self._action_counts)
        arrays["entity_offsets"] = self._offsets(entity_counts)
        arrays["option_offsets"] = self._offsets(option_counts)
        arrays["action_offsets"] = self._offsets(action_counts)
        validate_policy_codec_v1_shard(
            arrays,
            entity_capacity=self.entity_capacity,
            option_capacity=self.option_capacity,
        )
        index = len(self._shards)
        name = f"policy_codec_v1_{index:05d}.npz"
        path = self.output_dir / name
        _write_deterministic_npz(path, arrays)
        shard = {
            "path": name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "decisions": int(self._buffered_decisions),
            "entities": int(len(arrays["entity_cat"])),
            "options": int(len(arrays["option_cat"])),
            "actions": int(len(arrays["actions"])),
            "first_game_index": int(arrays["game_index"][0]),
            "last_game_index": int(arrays["game_index"][-1]),
            "first_step": int(arrays["step"][0]),
            "last_step": int(arrays["step"][-1]),
        }
        self._shards.append(shard)
        for key in self._totals:
            self._totals[key] += int(shard[key])
        self._chunks = {
            name: [] for name in ARRAY_DTYPES if name not in {"entity_offsets", "option_offsets", "action_offsets"}
        }
        self._entity_counts.clear()
        self._option_counts.clear()
        self._action_counts.clear()
        self._buffered_decisions = 0

    def close(self, *, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("corpus writer is already closed")
        self._flush()
        self._closed = True
        content_digest = hashlib.sha256()
        for shard in self._shards:
            content_digest.update(bytes.fromhex(str(shard["sha256"])))
        manifest = {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "codec_version": POLICY_CODEC_V1,
            "dimensions": {
                "global_cat": GLOBAL_CAT_DIM,
                "global_num": GLOBAL_NUM_DIM,
                "entity_cat": ENTITY_CAT_DIM,
                "entity_num": ENTITY_NUM_DIM,
                "option_cat": OPTION_CAT_DIM,
                "option_num": OPTION_NUM_DIM,
                "scalars": SCALAR_DIM,
            },
            "capacity": {
                "entities": self.entity_capacity,
                "options": self.option_capacity,
            },
            "shard_decisions": self.shard_decisions,
            "totals": dict(self._totals),
            "content_sha256": content_digest.hexdigest(),
            "metadata": self.metadata,
            "summary": dict(summary or {}),
            "shards": self._shards,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return manifest


def load_policy_codec_v1_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported corpus schema version")
    if manifest.get("codec_version") != POLICY_CODEC_V1:
        raise ValueError("manifest is not a PolicyCodecV1 corpus")
    if not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
        raise ValueError("corpus manifest has no shards")
    return manifest


def iter_policy_codec_v1_shards(
    path: str | Path,
    *,
    verify_sha256: bool = True,
) -> Iterator[CorpusShard]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = load_policy_codec_v1_manifest(manifest_path)
    root = manifest_path.parent
    entity_capacity = int(manifest["capacity"]["entities"])
    option_capacity = int(manifest["capacity"]["options"])
    for row in manifest["shards"]:
        shard_path = root / str(row["path"])
        if verify_sha256 and sha256_file(shard_path) != str(row["sha256"]):
            raise ValueError(f"corpus shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        validate_policy_codec_v1_shard(
            arrays,
            entity_capacity=entity_capacity,
            option_capacity=option_capacity,
        )
        if int(row["decisions"]) != int(arrays["global_cat"].shape[0]):
            raise ValueError(f"corpus shard decision count mismatch: {shard_path}")
        yield CorpusShard(path=shard_path, arrays=arrays)


def pad_policy_codec_v1_rows(
    arrays: Mapping[str, np.ndarray],
    begin: int,
    end: int,
    *,
    entity_capacity: int,
    option_capacity: int,
) -> dict[str, np.ndarray]:
    decisions = int(arrays["global_cat"].shape[0])
    if not 0 <= begin < end <= decisions:
        raise ValueError("invalid corpus row slice")
    batch_size = end - begin
    entity_cat = np.zeros((batch_size, entity_capacity, ENTITY_CAT_DIM), dtype=np.int64)
    entity_num = np.zeros((batch_size, entity_capacity, ENTITY_NUM_DIM), dtype=np.float32)
    entity_parent = np.full((batch_size, entity_capacity), -1, dtype=np.int64)
    entity_mask = np.zeros((batch_size, entity_capacity), dtype=np.bool_)
    option_cat = np.zeros((batch_size, option_capacity, OPTION_CAT_DIM), dtype=np.int64)
    option_num = np.zeros((batch_size, option_capacity, OPTION_NUM_DIM), dtype=np.float32)
    option_equiv = np.full((batch_size, option_capacity), -1, dtype=np.int64)
    option_mask = np.zeros((batch_size, option_capacity), dtype=np.bool_)
    scalars = arrays["scalars"][begin:end].copy()
    action_counts = np.diff(arrays["action_offsets"])[begin:end]
    max_actions = max(1, int(action_counts.max(initial=0)))
    actions = np.full((batch_size, max_actions), -1, dtype=np.int64)
    action_len = action_counts.astype(np.int64, copy=True)

    for output_index, source_index in enumerate(range(begin, end)):
        entity_begin = int(arrays["entity_offsets"][source_index])
        entity_end = int(arrays["entity_offsets"][source_index + 1])
        entity_count = entity_end - entity_begin
        entity_cat[output_index, :entity_count] = arrays["entity_cat"][entity_begin:entity_end]
        entity_num[output_index, :entity_count] = arrays["entity_num"][entity_begin:entity_end]
        entity_parent[output_index, :entity_count] = arrays["entity_parent"][entity_begin:entity_end]
        entity_mask[output_index, :entity_count] = True

        option_begin = int(arrays["option_offsets"][source_index])
        option_end = int(arrays["option_offsets"][source_index + 1])
        option_count = option_end - option_begin
        option_cat[output_index, :option_count] = arrays["option_cat"][option_begin:option_end]
        option_num[output_index, :option_count] = arrays["option_num"][option_begin:option_end]
        option_equiv[output_index, :option_count] = arrays["option_equiv"][option_begin:option_end]
        option_mask[output_index, :option_count] = True

        action_begin = int(arrays["action_offsets"][source_index])
        action_end = int(arrays["action_offsets"][source_index + 1])
        if action_end > action_begin:
            actions[output_index, : action_end - action_begin] = arrays["actions"][
                action_begin:action_end
            ]

    return {
        "global_cat": arrays["global_cat"][begin:end].copy(),
        "global_num": arrays["global_num"][begin:end].copy(),
        "entity_cat": entity_cat,
        "entity_num": entity_num,
        "entity_parent": entity_parent,
        "entity_mask": entity_mask,
        "option_cat": option_cat,
        "option_num": option_num,
        "option_equiv": option_equiv,
        "option_mask": option_mask,
        "min_count": scalars[:, 2],
        "max_count": scalars[:, 3],
        "action_family": scalars[:, 4],
        "actions": actions,
        "action_len": action_len,
        "state_digest": arrays["state_digest"][begin:end].copy(),
        "game_index": arrays["game_index"][begin:end].copy(),
        "step": arrays["step"][begin:end].copy(),
        "select_type": arrays["select_type"][begin:end].copy(),
        "select_context": arrays["select_context"][begin:end].copy(),
        "select_player": arrays["select_player"][begin:end].copy(),
        "acting_agent_id": arrays["acting_agent_id"][begin:end].copy(),
    }
