from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rl_environment.batch import UNIVERSAL_INPUT_KEYS, collate_encoded
from rl_environment.streaming import aligned_records, iter_jsonl


SUPPORTED_DATASET_VERSIONS = frozenset({"ptcg_kaggle_bc_v1", "ptcg_kaggle_bc_universal"})
MODEL_INPUT_KEYS = (
    "state_numeric",
    "state_card_ids",
    "action_type_ids",
    "action_card_ids",
    "action_target_ids",
    "action_numeric",
    "action_mask",
)


def _identity(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["episode_id"]),
        int(record["player_index"]),
        int(record["episode_step"]),
    )


def _validate_record(
    record: dict[str, Any], location: str, expected_width: int | None = None
) -> int:
    if record.get("dataset_version") not in SUPPORTED_DATASET_VERSIONS:
        raise ValueError(f"unsupported full-action record at {location}")
    encoded = record.get("encoded")
    targets = record.get("targets")
    if not isinstance(encoded, dict) or not isinstance(targets, list):
        raise ValueError(f"malformed full-action record at {location}")
    mask = encoded.get("action_mask")
    if not isinstance(mask, list) or not mask or not any(mask):
        raise ValueError(f"missing legal action mask at {location}")
    if expected_width is not None and len(mask) != expected_width:
        raise ValueError(f"inconsistent candidate width at {location}")
    minimum = int(record.get("selection_min_count", 0))
    maximum = int(record.get("selection_max_count", len(targets)))
    count = int(record.get("target_count", len(targets)))
    legal_count = sum(bool(value) for value in mask)
    if count != len(targets) or len(set(targets)) != len(targets):
        raise ValueError(f"invalid target cardinality at {location}")
    if not 0 <= minimum <= count <= maximum <= legal_count:
        raise ValueError(f"selection count is outside legal range at {location}")
    if any(
        isinstance(target, bool)
        or not isinstance(target, int)
        or target < 0
        or target >= len(mask)
        or not mask[target]
        for target in targets
    ):
        raise ValueError(f"illegal target at {location}")
    return len(mask)


def iter_full_action_records(path: str | Path) -> Iterator[dict[str, Any]]:
    expected_width: int | None = None
    for line_number, record in enumerate(iter_jsonl(path), 1):
        expected_width = _validate_record(record, f"line {line_number}", expected_width)
        yield record


def load_full_action_dataset(path: str | Path) -> list[dict[str, Any]]:
    records = list(iter_full_action_records(path))
    if not records:
        raise ValueError(f"dataset contains no records: {path}")
    experts = {
        str(record.get("expert_team_name"))
        for record in records
        if record.get("expert_team_name")
    }
    if len(experts) != 1:
        raise ValueError(f"stage-two BC requires exactly one expert team; found {sorted(experts)}")
    return records


def _split_path(path: Path, split: str) -> Path | None:
    candidate = path.with_name(f"{path.stem}.{split}{path.suffix}")
    return candidate if candidate.is_file() else None


def _reward_rows(path: Path) -> Iterator[dict[str, Any]]:
    for line_number, row in enumerate(iter_jsonl(path), 1):
        if not isinstance(row.get("reward_metrics"), dict):
            raise ValueError(f"invalid reward sidecar at {path}:{line_number}")
        yield row


def iter_split_records(
    dataset: Path,
    split: str,
    *,
    reward_sidecar: Path | None = None,
) -> Iterator[dict[str, Any]]:
    data_split_path = _split_path(dataset, split)
    data_path = data_split_path or dataset
    reward_split_path = _split_path(reward_sidecar, split) if reward_sidecar else None
    reward_path = (reward_split_path or reward_sidecar) if reward_sidecar else None
    if reward_sidecar is not None and bool(data_split_path) != bool(reward_split_path):
        raise ValueError(
            f"dataset and reward sidecar must both provide split file {split!r} or neither"
        )
    primary = iter_full_action_records(data_path)
    if reward_path is not None:
        paired = aligned_records(primary, _reward_rows(reward_path), identity=_identity)
        records = (
            {
                **record,
                "reward_schema_version": reward.get("reward_schema_version"),
                "reward_metrics": reward["reward_metrics"],
            }
            for record, reward in paired
        )
    else:
        records = primary
    for record in records:
        if data_split_path is None and str(record.get("split", "train")) != split:
            continue
        yield record


def scan_streaming_dataset(
    dataset: Path, *, reward_sidecar: Path | None = None
) -> tuple[dict[str, int], str]:
    counts = {"train": 0, "validation": 0, "test": 0}
    experts: set[str] = set()
    for split in counts:
        for record in iter_split_records(dataset, split, reward_sidecar=reward_sidecar):
            counts[split] += 1
            if record.get("expert_team_name"):
                experts.add(str(record["expert_team_name"]))
    if not counts["train"] or not counts["validation"]:
        raise ValueError("streaming dataset must include train and validation records")
    if len(experts) != 1:
        raise ValueError(f"stage-two BC requires exactly one expert team; found {sorted(experts)}")
    return counts, next(iter(experts))


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "validation": [], "test": []}
    for record in records:
        split = str(record.get("split", "train"))
        if split not in splits:
            raise ValueError(f"unknown dataset split: {split}")
        splits[split].append(record)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("dataset must include non-empty train and validation splits")
    return splits


def collate_records(records: list[dict[str, Any]]) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot collate empty full-action records")
    batch = collate_encoded([record["encoded"] for record in records])
    width = batch["action_mask"].shape[1]
    target_mask = torch.zeros((len(records), width), dtype=torch.float32)
    counts: list[int] = []
    minimums: list[int] = []
    maximums: list[int] = []
    for row, record in enumerate(records):
        targets = [int(target) for target in record["targets"]]
        for target in targets:
            target_mask[row, target] = 1.0
        counts.append(len(targets))
        minimums.append(int(record.get("selection_min_count", 0)))
        maximums.append(int(record.get("selection_max_count", len(targets))))
    batch["target_mask"] = target_mask
    batch["target_count"] = torch.tensor(counts, dtype=torch.long)
    batch["selection_min_count"] = torch.tensor(minimums, dtype=torch.long)
    batch["selection_max_count"] = torch.tensor(maximums, dtype=torch.long)
    return batch


def model_inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    keys = MODEL_INPUT_KEYS + tuple(key for key in UNIVERSAL_INPUT_KEYS if key in batch)
    return {key: batch[key] for key in keys}
