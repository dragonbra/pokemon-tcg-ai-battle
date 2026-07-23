from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rl_environment.batch import UNIVERSAL_INPUT_KEYS, collate_encoded


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


def load_full_action_dataset(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    expected_width: int | None = None
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid dataset JSON at line {line_number}") from exc
            if not isinstance(record, dict) or record.get("dataset_version") not in (
                SUPPORTED_DATASET_VERSIONS
            ):
                raise ValueError(f"unsupported full-action record at line {line_number}")
            encoded = record.get("encoded")
            targets = record.get("targets")
            if not isinstance(encoded, dict) or not isinstance(targets, list):
                raise ValueError(f"malformed full-action record at line {line_number}")
            mask = encoded.get("action_mask")
            if not isinstance(mask, list) or not mask or not any(mask):
                raise ValueError(f"missing legal action mask at line {line_number}")
            if expected_width is None:
                expected_width = len(mask)
            if len(mask) != expected_width:
                raise ValueError(f"inconsistent candidate width at line {line_number}")
            minimum = int(record.get("selection_min_count", 0))
            maximum = int(record.get("selection_max_count", len(targets)))
            count = int(record.get("target_count", len(targets)))
            legal_count = sum(bool(value) for value in mask)
            if count != len(targets) or len(set(targets)) != len(targets):
                raise ValueError(f"invalid target cardinality at line {line_number}")
            if not 0 <= minimum <= count <= maximum <= legal_count:
                raise ValueError(f"selection count is outside legal range at line {line_number}")
            if any(
                isinstance(target, bool)
                or not isinstance(target, int)
                or target < 0
                or target >= len(mask)
                or not mask[target]
                for target in targets
            ):
                raise ValueError(f"illegal target at line {line_number}")
            records.append(record)
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
