"""Lightweight arm views over the single shared 0015 R15 feature cache."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor

from ..ablation import remove_initial_deck_information

_ac_data = importlib.import_module(
    "train.0014_faithful_board_causal_features.training.ac_data"
)

ARM_BUILD_IDS = {
    "t0": frozenset({1}),
    "t1": frozenset({1, 2}),
    "t2": frozenset({1, 3}),
    "t3": frozenset({1, 2, 3}),
    "t4": frozenset({1, 2, 3, 4}),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(root: Path | str) -> dict:
    base = Path(root)
    reference = json.loads((base / "feature_dataset_reference.json").read_text())
    if reference.get("schema_version") != "0015_r15_feature_cache_v1":
        raise ValueError("unsupported 0015 feature cache")
    return reference


def _load_shard(root: Path, item: dict) -> dict[str, Tensor]:
    path = root / item["path"]
    if _sha256(path) != item["sha256"]:
        raise ValueError(f"feature shard hash mismatch: {path}")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or not all(
        isinstance(value, Tensor) for value in tensors.values()
    ):
        raise ValueError(f"invalid feature shard: {path}")
    if any(value.size(0) != item["count"] for value in tensors.values()):
        raise ValueError(f"feature shard leading dimensions disagree: {path}")
    return tensors


def _eligible_indices(tensors: dict[str, Tensor], arm: str, split: str) -> Tensor:
    if arm not in ARM_BUILD_IDS:
        raise ValueError(f"unknown arm: {arm}")
    allowed = torch.zeros_like(tensors["a0_eligible"])
    for build_id in ARM_BUILD_IDS[arm]:
        allowed |= tensors["build_id"].eq(build_id)
    if split == "validation":
        allowed &= tensors["build_id"].eq(1)
    return (tensors["a0_eligible"] & allowed).nonzero(as_tuple=False).flatten()


def select_batch(
    tensors: dict[str, Tensor], indices: Tensor, *, no_deck: bool
) -> dict[str, Tensor]:
    batch = _ac_data.select_ac_batch(tensors, indices)
    for name in (
        "source_id",
        "build_id",
        "outcome_id",
        "episode_id",
        "player_index",
        "first_player",
    ):
        batch[name] = tensors[name].index_select(0, indices).long()
    return remove_initial_deck_information(batch) if no_deck else batch


def iter_batches(
    root: Path | str,
    split: str,
    *,
    arm: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    no_deck: bool = False,
) -> Iterator[dict[str, Tensor]]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if no_deck and arm not in {"t1", "t3"}:
        raise ValueError("the registered-deck ablation is defined only for T1 or T3")
    base = Path(root)
    reference = load_reference(base)
    shards = list(reference["shards"][split])
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(shards)
    for item in shards:
        tensors = _load_shard(base, item)
        indices = _eligible_indices(tensors, arm, split)
        if shuffle and indices.numel():
            generator = torch.Generator().manual_seed(rng.randrange(2**63))
            indices = indices[torch.randperm(indices.numel(), generator=generator)]
        for start in range(0, indices.numel(), batch_size):
            yield select_batch(tensors, indices[start : start + batch_size], no_deck=no_deck)


def decision_count(root: Path | str, split: str, arm: str) -> int:
    reference = load_reference(root)
    allowed = ARM_BUILD_IDS[arm]
    total = 0
    for item in reference["shards"][split]:
        for build_id, count in item["eligible_by_build_id"].items():
            if int(build_id) in allowed and (split != "validation" or int(build_id) == 1):
                total += int(count)
    return total


__all__ = ["ARM_BUILD_IDS", "decision_count", "iter_batches", "load_reference", "select_batch"]
