"""Construct the exact 0034 focal/frozen resident CUDA runtime."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor

from ..model import ModelConfig, PodNativeActorCritic
from ..model.frozen_heads import FrozenDeckHeads
from ..model.routed_policy import RoutedResidentPolicy
from ..transfer import load_0031_initialization


ROOT = Path(__file__).resolve().parents[3]
CUDA_ROOT = ROOT / "engine_cuda"
FOCAL_DECK_ID = "dragapult_third_ptcg_club"
FOCAL_DECK_FILE_SHA256 = "5db1e0d52fc723e8b2f688d76f780715fd726171f4d804f1d4d55673ba9b0ac2"
FOCAL_DECK_MULTISET_SHA256 = "f0314f941a872d37dcbaf70371eb576945cc78dbd9aebb8896fbea5e6d8777eb"
FOCAL_DECK_PATH = (
    ROOT / "train" / "0040_dragapult_0809_action_boundary_rl" / "league" / "decks"
    / FOCAL_DECK_ID / "deck.csv"
)


def read_deck(path: Path) -> list[int]:
    cards = [int(line) for line in path.read_text(encoding="ascii").splitlines() if line]
    if len(cards) != 60 or any(card_id <= 0 for card_id in cards):
        raise ValueError(f"deck must contain exactly 60 cards: {path}")
    return cards


def deck_multiset_sha256(cards: list[int]) -> str:
    encoded = ",".join(
        str(card_id)
        for card_id, count in sorted(Counter(cards).items())
        for _ in range(count)
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    engine: Any
    learner: PodNativeActorCritic
    policy: RoutedResidentPolicy
    decks: Tensor
    seeds: Tensor
    opponent_head_indices: Tensor
    opponent_deck_ids: tuple[str, ...]
    base_opponent_deck_ids: tuple[str, ...]
    full_opponent_deck_ids: tuple[str, ...]
    transfer_report: dict[str, Any]


def build_runtime(
    *,
    rules: Path,
    extension_dir: Path,
    source_checkpoint: Path,
    support_report: Path,
    frozen_root: Path,
    checkpoint_root: Path,
    shared_foundation_checkpoint: Path | None = None,
    copies_per_opponent: int = 4,
    opponents_per_cohort: int = 5,
    cohort_index: int = 0,
    seed_start: int = 320320001,
    model_seed: int = 32032,
    device: str = "cuda",
    learner: PodNativeActorCritic | None = None,
) -> RuntimeBundle:
    if copies_per_opponent < 1:
        raise ValueError("copies_per_opponent must be positive")
    for path in (CUDA_ROOT / "python", extension_dir):
        if str(path.resolve()) not in sys.path:
            sys.path.insert(0, str(path.resolve()))
    native = importlib.import_module("ptcg_cuda_engine.native")
    importlib.import_module("_ptcg_cuda")
    support = json.loads(support_report.read_text(encoding="utf-8"))
    support_rows = support.get("decks", support.get("opponents", []))
    hashes = {
        row["deck_id"]: row["deck_sha256"]
        for row in support_rows
        if row.get("status", "supported") == "supported"
    }
    if shared_foundation_checkpoint is None:
        full_ids = tuple(
            deck_id
            for deck_id in sorted(hashes)
            if (checkpoint_root / f"{deck_id}.pt").is_file()
        )
        if len(full_ids) != 35:
            raise ValueError(f"expected 35 CUDA+0022 frozen heads, got {len(full_ids)}")
    else:
        full_ids = tuple(sorted(hashes))
        if len(full_ids) != 38:
            raise ValueError(f"expected 38 CUDA-supported frozen decks, got {len(full_ids)}")
    if not 1 <= opponents_per_cohort <= len(full_ids):
        raise ValueError("opponents_per_cohort is outside the frozen pool")
    begin = (cohort_index * opponents_per_cohort) % len(full_ids)
    base_ids = tuple(
        full_ids[(begin + offset) % len(full_ids)]
        for offset in range(opponents_per_cohort)
    )
    if shared_foundation_checkpoint is None:
        checkpoints = [checkpoint_root / f"{deck_id}.pt" for deck_id in base_ids]
        frozen_heads = FrozenDeckHeads.from_checkpoints(
            checkpoints, expected_deck_hashes=hashes
        )
    else:
        frozen_heads = FrozenDeckHeads.from_0019_foundation(
            shared_foundation_checkpoint
        )
    if learner is None:
        torch.manual_seed(model_seed)
        learner = PodNativeActorCritic(ModelConfig()).to(device).eval()
        transfer_report = load_0031_initialization(learner, source_checkpoint)
    else:
        learner.to(device).eval()
        transfer_report = {
            "role": "resident_learner_reuse",
            "source_checkpoint": str(source_checkpoint),
        }
    frozen_heads.to(device).eval()
    policy = RoutedResidentPolicy(learner, frozen_heads).to(device).eval()

    lane_ids = tuple(
        deck_id for deck_id in base_ids for _ in range(copies_per_opponent)
    )
    focal = read_deck(FOCAL_DECK_PATH)
    if deck_multiset_sha256(focal) != FOCAL_DECK_MULTISET_SHA256:
        raise ValueError("0034 focal deck identity mismatch")
    opponent_decks = {deck_id: read_deck(frozen_root / deck_id / "deck.csv") for deck_id in base_ids}
    mismatched = [
        deck_id for deck_id, deck in opponent_decks.items()
        if deck_multiset_sha256(deck) != hashes[deck_id]
    ]
    if mismatched:
        raise ValueError(f"Frozen opponent deck identity mismatch: {mismatched[0]}")
    pairs = [
        [focal, opponent_decks[deck_id]]
        for deck_id in lane_ids
    ]
    decks = torch.tensor(pairs, dtype=torch.int32, device=device)
    seeds = torch.arange(
        seed_start, seed_start + len(lane_ids), dtype=torch.int64, device=device
    )
    if shared_foundation_checkpoint is None:
        head_indices = torch.arange(
            len(base_ids), dtype=torch.long, device=device
        ).repeat_interleave(copies_per_opponent)
    else:
        head_indices = torch.zeros(len(lane_ids), dtype=torch.long, device=device)
    engine = native.create_official_engine(
        rules.read_bytes(), batch_size=len(lane_ids)
    )
    return RuntimeBundle(
        engine=engine,
        learner=learner,
        policy=policy,
        decks=decks,
        seeds=seeds,
        opponent_head_indices=head_indices,
        opponent_deck_ids=lane_ids,
        base_opponent_deck_ids=base_ids,
        full_opponent_deck_ids=full_ids,
        transfer_report=transfer_report,
    )


__all__ = [
    "FOCAL_DECK_ID",
    "FOCAL_DECK_PATH",
    "FOCAL_DECK_FILE_SHA256",
    "FOCAL_DECK_MULTISET_SHA256",
    "RuntimeBundle",
    "build_runtime",
    "deck_multiset_sha256",
    "read_deck",
]
