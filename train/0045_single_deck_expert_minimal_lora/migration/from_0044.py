"""Fail-closed 0044 checkpoint migration for the deck-007 specialist."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..assets import AssetRegistry, sha256_file
from ..policy.actor_critic import load_actor_critic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "0045_minimal_lora_model_only_v1"
RETIRED_PREFIXES = ("policy_strategy_adapter.", "meta_actor_residual.")
REBIND_KEYS = ("default_own_archetype_id",)


@dataclass(frozen=True, slots=True)
class MigrationReport:
    schema_version: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    source_schema_version: str
    source_update: int
    focal_deck_id: str
    focal_exact_deck_sha256: str
    copied_tensor_count: int
    copied_parameter_count: int
    copied_tensors: tuple[str, ...]
    dropped_tensor_count: int
    dropped_0044_only_tensors: tuple[str, ...]
    rebound_tensors: tuple[str, ...]
    base_materialized_tensors: tuple[str, ...]
    newly_initialized_tensors: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    equality_verified: bool
    destination_checkpoint: str
    destination_checkpoint_sha256: str


def _deck(deck_id: str) -> tuple[tuple[int, ...], str]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    row = next((item for item in registry.decks if item.deck_id == deck_id), None)
    if row is None:
        raise ValueError(f"unregistered focal deck: {deck_id}")
    path = PROJECT_ROOT / row.deck_path
    cards = tuple(int(value) for value in path.read_text(encoding="utf-8").splitlines())
    if len(cards) != 60:
        raise RuntimeError(f"focal deck {deck_id} is not exact 60-card")
    return cards, row.content_sha256


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def migrate_0044_checkpoint(
    source: Path,
    destination: Path,
    report_path: Path,
    *,
    deck_id: str = "007",
) -> MigrationReport:
    source = source.resolve()
    destination = destination.resolve()
    report_path = report_path.resolve()
    if deck_id != "007":
        raise ValueError("0045 V1 is fixed to exact focal deck 007")
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "0044_focal_v1_model_only_v1":
        raise RuntimeError("source is not an audited 0044 model-only checkpoint")
    source_state = payload.get("state_dict")
    if not isinstance(source_state, dict) or not source_state:
        raise RuntimeError("0044 checkpoint has no state_dict")

    cards, deck_hash = _deck(deck_id)
    model, base_identity = load_actor_critic(deck=cards, deck_id=deck_id)
    target_state = model.state_dict()
    dropped = tuple(sorted(
        name for name in source_state if name.startswith(RETIRED_PREFIXES)
    ))
    unclassified = tuple(sorted(
        name for name in source_state
        if name not in target_state
        and not name.startswith(RETIRED_PREFIXES)
    ))
    mismatches = tuple(sorted(
        f"{name}:source={tuple(source_state[name].shape)}:target={tuple(target_state[name].shape)}"
        for name in source_state
        if name in target_state
        and name not in REBIND_KEYS
        and tuple(source_state[name].shape) != tuple(target_state[name].shape)
    ))
    if unclassified or mismatches:
        raise RuntimeError(
            f"0044 migration classification failed: unexpected={unclassified}, "
            f"shape_mismatches={mismatches}"
        )
    copied = tuple(sorted(
        name for name in source_state
        if name in target_state and name not in REBIND_KEYS
    ))
    with torch.no_grad():
        for name in copied:
            target_state[name].copy_(source_state[name])
    incompatible = model.load_state_dict(target_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict target load failed: {incompatible}")
    loaded = model.state_dict()
    unequal = tuple(name for name in copied if not torch.equal(loaded[name], source_state[name]))
    if unequal:
        raise RuntimeError(f"copied migration tensors are not equal: {unequal[:8]}")

    base_materialized = tuple(sorted(
        name for name in target_state
        if name not in source_state and not name.startswith(RETIRED_PREFIXES)
    ))
    checkpoint_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if value.requires_grad
        or not name.startswith("actor.")
        or name.startswith("actor.action_decoder.")
    }
    migrated = {
        "schema_version": SCHEMA,
        "update": 0,
        "actor_schema": payload.get("actor_schema"),
        "state_dict": checkpoint_state,
        "adaptation": {
            "policy_only_option_lora": True,
            "option_block": 1,
            "attention_targets": ["self_attn.qv", "cross_attn.qv"],
            "rank": 4,
            "alpha": 8.0,
            "parameters": 10_240,
            "value_branch": "lora_free",
            "strategy_adapter": "removed",
            "meta_actor_residual": "removed",
        },
        "integrated_flags": payload.get("integrated_flags"),
        "metadata": {
            "project_id": "0045_single_deck_expert_minimal_lora",
            "version": "V1_minimal_lora_dragapult_007",
            "checkpoint_retention": "all",
            "focal_deck_id": deck_id,
            "focal_deck_ids": [deck_id],
            "focal_schedule": "fixed_007_v1",
            "source_0044_checkpoint": str(source),
            "source_0044_checkpoint_sha256": sha256_file(source),
            "source_0044_update": int(payload["update"]),
            "frozen_semantic_base_sha256": base_identity.checkpoint_sha256,
            "critic_only_own_archetype_conditioning": True,
        },
    }
    _atomic_torch_save(migrated, destination)

    report = MigrationReport(
        schema_version="0045_0044_migration_audit_v1",
        source_checkpoint=str(source),
        source_checkpoint_sha256=sha256_file(source),
        source_schema_version=str(payload["schema_version"]),
        source_update=int(payload["update"]),
        focal_deck_id=deck_id,
        focal_exact_deck_sha256=deck_hash,
        copied_tensor_count=len(copied),
        copied_parameter_count=sum(int(source_state[name].numel()) for name in copied),
        copied_tensors=copied,
        dropped_tensor_count=len(dropped),
        dropped_0044_only_tensors=dropped,
        rebound_tensors=REBIND_KEYS,
        base_materialized_tensors=base_materialized,
        newly_initialized_tensors=(),
        missing_keys=(),
        unexpected_keys=unclassified,
        shape_mismatches=mismatches,
        equality_verified=True,
        destination_checkpoint=str(destination),
        destination_checkpoint_sha256=sha256_file(destination),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["MigrationReport", "migrate_0044_checkpoint"]
