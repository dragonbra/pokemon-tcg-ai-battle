"""0045 Kaggle-equivalent FP16-storage/FP32-runtime materialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..assets import sha256_file
from ..policy.actor_critic import SemanticActorCritic, load_actor_critic
from ..rollout.deck_routing import exact_deck_sha256


CONTRACT_ID = "kaggle_fp16_storage_fp32_runtime_v1"
FIELDS = {
    "actor_state_dict": "actor.",
    "value_head_state_dict": "value_head.",
    "allocation_head_state_dict": "allocation_head.",
    "value_adapter_state_dict": "value_adapter.",
    "policy_option_lora_state_dict": "policy_option_lora.",
}
_PROTOTYPE_ALIASES = ("state_encoder.prototypes.", "option_encoder.prototypes.")


def _portable_actor_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """Store shared prototype parameters once under their canonical actor path."""
    return {
        name: value for name, value in state.items()
        if not name.startswith(_PROTOTYPE_ALIASES)
    }


def _expanded_actor_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore the two model aliases from the single canonical prototype copy."""
    expanded = dict(state)
    canonical = {
        name.removeprefix("prototype_encoder."): value
        for name, value in state.items()
        if name.startswith("prototype_encoder.")
    }
    if not canonical:
        raise RuntimeError("portable candidate has no canonical prototype encoder")
    for alias in _PROTOTYPE_ALIASES:
        expanded.update({f"{alias}{name}": value for name, value in canonical.items()})
    return expanded


@dataclass(frozen=True, slots=True)
class CandidateAudit:
    source_checkpoint_sha256: str
    portable_checkpoint_sha256: str
    effective_candidate_sha256: str
    checkpoint_update: int
    focal_exact_deck_sha256: str
    contract_id: str = CONTRACT_ID
    storage_dtype: str = "fp16"
    runtime_dtype: str = "fp32"
    conversion_order: str = "merge_full_effective_policy_then_fp16_storage_then_fp32_runtime"
    status: str = "PASS"
    schema_version: str = "0045_candidate_deployment_identity_audit_v1"

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_hash(payload: Mapping[str, Any], deck_hash: str) -> str:
    digest = hashlib.sha256(b"kaggle_fp16_storage_fp32_runtime_v1\0")
    digest.update(deck_hash.encode("ascii"))
    for field in FIELDS:
        values = dict(payload[field])
        if field == "actor_state_dict":
            canonical = {
                name.removeprefix("prototype_encoder."): value
                for name, value in values.items()
                if name.startswith("prototype_encoder.")
            }
            for alias in _PROTOTYPE_ALIASES:
                values.update({f"{alias}{name}": value for name, value in canonical.items()})
        for name, value in sorted(values.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{field}.{name}".encode()); digest.update(b"\0")
            digest.update(str(tensor.dtype).encode()); digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            digest.update(b"\0"); digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    return digest.hexdigest()


def materialize(
    *, checkpoint: Path, base_portable: Path, deck: Sequence[int], deck_id: str,
    own_archetype_id: int, output: Path, device: torch.device,
) -> tuple[SemanticActorCritic, CandidateAudit]:
    cards = tuple(map(int, deck))
    if len(cards) != 60 or not 0 <= own_archetype_id < 29:
        raise ValueError("candidate requires exact 60 cards and an own taxonomy ID")
    deck_hash = exact_deck_sha256(cards)
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if source.get("schema_version") != "0045_minimal_lora_model_only_v1":
        raise RuntimeError("unsupported 0045 model-only checkpoint")
    update = source.get("update")
    if not isinstance(update, int):
        raise RuntimeError("candidate checkpoint update is missing")
    base = torch.load(base_portable, map_location="cpu", weights_only=True)
    if base.get("schema_version") not in {
        "0043_focal_v1_kaggle_candidate_v1",
        "0044_policy_value_split_option_lora_meta_residual_candidate_v2",
    }:
        raise RuntimeError(
            "0044 portable base must be a supported immutable promoted Champion"
        )
    state = source["state_dict"]
    model, _ = load_actor_critic(
        checkpoint=base_portable, deck=cards, deck_id=deck_id, device="cpu",
        own_archetype_id_override=own_archetype_id,
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"0044 candidate has unexpected source tensors: {incompatible}")
    result = {
        "schema_version": "0045_minimal_lora_candidate_v1",
        "actor_state_dict": _portable_actor_state_dict(model.actor.state_dict()),
        "value_head_state_dict": model.value_head.state_dict(),
        "allocation_head_state_dict": model.allocation_head.state_dict(),
        "value_adapter_state_dict": model.value_adapter.state_dict(),
        "policy_option_lora_state_dict": model.policy_option_lora.state_dict(),
    }
    for field in FIELDS:
        result[field] = {
            name: value.half() if torch.is_floating_point(value) else value
            for name, value in result[field].items()
        }
    metadata = dict(base["metadata"])
    metadata.update({
        "checkpoint_update": update,
        "checkpoint_sha256": sha256_file(checkpoint),
        "own_archetype_id": own_archetype_id,
        "own_archetype_class_count": 29,
        "own_archetype_vocabulary_version": "own_archetypes_v2",
        "focal_deck_id": deck_id,
        "focal_exact_deck_sha256": deck_hash,
        "deployment_contract": CONTRACT_ID,
        "policy_value_option_split": True,
        "policy_option_lora": {
            "option_block": 1, "rank": 4, "alpha": 8.0,
            "attention_targets": ["self_attn.qv", "cross_attn.qv"],
            "parameters": 10240,
        },
        "policy_strategy_adapter": "removed",
        "meta_actor_residual": "removed",
        "critic_outputs_consumed_by_actor": False,
    })
    result["metadata"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(result, temporary); temporary.replace(output)
    strict = torch.load(output, map_location="cpu", weights_only=True)
    floating = [v for field in FIELDS for v in strict[field].values() if torch.is_floating_point(v)]
    if not floating or any(v.dtype != torch.float16 for v in floating):
        raise RuntimeError("portable candidate is not wholly FP16 stored")
    if strict.get("schema_version") != "0045_minimal_lora_candidate_v1":
        raise RuntimeError("0045 candidate schema changed during storage")
    model, _ = load_actor_critic(
        checkpoint=base_portable, deck=cards, deck_id=deck_id, device="cpu",
        own_archetype_id_override=own_archetype_id,
    )
    modules = {
        "actor_state_dict": model.actor,
        "value_head_state_dict": model.value_head,
        "allocation_head_state_dict": model.allocation_head,
        "value_adapter_state_dict": model.value_adapter,
        "policy_option_lora_state_dict": model.policy_option_lora,
    }
    for field, module in modules.items():
        stored = strict[field]
        if field == "actor_state_dict":
            stored = _expanded_actor_state_dict(stored)
        module.load_state_dict(
            {
                name: value.float() if torch.is_floating_point(value) else value
                for name, value in stored.items()
            },
            strict=True,
        )
    model.default_own_archetype_id.fill_(own_archetype_id)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    runtime = [v for v in model.parameters() if torch.is_floating_point(v)]
    if any(v.dtype != torch.float32 for v in runtime):
        raise RuntimeError("strict-loaded candidate runtime is not FP32")
    audit = CandidateAudit(
        source_checkpoint_sha256=sha256_file(checkpoint),
        portable_checkpoint_sha256=sha256_file(output),
        effective_candidate_sha256=_tensor_hash(strict, deck_hash),
        checkpoint_update=update,
        focal_exact_deck_sha256=deck_hash,
    )
    return model, audit


__all__ = ["CONTRACT_ID", "CandidateAudit", "materialize"]
