"""0043 Kaggle-equivalent FP16-storage/FP32-runtime candidate materialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..assets import sha256_file
from ..policy.actor_critic import SemanticActorCritic
from ..rollout.deck_routing import exact_deck_sha256
from ..semantic_runtime.deployment.compound_inference import PortableCompoundSemanticPolicy


CONTRACT_ID = "kaggle_fp16_storage_fp32_runtime_v1"
FIELDS = {
    "actor_state_dict": "actor.",
    "value_head_state_dict": "value_head.",
    "allocation_head_state_dict": "allocation_head.",
    "value_adapter_state_dict": "value_adapter.",
    "policy_strategy_adapter_state_dict": "policy_strategy_adapter.",
}


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
    schema_version: str = "0043_candidate_deployment_identity_audit_v1"

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_hash(payload: Mapping[str, Any], deck_hash: str) -> str:
    digest = hashlib.sha256(b"kaggle_fp16_storage_fp32_runtime_v1\0")
    digest.update(deck_hash.encode("ascii"))
    for field in FIELDS:
        for name, value in sorted(payload[field].items()):
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
    if source.get("schema_version") != "0043_focal_v1_model_only_v1":
        raise RuntimeError("unsupported 0043 model-only checkpoint")
    update = source.get("update")
    if not isinstance(update, int):
        raise RuntimeError("candidate checkpoint update is missing")
    payload = torch.load(base_portable, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "0043_focal_v1_kaggle_candidate_v1":
        raise RuntimeError("0043 portable base identity mismatch")
    state = source["state_dict"]
    result = dict(payload)
    for field, prefix in FIELDS.items():
        rows = dict(payload[field])
        for name in tuple(rows):
            key = prefix + name
            if key in state:
                rows[name] = state[key]
        result[field] = {
            name: value.half() if torch.is_floating_point(value) else value
            for name, value in rows.items()
        }
    metadata = dict(payload["metadata"])
    metadata.update({
        "checkpoint_update": update,
        "checkpoint_sha256": sha256_file(checkpoint),
        "own_archetype_id": own_archetype_id,
        "own_archetype_class_count": 29,
        "own_archetype_vocabulary_version": "own_archetypes_v2",
        "focal_deck_id": deck_id,
        "focal_exact_deck_sha256": deck_hash,
        "deployment_contract": CONTRACT_ID,
    })
    result["metadata"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(result, temporary); temporary.replace(output)
    strict = torch.load(output, map_location="cpu", weights_only=True)
    floating = [v for field in FIELDS for v in strict[field].values() if torch.is_floating_point(v)]
    if not floating or any(v.dtype != torch.float16 for v in floating):
        raise RuntimeError("portable candidate is not wholly FP16 stored")
    portable = PortableCompoundSemanticPolicy.from_checkpoint(output, cards)
    model = SemanticActorCritic(
        portable.actor, portable.value_head, own_archetype_id, 29
    )
    model.allocation_head.load_state_dict(
        {name: value.float() for name, value in strict["allocation_head_state_dict"].items()},
        strict=True,
    )
    model.value_adapter.load_state_dict(
        {name: value.float() for name, value in strict["value_adapter_state_dict"].items()},
        strict=True,
    )
    model.policy_strategy_adapter.load_state_dict(
        {name: value.float() for name, value in strict["policy_strategy_adapter_state_dict"].items()},
        strict=True,
    )
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
