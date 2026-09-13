"""0045 Kaggle-equivalent FP16-storage/FP32-runtime materialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..assets import sha256_file
from ..policy import AdaptationConfig
from ..policy.actor_critic import SemanticActorCritic, load_actor_critic
from ..rollout.deck_routing import exact_deck_sha256
from ..semantic_runtime.deployment.compound_inference import PortableCompoundSemanticPolicy
from ..training.checkpointing import load_model_only_delta


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


def _adaptation_from_checkpoint_metadata(metadata: Mapping[str, Any]) -> AdaptationConfig:
    targets = tuple(metadata.get("attention_targets", ()))
    if targets not in {
        ("self_attn.qv", "cross_attn.qv"),
        ("self_attn.qvo", "cross_attn.qvo"),
    }:
        raise RuntimeError("checkpoint Option LoRA targets are unsupported")
    adaptation = AdaptationConfig(
        rank=int(metadata["rank"]),
        alpha=float(metadata["alpha"]),
        output_projection=targets == ("self_attn.qvo", "cross_attn.qvo"),
        shared_state_encoder=bool(metadata.get("shared_state_encoder", False)),
        option_ffn_lora=bool(metadata.get("option_ffn_lora", False)),
        state_ffn_lora=bool(metadata.get("state_ffn_lora", False)),
        layernorm_tuning=bool(metadata.get("option_final_layernorm_tuning", False)),
    )
    adaptation.validate()
    return adaptation


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
    base_value_checkpoint: Path | None = None,
    adaptation: AdaptationConfig = AdaptationConfig(),
) -> tuple[SemanticActorCritic, CandidateAudit]:
    cards = tuple(map(int, deck))
    if len(cards) != 60 or not 0 <= own_archetype_id < 29:
        raise ValueError("candidate requires exact 60 cards and an own taxonomy ID")
    deck_hash = exact_deck_sha256(cards)
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    update = source.get("update")
    if not isinstance(update, int):
        raise RuntimeError("candidate checkpoint update is missing")
    checkpoint_adaptation = source.get("adaptation")
    if isinstance(checkpoint_adaptation, Mapping):
        # The checkpoint is authoritative. This prevents periodic eval from
        # instantiating a legacy 115-tensor structure for an expanded delta.
        adaptation = _adaptation_from_checkpoint_metadata(checkpoint_adaptation)
    base = torch.load(base_portable, map_location="cpu", weights_only=True)
    if base.get("schema_version") not in {
        "0031_model_only_checkpoint_v1",
        "0043_focal_v1_kaggle_candidate_v1",
        "0044_policy_value_split_option_lora_meta_residual_candidate_v2",
    }:
        raise RuntimeError(
            "0044 portable base must be a supported immutable promoted Champion"
        )
    model, _ = load_actor_critic(
        checkpoint=base_portable, deck=cards, deck_id=deck_id, device="cpu",
        own_archetype_id_override=own_archetype_id,
        value_checkpoint=base_value_checkpoint, adaptation=adaptation,
    )
    load_model_only_delta(model, source, allow_legacy=True)
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
    if "actor_metadata" not in metadata:
        model_config = metadata.get("model_config")
        if not isinstance(model_config, Mapping):
            raise RuntimeError("0045 base checkpoint has no audited actor model config")
        metadata["actor_metadata"] = {"model_config": model_config}
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
            "option_block": 1, "rank": adaptation.rank,
            "alpha": adaptation.alpha,
            "attention_targets": (
                ["self_attn.qvo", "cross_attn.qvo"]
                if adaptation.output_projection
                else ["self_attn.qv", "cross_attn.qv"]
            ),
            "shared_state_encoder": adaptation.shared_state_encoder,
            "option_ffn_lora": adaptation.option_ffn_lora,
            "state_ffn_lora": adaptation.state_ffn_lora,
            "option_final_layernorm_tuning": adaptation.layernorm_tuning,
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
        value_checkpoint=base_value_checkpoint, adaptation=adaptation,
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


def load_portable_candidate(
    *, portable: Path, deck: Sequence[int], deck_id: str, own_archetype_id: int,
    device: torch.device, expected_source_checkpoint_sha256: str,
    expected_portable_checkpoint_sha256: str,
    expected_effective_candidate_sha256: str,
) -> tuple[SemanticActorCritic, CandidateAudit]:
    """Strict-load one sealed 0045 deployment artifact for official evaluation."""
    cards = tuple(map(int, deck))
    if len(cards) != 60 or not 0 <= own_archetype_id < 29:
        raise ValueError("portable candidate requires exact 60 cards and an own taxonomy ID")
    deck_hash = exact_deck_sha256(cards)
    portable_hash = sha256_file(portable)
    if portable_hash != expected_portable_checkpoint_sha256:
        raise RuntimeError("portable candidate file identity mismatch")
    strict = torch.load(portable, map_location="cpu", weights_only=True)
    if (
        set(strict) != {"schema_version", "metadata", *FIELDS}
        or strict.get("schema_version") != "0045_minimal_lora_candidate_v1"
    ):
        raise RuntimeError("unsupported 0045 portable candidate")
    metadata = strict.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("portable candidate metadata is missing")
    update = metadata.get("checkpoint_update")
    if not isinstance(update, int):
        raise RuntimeError("portable candidate checkpoint update is missing")
    if metadata.get("checkpoint_sha256") != expected_source_checkpoint_sha256:
        raise RuntimeError("portable candidate source checkpoint identity mismatch")
    if (
        metadata.get("focal_deck_id") != deck_id
        or metadata.get("focal_exact_deck_sha256") != deck_hash
        or metadata.get("own_archetype_id") != own_archetype_id
        or metadata.get("deployment_contract") != CONTRACT_ID
    ):
        raise RuntimeError("portable candidate deck/deployment metadata mismatch")
    floating = [
        value for field in FIELDS for value in strict[field].values()
        if torch.is_floating_point(value)
    ]
    if not floating or any(value.dtype != torch.float16 for value in floating):
        raise RuntimeError("portable candidate is not wholly FP16 stored")
    effective_hash = _tensor_hash(strict, deck_hash)
    if effective_hash != expected_effective_candidate_sha256:
        raise RuntimeError("effective candidate identity mismatch")

    option = metadata.get("policy_option_lora")
    if not isinstance(option, Mapping):
        raise RuntimeError("portable candidate Option LoRA metadata is missing")
    targets = option.get("attention_targets")
    target_tuple = tuple(targets) if isinstance(targets, (list, tuple)) else ()
    output_projection = target_tuple == ("self_attn.qvo", "cross_attn.qvo")
    if target_tuple not in {
        ("self_attn.qv", "cross_attn.qv"),
        ("self_attn.qvo", "cross_attn.qvo"),
    }:
        raise RuntimeError("portable candidate Option LoRA targets are unsupported")
    adaptation = AdaptationConfig(
        rank=int(option.get("rank", -1)),
        alpha=float(option.get("alpha", -1.0)),
        output_projection=output_projection,
        shared_state_encoder=bool(option.get("shared_state_encoder", False)),
        option_ffn_lora=bool(option.get("option_ffn_lora", False)),
        state_ffn_lora=bool(option.get("state_ffn_lora", False)),
        layernorm_tuning=bool(option.get("option_final_layernorm_tuning", False)),
    )
    adaptation.validate()
    loaded = PortableCompoundSemanticPolicy.from_checkpoint(portable, cards)
    if loaded.policy_option_lora is None:
        raise RuntimeError("portable candidate did not reconstruct Policy Option LoRA")
    # The deployment loader has already installed and strict-loaded every
    # Actor-side parametrization. Bootstrap only the container here; applying
    # `adaptation` again would install State LoRA a second time.
    model = SemanticActorCritic(
        loaded.actor, loaded.value_head,
        default_own_archetype_id=own_archetype_id,
        own_archetype_classes=29,
        adaptation=AdaptationConfig(),
    )
    model.adaptation_config = adaptation
    model.policy_option_lora = loaded.policy_option_lora
    modules = {
        "allocation_head_state_dict": model.allocation_head,
        "value_adapter_state_dict": model.value_adapter,
        "policy_option_lora_state_dict": model.policy_option_lora,
    }
    for field, module in modules.items():
        module.load_state_dict(
            {
                name: value.float() if torch.is_floating_point(value) else value
                for name, value in strict[field].items()
            },
            strict=True,
        )
    model.default_own_archetype_id.fill_(own_archetype_id)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    runtime = [value for value in model.parameters() if torch.is_floating_point(value)]
    if not runtime or any(value.dtype != torch.float32 for value in runtime):
        raise RuntimeError("strict-loaded portable runtime is not FP32")
    audit = CandidateAudit(
        source_checkpoint_sha256=expected_source_checkpoint_sha256,
        portable_checkpoint_sha256=portable_hash,
        effective_candidate_sha256=effective_hash,
        checkpoint_update=update,
        focal_exact_deck_sha256=deck_hash,
    )
    return model, audit


__all__ = [
    "CONTRACT_ID", "CandidateAudit", "load_portable_candidate", "materialize",
]
