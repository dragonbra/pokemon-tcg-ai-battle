"""Critic-free 0045 Actor export and strict loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..semantic_runtime.domain.prototypes import PrototypeIndex
from ..semantic_runtime.model import ModelConfig, SemanticPolicy
from ..semantic_runtime.deployment.online_runtime import _prototype_paths
from ..semantic_runtime.deployment.inference import _expanded_portable_state_dict
from .allocation_head import DragapultAllocationHead
from .option_policy_lora import PolicyOnlyOptionLoRA


SCHEMA = "0045_minimal_lora_policy_only_export_v1"


def _effective_hash(states: dict[str, dict[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for namespace, state in sorted(states.items()):
        for name, value in sorted(state.items()):
            digest.update(f"{namespace}.{name}\0".encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyOnlyIdentity:
    schema_version: str
    effective_policy_sha256: str
    tensor_count: int
    parameter_count: int


class PolicyOnlySpecialist(nn.Module):
    """Complete deployed Actor; this class has no Critic attribute."""

    def __init__(self, actor: SemanticPolicy, policy_option_lora: PolicyOnlyOptionLoRA,
                 allocation_head: DragapultAllocationHead) -> None:
        super().__init__()
        self.actor = actor
        self.policy_option_lora = policy_option_lora
        self.allocation_head = allocation_head

    def encode_policy(self, batch):
        validated = self.actor.validate_batch(batch)
        prototype_memory = self.actor.prototype_memory()
        state = self.actor.state_encoder(validated, prototype_memory)
        encoder = self.actor.option_encoder
        inputs = encoder.encode_inputs(validated, state, prototype_memory)
        prefix = encoder.encode_prefix(validated, state, inputs)
        options = self.policy_option_lora(encoder, validated, state, prefix)
        if self.policy_option_lora.is_zero_delta():
            base = encoder.encode_final(validated, state, prefix)
            options = base + (options - options.detach())
        return validated, state, options

    def root_logits(self, batch) -> Tensor:
        validated, state, options = self.encode_policy(batch)
        decoder_state = self.actor.action_decoder.initialize(validated, state.summary)
        return self.actor.action_decoder.logits(validated, options, decoder_state)


def export_policy_only(model, destination: Path) -> PolicyOnlyIdentity:
    actor_state = {
        name: value.detach().cpu()
        for name, value in model.actor.state_dict().items()
        if not name.startswith(("state_encoder.prototypes.", "option_encoder.prototypes."))
    }
    states = {
        "actor": actor_state,
        "policy_option_lora": {
            name: value.detach().cpu() for name, value in model.policy_option_lora.state_dict().items()
        },
        "allocation_head": {
            name: value.detach().cpu() for name, value in model.allocation_head.state_dict().items()
        },
    }
    identity = PolicyOnlyIdentity(
        schema_version=SCHEMA,
        effective_policy_sha256=_effective_hash(states),
        tensor_count=sum(len(state) for state in states.values()),
        parameter_count=sum(
            int(value.numel())
            for module in (model.actor, model.policy_option_lora, model.allocation_head)
            for value in module.parameters()
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "model_config": asdict(model.actor.config),
        "state_dicts": states,
        "metadata": asdict(identity),
    }
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return identity


def load_policy_only(source: Path, *, device: str | torch.device = "cpu") -> PolicyOnlySpecialist:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "model_config", "state_dicts", "metadata"}:
        raise ValueError("policy-only payload has unexpected fields")
    if payload["schema_version"] != SCHEMA:
        raise ValueError("unsupported 0045 policy-only schema")
    states = payload["state_dicts"]
    if _effective_hash(states) != payload["metadata"]["effective_policy_sha256"]:
        raise RuntimeError("policy-only effective identity mismatch")
    public_path, engine_path = _prototype_paths()
    actor = SemanticPolicy(
        ModelConfig(**payload["model_config"]),
        PrototypeIndex.load(public_path, engine_path),
    )
    actor.load_state_dict(_expanded_portable_state_dict(states["actor"]), strict=True)
    lora = PolicyOnlyOptionLoRA(int(actor.config.d_model), rank=4, alpha=8.0)
    lora.load_state_dict(states["policy_option_lora"], strict=True)
    allocation = DragapultAllocationHead(int(actor.config.d_model))
    allocation.load_state_dict(states["allocation_head"], strict=True)
    policy = PolicyOnlySpecialist(actor, lora, allocation).to(device).eval()
    policy.requires_grad_(False)
    if any(hasattr(policy, name) for name in ("value_head", "value_adapter", "prize_aux")):
        raise RuntimeError("Critic leaked into policy-only export")
    return policy


__all__ = [
    "PolicyOnlyIdentity", "PolicyOnlySpecialist", "export_policy_only", "load_policy_only"
]
