"""One immutable 0028 representation with live and frozen decoder routes."""

from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn

from ..foundation import load_foundation


def create_value_head(width: int) -> nn.Sequential:
    head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1), nn.Tanh())
    nn.init.zeros_(head[-2].weight)
    nn.init.zeros_(head[-2].bias)
    return head


class DecoderPolicyHead(nn.Module):
    def __init__(self, action_decoder: nn.Module, value_head: nn.Module | None = None) -> None:
        super().__init__()
        self.action_decoder = action_decoder
        self.value_head = value_head

    @classmethod
    def copy_from(cls, model: "SemanticActorCritic", *, include_value: bool = True):
        head = cls(copy.deepcopy(model.actor.action_decoder), copy.deepcopy(model.value_head) if include_value else None)
        head.to(model.device).eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        return head


class SemanticActorCritic(nn.Module):
    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.actor = actor
        self.value_head = create_value_head(int(actor.config.d_model))
        self.opponent_head = DecoderPolicyHead(copy.deepcopy(actor.action_decoder), None)
        self._prototype_memory = None
        self.freeze_contract()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def head(self) -> DecoderPolicyHead:
        return DecoderPolicyHead(self.actor.action_decoder, self.value_head)

    def freeze_contract(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.actor.action_decoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.value_head.parameters():
            parameter.requires_grad_(True)

    freeze_representation = freeze_contract

    def encode(self, batch):
        validated = self.actor.validate_batch(batch)
        memory = self._prototype_memory
        if memory is None:
            memory = self.actor.prototype_encoder.encode_all()
        state = self.actor.state_encoder(validated, memory)
        options = self.actor.option_encoder(validated, state, memory)
        return validated, state.summary, options

    def prepare_inference_cache(self) -> None:
        if self._prototype_memory is None:
            with torch.inference_mode():
                self._prototype_memory = self.actor.prototype_encoder.encode_all()

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    def assert_trainable_contract(self) -> None:
        names = self.trainable_parameter_names()
        if not names or any(not name.startswith(("actor.action_decoder.", "value_head.")) for name in names):
            raise RuntimeError(f"invalid trainable contract: {names[:5]}")
        if any(parameter.requires_grad for parameter in self.opponent_head.parameters()):
            raise RuntimeError("opponent decoder must be immutable")

    def _hash(self, *, representation: bool) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.actor.state_dict().items()):
            is_decoder = name.startswith("action_decoder.")
            if representation == is_decoder:
                continue
            digest.update(name.encode()); digest.update(b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def representation_sha256(self) -> str:
        return self._hash(representation=True)

    def decoder_sha256(self) -> str:
        return self._hash(representation=False)

    def opponent_decoder_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.opponent_head.action_decoder.state_dict().items()):
            digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()


def load_actor_critic(device: str | torch.device = "cpu"):
    actor, prototypes, identity = load_foundation("cpu", eval_mode=False)
    model = SemanticActorCritic(actor).to(device)
    model.runtime_prototypes = prototypes
    model.assert_trainable_contract()
    return model, prototypes, identity


CanonicalActorCritic = SemanticActorCritic

__all__ = ["CanonicalActorCritic", "DecoderPolicyHead", "SemanticActorCritic", "load_actor_critic"]
