"""Frozen canonical representation with a trainable action decoder and value head."""

from __future__ import annotations

import copy
import hashlib

import torch
from torch import Tensor, nn

from ..focal import FocalIdentity, load_focal_actor
from ..focal.model.canonical.decoder import OrderedOptionDecoder


def create_value_head(width: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.LayerNorm(width),
        nn.Linear(width, width),
        nn.GELU(),
        nn.Linear(width, 1),
        nn.Tanh(),
    )
    output = head[-2]
    assert isinstance(output, nn.Linear)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    return head


class DecoderPolicyHead(nn.Module):
    """A decoder/value snapshot sharing no mutable tensors with the live head."""

    def __init__(self, action_decoder: OrderedOptionDecoder, value_head: nn.Module) -> None:
        super().__init__()
        self.action_decoder = action_decoder
        self.value_head = value_head

    @classmethod
    def copy_from(cls, model: "CanonicalActorCritic") -> "DecoderPolicyHead":
        snapshot = cls(
            copy.deepcopy(model.actor.action_decoder),
            copy.deepcopy(model.value_head),
        ).to(model.device).eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        return snapshot


class CanonicalActorCritic(nn.Module):
    """Expose one immutable semantic actor with only its final decoder trainable."""

    def __init__(self, actor: nn.Module, value_head: nn.Module | None = None) -> None:
        super().__init__()
        self.actor = actor
        self.value_head = value_head or create_value_head(int(actor.config.d_model))
        self.freeze_representation()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def head(self) -> DecoderPolicyHead:
        return DecoderPolicyHead(self.actor.action_decoder, self.value_head)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        summary, options = self.actor.encode(batch)
        return summary, options, self.value_head(summary).squeeze(-1)

    def freeze_representation(self) -> None:
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.actor.action_decoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.value_head.parameters():
            parameter.requires_grad_(True)

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def assert_trainable_contract(self) -> None:
        names = self.trainable_parameter_names()
        if not names or not any(name.startswith("actor.action_decoder.") for name in names):
            raise RuntimeError("canonical actor has no trainable action decoder")
        invalid = [
            name for name in names
            if not name.startswith("actor.action_decoder.") and not name.startswith("value_head.")
        ]
        if invalid:
            raise RuntimeError(f"canonical representation parameters are trainable: {invalid[:5]}")

    def representation_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.actor.state_dict().items()):
            if name.startswith("action_decoder."):
                continue
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def decoder_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.actor.action_decoder.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def load_actor_critic(
    device: str | torch.device = "cpu",
) -> tuple[CanonicalActorCritic, FocalIdentity]:
    actor, identity = load_focal_actor("cpu", eval_mode=False)
    model = CanonicalActorCritic(actor).to(device)
    model.assert_trainable_contract()
    return model, identity


__all__ = [
    "CanonicalActorCritic",
    "DecoderPolicyHead",
    "create_value_head",
    "load_actor_critic",
]
