"""The exact 0031 SemanticPolicy with only its decoder and a critic trainable."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn

from ..semantic_policy.deployment.inference import PortableSemanticPolicy


EXPECTED_SOURCE_SHA256 = "285b88f5e30c40ad07ad025b429c5bd1594f0359cc0d44849c368056222d63ae"
EXPECTED_PARAMETER_COUNT = 56_352_322
EXPECTED_TENSOR_COUNT = 293


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_sha256(module: nn.Module, *, exclude_decoder: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if exclude_decoder and name.startswith("action_decoder."):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    checkpoint_sha256: str
    schema_version: str
    project_id: str
    version: str
    epoch: int
    global_step: int
    actor_parameter_count: int
    actor_tensor_count: int


class DecoderPolicyHead(nn.Module):
    def __init__(self, action_decoder: nn.Module, value_head: nn.Module) -> None:
        super().__init__()
        self.action_decoder = action_decoder
        self.value_head = value_head

    @classmethod
    def copy_from(cls, model: "SemanticActorCritic") -> "DecoderPolicyHead":
        snapshot = cls(
            copy.deepcopy(model.actor.action_decoder),
            copy.deepcopy(model.value_head),
        ).to(model.device).eval()
        snapshot.requires_grad_(False)
        return snapshot


class SemanticActorCritic(nn.Module):
    """Preserve every original actor tensor and train only the original decoder."""

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

    def encode(self, batch: dict[str, Tensor]):
        validated, state, options = self.actor.encode(batch)
        value = self.value_head(state.summary).squeeze(-1)
        return validated, state.summary, options, value

    def freeze_representation(self) -> None:
        self.actor.requires_grad_(False)
        self.actor.action_decoder.requires_grad_(True)
        self.value_head.requires_grad_(True)

    def assert_trainable_contract(self) -> None:
        names = tuple(name for name, value in self.named_parameters() if value.requires_grad)
        invalid = [
            name for name in names
            if not name.startswith("actor.action_decoder.")
            and not name.startswith("value_head.")
        ]
        if invalid or not any(name.startswith("actor.action_decoder.") for name in names):
            raise RuntimeError(f"invalid full-semantic trainable boundary: {invalid[:5]}")

    def representation_sha256(self) -> str:
        return _module_sha256(self.actor, exclude_decoder=True)

    def decoder_sha256(self) -> str:
        return _module_sha256(self.actor.action_decoder)


def load_actor_critic(
    checkpoint: Path,
    deck: tuple[int, ...],
    device: str | torch.device = "cpu",
) -> tuple[SemanticActorCritic, SourceIdentity]:
    digest = _sha256(checkpoint)
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"PT0805 checkpoint SHA mismatch: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if len(payload.get("state_dict", {})) != EXPECTED_TENSOR_COUNT:
        raise ValueError("PT0805 checkpoint tensor inventory mismatch")
    portable = PortableSemanticPolicy.from_checkpoint(checkpoint, deck)
    actor = portable.model
    parameter_count = sum(value.numel() for value in actor.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"PT0805 actor parameter count mismatch: {parameter_count}")
    metadata = payload["metadata"]
    identity = SourceIdentity(
        checkpoint_sha256=digest,
        schema_version=str(payload["schema_version"]),
        project_id=str(metadata["project_id"]),
        version=str(metadata["version"]),
        epoch=int(metadata["epoch"]),
        global_step=int(metadata["global_step"]),
        actor_parameter_count=parameter_count,
        actor_tensor_count=len(payload["state_dict"]),
    )
    model = SemanticActorCritic(actor).to(device).eval()
    model.assert_trainable_contract()
    return model, identity


__all__ = [
    "DecoderPolicyHead",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_SOURCE_SHA256",
    "EXPECTED_TENSOR_COUNT",
    "SemanticActorCritic",
    "SourceIdentity",
    "create_value_head",
    "load_actor_critic",
]
