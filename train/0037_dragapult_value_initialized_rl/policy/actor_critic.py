"""The exact 0031 SemanticPolicy with only its decoder and a critic trainable."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn

from ..semantic_policy.deployment.inference import PortableSemanticPolicy
from .value_network import LatentQueryValueHead
from .adaptation import AdaptationConfig, apply_focal_adaptation


EXPECTED_SOURCE_SHA256 = "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8"
EXPECTED_PARAMETER_COUNT = 56_352_322
EXPECTED_TENSOR_COUNT = 293
EXPECTED_VALUE_SHA256 = "e88b2f18089911c4ebd810fa3abfba800a103189184b6265b9d38c03a9e36360"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VALUE_CHECKPOINT = (
    ROOT
    / "rl_runs/0037_dragapult_value_initialized_rl/source/value_v2_epoch0005/value_head.pt"
)


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


def _frozen_module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    rows = [
        (name, value) for name, value in module.named_parameters()
        if not value.requires_grad and not name.startswith("action_decoder.")
    ] + [
        (name, value) for name, value in module.named_buffers()
        if not name.startswith("action_decoder.")
    ]
    for name, value in sorted(rows):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_value_head(checkpoint: Path, actor: nn.Module) -> LatentQueryValueHead:
    digest = _sha256(checkpoint)
    if digest != EXPECTED_VALUE_SHA256:
        raise ValueError(f"0036 V2 epoch-5 checkpoint SHA mismatch: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "0036_value_model_only_checkpoint_v1":
        raise ValueError("0037 critic source schema mismatch")
    metadata = payload.get("metadata") or {}
    config = metadata.get("value_config") or {}
    expected = {"architecture": "latent_queries", "queries": 8, "layers": 2, "dropout": 0.0}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"0037 critic architecture mismatch: {config}")
    head = LatentQueryValueHead(
        int(actor.config.d_model),
        queries=8,
        layers=2,
        heads=int(actor.config.heads),
        dropout=0.0,
    )
    head.load_state_dict(payload["value_head_state_dict"], strict=True)
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
    def __init__(self, action_decoder: nn.Module) -> None:
        super().__init__()
        self.action_decoder = action_decoder

    @classmethod
    def copy_from(cls, model: "SemanticActorCritic") -> "DecoderPolicyHead":
        snapshot = cls(copy.deepcopy(model.actor.action_decoder)).to(model.device).eval()
        snapshot.requires_grad_(False)
        return snapshot


class SemanticActorCritic(nn.Module):
    """Preserve every original actor tensor and train only the original decoder."""

    def __init__(self, actor: nn.Module, value_head: LatentQueryValueHead,
                 adaptation: AdaptationConfig = AdaptationConfig()) -> None:
        super().__init__()
        self.actor = actor
        self.value_head = value_head
        self.adaptation_config = adaptation
        self.adaptation_inventory: dict[str, object] = {}
        self.freeze_representation()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def head(self) -> DecoderPolicyHead:
        return DecoderPolicyHead(self.actor.action_decoder)

    def encode(self, batch: dict[str, Tensor]):
        validated, state, options = self.actor.encode(batch)
        value = self.value_from_encoded(validated, state, options)
        return validated, state, options, value

    def value_from_encoded(self, validated, state, options: Tensor) -> Tensor:
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        logit = self.value_head.forward_value_logit(memory, memory_mask)
        return 2.0 * logit.sigmoid() - 1.0

    def freeze_representation(self) -> None:
        self.actor.requires_grad_(False)
        self.actor.action_decoder.requires_grad_(True)
        self.value_head.requires_grad_(False)
        self.value_head.queries.requires_grad_(True)
        self.value_head.blocks.requires_grad_(True)
        self.value_head.final_norm.requires_grad_(True)
        self.value_head.heads.value.requires_grad_(True)
        self.adaptation_inventory = apply_focal_adaptation(self.actor, self.adaptation_config)

    def assert_trainable_contract(self) -> None:
        names = tuple(name for name, value in self.named_parameters() if value.requires_grad)
        invalid = [
            name for name in names
            if not name.startswith("actor.action_decoder.")
            and not name.startswith("value_head.")
            and ".parametrizations." not in name
            and not (
                self.adaptation_config.layernorm_tuning
                and name.startswith(("actor.state_encoder.", "actor.option_encoder."))
                and name.endswith((".weight", ".bias"))
            )
        ]
        if invalid or not any(name.startswith("actor.action_decoder.") for name in names):
            raise RuntimeError(f"invalid full-semantic trainable boundary: {invalid[:5]}")

    def representation_sha256(self) -> str:
        return _frozen_module_sha256(self.actor)

    def decoder_sha256(self) -> str:
        return _module_sha256(self.actor.action_decoder)


def load_actor_critic(
    checkpoint: Path,
    deck: tuple[int, ...],
    device: str | torch.device = "cpu",
    value_checkpoint: Path = DEFAULT_VALUE_CHECKPOINT,
    adaptation: AdaptationConfig = AdaptationConfig(),
) -> tuple[SemanticActorCritic, SourceIdentity]:
    digest = _sha256(checkpoint)
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Large Model 0806 checkpoint SHA mismatch: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if len(payload.get("state_dict", {})) != EXPECTED_TENSOR_COUNT:
        raise ValueError("Large Model 0806 checkpoint tensor inventory mismatch")
    portable = PortableSemanticPolicy.from_checkpoint(checkpoint, deck)
    actor = portable.model
    parameter_count = sum(value.numel() for value in actor.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"Large Model 0806 actor parameter count mismatch: {parameter_count}")
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
    model = SemanticActorCritic(actor, load_value_head(value_checkpoint, actor), adaptation).to(device).eval()
    model.assert_trainable_contract()
    return model, identity


__all__ = [
    "DecoderPolicyHead",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_SOURCE_SHA256",
    "EXPECTED_TENSOR_COUNT",
    "SemanticActorCritic",
    "SourceIdentity",
    "DEFAULT_VALUE_CHECKPOINT",
    "EXPECTED_VALUE_SHA256",
    "load_value_head",
    "load_actor_critic",
]
