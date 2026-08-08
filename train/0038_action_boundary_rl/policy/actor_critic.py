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
from .allocation_head import DragapultAllocationHead
from ..source import ACTOR_SHA256, VALUE_CHECKPOINT, VALUE_SHA256
from ..integrated.config import IntegratedFlags
from ..integrated.opponent_meta import OpponentMetaConditioner, OpponentMetaHead
from ..integrated.prize import PrizeAuxHead


EXPECTED_SOURCE_SHA256 = ACTOR_SHA256
EXPECTED_PARAMETER_COUNT = 56_352_322
EXPECTED_TENSOR_COUNT = 293
EXPECTED_VALUE_SHA256 = VALUE_SHA256
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VALUE_CHECKPOINT = VALUE_CHECKPOINT


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
        raise ValueError(f"0038 value-source checkpoint SHA mismatch: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "0036_value_model_only_checkpoint_v1":
        raise ValueError("0038 critic source schema mismatch")
    metadata = payload.get("metadata") or {}
    config = metadata.get("value_config") or {}
    expected = {"architecture": "latent_queries", "queries": 8, "layers": 2, "dropout": 0.0}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"0038 critic architecture mismatch: {config}")
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
                 adaptation: AdaptationConfig = AdaptationConfig(),
                 integrated_flags: IntegratedFlags = IntegratedFlags()) -> None:
        super().__init__()
        self.actor = actor
        self.value_head = value_head
        self.allocation_head = DragapultAllocationHead(int(actor.config.d_model))
        self.adaptation_config = adaptation
        integrated_flags.validate()
        self.integrated_flags = integrated_flags
        width = int(actor.config.d_model)
        # Auxiliary construction uses a private RNG stream. Every preset therefore
        # shares exactly the same core and LoRA initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(380_038_211)
            self.prize_aux = PrizeAuxHead(width) if integrated_flags.enable_prize_aux else None
            classes = integrated_flags.opponent_meta_class_count
            self.opponent_meta_head = (
                OpponentMetaHead(width, classes) if integrated_flags.enable_opponent_meta else None
            )
            self.opponent_meta_conditioner = (
                OpponentMetaConditioner(width, classes)
                if integrated_flags.enable_opponent_meta_conditioning else None
            )
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
        value, _ = self.value_and_aux_from_encoded(validated, state, options)
        return value

    def value_and_aux_from_encoded(self, validated, state, options: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        queries = self.value_head.decode(memory, memory_mask)
        output: dict[str, Tensor] = {}
        meta_logits = None
        if self.opponent_meta_head is not None:
            meta_logits = self.opponent_meta_head(state.summary)
            output["opponent_meta_logits"] = meta_logits
        value_query = queries[:, 0]
        if meta_logits is not None and self.opponent_meta_conditioner is not None:
            conditioned = self.opponent_meta_conditioner(
                state.summary,
                meta_logits,
                detach=self.integrated_flags.opponent_meta_detach_to_actor,
            )
            value_query = value_query + (conditioned - state.summary)
        logit = self.value_head.heads.value(value_query).squeeze(-1)
        if self.prize_aux is not None:
            output["v_prize"] = self.prize_aux(queries)
        return 2.0 * logit.sigmoid() - 1.0, output

    def auxiliary_from_encoded(self, validated, state, options: Tensor) -> dict[str, Tensor]:
        _, output = self.value_and_aux_from_encoded(validated, state, options)
        return output

    def actor_summary(self, state) -> Tensor:
        if self.opponent_meta_head is None or self.opponent_meta_conditioner is None:
            return state.summary
        logits = self.opponent_meta_head(state.summary)
        return self.opponent_meta_conditioner(
            state.summary, logits, detach=self.integrated_flags.opponent_meta_detach_to_actor
        )

    def freeze_representation(self) -> None:
        self.actor.requires_grad_(False)
        self.actor.action_decoder.requires_grad_(True)
        self.value_head.requires_grad_(False)
        self.value_head.queries.requires_grad_(True)
        self.value_head.blocks.requires_grad_(True)
        self.value_head.final_norm.requires_grad_(True)
        self.value_head.heads.value.requires_grad_(True)
        self.allocation_head.requires_grad_(True)
        if self.prize_aux is not None:
            self.prize_aux.requires_grad_(True)
        if self.opponent_meta_head is not None:
            self.opponent_meta_head.requires_grad_(True)
        if self.opponent_meta_conditioner is not None:
            self.opponent_meta_conditioner.requires_grad_(True)
        self.adaptation_inventory = apply_focal_adaptation(self.actor, self.adaptation_config)

    def assert_trainable_contract(self) -> None:
        names = tuple(name for name, value in self.named_parameters() if value.requires_grad)
        invalid = [
            name for name in names
            if not name.startswith("actor.action_decoder.")
            and not name.startswith("value_head.")
            and not name.startswith("allocation_head.")
            and not name.startswith(("prize_aux.", "opponent_meta_head.", "opponent_meta_conditioner."))
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
    integrated_flags: IntegratedFlags = IntegratedFlags(),
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
    model = SemanticActorCritic(
        actor, load_value_head(value_checkpoint, actor), adaptation, integrated_flags
    ).to(device).eval()
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
