"""The paired-0809 SemanticPolicy and Value head under the 0042 trainable boundary."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from ..semantic_policy.deployment.inference import PortableSemanticPolicy
from .value_network import LatentQueryValueHead
from .adaptation import AdaptationConfig
from .allocation_head import DragapultAllocationHead
from .own_archetype import OwnArchetypeId, OwnArchetypeVocabulary
from .strategy_adapters import (
    PolicyStrategyAdapter,
    StrategyContext,
    ValueResidualAdapter,
)
from ..source import ACTOR_SHA256, VALUE_CHECKPOINT, VALUE_SHA256
from ..integrated.config import IntegratedFlags
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
        raise ValueError(f"0042 value-source checkpoint SHA mismatch: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "0036_value_model_only_checkpoint_v1":
        raise ValueError("0042 critic source schema mismatch")
    metadata = payload.get("metadata") or {}
    config = metadata.get("value_config") or {}
    expected = {"architecture": "latent_queries", "queries": 8, "layers": 2, "dropout": 0.0}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"0042 critic architecture mismatch: {config}")
    if metadata.get("source_checkpoint_sha256") != ACTOR_SHA256:
        raise ValueError("0042 Value checkpoint is not paired with the 0809 actor")
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
    def __init__(self, action_decoder: nn.Module, strategy_adapter: PolicyStrategyAdapter) -> None:
        super().__init__()
        self.action_decoder = action_decoder
        self.strategy_adapter = strategy_adapter

    def logits(self, batch, options: Tensor, state, context: StrategyContext, **kwargs) -> Tensor:
        readout, _ = self.strategy_adapter(state.hidden, context)
        return self.action_decoder.logits(
            batch, options, state, readout_hidden=readout, **kwargs
        )

    @classmethod
    def copy_from(cls, model: "SemanticActorCritic") -> "DecoderPolicyHead":
        snapshot = cls(
            copy.deepcopy(model.actor.action_decoder),
            copy.deepcopy(model.policy_strategy_adapter),
        ).to(model.device).eval()
        snapshot.requires_grad_(False)
        return snapshot


class SemanticActorCritic(nn.Module):
    """0042 strategy-conditioned actor-critic over frozen semantic encoders."""

    def __init__(self, actor: nn.Module, value_head: LatentQueryValueHead,
                 own_archetype_id: OwnArchetypeId,
                 adaptation: AdaptationConfig = AdaptationConfig(),
                 integrated_flags: IntegratedFlags = IntegratedFlags()) -> None:
        super().__init__()
        self.actor = actor
        self.value_head = value_head
        self.allocation_head = DragapultAllocationHead(int(actor.config.d_model))
        self.adaptation_config = adaptation
        if adaptation.lora or adaptation.layernorm_tuning:
            raise ValueError("0042 forbids Option LoRA and Option LayerNorm tuning")
        integrated_flags.validate()
        self.integrated_flags = integrated_flags
        width = int(actor.config.d_model)
        self.register_buffer(
            "default_own_archetype_id",
            torch.tensor(own_archetype_id.value, dtype=torch.long),
        )
        # New modules use a private stream so base actor/value construction is unaffected.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(420_042_211)
            self.value_adapter = ValueResidualAdapter(width)
            self.policy_strategy_adapter = PolicyStrategyAdapter(width)
            self.prize_aux = PrizeAuxHead(width) if integrated_flags.enable_prize_aux else None
        self.freeze_representation()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def head(self) -> DecoderPolicyHead:
        return DecoderPolicyHead(self.actor.action_decoder, self.policy_strategy_adapter)

    def own_archetype_ids(self, batch_size: int, device: torch.device) -> Tensor:
        return self.default_own_archetype_id.to(device).expand(batch_size)

    def encode(self, batch: dict[str, Tensor]):
        validated, state, options = self.actor.encode(batch)
        value = self.value_from_encoded(validated, state, options)
        return validated, state, options, value

    def encode_with_strategy(self, batch: dict[str, Tensor]):
        validated, state, options = self.actor.encode(batch)
        value, auxiliary = self.value_and_aux_from_encoded(validated, state, options)
        context = self.strategy_context(validated, value, auxiliary)
        return validated, state, options, value, auxiliary, context

    def value_from_encoded(self, validated, state, options: Tensor) -> Tensor:
        value, _ = self.value_and_aux_from_encoded(validated, state, options)
        return value

    def value_and_aux_from_encoded(self, validated, state, options: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        queries = self.value_head.decode(memory, memory_mask)
        z_value = queries[:, 0]
        z_meta = queries[:, 1]
        own_ids = self.own_archetype_ids(queries.shape[0], queries.device)
        adapted_value, value_delta = self.value_adapter(z_value, own_ids)
        meta_logits = self.value_head.heads.archetype(z_meta)
        logit = self.value_head.heads.value(adapted_value).squeeze(-1)
        value = 2.0 * logit.sigmoid() - 1.0
        output: dict[str, Tensor] = {
            "z_value": z_value,
            "z_meta": z_meta,
            "meta_logits": meta_logits,
            "opponent_meta_logits": meta_logits,
            "value_adapter_delta": value_delta,
            "value_adapter_residual_ratio": self.value_adapter.effective_residual_ratio(
                z_value, value_delta
            ),
        }
        if self.prize_aux is not None:
            output["v_prize"] = self.prize_aux(queries)
        return value, output

    def strategy_context(
        self, validated, value: Tensor, auxiliary: dict[str, Tensor]
    ) -> StrategyContext:
        return StrategyContext.build(
            relative_first_player=validated.global_cat[:, 2],
            z_meta=auxiliary["z_meta"],
            meta_logits=auxiliary["meta_logits"],
            value=value,
            own_archetype_id=self.own_archetype_ids(value.shape[0], value.device),
        )

    def auxiliary_from_encoded(self, validated, state, options: Tensor) -> dict[str, Tensor]:
        _, output = self.value_and_aux_from_encoded(validated, state, options)
        return output

    def actor_summary(self, state) -> Tensor:
        return state.summary

    def freeze_representation(self) -> None:
        self.actor.requires_grad_(False)
        self.actor.action_decoder.requires_grad_(True)
        self.value_head.requires_grad_(False)
        self.value_head.queries.requires_grad_(True)
        self.value_head.blocks.requires_grad_(True)
        self.value_head.final_norm.requires_grad_(True)
        self.value_head.heads.value.requires_grad_(True)
        self.value_head.heads.archetype.requires_grad_(False)
        self.allocation_head.requires_grad_(True)
        self.value_adapter.requires_grad_(True)
        self.policy_strategy_adapter.requires_grad_(True)
        if self.prize_aux is not None:
            self.prize_aux.requires_grad_(True)
        self.assert_no_option_adaptation()

    def assert_no_option_adaptation(self) -> None:
        option = self.actor.option_encoder
        trainable = [name for name, value in option.named_parameters() if value.requires_grad]
        parametrized = [
            name for name, module in option.named_modules()
            if any(parametrize.is_parametrized(module, field) for field in ("in_proj_weight", "weight"))
        ]
        lora_names = [name for name, _ in option.named_parameters() if "lora" in name.lower()]
        if trainable or parametrized or lora_names:
            raise RuntimeError(
                "0042 OptionEncoder must be frozen and unparametrized; "
                f"trainable={trainable[:3]}, parametrized={parametrized[:3]}, lora={lora_names[:3]}"
            )

    def assert_trainable_contract(self) -> None:
        names = tuple(name for name, value in self.named_parameters() if value.requires_grad)
        invalid = [
            name for name in names
            if not name.startswith("actor.action_decoder.")
            and not name.startswith("value_head.")
            and not name.startswith("allocation_head.")
            and not name.startswith(("prize_aux.", "value_adapter.", "policy_strategy_adapter."))
        ]
        if invalid or not any(name.startswith("actor.action_decoder.") for name in names):
            raise RuntimeError(f"invalid full-semantic trainable boundary: {invalid[:5]}")
        self.assert_no_option_adaptation()
        value_ids = {id(value) for value in self.value_adapter.parameters()}
        policy_ids = {id(value) for value in self.policy_strategy_adapter.parameters()}
        if value_ids.intersection(policy_ids):
            raise RuntimeError("0042 Value and Policy adapters share Parameter objects")

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
        raise ValueError(f"Policy-0809 checkpoint SHA mismatch: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if len(payload.get("state_dict", {})) != EXPECTED_TENSOR_COUNT:
        raise ValueError("Policy-0809 checkpoint tensor inventory mismatch")
    portable = PortableSemanticPolicy.from_checkpoint(checkpoint, deck)
    actor = portable.model
    parameter_count = sum(value.numel() for value in actor.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"Policy-0809 actor parameter count mismatch: {parameter_count}")
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
    vocabulary = OwnArchetypeVocabulary.load()
    model = SemanticActorCritic(
        actor,
        load_value_head(value_checkpoint, actor),
        vocabulary.classify_own_deck(deck),
        adaptation,
        integrated_flags,
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
