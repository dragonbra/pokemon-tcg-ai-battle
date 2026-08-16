"""0045 minimal specialist Actor with a strictly separate training-only Critic."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from ..semantic_runtime.deployment.compound_inference import PortableCompoundSemanticPolicy
from ..semantic_runtime.deployment.inference import PortableSemanticPolicy
from .value_network import LatentQueryValueHead
from .adaptation import AdaptationConfig, apply_focal_adaptation
from .option_policy_lora import DualOptionEncoding
from .allocation_head import DragapultAllocationHead
from .strategy_adapters import ValueResidualAdapter
from .shared_encoder_lora import (
    install_shared_encoder_lora,
    shared_encoder_lora_named_parameters,
    shared_encoder_lora_parameters,
)
from ..integrated.config import IntegratedFlags
from ..integrated.prize import PrizeAuxHead
from ..own_archetype import OwnArchetypeVocabulary


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOCAL_CHECKPOINT = (
    PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin"
)
DEFAULT_G2_SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "assets/policies/definitions/champion_g002/source_update_000407.pt"
)
DEFAULT_0814_ACTOR_CHECKPOINT = (
    PROJECT_ROOT / "assets/policies/definitions/policy_0814/model.pt"
)
DEFAULT_0814_VALUE_CHECKPOINT = (
    PROJECT_ROOT / "assets/policies/definitions/policy_0814/value_head.pt"
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
    """Policy-only decoder view.

    This object deliberately has no Critic, Value adapter, archetype embedding,
    or strategy-context field.  Keeping the API narrow makes inference-semantic
    coupling impossible rather than merely detached.
    """

    def __init__(self, action_decoder: nn.Module) -> None:
        super().__init__()
        self.action_decoder = action_decoder

    def logits(self, batch, options: Tensor, state, **kwargs) -> Tensor:
        return self.action_decoder.logits(batch, options, state, **kwargs)

    @classmethod
    def copy_from(cls, model: "SemanticActorCritic") -> "DecoderPolicyHead":
        snapshot = cls(copy.deepcopy(model.actor.action_decoder)).to(model.device).eval()
        snapshot.requires_grad_(False)
        return snapshot


class SemanticActorCritic(nn.Module):
    """0045 Actor/Critic container with no Critic-to-Actor forward edge."""

    def __init__(self, actor: nn.Module, value_head: LatentQueryValueHead,
                 default_own_archetype_id: int = 0,
                 own_archetype_classes: int = 29,
                 adaptation: AdaptationConfig = AdaptationConfig(),
                 integrated_flags: IntegratedFlags = IntegratedFlags()) -> None:
        super().__init__()
        self.actor = actor
        self.value_head = value_head
        self.allocation_head = DragapultAllocationHead(int(actor.config.d_model))
        self.adaptation_config = adaptation
        self.policy_option_lora = apply_focal_adaptation(actor, adaptation)
        self.shared_encoder_lora_inventory = (
            install_shared_encoder_lora(
                actor, rank=adaptation.rank, alpha=adaptation.alpha
            )
            if adaptation.shared_state_encoder else None
        )
        integrated_flags.validate()
        self.integrated_flags = integrated_flags
        width = int(actor.config.d_model)
        self.register_buffer(
            "default_own_archetype_id",
            torch.tensor(default_own_archetype_id, dtype=torch.long),
        )
        # New modules use a private stream so base actor/value construction is unaffected.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(420_042_211)
            self.value_adapter = ValueResidualAdapter(width, own_archetype_classes=own_archetype_classes)
            self.prize_aux = PrizeAuxHead(width) if integrated_flags.enable_prize_aux else None
        self.freeze_representation()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def head(self) -> DecoderPolicyHead:
        return DecoderPolicyHead(self.actor.action_decoder)

    def own_archetype_ids(self, batch_size: int, device: torch.device) -> Tensor:
        runtime = getattr(self, "_runtime_own_archetype_ids", None)
        if runtime is not None:
            if runtime.shape != (batch_size,):
                raise ValueError("runtime own-deck IDs do not match batch")
            return runtime.to(device)
        return self.default_own_archetype_id.to(device).expand(batch_size)

    def set_runtime_own_archetype_ids(self, values: Tensor | None) -> None:
        self._runtime_own_archetype_ids = values

    def encode(self, batch: dict[str, Tensor]):
        validated, state, options = self.encode_dual_options(batch)
        value = self.value_from_encoded(validated, state, options.value_options)
        return validated, state, options.policy_options, value

    def encode_policy(self, batch: dict[str, Tensor]):
        """Encode only tensors that can affect deployed Actor inference."""
        validated, state, options = self.encode_dual_options(batch)
        return validated, state, options.policy_options

    def encode_critic(self, batch: dict[str, Tensor]):
        """Encode the training-only Critic path independently of Actor logits."""
        validated, state, options = self.encode_dual_options(batch)
        value, auxiliary = self.value_and_aux_from_encoded(
            validated, state, options.value_options
        )
        return validated, state, options.value_options, value, auxiliary

    def encode_dual_options(self, batch: dict[str, Tensor]):
        """Share frozen Option input/block-0, then fork the immutable and LoRA paths."""
        validated = self.actor.validate_batch(batch)
        prototype_memory = self.actor.prototype_memory()
        state = self.actor.state_encoder(validated, prototype_memory)
        option_encoder = self.actor.option_encoder
        inputs = option_encoder.encode_inputs(validated, state, prototype_memory)
        prefix = option_encoder.encode_prefix(validated, state, inputs)
        value_options = option_encoder.encode_final(validated, state, prefix)
        policy_options = self.policy_option_lora(
            option_encoder, validated, state, prefix
        )
        if self.policy_option_lora.is_zero_delta():
            # Exact base Option forward at zero delta while retaining LoRA gradients.
            policy_options = value_options + (
                policy_options - policy_options.detach()
            )
        return validated, state, DualOptionEncoding(value_options, policy_options)

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

    def auxiliary_from_encoded(self, validated, state, options: Tensor) -> dict[str, Tensor]:
        _, output = self.value_and_aux_from_encoded(validated, state, options)
        return output

    def actor_summary(self, state) -> Tensor:
        return state.summary

    def freeze_representation(self) -> None:
        self.actor.requires_grad_(False)
        self.actor.action_decoder.requires_grad_(True)
        for parameter in shared_encoder_lora_parameters(self.actor):
            parameter.requires_grad_(True)
        self.value_head.requires_grad_(False)
        self.value_head.queries.requires_grad_(True)
        self.value_head.blocks.requires_grad_(True)
        self.value_head.final_norm.requires_grad_(True)
        self.value_head.heads.value.requires_grad_(True)
        self.value_head.heads.archetype.requires_grad_(False)
        self.allocation_head.requires_grad_(True)
        self.value_adapter.requires_grad_(True)
        self.policy_option_lora.requires_grad_(True)
        if self.prize_aux is not None:
            self.prize_aux.requires_grad_(True)
        self.assert_option_adaptation_contract()

    def assert_option_adaptation_contract(self) -> None:
        option = self.actor.option_encoder
        trainable = [name for name, value in option.named_parameters() if value.requires_grad]
        parametrized = [
            name for name, module in option.named_modules()
            if any(parametrize.is_parametrized(module, field) for field in ("in_proj_weight", "weight"))
        ]
        if trainable or parametrized:
            raise RuntimeError(
                "0045 immutable OptionEncoder base must be frozen and unparametrized; "
                f"trainable={trainable[:3]}, parametrized={parametrized[:3]}"
            )
        self.policy_option_lora.assert_inventory()

    def assert_trainable_contract(self) -> None:
        names = tuple(name for name, value in self.named_parameters() if value.requires_grad)
        invalid = [
            name for name in names
            if not name.startswith("actor.action_decoder.")
            and not name.startswith("value_head.")
            and not name.startswith("allocation_head.")
            and not name.startswith(("prize_aux.", "value_adapter.", "policy_option_lora."))
            and not (
                name.startswith("actor.state_encoder.")
                and ".parametrizations." in name
                and not name.endswith(".original")
            )
        ]
        if invalid or not any(name.startswith("actor.action_decoder.") for name in names):
            raise RuntimeError(f"invalid full-semantic trainable boundary: {invalid[:5]}")
        self.assert_option_adaptation_contract()
        critic_ids = {
            id(value) for module in (self.value_head, self.value_adapter, self.prize_aux)
            if module is not None for value in module.parameters()
        }
        policy_ids = {
            id(value) for module in (
                self.actor.action_decoder, self.allocation_head, self.policy_option_lora
            ) for value in module.parameters()
        }
        policy_ids.update(id(value) for value in shared_encoder_lora_parameters(self.actor))
        if critic_ids.intersection(policy_ids):
            raise RuntimeError("0045 Critic and Policy share Parameter objects")

    def representation_sha256(self) -> str:
        return _frozen_module_sha256(self.actor)

    def decoder_sha256(self) -> str:
        return _module_sha256(self.actor.action_decoder)


def load_actor_critic(
    checkpoint: Path = DEFAULT_FOCAL_CHECKPOINT,
    deck: tuple[int, ...] = (),
    deck_id: str = "007",
    device: str | torch.device = "cpu",
    adaptation: AdaptationConfig = AdaptationConfig(),
    integrated_flags: IntegratedFlags = IntegratedFlags(),
    own_archetype_id_override: int | None = None,
    value_checkpoint: Path | None = None,
) -> tuple[SemanticActorCritic, SourceIdentity]:
    if len(deck) != 60:
        raise ValueError("focal loader requires one exact 60-card deck")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor_only_0814 = payload.get("schema_version") == "0031_model_only_checkpoint_v1"
    if actor_only_0814:
        if value_checkpoint is None:
            raise ValueError("Policy-0814 focal initialization requires paired Value checkpoint")
        portable_actor = PortableSemanticPolicy.from_checkpoint(checkpoint, deck)
        actor = portable_actor.actor
        value_payload = torch.load(value_checkpoint, map_location="cpu", weights_only=True)
        if (
            value_payload.get("schema_version") != "0036_value_model_only_checkpoint_v1"
            or value_payload.get("metadata", {}).get("source_checkpoint_sha256")
            != _sha256(checkpoint)
        ):
            raise RuntimeError("Policy-0814 paired Value identity mismatch")
        value_head = LatentQueryValueHead(int(actor.config.d_model), queries=8, layers=2)
        value_head.load_state_dict(value_payload["value_head_state_dict"], strict=True)
        metadata = payload["metadata"]
    else:
        portable = PortableCompoundSemanticPolicy.from_checkpoint(checkpoint, deck)
        actor = portable.actor
        value_head = portable.value_head
        metadata = portable.metadata
    parameter_count = sum(value.numel() for value in actor.parameters())
    if parameter_count != 56_352_322:
        raise ValueError(f"0045 inherited focal actor parameter count mismatch: {parameter_count}")
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    if own_archetype_id_override is None:
        own_archetype_id = vocabulary.resolve_exact_deck(deck_id, deck).value
    else:
        if not 0 <= own_archetype_id_override < vocabulary.class_count:
            raise ValueError("own_archetype_id_override is outside the frozen vocabulary")
        own_archetype_id = own_archetype_id_override
    identity = SourceIdentity(
        checkpoint_sha256=_sha256(checkpoint),
        schema_version=str(payload.get("schema_version")),
        project_id=str(metadata["project_id"]),
        version=str(metadata["version"]),
        epoch=int(metadata.get("epoch", metadata.get("checkpoint_update", 407))),
        global_step=int(metadata.get("global_step", metadata.get("checkpoint_update", 407))),
        actor_parameter_count=parameter_count,
        actor_tensor_count=len(actor.state_dict()),
    )
    model = SemanticActorCritic(
        actor,
        value_head,
        own_archetype_id,
        29,
        adaptation,
        integrated_flags,
    )
    if actor_only_0814:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(450_814_013)
            model.allocation_head = DragapultAllocationHead(int(actor.config.d_model))
            model.prize_aux = PrizeAuxHead(int(actor.config.d_model))
    else:
        model.allocation_head = portable.allocation_head
        model.value_adapter.load_state_dict(portable.value_adapter.state_dict(), strict=True)
        model.prize_aux = PrizeAuxHead(int(actor.config.d_model))
    # Prize prediction is training-only and absent from the promoted portable
    # policy identity. Restore only that head from the FP32 U407 source; every
    # inference-effective tensor remains the strict-loaded FP16 G2 artifact.
    if not actor_only_0814:
        source = torch.load(
            DEFAULT_G2_SOURCE_CHECKPOINT, map_location="cpu", weights_only=True
        )
        if source.get("schema_version") != "0043_focal_v1_model_only_v1" or source.get("update") != 407:
            raise RuntimeError("0044 focal source is not the immutable Champion-G2 U407 checkpoint")
        prize_state = {
            name.removeprefix("prize_aux."): value
            for name, value in source["state_dict"].items()
            if name.startswith("prize_aux.")
        }
        model.prize_aux.load_state_dict(prize_state, strict=True)
    model.to(device).eval().freeze_representation()
    model.assert_trainable_contract()
    return model, identity


__all__ = [
    "DecoderPolicyHead",
    "SemanticActorCritic",
    "SourceIdentity",
    "DEFAULT_FOCAL_CHECKPOINT",
    "DEFAULT_G2_SOURCE_CHECKPOINT",
    "DEFAULT_0814_ACTOR_CHECKPOINT",
    "DEFAULT_0814_VALUE_CHECKPOINT",
    "load_actor_critic",
]
