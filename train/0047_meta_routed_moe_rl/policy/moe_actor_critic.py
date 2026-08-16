"""0047 seven-Actor policy with a shared frozen 0814 backbone and one Critic."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from ..integrated.config import IntegratedFlags
from ..integrated.prize import PrizeAuxHead
from ..semantic_runtime.deployment.inference import PortableSemanticPolicy
from .adaptation import AdaptationConfig, apply_focal_adaptation
from .allocation_head import DragapultAllocationHead
from .strategy_adapters import ValueResidualAdapter
from .value_network import LatentQueryValueHead


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_0814_ACTOR = PROJECT_ROOT / "assets/policies/definitions/policy_0814/model.pt"
DEFAULT_0814_VALUE = PROJECT_ROOT / "assets/policies/definitions/policy_0814/value_head.pt"
POLICY_0814_SHA256 = "d7921f420c8f12155119d6caa0fef414f51c0a8368cd5ecf07cd7d71312c897b"
VALUE_0814_SHA256 = "0ad6f57a32d37942cccdc78a8a9c8ef2f6f8784d08ea8ed77ca1513628c77d2d"
EFFECTIVE_0814_SHA256 = "476d57d55eb9c040fa4e75ce74ac5205af5d094a296db71cba4ecbf792902580"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    policy_id: str
    actor_checkpoint_sha256: str
    value_checkpoint_sha256: str
    effective_policy_sha256: str
    actor_parameter_count: int
    actor_tensor_count: int
    allocation_initialization: str


class ActorExpert(nn.Module):
    """The complete 0047 Actor-side trainable inventory for one expert."""

    def __init__(self, action_decoder: nn.Module, policy_option_lora: nn.Module,
                 allocation_head: nn.Module) -> None:
        super().__init__()
        self.action_decoder = action_decoder
        self.policy_option_lora = policy_option_lora
        self.allocation_head = allocation_head

    def parameter_manifest(self) -> dict[str, int]:
        return {
            "action_decoder": sum(p.numel() for p in self.action_decoder.parameters()),
            "allocation_head": sum(p.numel() for p in self.allocation_head.parameters()),
            "policy_option_lora": sum(p.numel() for p in self.policy_option_lora.parameters()),
        }


class MetaRoutedMoEActorCritic(nn.Module):
    expert_labels = ("E0", "E00", "E01", "E02", "E03", "E05", "E27")
    priority_meta_ids = (0, 1, 2, 3, 5, 27)
    meta_to_expert = {meta_id: index + 1 for index, meta_id in enumerate(priority_meta_ids)}
    expert_count = len(expert_labels)
    meta_count = 29

    def __init__(self, actor: nn.Module, value_head: LatentQueryValueHead,
                 experts: list[ActorExpert], *, own_archetype_id: int,
                 router_temperature: float = 1.0,
                 integrated_flags: IntegratedFlags = IntegratedFlags()) -> None:
        super().__init__()
        if len(experts) != self.expert_count:
            raise ValueError("0047 requires E0 plus six core-meta Actor specialists")
        self.actor = actor
        self.value_head = value_head
        self.experts = nn.ModuleList(experts)
        self.integrated_flags = integrated_flags
        integrated_flags.validate()
        width = int(actor.config.d_model)
        self.register_buffer("default_own_archetype_id", torch.tensor(own_archetype_id))
        self.register_buffer("router_temperature", torch.tensor(float(router_temperature)))
        self.register_buffer("routing_phase", torch.tensor(0, dtype=torch.long))
        initial_probabilities = torch.full((self.meta_count, self.expert_count), 1.0e-4)
        initial_probabilities[:, 0] = 1.0 - 1.0e-4 * (self.expert_count - 1)
        for meta_id, expert_index in self.meta_to_expert.items():
            initial_probabilities[meta_id].fill_(1.0e-4)
            initial_probabilities[meta_id, 0] = 0.2 - 5.0e-4
            initial_probabilities[meta_id, expert_index] = 0.8
        self.router_logits = nn.Parameter(
            initial_probabilities.log() * router_temperature
        )
        self.router_logits.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(470_814_001)
            self.value_adapter = ValueResidualAdapter(width, own_archetype_classes=29)
            self.prize_aux = PrizeAuxHead(width)
        self.freeze_contract()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def allocation_head(self) -> nn.Module:
        # Compatibility only. Mixture allocation never uses this shortcut.
        return self.experts[0].allocation_head

    @property
    def policy_option_lora(self) -> nn.Module:
        # Compatibility only. Mixture decoding iterates all experts explicitly.
        return self.experts[0].policy_option_lora

    def actor_summary(self, state) -> Tensor:
        return state.summary

    def set_runtime_own_archetype_ids(self, values: Tensor | None) -> None:
        self._runtime_own_archetype_ids = values

    def own_archetype_ids(self, batch_size: int, device: torch.device) -> Tensor:
        values = getattr(self, "_runtime_own_archetype_ids", None)
        if values is None:
            return self.default_own_archetype_id.to(device).expand(batch_size)
        if values.shape != (batch_size,):
            raise ValueError("runtime own-archetype IDs do not match batch")
        return values.to(device)

    def freeze_contract(self) -> None:
        self.actor.requires_grad_(False)
        self.value_head.requires_grad_(False)
        self.value_head.queries.requires_grad_(True)
        self.value_head.blocks.requires_grad_(True)
        self.value_head.final_norm.requires_grad_(True)
        self.value_head.heads.value.requires_grad_(True)
        self.value_head.heads.archetype.requires_grad_(False)
        for expert in self.experts:
            expert.requires_grad_(True)
        self.value_adapter.requires_grad_(True)
        self.prize_aux.requires_grad_(True)

    def set_soft_routing(self, enabled: bool = True) -> None:
        self.routing_phase.fill_(1 if enabled else 0)
        self.router_logits.requires_grad_(bool(enabled))

    @property
    def soft_routing(self) -> bool:
        return bool(self.routing_phase.item())

    def gate_probabilities(self, meta_ids: Tensor) -> Tensor:
        meta_ids = meta_ids.to(device=self.device, dtype=torch.long)
        if meta_ids.ndim != 1:
            raise ValueError("routing Meta IDs must be rank one")
        routes = torch.zeros((meta_ids.numel(), self.expert_count), device=self.device)
        routes[:, 0] = 1.0
        identified = meta_ids.ge(0) & meta_ids.lt(self.meta_count)
        if self.soft_routing:
            if bool(identified.any()):
                probabilities = torch.softmax(
                    self.router_logits / self.router_temperature.clamp_min(1.0e-6),
                    dim=1,
                )
                routes[identified] = probabilities.index_select(0, meta_ids[identified])
        else:
            for meta_id, expert_index in self.meta_to_expert.items():
                mask = meta_ids.eq(meta_id)
                if not bool(mask.any()):
                    continue
                routes[mask, 0] = 0.0
                routes[mask, expert_index] = 1.0
        return routes

    def router_table(self) -> list[dict[str, object]]:
        meta = torch.arange(self.meta_count, device=self.device)
        phase = int(self.routing_phase.item())
        probabilities = self.gate_probabilities(meta)
        return [
            {
                "meta_id": f"{index:02d}",
                "probabilities": [float(value) for value in probabilities[index].detach().cpu()],
                "argmax_expert": int(probabilities[index].argmax()),
                "phase": "soft" if phase else "hard_warmup",
            }
            for index in range(self.meta_count)
        ]

    def encode_shared(self, batch: dict[str, Tensor]):
        validated = self.actor.validate_batch(batch)
        prototypes = self.actor.prototype_memory()
        state = self.actor.state_encoder(validated, prototypes)
        option_encoder = self.actor.option_encoder
        inputs = option_encoder.encode_inputs(validated, state, prototypes)
        prefix = option_encoder.encode_prefix(validated, state, inputs)
        value_options = option_encoder.encode_final(validated, state, prefix)
        return validated, state, prefix, value_options

    def expert_options(self, expert_index: int, validated, state, prefix: Tensor,
                       value_options: Tensor | None = None) -> Tensor:
        options = self.experts[expert_index].policy_option_lora(
            self.actor.option_encoder, validated, state, prefix
        )
        if value_options is not None and self.experts[expert_index].policy_option_lora.is_zero_delta():
            options = value_options + (options - options.detach())
        return options

    def value_and_aux_from_encoded(self, validated, state, options: Tensor):
        memory = torch.cat((state.tokens, options), dim=1)
        mask = torch.cat((state.mask, validated.option_mask), dim=1)
        queries = self.value_head.decode(memory, mask)
        z_value, z_meta = queries[:, 0], queries[:, 1]
        own_ids = self.own_archetype_ids(queries.shape[0], queries.device)
        adapted, delta = self.value_adapter(z_value, own_ids)
        logit = self.value_head.heads.value(adapted).squeeze(-1)
        value = 2.0 * logit.sigmoid() - 1.0
        meta_logits = self.value_head.heads.archetype(z_meta)
        return value, {
            "z_value": z_value, "z_meta": z_meta,
            "meta_logits": meta_logits, "opponent_meta_logits": meta_logits,
            "value_adapter_delta": delta,
            "value_adapter_residual_ratio": self.value_adapter.effective_residual_ratio(z_value, delta),
            "v_prize": self.prize_aux(queries),
        }

    def value_from_encoded(self, validated, state, options: Tensor) -> Tensor:
        return self.value_and_aux_from_encoded(validated, state, options)[0]

    def representation_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.actor.state_dict().items()):
            if name.startswith("action_decoder."):
                continue
            digest.update(name.encode()); digest.update(b"\0")
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def assert_trainable_contract(self) -> None:
        allowed = ("experts.", "router_logits", "value_head.", "value_adapter.", "prize_aux.")
        invalid = [name for name, p in self.named_parameters() if p.requires_grad and not name.startswith(allowed)]
        if invalid:
            raise RuntimeError(f"0047 invalid trainable parameters: {invalid[:8]}")
        storages: set[int] = set()
        for expert in self.experts:
            for parameter in expert.parameters():
                pointer = parameter.untyped_storage().data_ptr()
                if pointer in storages:
                    raise RuntimeError("0047 experts share Parameter storage")
                storages.add(pointer)


def load_moe_actor_critic(
    *, deck: tuple[int, ...], own_archetype_id: int = 15,
    device: str | torch.device = "cpu",
    integrated_flags: IntegratedFlags = IntegratedFlags(),
) -> tuple[MetaRoutedMoEActorCritic, SourceIdentity]:
    if len(deck) != 60:
        raise ValueError("0047 focal loader requires exact deck 070")
    if _sha256(DEFAULT_0814_ACTOR) != POLICY_0814_SHA256:
        raise RuntimeError("Policy-0814 Actor hash mismatch")
    if _sha256(DEFAULT_0814_VALUE) != VALUE_0814_SHA256:
        raise RuntimeError("Policy-0814 Value hash mismatch")
    portable = PortableSemanticPolicy.from_checkpoint(DEFAULT_0814_ACTOR, deck)
    actor = portable.actor
    value_payload = torch.load(DEFAULT_0814_VALUE, map_location="cpu", weights_only=True)
    value_head = LatentQueryValueHead(int(actor.config.d_model), queries=8, layers=2)
    value_head.load_state_dict(value_payload["value_head_state_dict"], strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(470_814_002)
        seed_lora = apply_focal_adaptation(actor, AdaptationConfig())
        seed_allocation = DragapultAllocationHead(int(actor.config.d_model))
    seed = ActorExpert(copy.deepcopy(actor.action_decoder), seed_lora, seed_allocation)
    experts = [copy.deepcopy(seed) for _ in range(MetaRoutedMoEActorCritic.expert_count)]
    model = MetaRoutedMoEActorCritic(
        actor, value_head, experts, own_archetype_id=own_archetype_id,
        integrated_flags=integrated_flags,
    ).to(device).eval()
    model.assert_trainable_contract()
    identity = SourceIdentity(
        "Policy-0814", POLICY_0814_SHA256, VALUE_0814_SHA256,
        EFFECTIVE_0814_SHA256, sum(p.numel() for p in actor.parameters()),
        len(actor.state_dict()), "deterministic_seed_470814002_then_deepcopy",
    )
    return model, identity


def save_router_table(model: MetaRoutedMoEActorCritic, path: Path) -> None:
    payload = {
        "schema_version": "0047_meta29x7_lookup_softmax_router_v1",
        "temperature": float(model.router_temperature),
        "phase": "soft" if model.soft_routing else "hard_warmup",
        "logits": model.router_logits.detach().cpu().tolist(),
        "expert_labels": list(model.expert_labels),
        "rows": model.router_table(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


__all__ = [
    "ActorExpert", "MetaRoutedMoEActorCritic", "SourceIdentity",
    "load_moe_actor_critic", "save_router_table",
]
