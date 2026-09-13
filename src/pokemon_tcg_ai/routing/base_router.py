"""Identity-audited Policy-0814 public deck router V2 runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping as MappingABC
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[3]
_CUDA_PYTHON = _ROOT / "engine_cuda_2_0/python"
if str(_CUDA_PYTHON) not in sys.path:
    sys.path.insert(0, str(_CUDA_PYTHON))

from ptcg_cuda_engine.semantic0031_bridge import semantic0031_decode_device

from ..policy import AdaptationConfig
from ..evaluation.candidate import CandidateAudit, materialize as materialize_candidate
from .base_rules import (
    DEFAULT_UPDATE,
    PUBLIC_POLICY_ID,
    PublicDeckMemory,
    ROUTED_UPDATES,
    ROUTE_MANIFEST,
)
from .meta_router import PublicMetaResidentRouter


V14_ADAPTATION = AdaptationConfig(
    rank=16, alpha=16.0, output_projection=True, shared_state_encoder=True,
)
ROUTED_ACTOR_PREFIX = "action_decoder."


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tensor_digest(rows: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(rows.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(_canonical_json(list(tensor.shape))); digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_hash(*modules: nn.Module) -> str:
    rows: dict[str, torch.Tensor] = {}
    for index, module in enumerate(modules):
        rows.update({f"module{index}.{name}": value for name, value in module.state_dict().items()})
    return _tensor_digest(rows)


def _parameter_pointers(*modules: nn.Module) -> set[int]:
    return {
        parameter.untyped_storage().data_ptr()
        for module in modules for parameter in module.parameters()
    }


def _effective_actor_state(actor: nn.Module) -> dict[str, torch.Tensor]:
    """Canonicalize parametrized modules to the tensors used by forward()."""

    effective: dict[str, torch.Tensor] = {}
    parametrized_raw_prefixes: list[str] = []
    for module_name, module in actor.named_modules():
        values = getattr(module, "parametrizations", None)
        if values is None:
            continue
        for tensor_name in values.keys():
            prefix = f"{module_name}.{tensor_name}" if module_name else tensor_name
            raw_prefix = (
                f"{module_name}.parametrizations.{tensor_name}"
                if module_name else f"parametrizations.{tensor_name}"
            )
            parametrized_raw_prefixes.append(raw_prefix)
            effective[prefix] = getattr(module, tensor_name).detach()
    for name, value in actor.state_dict().items():
        if any(
            name.startswith(prefix + ".") for prefix in parametrized_raw_prefixes
        ):
            continue
        effective[name] = value
    return effective


def audit_shared_actor_states(
    actor_states: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    if set(actor_states) != set(ROUTED_UPDATES):
        raise ValueError(f"router actor audit requires exactly {ROUTED_UPDATES}")
    reference = actor_states[DEFAULT_UPDATE]
    names = set(reference)
    if any(set(state) != names for state in actor_states.values()):
        raise RuntimeError("router Actor tensor inventory mismatch")
    routed = sorted(name for name in names if name.startswith(ROUTED_ACTOR_PREFIX))
    shared = sorted(names - set(routed))
    for update, state in actor_states.items():
        for name in shared:
            if not torch.equal(reference[name], state[name]):
                raise RuntimeError(f"shared Actor tensor mismatch: U{update} {name}")
    return {
        "status": "PASS",
        "reference_update": DEFAULT_UPDATE,
        "shared_tensor_count": len(shared),
        "shared_tensor_names": shared,
        "shared_effective_sha256": _tensor_digest(
            {name: reference[name] for name in shared}
        ),
        "routed_actor_tensor_count": len(routed),
        "routed_actor_tensor_names": routed,
        "compared_updates": list(ROUTED_UPDATES),
        "comparison": "torch.equal_after_fp16_storage_fp32_runtime",
    }


def _select_rows(value: Any, rows: torch.Tensor, batch_size: int) -> Any:
    def selected(item: Any) -> Any:
        if isinstance(item, torch.Tensor) and item.ndim and item.shape[0] == batch_size:
            return item.index_select(0, rows)
        return item

    if isinstance(value, MappingABC):
        tensors = {name: selected(item) for name, item in value.items()}
        constructor = getattr(type(value), "from_mapping", None)
        return constructor(tensors) if callable(constructor) else tensors
    if is_dataclass(value):
        return replace(value, **{
            field.name: selected(getattr(value, field.name)) for field in fields(value)
        })
    return value


@dataclass(frozen=True, slots=True)
class RoutedActorBundle:
    checkpoint_update: int
    action_decoder: nn.Module
    policy_option_lora: nn.Module
    allocation_head: nn.Module
    component_sha256: str
    candidate_audit: CandidateAudit


class PublicDeckRouterRuntime:
    def __init__(
        self,
        *,
        model: nn.Module,
        heads: Mapping[int, RoutedActorBundle],
        source_audit: Mapping[str, Any],
        job_count: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.heads = heads
        self.source_audit = source_audit
        self.device = device
        self.memory = PublicDeckMemory(job_count, device)
        self.composite_effective_sha256 = str(
            source_audit["composite_effective_sha256"]
        )
        self.audit = source_audit

    def reset_memory(self, job_count: int) -> None:
        self.memory = PublicDeckMemory(job_count, self.device)

    def decode_compacted(
        self,
        batch: Any,
        *,
        job_indices: torch.Tensor,
        max_select: int,
        greedy: bool,
        compute_stats: bool,
        sampling_seeds: torch.Tensor | None,
        sampling_counters: torch.Tensor | None,
    ) -> dict[str, Any]:
        model = self.model
        actor = model.actor
        validated = actor.validate_batch(batch)
        routes = self.memory.observe(validated, job_indices)
        prototype_memory = actor.prototype_memory()
        state = actor.state_encoder(validated, prototype_memory)
        option_encoder = actor.option_encoder
        inputs = option_encoder.encode_inputs(validated, state, prototype_memory)
        prefix = option_encoder.encode_prefix(validated, state, inputs)
        value_options = option_encoder.encode_final(validated, state, prefix)
        value, auxiliary = model.value_and_aux_from_encoded(validated, state, value_options)
        auxiliary = {"value": value, **auxiliary}

        batch_size = int(routes.numel())
        policy_options = torch.zeros_like(value_options)
        decoded_full: dict[str, torch.Tensor] | None = None
        covered = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for update in ROUTED_UPDATES:
            rows = routes.eq(update).nonzero(as_tuple=False).flatten()
            if not rows.numel():
                continue
            selected_batch = _select_rows(validated, rows, batch_size)
            selected_state = _select_rows(state, rows, batch_size)
            selected_prefix = prefix.index_select(0, rows)
            bundle = self.heads[update]
            options = bundle.policy_option_lora(
                option_encoder, selected_batch, selected_state, selected_prefix
            )
            policy_options.index_copy_(0, rows, options)
            decoded = semantic0031_decode_device(
                bundle.action_decoder,
                selected_batch,
                options,
                selected_state.summary,
                max_select=max_select,
                greedy=greedy,
                compute_stats=compute_stats,
                sampling_seeds=(
                    None if sampling_seeds is None
                    else sampling_seeds.index_select(0, rows)
                ),
                sampling_counters=(
                    None if sampling_counters is None
                    else sampling_counters.index_select(0, rows)
                ),
            )
            if decoded_full is None:
                decoded_full = {
                    name: torch.zeros(
                        (batch_size, *tensor.shape[1:]),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    for name, tensor in decoded.items()
                }
            for name, tensor in decoded.items():
                decoded_full[name].index_copy_(0, rows, tensor)
            covered.index_fill_(0, rows, True)
        if decoded_full is None or not bool(covered.all()):
            raise RuntimeError("public deck router failed to cover every focal row")
        return {
            "decoded": decoded_full,
            "validated": validated,
            "state": state,
            "options": policy_options,
            "auxiliary": auxiliary,
        }

    def allocation_head_for_job(self, job_index: int) -> nn.Module:
        if not 0 <= int(job_index) < self.memory.job_count:
            raise IndexError("allocation route job outside public deck memory")
        update = int(self.memory.route_updates[int(job_index)].item())
        return self.heads[update].allocation_head


class PublicDeckResidentRouter(PublicMetaResidentRouter):
    """Named V2 boundary around the already-audited project-local callback router."""


def materialize_public_deck_router_v2(
    *,
    checkpoints: Mapping[int, Path],
    base_portable: Path,
    base_value_checkpoint: Path,
    deck: Sequence[int],
    deck_id: str,
    own_archetype_id: int,
    output_root: Path,
    device: torch.device,
    job_count: int,
) -> PublicDeckRouterRuntime:
    if set(checkpoints) != set(ROUTED_UPDATES):
        raise ValueError(f"public deck router requires exactly {ROUTED_UPDATES}")
    models: dict[int, nn.Module] = {}
    audits: dict[int, CandidateAudit] = {}
    missing_state_lora: dict[int, list[str]] = {}
    for update in ROUTED_UPDATES:
        model, audit = materialize_candidate(
            checkpoint=checkpoints[update],
            base_portable=base_portable,
            base_value_checkpoint=base_value_checkpoint,
            adaptation=V14_ADAPTATION,
            deck=deck,
            deck_id=deck_id,
            own_archetype_id=own_archetype_id,
            output=output_root / f"update-{update:06d}/model.pt",
            device=device,
        )
        if audit.checkpoint_update != update or audit.status != "PASS":
            raise RuntimeError(f"U{update} candidate identity failed")
        source = torch.load(checkpoints[update], map_location="cpu", weights_only=True)
        source_names = set(source["state_dict"])
        omitted = sorted(
            name for name, _ in model.named_parameters()
            if name.startswith("actor.state_encoder.")
            and ".parametrizations." in name
            and not name.endswith(".original")
            and name not in source_names
        )
        if len(omitted) != 16:
            raise RuntimeError(
                f"U{update} V14 checkpoint omission boundary changed: {omitted}"
            )
        models[update] = model
        audits[update] = audit
        missing_state_lora[update] = omitted

    shared_audit = audit_shared_actor_states({
        update: _effective_actor_state(model.actor) for update, model in models.items()
    })
    heads: dict[int, RoutedActorBundle] = {}
    pointers: dict[int, set[int]] = {}
    for update, model in models.items():
        modules = (
            model.actor.action_decoder,
            model.policy_option_lora,
            model.allocation_head,
        )
        heads[update] = RoutedActorBundle(
            checkpoint_update=update,
            action_decoder=modules[0],
            policy_option_lora=modules[1],
            allocation_head=modules[2],
            component_sha256=_module_hash(*modules),
            candidate_audit=audits[update],
        )
        pointers[update] = _parameter_pointers(*modules)
    for left_index, left in enumerate(ROUTED_UPDATES):
        for right in ROUTED_UPDATES[left_index + 1:]:
            if pointers[left] & pointers[right]:
                raise RuntimeError(f"routed modules U{left}/U{right} alias storage")

    shared_model = models[DEFAULT_UPDATE]
    shared_actor_pointers = {
        parameter.untyped_storage().data_ptr()
        for name, parameter in shared_model.actor.named_parameters()
        if not name.startswith(ROUTED_ACTOR_PREFIX)
    }
    if any(shared_actor_pointers & value for value in pointers.values()):
        raise RuntimeError("routed module aliases shared Actor storage")

    payload = {
        "policy_id": PUBLIC_POLICY_ID,
        "rules": ROUTE_MANIFEST,
        "shared_effective_sha256": shared_audit["shared_effective_sha256"],
        "routed_component_sha256": {
            str(update): heads[update].component_sha256 for update in ROUTED_UPDATES
        },
        "candidate_effective_sha256": {
            str(update): audits[update].effective_candidate_sha256
            for update in ROUTED_UPDATES
        },
        "omitted_state_encoder_lora_names": missing_state_lora[DEFAULT_UPDATE],
    }
    composite = hashlib.sha256(_canonical_json(payload)).hexdigest()
    audit = {
        "schema_version": "0045_public_deck_router_v2_identity_audit_v1",
        "status": "PASS",
        "policy_id": PUBLIC_POLICY_ID,
        "evidence_class": "public_information_routed_policy0814_cuda512_experimental",
        "default_checkpoint_update": DEFAULT_UPDATE,
        "rules": ROUTE_MANIFEST,
        "routed_modules": [
            "policy_option_lora", "actor.action_decoder", "allocation_head",
        ],
        "candidate_materializations": {
            str(update): asdict(audits[update]) for update in ROUTED_UPDATES
        },
        "shared_actor": shared_audit,
        "routed_component_sha256": payload["routed_component_sha256"],
        "shared_vs_routed_storage_aliases": 0,
        "cross_routed_storage_aliases": 0,
        "critic_source_update": DEFAULT_UPDATE,
        "critic_outputs_consumed_by_actor_or_router": False,
        "historical_checkpoint_boundary": {
            "status": "KNOWN_OMISSION",
            "omitted_trainable_tensor_count": 16,
            "omitted_state_encoder_lora_names": missing_state_lora[DEFAULT_UPDATE],
            "interpretation": (
                "scores identify reconstructable Decoder/Option-LoRA/Allocation "
                "candidates, not the complete in-memory V14 behavior policy"
            ),
        },
        "identity_payload": payload,
        "composite_effective_sha256": composite,
    }
    return PublicDeckRouterRuntime(
        model=shared_model,
        heads=heads,
        source_audit=audit,
        job_count=job_count,
        device=device,
    )


__all__ = [
    "PublicDeckResidentRouter",
    "PublicDeckRouterRuntime",
    "RoutedActorBundle",
    "V14_ADAPTATION",
    "audit_shared_actor_states",
    "materialize_public_deck_router_v2",
]
