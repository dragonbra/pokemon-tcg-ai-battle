"""Identity-audited complete-Actor Policy-0814 public deck router V3."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path
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
from .base_router import PublicDeckResidentRouter
from .rules import (
    PUBLIC_POLICY_ID,
    PublicDeckMemory,
    ROUTED_UPDATES,
    route_manifest,
)


V14_ADAPTATION = AdaptationConfig(
    rank=16,
    alpha=16.0,
    output_projection=True,
    shared_state_encoder=True,
)


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


def _effective_actor_state(actor: nn.Module) -> dict[str, torch.Tensor]:
    """Canonical tensors actually consumed by forward, excluding raw parametrizations."""

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
        if any(name.startswith(prefix + ".") for prefix in parametrized_raw_prefixes):
            continue
        effective[name] = value.detach()
    return effective


def audit_effective_actor_states(
    actor_states: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    """Classify only globally byte-equal effective tensors as shareable."""

    if set(actor_states) != set(ROUTED_UPDATES):
        raise ValueError(f"V3 Actor audit requires exactly {ROUTED_UPDATES}")
    reference = actor_states[ROUTED_UPDATES[0]]
    names = set(reference)
    if any(set(state) != names for state in actor_states.values()):
        raise RuntimeError("V3 complete Actor tensor inventory mismatch")
    globally_equal = sorted(
        name for name in names
        if all(
            torch.equal(reference[name], actor_states[update][name])
            for update in ROUTED_UPDATES[1:]
        )
    )
    routed = sorted(names - set(globally_equal))
    return {
        "status": "PASS",
        "comparison": "torch.equal_on_effective_fp16_storage_fp32_runtime_tensors",
        "tensor_count": len(names),
        "globally_equal_tensor_count": len(globally_equal),
        "globally_equal_tensor_names": globally_equal,
        "globally_equal_sha256": _tensor_digest(
            {name: reference[name] for name in globally_equal}
        ),
        "routed_tensor_count": len(routed),
        "routed_tensor_names": routed,
        "compared_updates": list(ROUTED_UPDATES),
    }


def _tensor_storage_pointers(module: nn.Module) -> set[int]:
    return {
        tensor.untyped_storage().data_ptr()
        for tensor in (*module.parameters(), *module.buffers())
    }


def assert_no_cross_route_storage_aliases(modules: Mapping[int, nn.Module]) -> None:
    if set(modules) != set(ROUTED_UPDATES):
        raise ValueError(f"storage audit requires exactly {ROUTED_UPDATES}")
    pointers = {
        update: _tensor_storage_pointers(module) for update, module in modules.items()
    }
    for index, left in enumerate(ROUTED_UPDATES):
        for right in ROUTED_UPDATES[index + 1:]:
            if pointers[left] & pointers[right]:
                raise RuntimeError(f"V3 routed policies U{left}/U{right} alias storage")


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


class _RouteModules(nn.Module):
    def __init__(
        self, actor: nn.Module, policy_option_lora: nn.Module,
        allocation_head: nn.Module,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.policy_option_lora = policy_option_lora
        self.allocation_head = allocation_head


@dataclass(frozen=True, slots=True)
class FullActorBundle:
    checkpoint_update: int
    source_version: str
    modules: _RouteModules
    component_sha256: str
    candidate_audit: CandidateAudit

    @property
    def actor(self) -> nn.Module:
        return self.modules.actor

    @property
    def policy_option_lora(self) -> nn.Module:
        return self.modules.policy_option_lora

    @property
    def allocation_head(self) -> nn.Module:
        return self.modules.allocation_head


class PublicDeckRouterV3Runtime:
    def __init__(
        self, *, default_model: nn.Module, bundles: Mapping[int, FullActorBundle],
        source_audit: Mapping[str, Any], job_count: int, device: torch.device,
        default_update: int,
    ) -> None:
        self.model = default_model
        self.bundles = dict(bundles)
        self.source_audit = source_audit
        self.audit = source_audit
        self.device = device
        self.default_update = int(default_update)
        self.memory = PublicDeckMemory(
            job_count, device, default_update=self.default_update
        )
        self.composite_effective_sha256 = str(
            source_audit["composite_effective_sha256"]
        )

    def reset_memory(self, job_count: int) -> None:
        self.memory = PublicDeckMemory(
            job_count, self.device, default_update=self.default_update
        )

    def decode_compacted(
        self, batch: Any, *, job_indices: torch.Tensor, max_select: int,
        greedy: bool, compute_stats: bool,
        sampling_seeds: torch.Tensor | None,
        sampling_counters: torch.Tensor | None,
    ) -> dict[str, Any]:
        # The default Critic is diagnostic only. Its outputs never enter routing or Actor decode.
        validated, critic_state, critic_options = self.model.encode_dual_options(batch)
        routes = self.memory.observe(validated, job_indices)
        value, auxiliary = self.model.value_and_aux_from_encoded(
            validated, critic_state, critic_options.value_options
        )
        auxiliary = {"value": value, **auxiliary}

        batch_size = int(routes.numel())
        policy_options = torch.zeros_like(critic_options.policy_options)
        decoded_full: dict[str, torch.Tensor] | None = None
        covered = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for update in ROUTED_UPDATES:
            rows = routes.eq(update).nonzero(as_tuple=False).flatten()
            if not rows.numel():
                continue
            selected = _select_rows(validated, rows, batch_size)
            bundle = self.bundles[update]
            actor = bundle.actor
            prototypes = actor.prototype_memory()
            state = actor.state_encoder(selected, prototypes)
            option_encoder = actor.option_encoder
            inputs = option_encoder.encode_inputs(selected, state, prototypes)
            prefix = option_encoder.encode_prefix(selected, state, inputs)
            options = bundle.policy_option_lora(
                option_encoder, selected, state, prefix
            )
            policy_options.index_copy_(0, rows, options)
            decoded = semantic0031_decode_device(
                actor.action_decoder,
                selected,
                options,
                state.summary,
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
            raise RuntimeError("public deck V3 router failed to cover every focal row")
        return {
            "decoded": decoded_full,
            "validated": validated,
            "state": critic_state,
            "options": policy_options,
            "auxiliary": auxiliary,
        }

    def allocation_head_for_job(self, job_index: int) -> nn.Module:
        if not 0 <= int(job_index) < self.memory.job_count:
            raise IndexError("allocation route job outside public deck V3 memory")
        update = int(self.memory.route_updates[int(job_index)].item())
        return self.bundles[update].allocation_head


def _route_component_state(model: nn.Module) -> dict[str, torch.Tensor]:
    rows = {
        f"actor.{name}": value
        for name, value in _effective_actor_state(model.actor).items()
    }
    rows.update({
        f"policy_option_lora.{name}": value.detach()
        for name, value in model.policy_option_lora.state_dict().items()
    })
    rows.update({
        f"allocation_head.{name}": value.detach()
        for name, value in model.allocation_head.state_dict().items()
    })
    return rows


def materialize_public_deck_router_v3(
    *, checkpoints: Mapping[int, Path], source_versions: Mapping[int, str],
    expected_source_sha256: Mapping[int, str], base_portable: Path,
    base_value_checkpoint: Path, deck: Sequence[int], deck_id: str,
    own_archetype_id: int, output_root: Path, device: torch.device,
    job_count: int, default_update: int,
) -> PublicDeckRouterV3Runtime:
    required = set(ROUTED_UPDATES)
    if any(set(values) != required for values in (
        checkpoints, source_versions, expected_source_sha256,
    )):
        raise ValueError(f"public deck V3 requires exactly {ROUTED_UPDATES}")
    rules = route_manifest(default_update)
    models: dict[int, nn.Module] = {}
    audits: dict[int, CandidateAudit] = {}
    actor_states: dict[int, Mapping[str, torch.Tensor]] = {}
    route_hashes: dict[int, str] = {}
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
            device=torch.device("cpu"),
        )
        if (
            audit.status != "PASS"
            or audit.checkpoint_update != update
            or audit.source_checkpoint_sha256 != expected_source_sha256[update]
        ):
            raise RuntimeError(f"U{update} complete candidate identity failed")
        models[update] = model
        audits[update] = audit
        actor_states[update] = _effective_actor_state(model.actor)
        route_hashes[update] = _tensor_digest(_route_component_state(model))

    actor_audit = audit_effective_actor_states(actor_states)
    bundles: dict[int, FullActorBundle] = {}
    route_modules: dict[int, _RouteModules] = {}
    for update, model in models.items():
        modules = _RouteModules(
            model.actor, model.policy_option_lora, model.allocation_head
        )
        route_modules[update] = modules
        bundles[update] = FullActorBundle(
            checkpoint_update=update,
            source_version=source_versions[update],
            modules=modules,
            component_sha256=route_hashes[update],
            candidate_audit=audits[update],
        )
    assert_no_cross_route_storage_aliases(route_modules)

    default_model = models[default_update]
    for model in models.values():
        model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    assert_no_cross_route_storage_aliases(route_modules)

    identity_payload = {
        "policy_id": PUBLIC_POLICY_ID,
        "rules": rules,
        "candidate_effective_sha256": {
            str(update): audits[update].effective_candidate_sha256
            for update in ROUTED_UPDATES
        },
        "complete_route_component_sha256": {
            str(update): route_hashes[update] for update in ROUTED_UPDATES
        },
        "actor_globally_equal_sha256": actor_audit["globally_equal_sha256"],
    }
    composite = hashlib.sha256(_canonical_json(identity_payload)).hexdigest()
    audit = {
        "schema_version": "0045_public_deck_router_v3_identity_audit_v1",
        "status": "PASS",
        "policy_id": PUBLIC_POLICY_ID,
        "evidence_class": "public_information_complete_actor_routed_policy0814",
        "default_checkpoint_update": int(default_update),
        "rules": rules,
        "routed_modules": [
            "actor.prototype_encoder",
            "actor.state_encoder_complete_effective",
            "actor.option_encoder_complete_effective",
            "policy_option_lora_complete",
            "actor.action_decoder",
            "allocation_head",
        ],
        "candidate_materializations": {
            str(update): asdict(audits[update]) for update in ROUTED_UPDATES
        },
        "source_versions": {
            str(update): source_versions[update] for update in ROUTED_UPDATES
        },
        "effective_actor_tensor_audit": actor_audit,
        "complete_route_component_sha256": identity_payload[
            "complete_route_component_sha256"
        ],
        "cross_route_storage_aliases": 0,
        "storage_strategy": "independent_complete_actor_per_route",
        "critic_source_update": int(default_update),
        "critic_outputs_consumed_by_actor_or_router": False,
        "historical_v14_missing_state_lora_behavior": (
            "zero_delta_reconstruction_from_immutable_policy0814_base"
        ),
        "identity_payload": identity_payload,
        "composite_effective_sha256": composite,
    }
    return PublicDeckRouterV3Runtime(
        default_model=default_model,
        bundles=bundles,
        source_audit=audit,
        job_count=job_count,
        device=device,
        default_update=default_update,
    )


__all__ = [
    "FullActorBundle",
    "PublicDeckResidentRouter",
    "PublicDeckRouterV3Runtime",
    "V14_ADAPTATION",
    "assert_no_cross_route_storage_aliases",
    "audit_effective_actor_states",
    "materialize_public_deck_router_v3",
]
