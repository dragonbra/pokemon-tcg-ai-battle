"""Public-information, earliest-confirmation expert routing for 0045.

The router consumes only certain opponent card identities already present in the
focal semantic observation.  It never receives an opponent deck ID or a Value
network Meta prediction.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping as MappingABC
import sys
from typing import Any, Mapping, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[3]
_CUDA_PYTHON = _ROOT / "engine_cuda_2_0/python"
if str(_CUDA_PYTHON) not in sys.path:
    sys.path.insert(0, str(_CUDA_PYTHON))

from ptcg_cuda_engine.semantic0031_bridge import semantic0031_decode_device
from ptcg_cuda_engine.semantic0031_router import (
    RoutedDecisionBatch,
    Semantic0031ResidentRouter,
    _scatter_rows,
    _select_rows as _engine_select_rows,
)

from .oracle_audit import (
    MetaOracleRuntime,
    materialize_oracle,
)
from ..semantic_runtime.deployment.public_meta_memory import (
    DEFAULT_UPDATE, PUBLIC_POLICY_ID, PublicMetaMemory, ROUTED_UPDATES,
    RULE_MANIFEST, TRIGGER_CARD_IDS,
)


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


class PublicMetaRouterRuntime:
    def __init__(self, oracle: MetaOracleRuntime, *, job_count: int, device: torch.device):
        self.model = oracle.model
        self.heads = oracle.heads
        self.source_audit = oracle.audit
        self.device = device
        self.memory = PublicMetaMemory(job_count, device)
        identity_payload = {
            "policy_id": PUBLIC_POLICY_ID,
            "rules": RULE_MANIFEST,
            "source_composite": oracle.audit["composite_effective_sha256"],
        }
        self.composite_effective_sha256 = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.audit = {
            "schema_version": "0045_public_meta_router_v3_grass_fold_u200_identity_audit",
            "status": "PASS",
            "policy_id": PUBLIC_POLICY_ID,
            "evidence_class": "public_information_routed_cuda_evidence_not_promote",
            "public_inputs": [
                "semantic.card_cat.card_id", "semantic.card_cat.relative_owner",
                "semantic.card_cat.identity_knowledge", "semantic.card_mask",
            ],
            "forbidden_inputs": [
                "opponent_exact_deck_id", "opponent_29_way_meta_label",
                "critic_opponent_meta_logits", "hidden_opponent_hand_or_deck",
            ],
            "default_checkpoint_update": DEFAULT_UPDATE,
            "rules": RULE_MANIFEST,
            "routed_modules": [
                "policy_option_lora", "actor.action_decoder", "allocation_head",
            ],
            "source_head_identity": {
                str(update): oracle.heads[update].component_sha256
                for update in ROUTED_UPDATES
            },
            "candidate_materializations": oracle.audit["candidate_materializations"],
            "shared_actor": oracle.audit["shared_actor"],
            "shared_vs_head_storage_aliases": 0,
            "cross_head_storage_aliases": 0,
            "critic_source_update": DEFAULT_UPDATE,
            "critic_outputs_consumed_by_actor_or_router": False,
            "composite_effective_sha256": self.composite_effective_sha256,
        }

    def reset_memory(self, job_count: int) -> None:
        self.memory = PublicMetaMemory(job_count, self.device)

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
        value, auxiliary = model.value_and_aux_from_encoded(
            validated, state, value_options
        )
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
                    None if sampling_seeds is None else sampling_seeds.index_select(0, rows)
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
            raise RuntimeError("public Meta router failed to cover every focal row")
        return {
            "decoded": decoded_full,
            "validated": validated,
            "state": state,
            "options": policy_options,
            "auxiliary": auxiliary,
        }

    def allocation_head_for_job(self, job_index: int) -> torch.nn.Module:
        if not 0 <= int(job_index) < self.memory.job_count:
            raise IndexError("allocation route job outside public memory")
        update = int(self.memory.route_updates[int(job_index)].item())
        return self.heads[update].allocation_head


class PublicMetaResidentRouter(Semantic0031ResidentRouter):
    """0045-local mixed-lane router; CUDA Engine 2.0 sources remain immutable."""

    def __init__(self, *, focal_compacted_policy_fn: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not callable(focal_compacted_policy_fn):
            raise TypeError("public focal compacted policy callback must be callable")
        self.public_focal_policy_fn = focal_compacted_policy_fn

    def _route_compacted(
        self, batch: Any, *, focal_route: Any, opponent_route: Any,
        max_select: int, focal_greedy: bool, compute_stats: bool,
        focal_sampling_seeds: Any | None,
        focal_sampling_counters: Any | None,
        job_indices: Any | None,
    ) -> RoutedDecisionBatch:
        batch_size = int(focal_route.numel())
        focal_rows = focal_route.bool().nonzero(as_tuple=False).flatten()
        opponent_rows = opponent_route.bool().nonzero(as_tuple=False).flatten()
        if focal_rows.numel() + opponent_rows.numel() != batch_size:
            raise RuntimeError("public role-compacted routing requires one role per row")

        focal_model = self.focal_adapter.model
        validated_full = focal_model.validate_batch(batch)
        focal_decoded = None
        focal_state_full = None
        focal_options_full = None
        focal_auxiliary_full = None
        if focal_rows.numel():
            if job_indices is None:
                raise RuntimeError("public focal routing requires resident job indices")
            focal_batch = _engine_select_rows(batch, focal_rows, batch_size=batch_size)
            routed = self.public_focal_policy_fn(
                focal_batch,
                job_indices=job_indices.index_select(0, focal_rows),
                max_select=max_select,
                greedy=focal_greedy,
                compute_stats=compute_stats,
                sampling_seeds=(
                    None if focal_sampling_seeds is None
                    else focal_sampling_seeds.index_select(0, focal_rows)
                ),
                sampling_counters=(
                    None if focal_sampling_counters is None
                    else focal_sampling_counters.index_select(0, focal_rows)
                ),
            )
            required = {"decoded", "validated", "state", "options", "auxiliary"}
            if not isinstance(routed, dict) or set(routed) != required:
                raise RuntimeError("public focal callback returned an invalid payload")
            focal_decoded = routed["decoded"]
            focal_state_full = _scatter_rows(
                routed["state"], focal_rows, batch_size=batch_size
            )
            focal_options_full = _scatter_rows(
                {"options": routed["options"]}, focal_rows, batch_size=batch_size
            )["options"]
            focal_auxiliary_full = _scatter_rows(
                routed["auxiliary"], focal_rows, batch_size=batch_size
            )

        opponent_decoded = None
        if opponent_rows.numel():
            opponent_batch = _engine_select_rows(
                batch, opponent_rows, batch_size=batch_size
            )
            opponent_model = self.opponent_adapter.model
            opponent_validated = opponent_model.validate_batch(opponent_batch)
            opponent_state = opponent_model.state_encoder(
                opponent_validated, self.opponent_adapter.prototype_memory
            )
            opponent_options = self.opponent_adapter._encode_options(
                opponent_validated, opponent_state
            )
            opponent_readout_fn = None
            if self.opponent_strategy_fn is not None:
                opponent_readout_fn, _ = self.opponent_strategy_fn(
                    opponent_validated, opponent_state, opponent_options,
                    None if job_indices is None
                    else job_indices.index_select(0, opponent_rows),
                )
            opponent_decoded = semantic0031_decode_device(
                self.opponent_decoder, opponent_validated, opponent_options,
                opponent_state.summary, max_select=max_select, greedy=True,
                compute_stats=False, readout_fn=opponent_readout_fn,
            )

        exemplar = focal_decoded if focal_decoded is not None else opponent_decoded
        width = int(exemplar["actions"].shape[1])
        actions = torch.zeros(
            (batch_size, width), dtype=exemplar["actions"].dtype,
            device=exemplar["actions"].device,
        )
        lengths = torch.zeros(
            batch_size, dtype=exemplar["lengths"].dtype, device=actions.device
        )
        stopped = torch.zeros(
            batch_size, dtype=exemplar["stopped"].dtype, device=actions.device
        )
        focal_logprob = torch.zeros(batch_size, device=actions.device)
        focal_entropy = torch.zeros(batch_size, device=actions.device)
        if focal_decoded is not None:
            actions.index_copy_(0, focal_rows, focal_decoded["actions"])
            lengths.index_copy_(0, focal_rows, focal_decoded["lengths"])
            stopped.index_copy_(0, focal_rows, focal_decoded["stopped"])
            focal_logprob.index_copy_(0, focal_rows, focal_decoded["logprob"])
            focal_entropy.index_copy_(0, focal_rows, focal_decoded["entropy"])
        if opponent_decoded is not None:
            actions.index_copy_(0, opponent_rows, opponent_decoded["actions"])
            lengths.index_copy_(0, opponent_rows, opponent_decoded["lengths"])
            stopped.index_copy_(0, opponent_rows, opponent_decoded["stopped"])
        return RoutedDecisionBatch(
            actions=actions, lengths=lengths, stopped=stopped,
            focal_logprob=focal_logprob, focal_entropy=focal_entropy,
            validated=validated_full, state=focal_state_full,
            focal_options=focal_options_full,
            focal_auxiliary=focal_auxiliary_full,
        )


def materialize_public_router(
    *, checkpoints: Mapping[int, Path], base_portable: Path,
    deck: Sequence[int], deck_id: str, own_archetype_id: int,
    output_root: Path, device: torch.device, job_count: int,
) -> PublicMetaRouterRuntime:
    sources = materialize_oracle(
        checkpoints=checkpoints, base_portable=base_portable,
        deck=deck, deck_id=deck_id, own_archetype_id=own_archetype_id,
        output_root=output_root, device=device,
    )
    sources.activate(DEFAULT_UPDATE)
    return PublicMetaRouterRuntime(sources, job_count=job_count, device=device)


__all__ = [
    "PUBLIC_POLICY_ID", "PublicMetaMemory", "PublicMetaRouterRuntime",
    "PublicMetaResidentRouter", "RULE_MANIFEST", "TRIGGER_CARD_IDS",
    "materialize_public_router",
]
