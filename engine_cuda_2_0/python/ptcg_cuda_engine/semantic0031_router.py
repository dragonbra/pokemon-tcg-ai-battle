"""Shared CUDA-resident routing for focal and frozen Semantic0031 policies."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields, is_dataclass, replace
import inspect
from types import SimpleNamespace
from typing import Any

from .semantic0031_bridge import semantic0031_decode_device


@dataclass(frozen=True)
class RoutedDecisionBatch:
    actions: Any
    lengths: Any
    stopped: Any
    focal_logprob: Any
    focal_entropy: Any
    validated: Any
    state: Any
    focal_options: Any
    focal_auxiliary: Any | None = None


def _select_rows(batch: Any, rows: Any, *, batch_size: int) -> Any:
    """Gather one role without changing scalar/schema metadata."""

    def selected(value: Any) -> Any:
        if hasattr(value, "ndim") and value.ndim > 0 and value.shape[0] == batch_size:
            return value.index_select(0, rows)
        return value

    if isinstance(batch, dict):
        return {name: selected(value) for name, value in batch.items()}
    if is_dataclass(batch):
        return replace(batch, **{
            field.name: selected(getattr(batch, field.name)) for field in fields(batch)
        })
    return SimpleNamespace(**{
        name: selected(value) for name, value in vars(batch).items()
    })


def _scatter_rows(value: Any, rows: Any, *, batch_size: int) -> Any:
    """Restore compact role outputs to mixed-batch row coordinates."""
    import torch

    role_size = int(rows.numel())

    def scattered(item: Any) -> Any:
        if (
            isinstance(item, torch.Tensor)
            and item.ndim > 0
            and item.shape[0] == role_size
        ):
            output = torch.zeros(
                (batch_size, *item.shape[1:]), dtype=item.dtype, device=item.device
            )
            return output.index_copy(0, rows, item)
        return item

    if isinstance(value, dict):
        return {name: scattered(item) for name, item in value.items()}
    if is_dataclass(value):
        return replace(value, **{
            field.name: scattered(getattr(value, field.name)) for field in fields(value)
        })
    return SimpleNamespace(**{
        name: scattered(item) for name, item in vars(value).items()
    })


@dataclass(frozen=True, slots=True)
class SharedTrunkIdentityProof:
    """Content identity proof for every component reused across policies."""

    focal_policy_id: str
    opponent_policy_id: str
    focal_component_sha256: dict[str, str]
    opponent_component_sha256: dict[str, str]

    def validate(self) -> None:
        shared = (
            "prototype_encoder",
            "state_encoder",
            "option_input_encoder",
            "option_transformer_layer_0",
        )
        mismatches = {
            name: (
                self.focal_component_sha256.get(name),
                self.opponent_component_sha256.get(name),
            )
            for name in shared
            if (
                not self.focal_component_sha256.get(name)
                or self.focal_component_sha256.get(name)
                != self.opponent_component_sha256.get(name)
            )
        }
        if mismatches:
            raise RuntimeError(
                "FATAL: cross-policy shared trunk has no effective-weight "
                f"identity proof: {mismatches}"
            )


def branched_option_outputs(
    focal_adapter: Any,
    opponent_last_layer: Any,
    opponent_norm: Any | None,
    batch: Any,
    state: Any,
    *,
    identity_proof: SharedTrunkIdentityProof,
) -> tuple[Any, Any]:
    """Share an exactly identical trunk, proven before optimized execution."""

    identity_proof.validate()

    focal_transformer = (
        focal_adapter.model.option_encoder.cross_attention_transformer
    )
    if len(focal_transformer.layers) != 2:
        raise RuntimeError("shared Option routing requires exactly two focal blocks")
    inputs = focal_adapter._encode_option_inputs(batch, state)
    masks = {
        "tgt_key_padding_mask": ~batch.option_mask,
        "memory_key_padding_mask": ~state.mask,
    }
    shared = focal_transformer.layers[0](inputs, state.tokens, **masks)
    focal = focal_transformer.layers[1](shared, state.tokens, **masks)
    opponent = opponent_last_layer(shared, state.tokens, **masks)
    if focal_transformer.norm is not None:
        focal = focal_transformer.norm(focal)
    if opponent_norm is not None:
        opponent = opponent_norm(opponent)
    mask = batch.option_mask.unsqueeze(-1)
    return focal * mask, opponent * mask


class Semantic0031ResidentRouter:
    """Route complete effective policies after fail-closed identity resolution."""

    def __init__(
        self,
        *,
        focal_adapter: Any,
        same_policy: bool,
        opponent_adapter: Any | None = None,
        opponent_last_option_layer: Any | None = None,
        opponent_option_norm: Any | None = None,
        opponent_decoder: Any | None = None,
        requested_opponent_policy_id: str | None = None,
        opponent_identity_audit: Any | None = None,
        shared_trunk_identity_proof: SharedTrunkIdentityProof | None = None,
        focal_summary_fn: Any | None = None,
        focal_strategy_fn: Any | None = None,
        opponent_strategy_fn: Any | None = None,
        role_compacted: bool = False,
    ) -> None:
        self.focal_adapter = focal_adapter
        self.opponent_last_option_layer = opponent_last_option_layer
        self.opponent_option_norm = opponent_option_norm
        self.opponent_decoder = opponent_decoder
        self.same_policy = bool(same_policy)
        self.opponent_adapter = opponent_adapter
        self.requested_opponent_policy_id = requested_opponent_policy_id
        self.opponent_identity_audit = opponent_identity_audit
        self.shared_trunk_identity_proof = shared_trunk_identity_proof
        self.focal_summary_fn = focal_summary_fn
        self.focal_strategy_fn = focal_strategy_fn
        self._focal_strategy_accepts_job_indices = (
            focal_strategy_fn is not None
            and len(inspect.signature(focal_strategy_fn).parameters) >= 4
        )
        self.opponent_strategy_fn = opponent_strategy_fn
        self.role_compacted = bool(role_compacted)
        if self.same_policy:
            if opponent_adapter is not None:
                raise RuntimeError("same-policy routing must use the focal full adapter")
            self.opponent_decoder = focal_adapter.model.action_decoder
        else:
            if opponent_adapter is None:
                raise RuntimeError(
                    "FATAL: cross-policy routing requires a complete opponent adapter; "
                    "partial opponent layer/head routing is forbidden"
                )
            audit = opponent_identity_audit
            status = (
                audit.get("status") if isinstance(audit, dict)
                else getattr(audit, "status", None)
            )
            audited_policy_id = (
                audit.get("requested_policy_id") if isinstance(audit, dict)
                else getattr(audit, "requested_policy_id", None)
            )
            effective_hash = (
                audit.get("effective_policy_sha256") if isinstance(audit, dict)
                else getattr(audit, "effective_policy_sha256", None)
            )
            if (
                status != "PASS"
                or not requested_opponent_policy_id
                or audited_policy_id != requested_opponent_policy_id
                or not isinstance(effective_hash, str)
                or len(effective_hash) != 64
            ):
                raise RuntimeError(
                    "FATAL: complete opponent adapter lacks a passing policy identity audit"
                )
            adapter_decoder = opponent_adapter.model.action_decoder
            if opponent_decoder is not None and opponent_decoder is not adapter_decoder:
                raise RuntimeError(
                    "FATAL: supplied opponent decoder is not owned by the audited full policy"
                )
            self.opponent_decoder = adapter_decoder

    def _route_compacted(
        self,
        batch: Any,
        *,
        focal_route: Any,
        opponent_route: Any,
        max_select: int,
        focal_greedy: bool,
        compute_stats: bool,
        focal_sampling_seeds: Any | None,
        focal_sampling_counters: Any | None,
        job_indices: Any | None,
    ) -> RoutedDecisionBatch:
        """Run each complete policy only on rows assigned to that role."""
        import torch

        batch_size = int(focal_route.numel())
        focal_rows = focal_route.bool().nonzero(as_tuple=False).flatten()
        opponent_rows = opponent_route.bool().nonzero(as_tuple=False).flatten()
        if focal_rows.numel() + opponent_rows.numel() != batch_size:
            raise RuntimeError("role-compacted routing requires exactly one role per row")

        focal_model = self.focal_adapter.model
        validated_full = focal_model.validate_batch(batch)
        focal_decoded = None
        focal_state_full = None
        focal_options_full = None
        focal_auxiliary_full = None
        if focal_rows.numel():
            focal_batch = _select_rows(batch, focal_rows, batch_size=batch_size)
            focal_validated = focal_model.validate_batch(focal_batch)
            focal_state = focal_model.state_encoder(
                focal_validated, self.focal_adapter.prototype_memory
            )
            focal_options = self.focal_adapter._encode_options(
                focal_validated, focal_state
            )
            focal_summary = (
                self.focal_summary_fn(focal_state)
                if self.focal_summary_fn is not None else focal_state.summary
            )
            focal_readout_fn = None
            focal_auxiliary = None
            if self.focal_strategy_fn is not None:
                focal_args = (focal_validated, focal_state, focal_options)
                if self._focal_strategy_accepts_job_indices:
                    if job_indices is None:
                        raise RuntimeError(
                            "focal strategy requires resident job indices"
                        )
                    focal_args += (job_indices.index_select(0, focal_rows),)
                focal_readout_fn, focal_auxiliary = self.focal_strategy_fn(*focal_args)
                if not callable(focal_readout_fn) or not isinstance(
                    focal_auxiliary, dict
                ):
                    raise RuntimeError(
                        "focal strategy function must return (readout_fn, auxiliary dict)"
                    )
            focal_decoded = semantic0031_decode_device(
                focal_model.action_decoder,
                focal_validated,
                focal_options,
                focal_summary,
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
                readout_fn=focal_readout_fn,
            )
            focal_state_full = _scatter_rows(
                focal_state, focal_rows, batch_size=batch_size
            )
            focal_options_full = _scatter_rows(
                {"options": focal_options}, focal_rows, batch_size=batch_size
            )["options"]
            if focal_auxiliary is not None:
                focal_auxiliary_full = _scatter_rows(
                    focal_auxiliary, focal_rows, batch_size=batch_size
                )

        opponent_decoded = None
        if opponent_rows.numel():
            opponent_batch = _select_rows(batch, opponent_rows, batch_size=batch_size)
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
                    None if job_indices is None else job_indices.index_select(0, opponent_rows),
                )
            opponent_decoded = semantic0031_decode_device(
                self.opponent_decoder,
                opponent_validated,
                opponent_options,
                opponent_state.summary,
                max_select=max_select,
                greedy=True,
                compute_stats=False,
                readout_fn=opponent_readout_fn,
            )

        exemplar = focal_decoded if focal_decoded is not None else opponent_decoded
        width = int(exemplar["actions"].shape[1])
        actions = torch.zeros(
            (batch_size, width), dtype=exemplar["actions"].dtype,
            device=exemplar["actions"].device,
        )
        lengths = torch.zeros(
            batch_size, dtype=exemplar["lengths"].dtype,
            device=exemplar["lengths"].device,
        )
        stopped = torch.zeros(
            batch_size, dtype=exemplar["stopped"].dtype,
            device=exemplar["stopped"].device,
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
            actions=actions,
            lengths=lengths,
            stopped=stopped,
            focal_logprob=focal_logprob,
            focal_entropy=focal_entropy,
            validated=validated_full,
            state=focal_state_full,
            focal_options=focal_options_full,
            focal_auxiliary=focal_auxiliary_full,
        )

    def encode(self, batch: Any) -> tuple[Any, Any, Any, Any]:
        model = self.focal_adapter.model
        validated = model.validate_batch(batch)
        state = model.state_encoder(
            validated, self.focal_adapter.prototype_memory
        )
        if self.same_policy or self.opponent_adapter is not None:
            focal_options = self.focal_adapter._encode_options(validated, state)
            opponent_options = focal_options
        else:
            focal_options, opponent_options = branched_option_outputs(
                self.focal_adapter,
                self.opponent_last_option_layer,
                self.opponent_option_norm,
                validated,
                state,
            )
        return validated, state, focal_options, opponent_options

    def route(
        self,
        batch: Any,
        *,
        focal_route: Any,
        opponent_route: Any,
        max_select: int,
        focal_greedy: bool,
        compute_stats: bool = True,
        focal_sampling_seeds: Any | None = None,
        focal_sampling_counters: Any | None = None,
        job_indices: Any | None = None,
    ) -> RoutedDecisionBatch:
        import torch

        if self.role_compacted and not self.same_policy:
            return self._route_compacted(
                batch,
                focal_route=focal_route,
                opponent_route=opponent_route,
                max_select=max_select,
                focal_greedy=focal_greedy,
                compute_stats=compute_stats,
                focal_sampling_seeds=focal_sampling_seeds,
                focal_sampling_counters=focal_sampling_counters,
                job_indices=job_indices,
            )

        validated, state, focal_options, opponent_options = self.encode(batch)
        opponent_state = state
        opponent_validated = validated
        if self.opponent_adapter is not None:
            opponent_model = self.opponent_adapter.model
            opponent_validated = opponent_model.validate_batch(batch)
            opponent_state = opponent_model.state_encoder(
                opponent_validated, self.opponent_adapter.prototype_memory
            )
            opponent_options = self.opponent_adapter._encode_options(
                opponent_validated, opponent_state
            )
        opponent_readout_fn = None
        if self.opponent_strategy_fn is not None:
            opponent_readout_fn, _ = self.opponent_strategy_fn(
                opponent_validated, opponent_state, opponent_options, job_indices
            )
        ready = focal_route.bool() | opponent_route.bool()
        focal_decoder = self.focal_adapter.model.action_decoder
        focal_summary = (
            self.focal_summary_fn(state)
            if self.focal_summary_fn is not None else state.summary
        )
        focal_readout_fn = None
        focal_auxiliary = None
        if self.focal_strategy_fn is not None:
            focal_args = (validated, state, focal_options)
            if self._focal_strategy_accepts_job_indices:
                if job_indices is None:
                    raise RuntimeError("focal strategy requires resident job indices")
                focal_args += (job_indices,)
            focal_readout_fn, focal_auxiliary = self.focal_strategy_fn(*focal_args)
            if not callable(focal_readout_fn) or not isinstance(
                focal_auxiliary, dict
            ):
                raise RuntimeError(
                    "focal strategy function must return (readout_fn, auxiliary dict)"
                )
        if self.same_policy and focal_greedy:
            if focal_readout_fn is not None:
                raise RuntimeError(
                    "same-policy shared decode cannot apply a focal-only strategy adapter"
                )
            shared = semantic0031_decode_device(
                focal_decoder,
                validated,
                focal_options,
                focal_summary,
                max_select=max_select,
                greedy=True,
                route_mask=ready,
                compute_stats=compute_stats,
            )
            focal = shared
            opponent = shared
        else:
            focal = semantic0031_decode_device(
                focal_decoder,
                validated,
                focal_options,
                focal_summary,
                max_select=max_select,
                greedy=focal_greedy,
                route_mask=focal_route,
                compute_stats=compute_stats,
                sampling_seeds=focal_sampling_seeds,
                sampling_counters=focal_sampling_counters,
                readout_fn=focal_readout_fn,
            )
            opponent = semantic0031_decode_device(
                self.opponent_decoder,
                opponent_validated,
                opponent_options,
                opponent_state.summary,
                max_select=max_select,
                greedy=True,
                route_mask=opponent_route,
                compute_stats=False,
                readout_fn=opponent_readout_fn,
            )
        actions = torch.where(
            focal_route.bool().view(-1, 1), focal["actions"], opponent["actions"]
        ).contiguous()
        lengths = torch.where(
            focal_route.bool(), focal["lengths"], opponent["lengths"]
        )
        stopped = torch.where(
            focal_route.bool(), focal["stopped"], opponent["stopped"]
        )
        return RoutedDecisionBatch(
            actions=actions,
            lengths=lengths,
            stopped=stopped,
            focal_logprob=focal["logprob"],
            focal_entropy=focal["entropy"],
            validated=validated,
            state=state,
            focal_options=focal_options,
            focal_auxiliary=focal_auxiliary,
        )


__all__ = [
    "RoutedDecisionBatch",
    "Semantic0031ResidentRouter",
    "SharedTrunkIdentityProof",
    "branched_option_outputs",
]
