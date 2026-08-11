"""Shared CUDA-resident routing for focal and frozen Semantic0031 policies."""

from __future__ import annotations

from dataclasses import dataclass
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
    ) -> RoutedDecisionBatch:
        import torch

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
        ready = focal_route.bool() | opponent_route.bool()
        focal_decoder = self.focal_adapter.model.action_decoder
        focal_summary = (
            self.focal_summary_fn(state)
            if self.focal_summary_fn is not None else state.summary
        )
        focal_readout_fn = None
        focal_auxiliary = None
        if self.focal_strategy_fn is not None:
            focal_readout_fn, focal_auxiliary = self.focal_strategy_fn(
                validated, state, focal_options
            )
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
