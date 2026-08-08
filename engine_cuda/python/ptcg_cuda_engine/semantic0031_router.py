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


def branched_option_outputs(
    focal_adapter: Any,
    opponent_last_layer: Any,
    opponent_norm: Any | None,
    batch: Any,
    state: Any,
) -> tuple[Any, Any]:
    """Share Option inputs and block 0, branching only at the adapted final block."""

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
    """One frozen trunk with focal and opponent Option/Decoder endpoints."""

    def __init__(
        self,
        *,
        focal_adapter: Any,
        opponent_last_option_layer: Any,
        opponent_option_norm: Any | None,
        opponent_decoder: Any,
        same_policy: bool,
        focal_summary_fn: Any | None = None,
    ) -> None:
        self.focal_adapter = focal_adapter
        self.opponent_last_option_layer = opponent_last_option_layer
        self.opponent_option_norm = opponent_option_norm
        self.opponent_decoder = opponent_decoder
        self.same_policy = bool(same_policy)
        self.focal_summary_fn = focal_summary_fn

    def encode(self, batch: Any) -> tuple[Any, Any, Any, Any]:
        model = self.focal_adapter.model
        validated = model.validate_batch(batch)
        state = model.state_encoder(
            validated, self.focal_adapter.prototype_memory
        )
        if self.same_policy:
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
        ready = focal_route.bool() | opponent_route.bool()
        focal_decoder = self.focal_adapter.model.action_decoder
        focal_summary = (
            self.focal_summary_fn(state)
            if self.focal_summary_fn is not None else state.summary
        )
        if self.same_policy and focal_greedy:
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
            )
            opponent = semantic0031_decode_device(
                self.opponent_decoder,
                validated,
                opponent_options,
                state.summary,
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
        )


__all__ = [
    "RoutedDecisionBatch",
    "Semantic0031ResidentRouter",
    "branched_option_outputs",
]
