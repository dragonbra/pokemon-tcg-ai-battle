"""CUDA-resident focal runtime for public-meta hard identification + soft MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Categorical

from ..evaluation.public_meta_router_v1 import PublicMetaResidentRouter
from ..policy.moe_actor_critic import MetaRoutedMoEActorCritic
from ..policy.moe_distribution import decode_moe_device, routing_groups, select_rows
from ..semantic_runtime.action_boundary.dragapult import enumerate_allocations
from ..semantic_runtime.action_boundary.macro_planner import MacroPlanner, PlannedMacro
from ..semantic_runtime.deployment.public_meta29_memory import PublicMeta29Memory


class MoEResidentRouter(PublicMetaResidentRouter):
    """Callback-only reuse of the audited mixed-role scatter/gather router."""


@dataclass(slots=True)
class _DecisionContext:
    expert_options: dict[int, torch.Tensor]
    posterior: torch.Tensor


class MoEFocalRuntime:
    def __init__(self, model: MetaRoutedMoEActorCritic, *, job_count: int,
                 device: torch.device) -> None:
        self.model = model
        self.device = device
        self.memory = PublicMeta29Memory(job_count, device)
        self.contexts: dict[int, _DecisionContext] = {}
        self.usage_sum = torch.zeros(model.expert_count, dtype=torch.float64)
        self.decision_count = 0
        self.unknown_decisions = 0

    def reset(self, job_count: int) -> None:
        self.memory = PublicMeta29Memory(job_count, self.device)
        self.contexts.clear()
        self.usage_sum.zero_()
        self.decision_count = 0
        self.unknown_decisions = 0

    def decode_compacted(
        self, batch: Any, *, job_indices: torch.Tensor, max_select: int,
        greedy: bool, compute_stats: bool,
        sampling_seeds: torch.Tensor | None,
        sampling_counters: torch.Tensor | None,
    ) -> dict[str, Any]:
        validated, state, prefix, value_options = self.model.encode_shared(batch)
        meta_ids = self.memory.observe(validated, job_indices)
        gate = self.model.gate_probabilities(meta_ids)
        batch_size = validated.batch_size
        decoded = {
            "actions": torch.full(
                (batch_size, max_select), -1, dtype=torch.long, device=self.device
            ),
            "lengths": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            "stopped": torch.zeros(batch_size, dtype=torch.bool, device=self.device),
            "logprob": torch.zeros(batch_size, dtype=torch.float32, device=self.device),
            "entropy": torch.zeros(batch_size, dtype=torch.float32, device=self.device),
        }
        focal_options = torch.empty_like(value_options)
        for rows, active_experts in routing_groups(self.model, meta_ids):
            partial, expert_options, posterior = decode_moe_device(
                self.model,
                select_rows(validated, rows, batch_size),
                select_rows(state, rows, batch_size),
                prefix.index_select(0, rows), value_options.index_select(0, rows),
                meta_ids.index_select(0, rows),
                max_select=max_select, greedy=greedy, compute_stats=compute_stats,
                sampling_seeds=(
                    None if sampling_seeds is None else sampling_seeds.index_select(0, rows)
                ),
                sampling_counters=(
                    None if sampling_counters is None else sampling_counters.index_select(0, rows)
                ),
                active_experts=active_experts,
            )
            for name, value in partial.items():
                decoded[name].index_copy_(0, rows, value)
            primary_options = expert_options[active_experts[-1]]
            focal_options.index_copy_(0, rows, primary_options)
            for local_row, global_row in enumerate(rows.tolist()):
                job = int(job_indices[global_row])
                self.contexts[job] = _DecisionContext(
                    {
                        index: options[local_row].detach().clone()
                        for index, options in expert_options.items()
                    },
                    posterior[local_row].detach().clone(),
                )
        value, auxiliary = self.model.value_and_aux_from_encoded(
            validated, state, value_options
        )
        auxiliary = {"value": value, **auxiliary, "routing_meta_id": meta_ids}
        self.usage_sum += gate.detach().double().sum(dim=0).cpu()
        self.decision_count += int(meta_ids.numel())
        self.unknown_decisions += int(meta_ids.lt(0).sum())
        return {
            "decoded": decoded, "validated": validated, "state": state,
            # Boundary uses this only as a compatibility placeholder. The
            # allocation callback below consumes all expert-local options.
            "options": focal_options, "auxiliary": auxiliary,
        }

    def plan_allocation(self, job_index: int, *, state_summary: torch.Tensor,
                        root_index: int, root_option: torch.Tensor,
                        target_embeddings: torch.Tensor, target_identities,
                        visible_targets: list[dict[str, Any]], greedy: bool,
                        generator: torch.Generator | None) -> PlannedMacro:
        del root_option
        context = self.contexts.get(int(job_index))
        if context is None:
            raise RuntimeError("0047 allocation has no same-decision MoE context")
        allocations = enumerate_allocations(target_identities)
        if len(allocations) == 1:
            return PlannedMacro(allocations[0], 0, 0.0, 0.0)
        features = MacroPlanner.visible_features(visible_targets, allocations).to(
            device=target_embeddings.device, dtype=target_embeddings.dtype
        )
        count = len(allocations)
        embeddings = target_embeddings.unsqueeze(0).expand(count, -1, -1).unsqueeze(0)
        batched_features = features.unsqueeze(0)
        target_mask = torch.ones(
            batched_features.shape[:-1], dtype=torch.bool, device=self.device
        )
        allocation_mask = torch.ones((1, count), dtype=torch.bool, device=self.device)
        expert_probabilities = []
        with torch.inference_mode():
            for index, options in context.expert_options.items():
                expert = self.model.experts[index]
                logits = expert.allocation_head(
                    state_summary.unsqueeze(0),
                    options[root_index].unsqueeze(0),
                    embeddings, batched_features, target_mask, allocation_mask,
                )[0]
                expert_probabilities.append(logits.float().softmax(dim=-1))
            active = torch.tensor(
                tuple(context.expert_options), device=self.device, dtype=torch.long
            )
            probabilities = (
                context.posterior.index_select(0, active)[:, None]
                * torch.stack(expert_probabilities, dim=0)
            ).sum(dim=0)
            distribution = Categorical(probs=probabilities)
            index = (
                probabilities.argmax() if greedy
                else torch.multinomial(probabilities, 1, generator=generator).squeeze(0)
            )
        return PlannedMacro(
            allocations[int(index)], int(index), float(distribution.log_prob(index)),
            float(distribution.entropy()),
        )

    def telemetry(self) -> dict[str, Any]:
        memory = self.memory.telemetry()
        denominator = max(1, self.decision_count)
        return {
            **memory,
            "effective_usage": [float(value / denominator) for value in self.usage_sum],
            "unknown_decision_fraction": self.unknown_decisions / denominator,
            "total_decisions": self.decision_count,
        }


__all__ = ["MoEFocalRuntime", "MoEResidentRouter"]
