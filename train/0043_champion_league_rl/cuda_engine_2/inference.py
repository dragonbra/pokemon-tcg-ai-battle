"""Project-local FP32 policy cohorts over CUDA Engine 2.0 semantic tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ..policy_identity import materialize_policy_bundle
from ..runtime import load_policy
from ..semantic_runtime.contracts.batch import DecisionBatch
from ..semantic_runtime.contracts.fields import EXPECTED_BATCH_KEYS
from ..semantic_runtime.deployment.inference import PortableSemanticPolicy


_MASKS = {
    "card_mask", "resource_mask", "event_mask", "option_mask",
    "option_skill_mask", "option_effect_mask",
}
_FLOATS = {"global_num", "card_num", "resource_num", "event_num", "option_num"}


def decision_batch_from_cuda_codec(encoded: Mapping[str, torch.Tensor]) -> DecisionBatch:
    missing = sorted(EXPECTED_BATCH_KEYS - set(encoded))
    if missing:
        raise ValueError(f"CUDA semantic codec is missing actor fields: {missing}")
    option_mask = encoded["option_mask"]
    if option_mask.device.type != "cuda":
        raise ValueError("CUDA semantic codec tensors must remain device resident")
    tensors: dict[str, torch.Tensor] = {}
    for name in EXPECTED_BATCH_KEYS:
        value = encoded[name]
        if value.device != option_mask.device:
            raise ValueError(f"CUDA semantic field is on a different device: {name}")
        if name in _MASKS:
            tensors[name] = value.bool()
        elif name in _FLOATS:
            tensors[name] = value.float()
        else:
            tensors[name] = value.long()
    return DecisionBatch.from_mapping(tensors)


def _modules(policy: Any) -> tuple[torch.nn.Module, ...]:
    if isinstance(policy, PortableSemanticPolicy):
        return (policy.model,)
    return (
        policy.actor, policy.value_head, policy.allocation_head,
        policy.value_adapter, policy.policy_strategy_adapter,
    )


@dataclass(slots=True)
class CudaPolicyCohort:
    policy_id: str
    deck_id: str
    effective_policy_sha256: str
    policy: Any
    device: torch.device

    @classmethod
    def load(
        cls, project_root, *, policy_id: str, deck_id: str,
        device: str | torch.device = "cuda:0",
    ) -> "CudaPolicyCohort":
        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA policy cohort requires an available CUDA device")
        # Full identity materialization precedes model construction/routing.
        bundle = materialize_policy_bundle(
            project_root, policy_id, purpose="0043_cuda_engine_2_inference"
        )
        policy = load_policy(policy_id, deck_id=deck_id)
        for module in _modules(policy):
            module.to(resolved).eval()
        return cls(
            policy_id=policy_id, deck_id=deck_id,
            effective_policy_sha256=bundle.audit.effective_policy_sha256,
            policy=policy, device=resolved,
        )

    def greedy(self, encoded: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = decision_batch_from_cuda_codec(encoded)
        if batch.option_mask.device != self.device:
            raise ValueError("CUDA cohort received tensors on the wrong device")
        with torch.inference_mode():
            if isinstance(self.policy, PortableSemanticPolicy):
                result = self.policy.model.deterministic_action_tensors(batch)
                if not bool(result.legal.all()):
                    raise RuntimeError("Policy-0809 CUDA greedy decode is illegal")
                return result.sequences, result.lengths
            validated, state, options, _, _, context = self.policy.encode_with_strategy(batch)
            sequences, lengths, legal = self.policy._greedy_strategy(
                validated, state, options, context
            )
            if not bool(legal.all()):
                raise RuntimeError(f"{self.policy_id} CUDA greedy decode is illegal")
            return sequences, lengths


__all__ = ["CudaPolicyCohort", "decision_batch_from_cuda_codec"]
