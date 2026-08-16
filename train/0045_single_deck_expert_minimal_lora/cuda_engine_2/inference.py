"""Project-local FP32 policy cohorts over CUDA Engine 2.0 semantic tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ..policy_identity import materialize_policy_bundle
from ..runtime import load_focal_seed, load_policy
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
    modules = [
        policy.actor, policy.value_head, policy.allocation_head,
        policy.value_adapter,
    ]
    for name in ("policy_option_lora", "policy_strategy_adapter", "meta_actor_residual"):
        module = getattr(policy, name, None)
        if module is not None:
            modules.append(module)
    return tuple(modules)


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
            project_root, policy_id, purpose="0044_cuda_engine_2_inference"
        )
        policy = load_policy(policy_id, deck_id=deck_id)
        for module in _modules(policy):
            module.to(resolved).eval()
        return cls(
            policy_id=policy_id, deck_id=deck_id,
            effective_policy_sha256=bundle.audit.effective_policy_sha256,
            policy=policy, device=resolved,
        )

    @classmethod
    def load_focal(
        cls, project_root, *, deck_id: str = "002",
        device: str | torch.device = "cuda:0",
    ) -> "CudaPolicyCohort":
        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA focal cohort requires an available CUDA device")
        policy = load_focal_seed(deck_id=deck_id)
        for module in _modules(policy):
            module.to(resolved).eval()
        from ..assets import sha256_file
        seed = (project_root.parents[1] / "rl_runs/0044_g2_dragapult_policy_option_lora/versions/"
                "V1_focal_002_007/artifact/focal_seed/model.bin")
        return cls("V1-Focal-Seed", deck_id, sha256_file(seed), policy, resolved)

    def greedy(
        self, encoded: Mapping[str, torch.Tensor], *,
        own_archetype_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = decision_batch_from_cuda_codec(encoded)
        if batch.option_mask.device != self.device:
            raise ValueError("CUDA cohort received tensors on the wrong device")
        with torch.inference_mode():
            if isinstance(self.policy, PortableSemanticPolicy):
                result = self.policy.model.deterministic_action_tensors(batch)
                if not bool(result.legal.all()):
                    raise RuntimeError("Policy-0809 CUDA greedy decode is illegal")
                return result.sequences, result.lengths
            validated, state, options, _, _, context = self.policy.encode_with_strategy(
                batch, own_archetype_id=own_archetype_ids
            )
            sequences, lengths, legal = self.policy._greedy_strategy(
                validated, state, options, context
            )
            if not bool(legal.all()):
                raise RuntimeError(f"{self.policy_id} CUDA greedy decode is illegal")
            return sequences, lengths


class ResidentPolicyPool:
    """Run-scoped policy cache; deck routing never reloads effective weights."""

    def __init__(self, project_root, *, device: str | torch.device = "cuda:0") -> None:
        self.project_root = project_root
        self.device = torch.device(device)
        self._cohorts: dict[str, CudaPolicyCohort] = {}
        self._loads: dict[str, int] = {}

    def get(self, policy_id: str) -> CudaPolicyCohort:
        admitted = {
            policy.policy_id for policy in __import__(
                "train.0045_single_deck_expert_minimal_lora.assets", fromlist=["AssetRegistry"]
            ).AssetRegistry.load(self.project_root).policies
        }
        if policy_id != "V1-Focal-Seed" and policy_id not in admitted:
            raise ValueError(f"non-admitted resident policy: {policy_id}")
        if policy_id not in self._cohorts:
            self._cohorts[policy_id] = (
                CudaPolicyCohort.load_focal(self.project_root, device=self.device)
                if policy_id == "V1-Focal-Seed" else
                CudaPolicyCohort.load(
                    self.project_root, policy_id=policy_id, deck_id="001", device=self.device
                )
            )
            self._loads[policy_id] = self._loads.get(policy_id, 0) + 1
        return self._cohorts[policy_id]

    def warm(self) -> None:
        from ..assets import AssetRegistry
        for policy_id in ("V1-Focal-Seed", *AssetRegistry.load(self.project_root).active_policy_ids):
            self.get(policy_id)

    @property
    def load_counts(self) -> dict[str, int]:
        return dict(self._loads)

    def own_ids(self, policy_id: str, deck_ids: Sequence[str]) -> torch.Tensor | None:
        if policy_id in {"Policy-0809", "Policy-0814"}:
            return None
        from ..own_archetype import OwnArchetypeVocabulary

        vocabulary = OwnArchetypeVocabulary.load_version(
            "own_archetypes_v2", project_root=self.project_root
        )
        cohort = self.get(policy_id)
        embedding_count = cohort.policy.value_adapter.own_embedding.num_embeddings
        mapping = {
            row.deck_id: (
                row.archetype_id if embedding_count == vocabulary.class_count
                else vocabulary.classes[row.archetype_id].embedding_init_from
            ) for row in vocabulary.mappings
        }
        try:
            values = [mapping[deck_id] for deck_id in deck_ids]
        except KeyError as error:
            raise ValueError(f"unmapped exact opponent deck: {error.args[0]}") from error
        return torch.tensor(values, dtype=torch.long, device=self.device)


__all__ = [
    "CudaPolicyCohort", "ResidentPolicyPool", "decision_batch_from_cuda_codec",
]
