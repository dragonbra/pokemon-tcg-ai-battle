"""Kaggle-equivalent candidate materialization for formal strength evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn

from .export_full_semantic_candidate import (
    PORTABLE_STATE_FIELDS,
    deployment_effective_sha256,
    export_candidate,
)


CONTRACT_ID = "kaggle_fp16_storage_fp32_runtime_v1"
AUDIT_SCHEMA = "0042_candidate_deployment_identity_audit_v1"
PORTABLE_SCHEMA = "0042_strategy_conditioned_kaggle_candidate_v1"


class CandidateDeploymentIdentityViolation(RuntimeError):
    """The evaluated candidate differs from the Kaggle deployment contract."""


@dataclass(frozen=True, slots=True)
class CandidateDeploymentAudit:
    contract_id: str
    source_checkpoint_sha256: str
    portable_checkpoint_sha256: str
    effective_candidate_sha256: str
    checkpoint_update: int
    storage_dtype: str = "fp16"
    runtime_dtype: str = "fp32"
    conversion_order: str = "merge_full_effective_policy_then_fp16_storage_then_fp32_runtime"
    status: str = "PASS"
    schema_version: str = AUDIT_SCHEMA

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def _fatal(detail: str) -> CandidateDeploymentIdentityViolation:
    return CandidateDeploymentIdentityViolation(
        f"FATAL: candidate deployment identity violation. {detail}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_tensors(payload: Mapping[str, Any]) -> Iterator[tuple[str, Tensor]]:
    for field in PORTABLE_STATE_FIELDS:
        state = payload.get(field)
        if not isinstance(state, Mapping):
            raise _fatal(f"portable checkpoint is missing {field}")
        for name, value in sorted(state.items()):
            if not isinstance(value, Tensor):
                raise _fatal(f"portable tensor is not a Tensor: {field}.{name}")
            yield f"{field}.{name}", value


def _runtime_floating_tensors(model: nn.Module) -> Iterator[tuple[str, Tensor]]:
    yield from model.named_parameters()
    yield from (
        (name, value) for name, value in model.named_buffers()
        if torch.is_floating_point(value)
    )


def audit_candidate_deployment(
    *,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    runtime_model: nn.Module,
    source_checkpoint_sha256: str,
    portable_checkpoint_sha256: str,
) -> CandidateDeploymentAudit:
    if payload.get("schema_version") != PORTABLE_SCHEMA:
        raise _fatal("portable checkpoint schema is not the Kaggle compound contract")
    if manifest.get("storage_dtype") != "fp16":
        raise _fatal("manifest does not require FP16 storage")
    if manifest.get("runtime_dtype") != "fp32":
        raise _fatal("manifest does not require FP32 runtime")
    if manifest.get("rl_checkpoint_sha256") != source_checkpoint_sha256:
        raise _fatal("source checkpoint SHA-256 does not match the package manifest")
    if manifest.get("portable_checkpoint_sha256") != portable_checkpoint_sha256:
        raise _fatal("portable checkpoint SHA-256 does not match the package manifest")
    stored = [
        (name, value) for name, value in _payload_tensors(payload)
        if torch.is_floating_point(value)
    ]
    if not stored or any(value.dtype is not torch.float16 for _, value in stored):
        raise _fatal("every deployed floating tensor must use FP16 storage")
    runtime = list(_runtime_floating_tensors(runtime_model))
    if not runtime or any(value.dtype is not torch.float32 for _, value in runtime):
        raise _fatal("every materialized floating tensor must use FP32 runtime")
    checkpoint_update = manifest.get("checkpoint_update")
    if not isinstance(checkpoint_update, int) or checkpoint_update < 0:
        raise _fatal("checkpoint update identity is missing")
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and "focal_exact_deck_sha256" in metadata:
        focal_deck_sha256 = metadata["focal_exact_deck_sha256"]
        if (
            not isinstance(focal_deck_sha256, str)
            or len(focal_deck_sha256) != 64
            or manifest.get("deck_sha256") != focal_deck_sha256
        ):
            raise _fatal(
                "custom focal exact-deck identity differs between portable metadata "
                "and package manifest"
            )
    return CandidateDeploymentAudit(
        contract_id=CONTRACT_ID,
        source_checkpoint_sha256=source_checkpoint_sha256,
        portable_checkpoint_sha256=portable_checkpoint_sha256,
        effective_candidate_sha256=deployment_effective_sha256(payload, manifest),
        checkpoint_update=checkpoint_update,
    )


class KaggleEvaluationActorCritic(nn.Module):
    """Collector-compatible view of the strict-loaded Kaggle policy payload."""

    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.actor = policy.actor
        self.value_head = policy.value_head
        self.allocation_head = policy.allocation_head
        self.value_adapter = policy.value_adapter
        self.policy_strategy_adapter = policy.policy_strategy_adapter
        self.register_buffer(
            "default_own_archetype_id",
            torch.tensor(int(policy.metadata["own_archetype_id"]), dtype=torch.long),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def actor_summary(self, state: Any) -> Tensor:
        return state.summary

    @property
    def head(self):
        from .policy.actor_critic import DecoderPolicyHead
        return DecoderPolicyHead(self.actor.action_decoder, self.policy_strategy_adapter)

    def own_archetype_ids(self, batch_size: int, device: torch.device) -> Tensor:
        return self.default_own_archetype_id.to(device).expand(batch_size)

    def value_and_aux_from_encoded(
        self, validated: Any, state: Any, options: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        queries = self.value_head.decode(memory, memory_mask)
        z_value, z_meta = queries[:, 0], queries[:, 1]
        own_ids = self.own_archetype_ids(queries.shape[0], queries.device)
        value_query, value_delta = self.value_adapter(z_value, own_ids)
        meta_logits = self.value_head.heads.archetype(z_meta)
        logit = self.value_head.heads.value(value_query).squeeze(-1)
        return 2.0 * logit.sigmoid() - 1.0, {
            "z_value": z_value,
            "z_meta": z_meta,
            "meta_logits": meta_logits,
            "opponent_meta_logits": meta_logits,
            "value_adapter_delta": value_delta,
        }

    def strategy_context(self, validated: Any, value: Tensor, auxiliary: Mapping[str, Tensor]):
        from .policy.strategy_adapters import StrategyContext
        return StrategyContext.build(
            relative_first_player=validated.global_cat[:, 2],
            z_meta=auxiliary["z_meta"],
            meta_logits=auxiliary["meta_logits"],
            value=value,
            own_archetype_id=self.own_archetype_ids(value.shape[0], value.device),
        )

    def encode_with_strategy(self, features: Mapping[str, Tensor]):
        validated, state, options = self.actor.encode(features)
        value, auxiliary = self.value_and_aux_from_encoded(validated, state, options)
        context = self.strategy_context(validated, value, auxiliary)
        return validated, state, options, value, auxiliary, context


def require_kaggle_candidate_deployment(model: nn.Module) -> CandidateDeploymentAudit:
    audit = getattr(model, "_candidate_deployment_audit", None)
    if (
        not isinstance(audit, CandidateDeploymentAudit)
        or audit.status != "PASS"
        or audit.contract_id != CONTRACT_ID
        or audit.storage_dtype != "fp16"
        or audit.runtime_dtype != "fp32"
    ):
        raise _fatal(
            "formal Kaggle-facing evaluation received no passing "
            f"{CONTRACT_ID} audit"
        )
    return audit


def materialize_kaggle_evaluation_candidate(
    *,
    source: Path,
    checkpoint: Path,
    deck: Sequence[int],
    device: str | torch.device,
    temporary_root: Path,
    deck_id: str | None = None,
    deck_display_name: str | None = None,
    deck_source: str | None = None,
) -> tuple[KaggleEvaluationActorCritic, CandidateDeploymentAudit]:
    source = source.resolve()
    checkpoint = checkpoint.resolve()
    temporary_root = temporary_root.resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256_file(checkpoint)
    with tempfile.TemporaryDirectory(
        prefix="kaggle-candidate-", dir=temporary_root
    ) as directory:
        package = Path(directory) / "candidate"
        manifest = export_candidate(
            source=source,
            checkpoint=checkpoint,
            output=package,
            require_frozen_selection=False,
            deployment_deck=(deck if deck_id is not None else None),
            deployment_deck_id=deck_id,
            deployment_deck_display_name=deck_display_name,
            deployment_deck_source=deck_source,
        )
        portable_checkpoint = package / "strategy/model.bin"
        portable_sha256 = _sha256_file(portable_checkpoint)
        payload = torch.load(portable_checkpoint, map_location="cpu", weights_only=True)
        package_text = str(package)
        stale = [name for name in sys.modules if name == "strategy" or name.startswith("strategy.")]
        if stale:
            raise _fatal(f"stale Kaggle strategy modules are already loaded: {stale[:3]}")
        sys.path.insert(0, package_text)
        previous_threads = torch.get_num_threads()
        try:
            runtime = importlib.import_module("strategy.deployment.compound_inference")
            policy = runtime.PortableCompoundSemanticPolicy.from_checkpoint(
                portable_checkpoint, deck
            )
        finally:
            torch.set_num_threads(previous_threads)
            sys.path.remove(package_text)
            for name in tuple(sys.modules):
                if name == "strategy" or name.startswith("strategy."):
                    sys.modules.pop(name, None)
        model = KaggleEvaluationActorCritic(policy).to(
            device=device, dtype=torch.float32
        ).eval().requires_grad_(False)
        audit = audit_candidate_deployment(
            manifest=manifest,
            payload=payload,
            runtime_model=model,
            source_checkpoint_sha256=source_sha256,
            portable_checkpoint_sha256=portable_sha256,
        )
    model._candidate_deployment_audit = audit
    return model, audit


__all__ = [
    "AUDIT_SCHEMA",
    "CONTRACT_ID",
    "CandidateDeploymentAudit",
    "CandidateDeploymentIdentityViolation",
    "KaggleEvaluationActorCritic",
    "audit_candidate_deployment",
    "materialize_kaggle_evaluation_candidate",
    "require_kaggle_candidate_deployment",
]
