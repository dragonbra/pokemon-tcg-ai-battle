"""Kaggle-equivalent candidate materialization for formal strength evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
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
AUDIT_SCHEMA = "0040_candidate_deployment_identity_audit_v1"
PORTABLE_SCHEMA = "0038_compound_kaggle_candidate_v4"


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
        self.opponent_meta_head = policy.meta_head
        self.opponent_meta_conditioner = policy.meta_conditioner

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def actor_summary(self, state: Any) -> Tensor:
        logits = self.opponent_meta_head(state.summary)
        return self.opponent_meta_conditioner(state.summary, logits)

    def value_and_aux_from_encoded(
        self, validated: Any, state: Any, options: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        queries = self.value_head.decode(memory, memory_mask)
        meta_logits = self.opponent_meta_head(state.summary)
        conditioned = self.opponent_meta_conditioner(state.summary, meta_logits)
        value_query = queries[:, 0] + (conditioned - state.summary)
        logit = self.value_head.heads.value(value_query).squeeze(-1)
        return 2.0 * logit.sigmoid() - 1.0, {
            "opponent_meta_logits": meta_logits
        }


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
        )
        model, audit = load_kaggle_evaluation_candidate(
            package=package, checkpoint=checkpoint, deck=deck, device=device
        )
    return model, audit


def load_kaggle_evaluation_candidate(
    *,
    package: Path,
    checkpoint: Path,
    deck: Sequence[int],
    device: str | torch.device,
) -> tuple[KaggleEvaluationActorCritic, CandidateDeploymentAudit]:
    """Strict-load and audit a persistent Kaggle-equivalent candidate package."""

    package = package.resolve()
    checkpoint = checkpoint.resolve()
    manifest_path = package / "manifest.json"
    portable_checkpoint = package / "strategy/model.bin"
    if not manifest_path.is_file() or not portable_checkpoint.is_file():
        raise _fatal("persistent candidate package is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        source_checkpoint_sha256=_sha256_file(checkpoint),
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
    "load_kaggle_evaluation_candidate",
    "materialize_kaggle_evaluation_candidate",
    "require_kaggle_candidate_deployment",
]
