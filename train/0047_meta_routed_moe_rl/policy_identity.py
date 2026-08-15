"""Project-local full-tensor policy identity and isolation gates.

This module deliberately materializes tensor bundles without importing executable
code from an older numbered project. Model construction/forward is a later gate;
these audits establish immutable effective content, dtype and storage isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor

from .assets import AssetIntegrityError, AssetRegistry, sha256_file


EFFECTIVE_0809_COMPONENTS = (
    "prototype_encoder", "state_encoder", "option_input_encoder",
    "option_transformer_layer_0", "option_transformer_layer_1", "final_norm",
    "lora", "action_decoder",
)
PORTABLE_FIELDS = (
    "actor_state_dict", "value_head_state_dict", "allocation_head_state_dict",
    "value_adapter_state_dict", "policy_strategy_adapter_state_dict",
)
COMPLETE_PORTABLE_FIELDS = PORTABLE_FIELDS + (
    "policy_option_lora_state_dict",
    "meta_actor_residual_state_dict",
)


class PolicyIdentityViolation(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PolicyBundleAudit:
    requested_policy_id: str
    artifact_sha256: dict[str, str]
    tensor_count: int
    component_sha256: dict[str, str]
    storage_dtype: str
    runtime_dtype: str
    effective_policy_sha256: str
    purpose: str
    status: str = "PASS"
    schema_version: str = "0044_policy_bundle_audit_v1"


@dataclass(slots=True)
class MaterializedPolicyBundle:
    policy_id: str
    tensors: dict[str, Tensor]
    audit: PolicyBundleAudit

    def clone_to(self, device: str | torch.device, *, runtime_dtype: torch.dtype) -> "MaterializedPolicyBundle":
        copied = {
            name: value.detach().clone().to(
                device=device,
                dtype=runtime_dtype if torch.is_floating_point(value) else value.dtype,
            )
            for name, value in self.tensors.items()
        }
        return MaterializedPolicyBundle(
            self.policy_id,
            copied,
            PolicyBundleAudit(
                **{
                    **asdict(self.audit),
                    "runtime_dtype": str(runtime_dtype).removeprefix("torch."),
                    "purpose": self.audit.purpose + ":clone_to",
                }
            ),
        )


def _tensor_hash(rows: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(rows.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _component(name: str) -> str:
    if name.startswith("prototype_encoder."):
        return "prototype_encoder"
    if name.startswith("state_encoder."):
        return "state_encoder"
    if name.startswith("action_decoder."):
        return "action_decoder"
    prefix = "option_encoder.cross_attention_transformer.layers."
    if name.startswith(prefix + "0."):
        return "option_transformer_layer_0"
    if name.startswith(prefix + "1."):
        return "option_transformer_layer_1"
    if name.startswith("option_encoder.cross_attention_transformer.norm."):
        return "final_norm"
    if name.startswith("option_encoder."):
        return "option_input_encoder"
    raise PolicyIdentityViolation(f"unclassified Policy-0809 tensor: {name}")


def _0809_components(state: Mapping[str, Tensor]) -> dict[str, str]:
    rows: dict[str, dict[str, Tensor]] = {name: {} for name in EFFECTIVE_0809_COMPONENTS}
    for name, tensor in state.items():
        component = _component(name)
        rows[component][name] = tensor
        if ".parametrizations." in name:
            rows["lora"][name] = tensor
    if any(not rows[name] for name in EFFECTIVE_0809_COMPONENTS if name != "lora"):
        raise PolicyIdentityViolation("Policy-0809 component inventory is incomplete")
    return {name: _tensor_hash(rows[name]) for name in EFFECTIVE_0809_COMPONENTS}


def _policy(registry: AssetRegistry, policy_id: str):
    try:
        return next(policy for policy in registry.policies if policy.policy_id == policy_id)
    except StopIteration as error:
        raise PolicyIdentityViolation(f"unregistered policy: {policy_id}") from error


def _artifact(policy, purpose: str, project_root: Path) -> Path:
    try:
        identity = next(row for row in policy.artifacts if row.purpose == purpose)
    except StopIteration as error:
        raise PolicyIdentityViolation(f"{policy.policy_id} missing artifact: {purpose}") from error
    path = project_root / identity.path
    if sha256_file(path) != identity.sha256:
        raise PolicyIdentityViolation(f"{policy.policy_id} artifact hash mismatch: {purpose}")
    return path


def materialize_policy_bundle(
    project_root: Path, policy_id: str, *, purpose: str = "preflight",
) -> MaterializedPolicyBundle:
    registry = AssetRegistry.load(project_root)
    registry.validate_all()
    policy = _policy(registry, policy_id)
    manifest = json.loads((project_root / policy.manifest_path).read_text())
    artifacts = {row.purpose: row.sha256 for row in policy.artifacts}
    if policy_id in {"Policy-0809", "Policy-0814"}:
        payload = torch.load(
            _artifact(policy, "complete_base_checkpoint", project_root),
            map_location="cpu", weights_only=True,
        )
        if payload.get("schema_version") != "0031_model_only_checkpoint_v1":
            raise PolicyIdentityViolation("Policy-0809 checkpoint schema mismatch")
        state = payload.get("state_dict")
        if not isinstance(state, dict) or len(state) != 293:
            raise PolicyIdentityViolation("Policy-0809 full tensor inventory mismatch")
        components = _0809_components(state)
        if policy_id == "Policy-0809":
            declared = {
                name: manifest["components"][name]["effective_sha256"]
                for name in EFFECTIVE_0809_COMPONENTS
            }
            if components != declared:
                raise PolicyIdentityViolation("Policy-0809 effective component hashes mismatch")
        tensors = {name: value.detach().clone() for name, value in state.items()}
        storage_dtype = "fp32"
    elif policy_id.startswith("Champion-G") and policy.generation is not None:
        delta_purpose = manifest.get("source_checkpoint_artifact_purpose", "model_only_delta")
        delta = torch.load(
            _artifact(policy, delta_purpose, project_root),
            map_location="cpu", weights_only=True,
        )
        portable = torch.load(
            _artifact(policy, "portable_fp16_artifact", project_root),
            map_location="cpu", weights_only=True,
        )
        expected_delta_schema = manifest.get(
            "source_checkpoint_schema", "0042_strategy_conditioned_model_only_v1"
        )
        expected_portable_schema = manifest.get(
            "portable_checkpoint_schema", "0042_strategy_conditioned_kaggle_candidate_v1"
        )
        if (
            delta.get("schema_version") != expected_delta_schema
            or portable.get("schema_version") != expected_portable_schema
            or portable.get("metadata", {}).get("checkpoint_sha256")
            != artifacts[delta_purpose]
        ):
            raise PolicyIdentityViolation(f"{policy_id} reconstruction provenance mismatch")
        base_purpose = manifest.get("source_base_artifact_purpose", "complete_base_checkpoint")
        if base_purpose and (
            delta.get("metadata", {}).get("source_actor_sha256") != artifacts[base_purpose]
        ):
            raise PolicyIdentityViolation(f"{policy_id} base provenance mismatch")
        tensors = {}
        components = {}
        portable_fields = tuple(manifest.get("portable_fields", PORTABLE_FIELDS))
        if portable_fields not in {PORTABLE_FIELDS, COMPLETE_PORTABLE_FIELDS}:
            raise PolicyIdentityViolation(f"{policy_id} portable field contract mismatch")
        for field in portable_fields:
            state = portable.get(field)
            if not isinstance(state, dict) or not state:
                raise PolicyIdentityViolation(f"{policy_id} missing portable field: {field}")
            floating = [value for value in state.values() if torch.is_floating_point(value)]
            if not floating or any(value.dtype != torch.float16 for value in floating):
                raise PolicyIdentityViolation(f"{policy_id} portable floating tensors must be FP16")
            components[field] = _tensor_hash(state)
            tensors.update({f"{field}.{name}": value.detach().clone() for name, value in state.items()})
        storage_dtype = "fp16"
        declared_tensor_hash = manifest.get("portable_tensor_sha256")
        if declared_tensor_hash is not None and _tensor_hash(tensors) != declared_tensor_hash:
            raise PolicyIdentityViolation(f"{policy_id} portable tensor identity mismatch")
    else:
        raise PolicyIdentityViolation(f"unsupported 0044 policy: {policy_id}")
    audit = PolicyBundleAudit(
        requested_policy_id=policy_id,
        artifact_sha256=artifacts,
        tensor_count=len(tensors),
        component_sha256=components,
        storage_dtype=storage_dtype,
        runtime_dtype=manifest["inference_semantics"]["runtime_dtype"],
        effective_policy_sha256=policy.effective_policy_sha256,
        purpose=purpose,
    )
    return MaterializedPolicyBundle(policy_id, tensors, audit)


def assert_storage_isolation(*bundles: MaterializedPolicyBundle) -> None:
    seen: dict[int, tuple[str, str]] = {}
    for bundle in bundles:
        for name, tensor in bundle.tensors.items():
            pointer = tensor.untyped_storage().data_ptr()
            if pointer in seen:
                owner = seen[pointer]
                raise PolicyIdentityViolation(
                    f"policy tensor storage alias: {owner} and {(bundle.policy_id, name)}"
                )
            seen[pointer] = (bundle.policy_id, name)


__all__ = [
    "COMPLETE_PORTABLE_FIELDS",
    "MaterializedPolicyBundle", "PolicyBundleAudit", "PolicyIdentityViolation",
    "assert_storage_isolation", "materialize_policy_bundle",
]
