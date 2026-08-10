"""Immutable policy registry, materialization, and fail-closed identity audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .semantic_policy.deployment.inference import PortableSemanticPolicy
from .policy.adaptation import AdaptationConfig, apply_focal_adaptation
from .checkpoint import validate_model_only_payload


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).with_name("policy_registry.json")
REGISTRY_SCHEMA = "0040_policy_identity_registry_v1"
AUDIT_SCHEMA = "rl_policy_identity_audit_v1"
EFFECTIVE_COMPONENTS = (
    "prototype_encoder",
    "state_encoder",
    "option_input_encoder",
    "option_transformer_layer_0",
    "option_transformer_layer_1",
    "final_norm",
    "lora",
    "action_decoder",
)
_HASH_FIELDS = (
    "policy_id",
    "policy_kind",
    "parent_policy_id",
    "model_schema_version",
    "observation_schema_version",
    "action_schema_version",
    "base_checkpoint",
    "trained_checkpoint",
    "components",
    "created_from_update",
)


class PolicyIdentityViolation(RuntimeError):
    """A requested policy does not match its materialized inference function."""


@dataclass(frozen=True, slots=True)
class PolicyIdentityAudit:
    requested_policy_id: str
    policy_kind: str
    effective_policy_sha256: str
    checkpoint_sha256: str
    components: dict[str, dict[str, str]]
    model_schema_version: str
    observation_schema_version: str
    action_schema_version: str
    purpose: str
    status: str = "PASS"
    schema_version: str = AUDIT_SCHEMA

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResolvedPolicy:
    policy_id: str
    model: nn.Module
    audit: PolicyIdentityAudit
    checkpoint_path: Path


def _fatal(policy_id: str, detail: str) -> PolicyIdentityViolation:
    return PolicyIdentityViolation(
        "FATAL: Opponent policy identity violation. "
        f"Requested: {policy_id}. {detail}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.view(torch.uint8).numpy().tobytes()


def _tensor_inventory_sha256(rows: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(rows.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _component_for_tensor(name: str) -> str:
    if name.startswith("prototype_encoder."):
        return "prototype_encoder"
    if name.startswith("state_encoder."):
        return "state_encoder"
    if name.startswith("action_decoder."):
        return "action_decoder"
    layer_prefix = "option_encoder.cross_attention_transformer.layers."
    if name.startswith(f"{layer_prefix}0."):
        return "option_transformer_layer_0"
    if name.startswith(f"{layer_prefix}1."):
        return "option_transformer_layer_1"
    if name.startswith("option_encoder.cross_attention_transformer.norm."):
        return "final_norm"
    if name.startswith("option_encoder."):
        return "option_input_encoder"
    raise PolicyIdentityViolation(f"unclassified effective policy tensor: {name}")


def component_hashes(state_dict: Mapping[str, Tensor]) -> dict[str, str]:
    rows: dict[str, dict[str, Tensor]] = {
        name: {} for name in EFFECTIVE_COMPONENTS
    }
    for name, value in state_dict.items():
        component = _component_for_tensor(name)
        rows[component][name] = value
        if ".parametrizations." in name:
            rows["lora"][name] = value
    missing = [
        name for name in EFFECTIVE_COMPONENTS
        if name != "lora" and not rows[name]
    ]
    if missing:
        raise PolicyIdentityViolation(
            f"effective policy state is missing components: {missing}"
        )
    return {name: _tensor_inventory_sha256(rows[name]) for name in EFFECTIVE_COMPONENTS}


def _effective_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {name: manifest.get(name) for name in _HASH_FIELDS}


def effective_policy_sha256(manifest: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(_effective_payload(manifest))


def validate_policy_manifest(manifest: Mapping[str, Any]) -> None:
    policy_id = str(manifest.get("policy_id") or "<missing>")
    if not policy_id.startswith(("Policy-", "Experimental-Hybrid-")):
        raise _fatal(policy_id, "policy_id is not registered under an allowed namespace")
    components = manifest.get("components")
    if not isinstance(components, Mapping) or set(components) != set(EFFECTIVE_COMPONENTS):
        raise _fatal(policy_id, "effective component inventory is incomplete")
    for name in EFFECTIVE_COMPONENTS:
        component = components[name]
        if (
            not isinstance(component, Mapping)
            or not isinstance(component.get("source_policy_id"), str)
            or len(str(component.get("effective_sha256", ""))) != 64
        ):
            raise _fatal(policy_id, f"invalid component identity: {name}")
    for field in (
        "model_schema_version", "observation_schema_version", "action_schema_version"
    ):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise _fatal(policy_id, f"missing schema identity: {field}")
    base = manifest.get("base_checkpoint")
    if (
        not isinstance(base, Mapping)
        or not isinstance(base.get("path"), str)
        or len(str(base.get("sha256", ""))) != 64
    ):
        raise _fatal(policy_id, "base checkpoint identity is incomplete")
    actual = effective_policy_sha256(manifest)
    if manifest.get("effective_policy_sha256") != actual:
        raise _fatal(
            policy_id,
            "manifest effective hash does not match its immutable composition; "
            f"declared={manifest.get('effective_policy_sha256')} actual={actual}",
        )


def load_policy_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != REGISTRY_SCHEMA:
        raise PolicyIdentityViolation("unsupported policy identity registry schema")
    policies = payload.get("policies")
    if not isinstance(policies, list) or not policies:
        raise PolicyIdentityViolation("policy identity registry is empty")
    registry: dict[str, dict[str, Any]] = {}
    for raw in policies:
        if not isinstance(raw, dict):
            raise PolicyIdentityViolation("policy registry entry is not an object")
        validate_policy_manifest(raw)
        policy_id = raw["policy_id"]
        if policy_id in registry:
            raise PolicyIdentityViolation(f"duplicate registered policy: {policy_id}")
        registry[policy_id] = raw
    return registry


def _entry(policy_id: str, registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    try:
        return load_policy_registry(registry_path)[policy_id]
    except KeyError as error:
        raise _fatal(policy_id, "policy_id is not present in the immutable registry") from error


def checkpoint_state_dict(
    policy_id: str, *, registry_path: Path = REGISTRY_PATH
) -> dict[str, Tensor]:
    entry = _entry(policy_id, registry_path)
    path = ROOT / entry["base_checkpoint"]["path"]
    if not path.is_file():
        raise _fatal(policy_id, f"checkpoint is missing: {path}")
    digest = _sha256_file(path)
    if digest != entry["base_checkpoint"]["sha256"]:
        raise _fatal(
            policy_id,
            f"checkpoint SHA-256 mismatch; expected={entry['base_checkpoint']['sha256']} "
            f"actual={digest}",
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise _fatal(policy_id, "checkpoint has no state_dict")
    return state


def audit_materialized_state_dict(
    policy_id: str,
    state_dict: Mapping[str, Tensor],
    *,
    purpose: str = "preflight",
    registry_path: Path = REGISTRY_PATH,
) -> PolicyIdentityAudit:
    entry = _entry(policy_id, registry_path)
    actual_hashes = component_hashes(state_dict)
    expected = {
        name: str(entry["components"][name]["effective_sha256"])
        for name in EFFECTIVE_COMPONENTS
    }
    mismatches = {
        name: {"expected": expected[name], "actual": actual_hashes[name]}
        for name in EFFECTIVE_COMPONENTS
        if actual_hashes[name] != expected[name]
    }
    if mismatches:
        raise _fatal(policy_id, f"materialized component hash mismatch: {mismatches}")
    components = {
        name: {
            "source_policy_id": str(entry["components"][name]["source_policy_id"]),
            "effective_sha256": actual_hashes[name],
        }
        for name in EFFECTIVE_COMPONENTS
    }
    return PolicyIdentityAudit(
        requested_policy_id=policy_id,
        policy_kind=str(entry["policy_kind"]),
        effective_policy_sha256=str(entry["effective_policy_sha256"]),
        checkpoint_sha256=str(entry["base_checkpoint"]["sha256"]),
        components=components,
        model_schema_version=str(entry["model_schema_version"]),
        observation_schema_version=str(entry["observation_schema_version"]),
        action_schema_version=str(entry["action_schema_version"]),
        purpose=purpose,
    )


def audit_checkpoint(
    policy_id: str,
    *,
    focal_policy_id: str | None = None,
    purpose: str = "checkpoint_preflight",
    registry_path: Path = REGISTRY_PATH,
) -> PolicyIdentityAudit:
    del focal_policy_id  # Focal identity is deliberately irrelevant to opponent resolution.
    return audit_materialized_state_dict(
        policy_id,
        checkpoint_state_dict(policy_id, registry_path=registry_path),
        purpose=purpose,
        registry_path=registry_path,
    )


def resolve_policy_identity(
    policy_id: str,
    *,
    purpose: str,
    registry_path: Path = REGISTRY_PATH,
) -> PolicyIdentityAudit:
    return audit_checkpoint(
        policy_id, purpose=purpose, registry_path=registry_path
    )


def materialize_promoted_actor(
    parent_policy_id: str,
    trained_checkpoint_path: Path,
    deck: Sequence[int],
    device: str | torch.device,
    *,
    registry_path: Path = REGISTRY_PATH,
) -> nn.Module:
    parent = _entry(parent_policy_id, registry_path)
    if parent["policy_kind"] != "immutable_pretrained":
        raise _fatal(parent_policy_id, "promoted snapshot parent must be immutable pretrained")
    base_checkpoint = ROOT / parent["base_checkpoint"]["path"]
    policy = PortableSemanticPolicy.from_checkpoint(base_checkpoint, deck)
    model = policy.model.to(device).eval()
    payload = torch.load(
        Path(trained_checkpoint_path), map_location="cpu", weights_only=True
    )
    validate_model_only_payload(payload)
    adaptation_payload = payload.get("adaptation") or {}
    allowed = {"lora", "layernorm_tuning", "rank", "alpha", "option_block"}
    adaptation = AdaptationConfig(**{
        name: adaptation_payload[name]
        for name in allowed if name in adaptation_payload
    })
    apply_focal_adaptation(model, adaptation)
    current = model.state_dict()
    actor_delta = {
        name.removeprefix("actor."): value
        for name, value in (payload.get("state_dict") or {}).items()
        if name.startswith("actor.")
    }
    unknown = sorted(set(actor_delta) - set(current))
    if unknown:
        raise _fatal(
            parent_policy_id,
            f"promoted actor delta has unknown tensors: {unknown[:5]}",
        )
    with torch.no_grad():
        for name, value in actor_delta.items():
            current[name].copy_(value.to(current[name].device, current[name].dtype))
    return model.eval()


def materialize_policy(
    policy_id: str,
    deck: Sequence[int],
    device: str | torch.device,
    *,
    purpose: str,
    registry_path: Path = REGISTRY_PATH,
) -> ResolvedPolicy:
    entry = _entry(policy_id, registry_path)
    checkpoint = ROOT / entry["base_checkpoint"]["path"]
    if entry["policy_kind"] == "promoted_snapshot":
        trained = entry.get("trained_checkpoint")
        if not isinstance(trained, Mapping):
            raise _fatal(policy_id, "promoted snapshot has no trained checkpoint")
        trained_path = ROOT / str(trained.get("path", ""))
        if not trained_path.is_file():
            raise _fatal(policy_id, f"promoted checkpoint is missing: {trained_path}")
        trained_digest = _sha256_file(trained_path)
        if trained_digest != trained.get("sha256"):
            raise _fatal(policy_id, "promoted checkpoint SHA-256 mismatch")
        model = materialize_promoted_actor(
            str(entry["parent_policy_id"]), trained_path, deck, device,
            registry_path=registry_path,
        )
    elif entry["policy_kind"] != "immutable_pretrained":
        raise _fatal(policy_id, "unsupported registered policy kind")
    else:
        policy = PortableSemanticPolicy.from_checkpoint(checkpoint, deck)
        model = policy.model.to(device).eval()
    model.requires_grad_(False)
    audit = audit_materialized_state_dict(
        policy_id,
        model.state_dict(),
        purpose=purpose,
        registry_path=registry_path,
    )
    return ResolvedPolicy(policy_id, model, audit, checkpoint)


def build_promoted_snapshot_manifest(
    *,
    policy_id: str,
    parent_policy_id: str,
    base_checkpoint_sha256: str,
    trained_checkpoint_sha256: str,
    trained_checkpoint_path: str = "registry-managed-promoted-checkpoint",
    components: Mapping[str, Mapping[str, str]],
    model_schema_version: str,
    observation_schema_version: str,
    action_schema_version: str,
    created_from_update: int,
) -> dict[str, Any]:
    parent = _entry(parent_policy_id)
    if parent["base_checkpoint"]["sha256"] != base_checkpoint_sha256:
        raise _fatal(policy_id, "promoted snapshot base checkpoint differs from parent")
    manifest: dict[str, Any] = {
        "policy_id": policy_id,
        "policy_kind": "promoted_snapshot",
        "parent_policy_id": parent_policy_id,
        "model_schema_version": model_schema_version,
        "observation_schema_version": observation_schema_version,
        "action_schema_version": action_schema_version,
        "base_checkpoint": dict(parent["base_checkpoint"]),
        "trained_checkpoint": {
            "path": trained_checkpoint_path,
            "sha256": trained_checkpoint_sha256,
        },
        "components": {name: dict(components[name]) for name in EFFECTIVE_COMPONENTS},
        "created_from_update": int(created_from_update),
    }
    manifest["effective_policy_sha256"] = effective_policy_sha256(manifest)
    validate_policy_manifest(manifest)
    return manifest


__all__ = [
    "AUDIT_SCHEMA",
    "EFFECTIVE_COMPONENTS",
    "PolicyIdentityAudit",
    "PolicyIdentityViolation",
    "REGISTRY_SCHEMA",
    "ResolvedPolicy",
    "audit_checkpoint",
    "audit_materialized_state_dict",
    "build_promoted_snapshot_manifest",
    "checkpoint_state_dict",
    "component_hashes",
    "effective_policy_sha256",
    "load_policy_registry",
    "materialize_promoted_actor",
    "materialize_policy",
    "resolve_policy_identity",
    "validate_policy_manifest",
]
