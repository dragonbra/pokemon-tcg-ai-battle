"""Generate the immutable, fail-closed 0038 submission release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *, inventory: Mapping[str, Any], gates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    checkpoint = inventory["checkpoint"]
    package = inventory["package"]
    runtime = inventory["runtime"]
    contracts = inventory["contracts"]
    gate_status = {name: gate.get("status") for name, gate in gates.items()}
    blockers: list[str] = []
    if checkpoint["critical_missing_keys"]:
        blockers.append("critical checkpoint keys are missing")
    if checkpoint["critical_unexpected_keys"]:
        blockers.append("critical checkpoint keys are unexpected")
    if not package["strict_load"]:
        blockers.append("portable model is not strict-load compatible")
    if package["silent_legacy_fallback"]:
        blockers.append("package contains silent macro-to-policy fallback")
    if not package["model_eval"]:
        blockers.append("package does not set eval mode")
    if int(package["checkpoint_update"]) != int(checkpoint["update"]):
        blockers.append("package/checkpoint update mismatch")
    version_pairs = {
        "action boundary": (package["action_boundary"], contracts["action_boundary"]),
        "DecisionGate": (package["decision_gate"], contracts["decision_gate"]),
        "canonicalizer": (package["canonicalizer"], contracts["canonicalizer"]),
        "official protocol adapter": (
            package["official_protocol_adapter"],
            contracts["official_protocol_adapter"],
        ),
    }
    for label, (deployed, expected) in version_pairs.items():
        if deployed != expected:
            blockers.append(f"package/checkpoint {label} version mismatch")
    for name in ("A", "B", "C", "D"):
        if gate_status.get(name) != "PASS":
            blockers.append(f"Gate {name} is {gate_status.get(name, 'MISSING')}")
    if inventory.get("git_dirty"):
        blockers.append("diagnostic source tree is dirty; no immutable release commit")

    result = {
        "schema_version": "0038_submission_release_manifest_v1",
        "release_ready": not blockers,
        "release_blockers": blockers,
        "checkpoint": {
            "id": f"update-{int(checkpoint['update']):06d}",
            "sha256": checkpoint["sha256"],
            "model_schema_version": checkpoint["schema_version"],
            "base_model_sha256": checkpoint["source_actor_sha256"],
            "base_encoder_component_sha256": checkpoint[
                "component_sha256"
            ]["base_encoder"],
            "lora_sha256": checkpoint["component_sha256"]["option_encoder_lora"],
            "action_decoder_sha256": checkpoint["component_sha256"]["action_decoder"],
            "value_sha256": checkpoint["component_sha256"]["value"],
            "allocation_head_sha256": checkpoint["component_sha256"][
                "allocation_head"
            ],
            "critical_missing_keys": checkpoint["critical_missing_keys"],
            "critical_unexpected_keys": checkpoint["critical_unexpected_keys"],
        },
        "package": {
            "root": package["root"],
            "model_sha256": package["model_sha256"],
            "manifest_sha256": package["manifest_sha256"],
            "component_sha256": package["component_sha256"],
            "portable_schema": package["portable_schema"],
            "checkpoint_update": package["checkpoint_update"],
            "strict_load": package["strict_load"],
            "critic_deployed": package["critic_deployed"],
            "value_runtime_role": (
                "training/evaluation critic only; excluded from actor action path"
            ),
            "silent_legacy_fallback": package["silent_legacy_fallback"],
        },
        "source": {
            "git_commit": inventory["git_commit"],
            "git_dirty": inventory["git_dirty"],
        },
        "contracts": {
            "action_boundary_schema_version": contracts["action_boundary"],
            "decision_gate_version": contracts["decision_gate"],
            "canonical_allocation_version": contracts["canonicalizer"],
            "trajectory_schema_version": contracts["trajectory"],
            "official_protocol_adapter_version": contracts[
                "official_protocol_adapter"
            ],
            "feature_preprocessing_version": (
                "0031_rule_faithful_semantic_decision_v2"
            ),
            "card_database_sha256": runtime["card_database_sha256"],
            "cuda_rules_sha256": runtime["cuda_rules_sha256"],
            "cuda_extension_sha256": runtime["cuda_extension_sha256"],
            "official_rule_engine_sha256": runtime[
                "official_cpu_library_sha256"
            ],
        },
        "decision_gate_config": {
            "forced_shortcut": True,
            "legal_empty_pass": "explicit_only",
            "mask_error": "fail_closed",
        },
        "precision_backend": {
            "package_storage": package["storage_dtype"],
            "package_runtime": package["runtime_dtype"],
            "cuda_training": "fp32",
            "cuda_state_backend": "semantic0031_v2_lanes",
        },
        "inference": {
            "greedy": True,
            "sampling": False,
            "dropout": False,
            "model_eval": package["model_eval"],
        },
        "gates": gate_status,
    }
    validate_release_manifest(result)
    return result


def validate_release_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on a missing/malformed release-manifest field.

    This validator is suitable for package startup once a candidate has four
    passing Gates.  A blocked diagnostic manifest is still structurally valid;
    it simply has ``release_ready=false`` and a non-empty blocker list.
    """

    required_top = {
        "schema_version", "release_ready", "release_blockers", "checkpoint",
        "package", "source", "contracts", "decision_gate_config",
        "precision_backend", "inference", "gates",
    }
    missing = sorted(required_top.difference(manifest))
    if missing:
        raise ValueError(f"release manifest missing fields: {missing}")
    if manifest["schema_version"] != "0038_submission_release_manifest_v1":
        raise ValueError("release manifest schema mismatch")
    checkpoint = manifest["checkpoint"]
    package = manifest["package"]
    hash_fields = (
        checkpoint["sha256"], checkpoint["base_model_sha256"],
        checkpoint["base_encoder_component_sha256"], checkpoint["lora_sha256"],
        checkpoint["action_decoder_sha256"], checkpoint["value_sha256"],
        checkpoint["allocation_head_sha256"], package["model_sha256"],
        package["manifest_sha256"],
    )
    if any(re.fullmatch(r"[0-9a-f]{64}", value or "") is None for value in hash_fields):
        raise ValueError("release manifest contains a malformed component hash")
    gates = manifest["gates"]
    if set(gates) != set("ABCD"):
        raise ValueError("release manifest gate inventory mismatch")
    ready = bool(manifest["release_ready"])
    blockers = list(manifest["release_blockers"])
    if ready and (
        blockers
        or any(gates[name] != "PASS" for name in "ABCD")
        or package["silent_legacy_fallback"]
        or not package["strict_load"]
        or manifest["source"]["git_dirty"]
    ):
        raise ValueError("release_ready contradicts fail-closed release conditions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    for gate in "abcd":
        parser.add_argument(f"--gate-{gate}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = {
        "inventory": args.inventory.resolve(),
        **{name.upper(): getattr(args, f"gate_{name}").resolve() for name in "abcd"},
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in inputs.items()
    }
    result = build_manifest(
        inventory=payloads.pop("inventory"), gates=payloads
    )
    result["evidence"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in inputs.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_manifest", "validate_release_manifest"]
