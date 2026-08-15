"""Verify the complete immutable Policy-0814 actor and paired Value release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import Tensor

sys.dont_write_bytecode = True

from loader import (
    ARCHIVE_ROOT,
    POLICY_ID,
    POLICY_SHA256,
    VALUE_SHA256,
    load_policy,
    load_policy_checkpoint,
    load_value_checkpoint,
    load_value_network,
)
from model_source.contracts.fields import WIDTHS


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
POLICY_HASH_FIELDS = (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _tensor_inventory_sha256(rows: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(rows.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
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
    raise ValueError(f"unclassified Policy-0814 tensor: {name}")


def _component_hashes(state_dict: Mapping[str, Tensor]) -> dict[str, str]:
    rows: dict[str, dict[str, Tensor]] = {name: {} for name in EFFECTIVE_COMPONENTS}
    for name, value in state_dict.items():
        rows[_component_for_tensor(name)][name] = value
        if ".parametrizations." in name:
            rows["lora"][name] = value
    missing = [name for name in EFFECTIVE_COMPONENTS if name != "lora" and not rows[name]]
    if missing:
        raise ValueError(f"Policy-0814 is missing effective components: {missing}")
    return {name: _tensor_inventory_sha256(rows[name]) for name in EFFECTIVE_COMPONENTS}


def _effective_policy_sha256(identity: Mapping[str, Any]) -> str:
    payload = {name: identity.get(name) for name in POLICY_HASH_FIELDS}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _structural_sha256(state_dict: Mapping[str, Tensor]) -> str:
    rows = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(state_dict.items())
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _reject_resume_state(payload: dict[str, Any]) -> None:
    forbidden = {
        "optimizer",
        "optimizer_state_dict",
        "scheduler",
        "scheduler_state_dict",
        "scaler",
        "grad_scaler_state_dict",
        "rng",
        "rng_state",
        "rollout",
        "rollout_buffer",
        "replay",
        "replay_buffer",
    }
    if forbidden & set(payload) or forbidden & set(payload.get("metadata", {})):
        raise ValueError("Policy-0814 release contains resumable training state")


def _smoke_batch() -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        "global_cat": torch.zeros(1, WIDTHS.global_cat, dtype=torch.long),
        "global_num": torch.zeros(1, WIDTHS.global_num),
        "global_state": torch.ones(1, WIDTHS.global_state, dtype=torch.long),
        "min_count": torch.zeros(1, dtype=torch.long),
        "max_count": torch.ones(1, dtype=torch.long),
        "targets": torch.ones(1, 1, dtype=torch.long),
    }
    for prefix, cat_width, num_width, valid in (
        ("card", WIDTHS.card_cat, WIDTHS.card_num, False),
        ("resource", WIDTHS.resource_cat, WIDTHS.resource_num, False),
        ("event", WIDTHS.event_cat, WIDTHS.event_num, False),
        ("option", WIDTHS.option_cat, WIDTHS.option_num, True),
    ):
        batch[f"{prefix}_cat"] = torch.zeros(1, 1, cat_width, dtype=torch.long)
        batch[f"{prefix}_num"] = torch.zeros(1, 1, num_width)
        batch[f"{prefix}_mask"] = torch.full((1, 1), valid, dtype=torch.bool)
        if prefix != "option":
            batch[f"{prefix}_state"] = torch.ones(
                1, 1, getattr(WIDTHS, f"{prefix}_state"), dtype=torch.long
            )
    batch["card_parent"] = torch.zeros(1, 1, dtype=torch.long)
    for relation in ("source", "target", "before", "after"):
        batch[f"event_{relation}"] = torch.zeros(1, 1, dtype=torch.long)
    batch["option_state"] = torch.zeros(1, 1, WIDTHS.option_state, dtype=torch.long)
    batch["option_source"] = torch.zeros(1, 1, dtype=torch.long)
    batch["option_target"] = torch.zeros(1, 1, dtype=torch.long)
    batch["option_context"] = torch.zeros(1, 1, dtype=torch.long)
    batch["option_effect_card"] = torch.zeros(1, 1, dtype=torch.long)
    for prefix in ("option_skill", "option_effect"):
        batch[f"{prefix}_id"] = torch.zeros(1, 1, dtype=torch.long)
        batch[f"{prefix}_role"] = torch.zeros(1, 1, dtype=torch.long)
        batch[f"{prefix}_parent"] = torch.zeros(1, 1, dtype=torch.long)
        batch[f"{prefix}_mask"] = torch.zeros(1, 1, dtype=torch.bool)
    return batch


def main() -> None:
    manifest = json.loads((ARCHIVE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    policy_payload = load_policy_checkpoint()
    value_payload = load_value_checkpoint()
    _reject_resume_state(policy_payload)
    _reject_resume_state(value_payload)
    if value_payload["metadata"]["source_checkpoint_sha256"] != POLICY_SHA256:
        raise ValueError("Policy-0814 Value checkpoint is not bound to the released actor")
    if manifest["weights"]["policy"]["sha256"] != POLICY_SHA256:
        raise ValueError("Policy-0814 manifest actor identity mismatch")
    if manifest["weights"]["value"]["sha256"] != VALUE_SHA256:
        raise ValueError("Policy-0814 manifest Value identity mismatch")
    for relative_path, expected in manifest["file_sha256"].items():
        actual = _sha256(ARCHIVE_ROOT / relative_path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative_path}: {actual}")
    for bundle, expected in manifest["implementation_bundle_sha256"].items():
        actual = _tree_sha256(ARCHIVE_ROOT / bundle)
        if actual != expected:
            raise ValueError(f"implementation bundle SHA-256 mismatch for {bundle}: {actual}")
    if any(path.is_symlink() for path in ARCHIVE_ROOT.rglob("*")):
        raise ValueError("Policy-0814 release must not contain symlinks")
    if any(
        "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store"
        for path in ARCHIVE_ROOT.rglob("*")
    ):
        raise ValueError("Policy-0814 release contains cache or platform-junk files")

    identity = manifest["policy_identity"]
    if identity["policy_id"] != POLICY_ID:
        raise ValueError("Policy-0814 manifest policy_id mismatch")
    actual_components = _component_hashes(policy_payload["state_dict"])
    declared_components = {
        name: row["effective_sha256"] for name, row in identity["components"].items()
    }
    if actual_components != declared_components:
        raise ValueError("Policy-0814 effective component identity mismatch")
    if _effective_policy_sha256(identity) != identity["effective_policy_sha256"]:
        raise ValueError("Policy-0814 effective policy hash mismatch")

    parity = manifest["architecture_parity_with_policy_0809"]
    actor_structure = _structural_sha256(policy_payload["state_dict"])
    value_structure = _structural_sha256(value_payload["value_head_state_dict"])
    if actor_structure != parity["actor_structure_sha256_0814"]:
        raise ValueError("Policy-0814 actor structural inventory mismatch")
    if value_structure != parity["value_structure_sha256_0814"]:
        raise ValueError("Policy-0814 Value structural inventory mismatch")
    if actor_structure != parity["actor_structure_sha256_0809"]:
        raise ValueError("Policy-0814 actor architecture differs from Policy-0809")
    if value_structure != parity["value_structure_sha256_0809"]:
        raise ValueError("Policy-0814 Value architecture differs from Policy-0809")

    batch = _smoke_batch()
    policy = load_policy("cpu")
    critic = load_value_network("cpu")
    with torch.inference_mode():
        logits = policy(batch)
        outputs = critic(batch)
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError("minimal Policy-0814 actor forward failed")
    expected_shapes = {
        "value_logit": (1,),
        "archetype_logits": (1, 15),
        "final_diff_logits": (1, 13),
        "win_probability": (1,),
        "value": (1,),
    }
    for name, shape in expected_shapes.items():
        tensor = getattr(outputs, name)
        if tensor.shape != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"minimal Policy-0814 Value forward failed for {name}")
    print(
        "verified Policy-0814 internal pretrained release: "
        f"policy_epoch={policy_payload['metadata']['epoch']}, "
        f"value_epoch={value_payload['metadata']['epoch']}, "
        f"effective_policy_sha256={identity['effective_policy_sha256']}, "
        f"policy_parameters={sum(p.numel() for p in policy.parameters()):,}, "
        f"value_parameters={sum(p.numel() for p in critic.value_head.parameters()):,}"
    )


if __name__ == "__main__":
    main()
