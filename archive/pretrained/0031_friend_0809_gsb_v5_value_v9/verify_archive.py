"""Verify hashes, model-only boundaries, strict pair loading, and minimal forwards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from loader import (
    ARCHIVE_ROOT,
    POLICY_SHA256,
    VALUE_SHA256,
    load_policy,
    load_policy_checkpoint,
    load_value_checkpoint,
    load_value_network,
)
from model_source.contracts.fields import WIDTHS


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
    if forbidden & set(payload):
        raise ValueError("release checkpoint contains resumable training state")


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
    manifest = json.loads((ARCHIVE_ROOT / "manifest.json").read_text(encoding="ascii"))
    policy_payload = load_policy_checkpoint()
    value_payload = load_value_checkpoint()
    _reject_resume_state(policy_payload)
    _reject_resume_state(value_payload)
    if value_payload["metadata"]["source_checkpoint_sha256"] != POLICY_SHA256:
        raise ValueError("Value checkpoint is not bound to the released policy")
    if manifest["weights"]["policy"]["sha256"] != POLICY_SHA256:
        raise ValueError("policy manifest identity mismatch")
    if manifest["weights"]["value"]["sha256"] != VALUE_SHA256:
        raise ValueError("Value manifest identity mismatch")
    for relative_path, expected in manifest["file_sha256"].items():
        actual = _sha256(ARCHIVE_ROOT / relative_path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative_path}: {actual}")
    for bundle, expected in manifest["implementation_bundle_sha256"].items():
        actual = _tree_sha256(ARCHIVE_ROOT / bundle)
        if actual != expected:
            raise ValueError(f"implementation bundle SHA-256 mismatch for {bundle}: {actual}")
    if any(path.is_symlink() for path in ARCHIVE_ROOT.rglob("*")):
        raise ValueError("pretrained release must not contain symlinks")

    batch = _smoke_batch()
    policy = load_policy("cpu")
    critic = load_value_network("cpu")
    with torch.inference_mode():
        logits = policy(batch)
        outputs = critic(batch)
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError("minimal 0031 policy forward failed")
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
            raise ValueError(f"minimal 0036 Value forward failed for {name}")
    print(
        "verified 0031-friend-0809-gsb-v5-value-v9: "
        f"policy_epoch={policy_payload['metadata']['epoch']}, "
        f"value_epoch={value_payload['metadata']['epoch']}, "
        f"policy_parameters={sum(p.numel() for p in policy.parameters()):,}, "
        f"value_parameters={sum(p.numel() for p in critic.value_head.parameters()):,}"
    )


if __name__ == "__main__":
    main()
