"""Audited initialization of a new rule reader from a frozen R15 policy."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn


_RULE_PREFIXES = (
    "rule_",
    "turn_budget_reader.",
    "prize_race_reader.",
    "library_pressure_reader.",
    "board_relay_reader.",
    "option_rule_attention.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_r15_base_checkpoint(model: nn.Module, checkpoint: Path | str) -> dict[str, object]:
    """Load every inherited R15/source tensor and leave only the new rule reader fresh."""

    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    if metadata.get("schema_version") != "0015_r15_source_conditioned_training_v1":
        raise ValueError("warm start requires an 0015 source-conditioned R15 checkpoint")
    if metadata.get("model_family", "r15") != "r15":
        raise ValueError("warm start source must use the frozen r15 model family")
    source = payload.get("model")
    if not isinstance(source, dict) or not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in source.items()
    ):
        raise ValueError("warm start checkpoint has no tensor state dictionary")

    target = model.state_dict()
    unexpected = sorted(set(source) - set(target))
    new_model_keys = sorted(set(target) - set(source))
    invalid_new = [
        name for name in new_model_keys if not name.startswith(_RULE_PREFIXES)
    ]
    shape_mismatches = sorted(
        name for name in set(source) & set(target) if source[name].shape != target[name].shape
    )
    if unexpected or invalid_new or shape_mismatches or not new_model_keys:
        raise ValueError(
            "warm start key contract mismatch: "
            f"unexpected={unexpected}, invalid_new={invalid_new}, shapes={shape_mismatches}"
        )

    incompatible = model.load_state_dict(source, strict=False)
    if incompatible.unexpected_keys or sorted(incompatible.missing_keys) != new_model_keys:
        raise RuntimeError("warm start load result disagrees with the audited key contract")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "source_version": metadata.get("version"),
        "source_model_family": "r15",
        "loaded_key_count": len(source),
        "new_model_keys": new_model_keys,
    }


def freeze_to_new_model_keys(
    model: nn.Module,
    new_model_keys: Iterable[str],
) -> dict[str, object]:
    """Freeze the inherited policy and keep exactly the audited new rule tensors trainable."""

    trainable_names = set(new_model_keys)
    parameters = dict(model.named_parameters())
    invalid = sorted(trainable_names - set(parameters))
    if invalid or not trainable_names:
        raise ValueError(f"rule-only trainable-key contract is invalid: {invalid}")
    if any(not name.startswith(_RULE_PREFIXES) for name in trainable_names):
        raise ValueError("rule-only training received a non-rule parameter")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in trainable_names)
    trainable = [parameter for parameter in parameters.values() if parameter.requires_grad]
    frozen = [parameter for parameter in parameters.values() if not parameter.requires_grad]
    return {
        "mode": "new_rule_parameters_only",
        "trainable_tensor_count": len(trainable),
        "frozen_tensor_count": len(frozen),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "frozen_parameter_count": sum(parameter.numel() for parameter in frozen),
        "trainable_names": sorted(trainable_names),
    }


__all__ = ["freeze_to_new_model_keys", "load_r15_base_checkpoint"]
