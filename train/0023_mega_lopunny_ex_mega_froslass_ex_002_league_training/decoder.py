"""Deck-specific decoder and value-head model-only checkpoints."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


DECODER_COMPONENTS = (
    "pointer_key",
    "pointer_query",
    "option_bias",
    "decoder_init",
    "decoder",
    "stop",
)
CHECKPOINT_SCHEMA = "0023_league_decoder_model_only_v1"
READABLE_CHECKPOINT_SCHEMAS = frozenset({
    "0022_league_decoder_model_only_v1",
    CHECKPOINT_SCHEMA,
})
FORBIDDEN_FIELDS = frozenset(
    {
        "optimizer",
        "optimizer_state",
        "scheduler",
        "scaler",
        "rng",
        "rng_state",
        "replay",
        "rollout",
        "rollout_buffer",
    }
)


@dataclass(frozen=True)
class DecoderLoadAudit:
    path: Path
    checkpoint_sha256: str
    deck_id: str
    deck_sha256: str
    foundation_sha256: str
    policy_role: str
    policy_version: str
    update: int
    tensor_count: int
    includes_value_head: bool


def create_value_head(width: int) -> nn.Sequential:
    if width < 1:
        raise ValueError("value head width must be positive")
    head = nn.Sequential(
        nn.LayerNorm(width),
        nn.Linear(width, width),
        nn.GELU(),
        nn.Linear(width, 1),
        nn.Tanh(),
    )
    output = head[-2]
    assert isinstance(output, nn.Linear)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    return head


def extract_decoder_state(
    model: nn.Module, value_head: nn.Module | None = None
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for component in DECODER_COMPONENTS:
        module = getattr(model, component, None)
        if not isinstance(module, nn.Module):
            raise ValueError(f"model is missing decoder component: {component}")
        for name, tensor in module.state_dict().items():
            state[f"decoder.{component}.{name}"] = tensor.detach().cpu().clone()
    if value_head is not None:
        for name, tensor in value_head.state_dict().items():
            state[f"value_head.{name}"] = tensor.detach().cpu().clone()
    return state


def _expected_keys(model: nn.Module, value_head: nn.Module | None) -> set[str]:
    return set(extract_decoder_state(model, value_head))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_payload(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "foundation_sha256",
        "deck_id",
        "deck_sha256",
        "policy_role",
        "policy_version",
        "update",
        "state",
    }
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown & FORBIDDEN_FIELDS:
        forbidden = sorted(unknown & FORBIDDEN_FIELDS)
        raise ValueError(f"checkpoint contains forbidden training state: {forbidden}")
    if unknown or missing:
        raise ValueError(
            f"checkpoint fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if payload.get("schema_version") not in READABLE_CHECKPOINT_SCHEMAS:
        raise ValueError(
            "checkpoint schema must be one of "
            f"{sorted(READABLE_CHECKPOINT_SCHEMAS)}"
        )
    if payload.get("policy_role") not in {"live", "frozen_snapshot"}:
        raise ValueError("checkpoint policy_role must be live or frozen_snapshot")
    if not isinstance(payload.get("update"), int) or payload["update"] < 0:
        raise ValueError("checkpoint update must be a nonnegative integer")
    state = payload.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint state must be a nonempty tensor mapping")
    for key, tensor in state.items():
        if not isinstance(key, str) or not (
            key.startswith("decoder.") or key.startswith("value_head.")
        ):
            raise ValueError(f"checkpoint contains unknown tensor key: {key!r}")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint value is not a tensor: {key}")


def save_decoder_checkpoint(
    path: Path,
    model: nn.Module,
    value_head: nn.Module,
    *,
    foundation_sha256: str,
    deck_id: str,
    deck_sha256: str,
    policy_role: str,
    policy_version: str,
    update: int,
) -> str:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "foundation_sha256": foundation_sha256,
        "deck_id": deck_id,
        "deck_sha256": deck_sha256,
        "policy_role": policy_role,
        "policy_version": policy_version,
        "update": int(update),
        "state": extract_decoder_state(model, value_head),
    }
    _validate_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"decoder checkpoint already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    try:
        sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
        sidecar_temporary.replace(sidecar)
    finally:
        sidecar_temporary.unlink(missing_ok=True)
    return digest


def load_decoder_checkpoint(
    path: Path,
    *,
    expected_foundation_sha256: str,
    expected_deck_id: str,
    expected_deck_sha256: str,
    model: nn.Module | None = None,
    value_head: nn.Module | None = None,
) -> DecoderLoadAudit:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("decoder checkpoint payload must be an object")
    _validate_payload(payload)
    identity_checks = {
        "Foundation SHA": (payload["foundation_sha256"], expected_foundation_sha256),
        "deck ID": (payload["deck_id"], expected_deck_id),
        "deck SHA": (payload["deck_sha256"], expected_deck_sha256),
    }
    for label, (actual, expected) in identity_checks.items():
        if actual != expected:
            raise ValueError(f"decoder checkpoint {label} mismatch: {actual!r} != {expected!r}")
    state: dict[str, torch.Tensor] = payload["state"]
    includes_value = any(key.startswith("value_head.") for key in state)
    if (model is None) != (value_head is None):
        raise ValueError("model and value_head must be supplied together")
    if model is not None and value_head is not None:
        expected_keys = _expected_keys(model, value_head)
        if set(state) != expected_keys:
            raise ValueError(
                "decoder tensor keys mismatch; "
                f"missing={sorted(expected_keys - set(state))}, "
                f"unknown={sorted(set(state) - expected_keys)}"
            )
        for component in DECODER_COMPONENTS:
            prefix = f"decoder.{component}."
            component_state = {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            getattr(model, component).load_state_dict(component_state, strict=True)
        value_state = {
            key.removeprefix("value_head."): value
            for key, value in state.items()
            if key.startswith("value_head.")
        }
        value_head.load_state_dict(value_state, strict=True)
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.is_file() and sidecar.read_text(encoding="ascii").split()[0] != digest:
        raise ValueError("decoder checkpoint SHA sidecar mismatch")
    return DecoderLoadAudit(
        path=path,
        checkpoint_sha256=digest,
        deck_id=payload["deck_id"],
        deck_sha256=payload["deck_sha256"],
        foundation_sha256=payload["foundation_sha256"],
        policy_role=payload["policy_role"],
        policy_version=payload["policy_version"],
        update=payload["update"],
        tensor_count=len(state),
        includes_value_head=includes_value,
    )


__all__ = [
    "DECODER_COMPONENTS",
    "DecoderLoadAudit",
    "create_value_head",
    "extract_decoder_state",
    "load_decoder_checkpoint",
    "save_decoder_checkpoint",
]
