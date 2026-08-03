"""Atomic decoder/value-only checkpoints bound to the immutable representation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .foundation.contract import EXPECTED_CHECKPOINT_SHA256
from .policy.actor_critic import CanonicalActorCritic


SCHEMA_VERSION = "0030_dragapult_decoder_rl_model_only_v1"
FORBIDDEN_FIELDS = frozenset({
    "optimizer", "scheduler", "scaler", "rng", "dataloader", "rollout", "replay"
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(model: CanonicalActorCritic) -> dict[str, torch.Tensor]:
    output = {
        f"action_decoder.{name}": tensor.detach().cpu().clone()
        for name, tensor in model.actor.action_decoder.state_dict().items()
    }
    output.update({
        f"value_head.{name}": tensor.detach().cpu().clone()
        for name, tensor in model.value_head.state_dict().items()
    })
    return output


def _validate(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "initial_checkpoint_sha256", "representation_sha256",
        "decoder_sha256", "policy_version", "update", "state",
    }
    unknown = set(payload) - required
    missing = required - set(payload)
    if unknown & FORBIDDEN_FIELDS:
        raise ValueError(f"checkpoint contains forbidden training state: {sorted(unknown & FORBIDDEN_FIELDS)}")
    if unknown or missing:
        raise ValueError(f"checkpoint fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if payload.get("initial_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint initialization identity mismatch")
    if not isinstance(payload.get("update"), int) or payload["update"] < 0:
        raise ValueError("checkpoint update must be nonnegative")
    state = payload.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint state must be a nonempty tensor mapping")
    for name, tensor in state.items():
        if not isinstance(name, str) or not name.startswith(("action_decoder.", "value_head.")):
            raise ValueError(f"unexpected checkpoint tensor: {name!r}")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint value is not a tensor: {name}")


def save_model_checkpoint(
    path: Path, model: CanonicalActorCritic, *, policy_version: str, update: int
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "initial_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "representation_sha256": model.representation_sha256(),
        "decoder_sha256": model.decoder_sha256(),
        "policy_version": policy_version,
        "update": int(update),
        "state": _state(model),
    }
    _validate(payload)
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def load_model_checkpoint(path: Path, model: CanonicalActorCritic) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    _validate(payload)
    if payload["representation_sha256"] != model.representation_sha256():
        raise ValueError("checkpoint representation SHA mismatch")
    decoder = {
        name.removeprefix("action_decoder."): tensor
        for name, tensor in payload["state"].items()
        if name.startswith("action_decoder.")
    }
    value = {
        name.removeprefix("value_head."): tensor
        for name, tensor in payload["state"].items()
        if name.startswith("value_head.")
    }
    model.actor.action_decoder.load_state_dict(decoder, strict=True)
    model.value_head.load_state_dict(value, strict=True)
    if payload["decoder_sha256"] != model.decoder_sha256():
        raise ValueError("loaded decoder SHA mismatch")
    return {
        "checkpoint_sha256": _sha256(path),
        "decoder_sha256": payload["decoder_sha256"],
        "policy_version": payload["policy_version"],
        "update": payload["update"],
    }


__all__ = ["load_model_checkpoint", "save_model_checkpoint"]
