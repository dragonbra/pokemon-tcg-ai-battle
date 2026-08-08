"""Atomic model-only decoder/value checkpoints for full-semantic V3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from ..policy.actor_critic import SemanticActorCritic


FORBIDDEN_KEYS = {
    "optimizer",
    "scheduler",
    "scaler",
    "rng_state",
    "rollout_buffer",
    "replay",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_only(
    model: SemanticActorCritic,
    path: Path,
    *,
    update: int,
    metadata: dict[str, Any],
) -> str:
    if update < 0:
        raise ValueError("checkpoint update must be nonnegative")
    if model.adaptation_config.lora:
        state = {
            name: value.detach().cpu()
            for name, value in model.named_parameters()
            if value.requires_grad
        }
        schema = "0037_value_initialized_adapted_model_only_v2"
    else:
        state = {
            **{
            f"action_decoder.{name}": value.detach().cpu()
            for name, value in model.actor.action_decoder.state_dict().items()
            },
            **{
            f"value_head.{name}": value.detach().cpu()
            for name, value in model.value_head.state_dict().items()
            },
        }
        schema = "0037_value_initialized_decoder_critic_model_only_v1"
    payload = {
        "schema_version": schema,
        "update": update,
        "actor_schema": "0031_rule_faithful_semantic_decision_v2",
        "state_dict": state,
        "adaptation": {
            "lora": model.adaptation_config.lora,
            "layernorm_tuning": model.adaptation_config.layernorm_tuning,
            "rank": model.adaptation_config.rank,
            "alpha": model.adaptation_config.alpha,
            "board_layers": model.adaptation_config.board_layers,
        },
        "metadata": metadata,
    }
    if FORBIDDEN_KEYS & set(payload):
        raise RuntimeError("model-only checkpoint contains recovery state")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_tmp.write_text(digest + "\n", encoding="ascii")
    sidecar_tmp.replace(sidecar)
    return digest


def load_adapted_model_only(model: SemanticActorCritic, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "0037_value_initialized_adapted_model_only_v2":
        raise ValueError("unexpected adapted checkpoint schema")
    current = dict(model.named_parameters())
    state = payload.get("state_dict") or {}
    expected = {name for name, value in current.items() if value.requires_grad}
    if set(state) != expected:
        raise ValueError("adapted checkpoint trainable inventory mismatch")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(current[name].device, current[name].dtype))
    return payload


__all__ = ["FORBIDDEN_KEYS", "load_adapted_model_only", "save_model_only"]
