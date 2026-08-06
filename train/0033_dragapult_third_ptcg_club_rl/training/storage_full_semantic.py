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
    payload = {
        "schema_version": "0033_full_semantic_decoder_value_model_only_v1",
        "update": update,
        "actor_schema": "0031_rule_faithful_semantic_decision_v2",
        "state_dict": state,
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


__all__ = ["FORBIDDEN_KEYS", "save_model_only"]
