"""Atomic model-only decoder/value checkpoints for full-semantic V3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from ..policy.actor_critic import SemanticActorCritic
from ..checkpoint import CHECKPOINT_SCHEMA_VERSION, checkpoint_metadata, validate_model_only_payload
from ..source import ACTOR_SHA256, VALUE_SHA256


FORBIDDEN_KEYS = {
    "optimizer",
    "scheduler",
    "scaler",
    "rng_state",
    "rollout_buffer",
    "replay",
}

AUXILIARY_PREFIXES = (
    "prize_aux.", "opponent_meta_head.", "opponent_meta_conditioner.", "tempo_aux_head.",
)


def _checkpoint_tensor(model: SemanticActorCritic, name: str) -> bool:
    return (
        name.startswith(("actor.action_decoder.", "value_head.", "allocation_head."))
        or name.startswith(AUXILIARY_PREFIXES)
        or (".parametrizations." in name and not name.endswith(".original"))
        or (
            model.adaptation_config.layernorm_tuning
            and name.startswith("actor.option_encoder.cross_attention_transformer.norm.")
        )
    )


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
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if _checkpoint_tensor(model, name)
    }
    version_metadata = checkpoint_metadata(
        source_actor_sha256=ACTOR_SHA256, source_value_sha256=VALUE_SHA256
    )
    version_metadata.update(metadata)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "update": update,
        "actor_schema": "0031_rule_faithful_semantic_decision_v2",
        "state_dict": state,
        "adaptation": {
            "lora": model.adaptation_config.lora,
            "layernorm_tuning": model.adaptation_config.layernorm_tuning,
            "rank": model.adaptation_config.rank,
            "alpha": model.adaptation_config.alpha,
            "option_block": model.adaptation_config.option_block,
        },
        "integrated_flags": model.integrated_flags.metadata(),
        "metadata": version_metadata,
    }
    if FORBIDDEN_KEYS & set(payload):
        raise RuntimeError("model-only checkpoint contains recovery state")
    validate_model_only_payload(payload)
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
    validate_model_only_payload(payload)
    current = model.state_dict()
    state = payload.get("state_dict") or {}
    expected = {
        name for name in current
        if _checkpoint_tensor(model, name)
    }
    if set(state) != expected:
        raise ValueError("0038 checkpoint state inventory mismatch")
    with torch.no_grad():
        for name in expected:
            target = current[name]
            target.copy_(state[name].to(target.device, target.dtype))
    return payload


__all__ = ["FORBIDDEN_KEYS", "load_adapted_model_only", "save_model_only"]
