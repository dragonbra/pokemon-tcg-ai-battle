from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..constants import (
    ONTOLOGY_PATH,
    SOURCE_CHECKPOINT,
    SOURCE_CHECKPOINT_SHA256,
)
from .ac_model import ACModelConfig
from .base_model import IDOnlyConfig
from .r15_model import R15ModelConfig
from .source_model import SourceModelConfig
from .source_r15_model import SourceConditionedR15Policy


class AlakazamActorCritic(nn.Module):
    """Exact source actor plus a separately initialized scalar terminal-value head."""

    def __init__(self, actor: SourceConditionedR15Policy) -> None:
        super().__init__()
        self.actor = actor
        width = actor.config.d_model
        self.value_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
            nn.Tanh(),
        )
        output = self.value_head[-2]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        state, options = self.actor.encode(batch)
        return state, options, self.value_head(state).squeeze(-1)

    def value(self, batch: dict[str, Tensor]) -> Tensor:
        state, _ = self.actor.encode(batch)
        return self.value_head(state).squeeze(-1)

    def actor_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.actor.parameters())

    def freeze_actor(self) -> None:
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    def unfreeze_decoder(self) -> None:
        self.freeze_actor()
        names = (
            "pointer_key",
            "pointer_query",
            "option_bias",
            "decoder_init",
            "decoder",
            "stop",
        )
        for name in names:
            for parameter in getattr(self.actor, name).parameters():
                parameter.requires_grad_(True)


def _config_from_metadata(metadata: dict[str, Any]) -> tuple[R15ModelConfig, SourceModelConfig]:
    raw = metadata.get("model")
    source_raw = metadata.get("source_model")
    if not isinstance(raw, dict) or not isinstance(raw.get("ac"), dict):
        raise ValueError("source checkpoint is missing model.ac metadata")
    if not isinstance(source_raw, dict):
        raise ValueError("source checkpoint is missing source_model metadata")
    ac_raw = raw["ac"]
    base = IDOnlyConfig(**ac_raw["base"])
    ac = ACModelConfig(
        base=base,
        event_layers=int(ac_raw["event_layers"]),
        auxiliary_ffn_multiplier=int(ac_raw["auxiliary_ffn_multiplier"]),
        goal_roles=int(ac_raw["goal_roles"]),
    )
    return R15ModelConfig(
        ac=ac,
        scenario_layers=int(raw["scenario_layers"]),
        scenario_ffn_multiplier=int(raw["scenario_ffn_multiplier"]),
        scale_gate_ffn_multiplier=int(raw["scale_gate_ffn_multiplier"]),
        option_initial_scale=float(raw["option_initial_scale"]),
    ), SourceModelConfig(
        vocabulary_size=int(source_raw["vocabulary_size"]),
        initial_scale=float(source_raw["initial_scale"]),
    )


def load_source_actor_critic(
    checkpoint: str | Path = SOURCE_CHECKPOINT,
    *,
    ontology_path: str | Path = ONTOLOGY_PATH,
    verify_hash: bool = True,
) -> tuple[AlakazamActorCritic, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    if verify_hash:
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != SOURCE_CHECKPOINT_SHA256:
            raise ValueError(f"source checkpoint SHA-256 mismatch: {digest}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    if metadata.get("schema_version") != "0016_source_r15_training_v1":
        raise ValueError("0018 source checkpoint must be the 0016 R15 family")
    model_config, source_config = _config_from_metadata(metadata)
    actor = SourceConditionedR15Policy(
        model_config,
        source_config,
        ontology_path=Path(ontology_path),
    )
    actor.load_state_dict(payload["model"], strict=True)
    model = AlakazamActorCritic(actor)
    if model.actor_parameter_count() != 17_601_282:
        raise ValueError(
            f"unexpected source actor parameter count: {model.actor_parameter_count()}"
        )
    return model, metadata


__all__ = [
    "AlakazamActorCritic",
    "SourceConditionedR15Policy",
    "load_source_actor_critic",
]
