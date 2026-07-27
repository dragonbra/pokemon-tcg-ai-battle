from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
from .r15_model import R15DeterministicGradualOptionPolicy, R15ModelConfig


@dataclass(frozen=True)
class SourceConditionedR15Config:
    r15: R15ModelConfig
    source_vocabulary_size: int
    source_initial_scale: float = 0.10


class SourceConditionedR15Policy(R15DeterministicGradualOptionPolicy):
    """The exact 0015 V2 actor graph, frozen into the 0017 namespace."""

    def __init__(self, config: SourceConditionedR15Config, *, ontology_path: Path) -> None:
        super().__init__(config.r15, ontology_path=ontology_path)
        self.source_config = config
        width = self.config.d_model
        self.source_persona = nn.Embedding(config.source_vocabulary_size + 1, width)
        self.source_state_norm = nn.LayerNorm(width)
        self.source_option_norm = nn.LayerNorm(width)
        initial = torch.tensor(config.source_initial_scale)
        self.source_gate = nn.Parameter(torch.full((width,), float(torch.atanh(initial))))
        nn.init.normal_(self.source_persona.weight, std=0.02)
        with torch.no_grad():
            self.source_persona.weight[0].zero_()

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        state, options = super().encode(batch)
        source_id = batch.get("source_id")
        if source_id is None or source_id.ndim != 1 or source_id.size(0) != state.size(0):
            raise ValueError("source_id must have one entry per observation")
        persona = self.source_persona(source_id)
        delta = torch.tanh(self.source_gate) * persona
        return (
            self.source_state_norm(state + delta),
            self.source_option_norm(options + delta.unsqueeze(1)),
        )


class DragapultActorCritic(nn.Module):
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


def _config_from_metadata(metadata: dict[str, Any]) -> SourceConditionedR15Config:
    raw = metadata.get("model")
    if not isinstance(raw, dict) or not isinstance(raw.get("r15"), dict):
        raise ValueError("source checkpoint is missing model.r15 metadata")
    r15_raw = raw["r15"]
    ac_raw = r15_raw["ac"]
    base = IDOnlyConfig(**ac_raw["base"])
    ac = ACModelConfig(
        base=base,
        event_layers=int(ac_raw["event_layers"]),
        auxiliary_ffn_multiplier=int(ac_raw["auxiliary_ffn_multiplier"]),
        goal_roles=int(ac_raw["goal_roles"]),
    )
    r15 = R15ModelConfig(
        ac=ac,
        scenario_layers=int(r15_raw["scenario_layers"]),
        scenario_ffn_multiplier=int(r15_raw["scenario_ffn_multiplier"]),
        scale_gate_ffn_multiplier=int(r15_raw["scale_gate_ffn_multiplier"]),
        option_initial_scale=float(r15_raw["option_initial_scale"]),
    )
    return SourceConditionedR15Config(
        r15=r15,
        source_vocabulary_size=int(raw["source_vocabulary_size"]),
        source_initial_scale=float(raw["source_initial_scale"]),
    )


def load_source_actor_critic(
    checkpoint: str | Path = SOURCE_CHECKPOINT,
    *,
    ontology_path: str | Path = ONTOLOGY_PATH,
    verify_hash: bool = True,
) -> tuple[DragapultActorCritic, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    if verify_hash:
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != SOURCE_CHECKPOINT_SHA256:
            raise ValueError(f"source checkpoint SHA-256 mismatch: {digest}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    if metadata.get("model_family", "r15") != "r15":
        raise ValueError("0017 source checkpoint must be the R15 family")
    actor = SourceConditionedR15Policy(
        _config_from_metadata(metadata), ontology_path=Path(ontology_path)
    )
    actor.load_state_dict(payload["model"], strict=True)
    model = DragapultActorCritic(actor)
    if model.actor_parameter_count() != 17_416_642:
        raise ValueError(
            f"unexpected source actor parameter count: {model.actor_parameter_count()}"
        )
    return model, metadata


__all__ = [
    "DragapultActorCritic",
    "SourceConditionedR15Config",
    "SourceConditionedR15Policy",
    "load_source_actor_critic",
]
