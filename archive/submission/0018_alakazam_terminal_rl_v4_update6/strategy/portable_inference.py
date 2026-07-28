"""Portable CPU greedy inference for 0018 candidate packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .ac_model import ACModelConfig
from .base_model import IDOnlyConfig
from .online_runtime import OnlineCausalEncoder
from .r15_model import R15ModelConfig
from .source_model import SourceModelConfig
from .source_r15_model import SourceConditionedR15Policy


class PortableActorCritic(nn.Module):
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


def _fixed_config() -> tuple[R15ModelConfig, SourceModelConfig]:
    base = IDOnlyConfig(
        max_card_id=2048,
        d_model=320,
        heads=8,
        encoder_layers=4,
        ffn_multiplier=3,
        dropout=0.1,
        max_entities=192,
        max_options=128,
        max_action_steps=64,
    )
    return (
        R15ModelConfig(
            ac=ACModelConfig(
                base=base,
                event_layers=1,
                auxiliary_ffn_multiplier=2,
                goal_roles=4,
            ),
            scenario_layers=2,
            scenario_ffn_multiplier=3,
            scale_gate_ffn_multiplier=2,
            option_initial_scale=0.35,
        ),
        SourceModelConfig(vocabulary_size=25, initial_scale=0.05),
    )


class PortablePolicy:
    def __init__(
        self,
        actor: SourceConditionedR15Policy,
        deck: Sequence[int],
        *,
        source_id: int = 0,
    ) -> None:
        torch.set_num_threads(1)
        self.actor = actor.eval()
        self.deck = tuple(int(value) for value in deck)
        self.source_id = source_id
        self.encoder: OnlineCausalEncoder | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        ontology_path: Path,
        deck: Sequence[int],
    ) -> "PortablePolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_config, source_config = _fixed_config()
        actor = SourceConditionedR15Policy(
            model_config,
            source_config,
            ontology_path=ontology_path,
        )
        state = payload["model"]
        if any(key.startswith("actor.") for key in state):
            model = PortableActorCritic(actor)
            model.load_state_dict(state, strict=True)
            actor = model.actor
        else:
            actor.load_state_dict(state, strict=True)
        return cls(actor, deck)

    def reset(self) -> None:
        self.encoder = None

    def select(self, observation: dict[str, Any]) -> list[int]:
        current = observation.get("current") or {}
        actor_index = current.get("yourIndex")
        if actor_index not in (0, 1):
            raise ValueError("observation has no valid actor")
        if self.encoder is None or self.encoder.actor != actor_index:
            self.encoder = OnlineCausalEncoder(
                actor_index, self.deck, self.actor.config
            )
        batch = self.encoder.encode(observation)
        batch["source_id"] = torch.tensor([self.source_id], dtype=torch.long)
        with torch.inference_mode():
            return self.actor.greedy_action(batch)


__all__ = ["PortablePolicy"]
