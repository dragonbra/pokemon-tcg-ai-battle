"""One frozen Foundation encoder with isolated deck-local decoder/value heads."""
from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn

from ..decoder import DECODER_COMPONENTS, create_value_head, load_decoder_checkpoint
from ..decks import DeckPlugin
from ..foundation import load_foundation
from .actor_critic import LeagueActorCritic


class RoutedActor(nn.Module):
    """Delegate encoding to one unregistered shared actor and own only a decoder."""

    def __init__(self, shared_actor: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "shared_actor", shared_actor)
        for component in DECODER_COMPONENTS:
            self.add_module(component, copy.deepcopy(getattr(shared_actor, component)))

    @property
    def config(self):
        return self.shared_actor.config

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        encoded = self.shared_actor.encode_backbone(batch)
        source_id = batch.get("source_id")
        if source_id is None:
            raise ValueError("source-conditioned Foundation requires source_id")
        return self.shared_actor.condition_encoding(encoded, source_id)

    def deterministic_action_tensors(
        self, batch: dict[str, Tensor], *, encoded: tuple[Tensor, Tensor] | None = None
    ):
        # The Foundation helper reads decoder attributes from self, so bind its
        # implementation to this routed actor instead of delegating the call.
        method = type(self.shared_actor).deterministic_action_tensors
        return method(self, batch, encoded=encoded)


class LeaguePolicyPool(nn.Module):
    def __init__(
        self, shared_actor: nn.Module, policies: dict[str, LeagueActorCritic]
    ) -> None:
        super().__init__()
        self.shared_actor = shared_actor
        self.policies = nn.ModuleDict(policies)
        for parameter in self.shared_actor.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_foundation(
        cls,
        plugins: Iterable[DeckPlugin],
        *,
        device: torch.device,
        decoder_checkpoints: dict[str, str] | None = None,
        foundation_sha256: str | None = None,
    ) -> "LeaguePolicyPool":
        shared_actor, _ = load_foundation(device, eval_mode=True)
        policies: dict[str, LeagueActorCritic] = {}
        for plugin in plugins:
            routed = RoutedActor(shared_actor)
            policy = LeagueActorCritic(
                routed, create_value_head(int(shared_actor.config.d_model))
            ).to(device)
            checkpoint = (decoder_checkpoints or {}).get(plugin.deck_id)
            if checkpoint is not None:
                if foundation_sha256 is None:
                    raise ValueError("foundation SHA is required for decoder loading")
                load_decoder_checkpoint(
                    Path(checkpoint),
                    expected_foundation_sha256=foundation_sha256,
                    expected_deck_id=plugin.deck_id,
                    expected_deck_sha256=plugin.deck_sha256,
                    model=policy.actor,
                    value_head=policy.value_head,
                )
            policies[plugin.deck_id] = policy
        return cls(shared_actor, policies).to(device)

    def policy(self, deck_id: str) -> LeagueActorCritic:
        if deck_id not in self.policies:
            raise KeyError(f"unknown League policy deck: {deck_id}")
        return self.policies[deck_id]

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def total_parameter_storage_count(self) -> int:
        unique: dict[int, nn.Parameter] = {}
        for parameter in self.parameters():
            unique.setdefault(parameter.data_ptr(), parameter)
        return sum(parameter.numel() for parameter in unique.values())


__all__ = ["LeaguePolicyPool", "RoutedActor"]
