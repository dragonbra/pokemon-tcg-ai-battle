"""Frozen 0019 encoder with a trainable deck decoder and value head."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn

from ..decoder import DECODER_COMPONENTS, create_value_head, load_decoder_checkpoint
from ..foundation import FoundationIdentity, load_foundation


class LeagueActorCritic(nn.Module):
    """Compose one immutable Foundation actor with deck-local trainable heads."""

    def __init__(self, actor: nn.Module, value_head: nn.Module | None = None) -> None:
        super().__init__()
        self.actor = actor
        self.value_head = value_head or create_value_head(int(actor.config.d_model))
        self.freeze_encoder()

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        state, options = self.actor.encode(batch)
        return state, options, self.value_head(state).squeeze(-1)

    def value(self, batch: dict[str, Tensor]) -> Tensor:
        state, _ = self.actor.encode(batch)
        return self.value_head(state).squeeze(-1)

    def freeze_encoder(self) -> None:
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        for component in DECODER_COMPONENTS:
            for parameter in getattr(self.actor, component).parameters():
                parameter.requires_grad_(True)
        for parameter in self.value_head.parameters():
            parameter.requires_grad_(True)

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    def assert_trainable_contract(self) -> None:
        names = self.trainable_parameter_names()
        if not names:
            raise RuntimeError("League actor-critic has no trainable parameters")
        allowed = tuple(f"actor.{component}." for component in DECODER_COMPONENTS)
        invalid = [
            name for name in names
            if not name.startswith(allowed) and not name.startswith("value_head.")
        ]
        if invalid:
            raise RuntimeError(f"encoder parameters are trainable: {invalid[:5]}")

    def encoder_sha256(self) -> str:
        digest = hashlib.sha256()
        decoder_prefixes = tuple(f"{component}." for component in DECODER_COMPONENTS)
        for name, tensor in sorted(self.actor.state_dict().items()):
            if name.startswith(decoder_prefixes):
                continue
            digest.update(name.encode("utf-8")); digest.update(b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def load_league_actor_critic(
    device: str = "cpu", *, decoder_checkpoint: str | Path | None = None,
    deck_id: str | None = None, deck_sha256: str | None = None,
) -> tuple[LeagueActorCritic, FoundationIdentity]:
    actor, identity = load_foundation("cpu", eval_mode=False)
    model = LeagueActorCritic(actor)
    if decoder_checkpoint is not None:
        if deck_id is None or deck_sha256 is None:
            raise ValueError("deck identity is required when loading a decoder checkpoint")
        load_decoder_checkpoint(
            decoder_checkpoint,
            expected_foundation_sha256=identity.weights_sha256,
            expected_deck_id=deck_id,
            expected_deck_sha256=deck_sha256,
            model=model.actor,
            value_head=model.value_head,
        )
    model.assert_trainable_contract()
    model.to(device)
    return model, identity


__all__ = ["LeagueActorCritic", "load_league_actor_critic"]
