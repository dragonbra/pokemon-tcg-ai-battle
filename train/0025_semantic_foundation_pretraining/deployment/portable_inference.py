"""Portable model-only inference for the 0025 legacy-default checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from .base_model import IDOnlyConfig, IDOnlyPointerPolicy
from .inference import legal_fallback
from .online_runtime import OnlineCausalEncoder


@dataclass(frozen=True)
class DecodedActions:
    sequences: Tensor
    lengths: Tensor
    legal: Tensor


class InferenceIDOnlyPointerPolicy(IDOnlyPointerPolicy):
    """Add the repository batched inference service contract without parameters."""

    def deterministic_action_tensors(
        self, batch: dict[str, Tensor]
    ) -> DecodedActions:
        state, options = self.encode(batch)
        batch_size, option_count, _ = options.shape
        hidden = torch.tanh(self.decoder_init(state))
        keys = self.pointer_key(options)
        chosen = torch.zeros(
            (batch_size, option_count), dtype=torch.bool, device=options.device
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
        legal = torch.ones(batch_size, dtype=torch.bool, device=options.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=options.device)
        maximum_steps = min(option_count, int(batch["max_count"].max().item()))
        sequences = torch.full(
            (batch_size, maximum_steps),
            -1,
            dtype=torch.long,
            device=options.device,
        )
        rows = torch.arange(batch_size, device=options.device)
        for step in range(maximum_steps):
            pointer = (
                self.pointer_query(hidden).unsqueeze(1) * keys
            ).sum(-1) / math.sqrt(self.config.d_model)
            pointer = pointer + self.option_bias(options).squeeze(-1)
            available = batch["option_mask"] & ~chosen
            pointer = pointer.masked_fill(
                ~available, torch.finfo(pointer.dtype).min
            )
            best_score, best_index = pointer.max(1)
            stop_score = self.stop(hidden).squeeze(1)
            at_maximum = active & lengths.ge(batch["max_count"])
            may_stop = lengths.ge(batch["min_count"])
            chose_stop = active & may_stop & stop_score.ge(best_score)
            no_option = active & ~available.any(1)
            legal &= ~(no_option & ~may_stop)
            active &= ~(at_maximum | chose_stop | no_option)
            selecting = active.clone()
            if not selecting.any():
                break
            sequences[selecting, step] = best_index[selecting]
            chosen[rows[selecting], best_index[selecting]] = True
            selected = options[rows, best_index]
            updated = self.decoder(selected, hidden)
            hidden = torch.where(selecting.unsqueeze(1), updated, hidden)
            lengths += selecting.long()
        legal &= lengths.ge(batch["min_count"]) & lengths.le(batch["max_count"])
        return DecodedActions(sequences, lengths, legal)


class PortablePolicy:
    def __init__(
        self, model: InferenceIDOnlyPointerPolicy, deck: Sequence[int]
    ) -> None:
        torch.set_num_threads(1)
        self.model = model.eval()
        self.actor = self.model
        self.config = model.config
        self.deck = tuple(int(card_id) for card_id in deck)
        self.encoder: OnlineCausalEncoder | None = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, deck: Sequence[int]
    ) -> "PortablePolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if set(payload) != {"schema_version", "state_dict", "metadata"}:
            raise ValueError("checkpoint is not the 0025 model-only contract")
        metadata = payload["metadata"]
        if not isinstance(metadata, dict) or metadata.get("arm") != "legacy_default":
            raise ValueError("checkpoint is not the legacy_default arm")
        model_contract = metadata.get("model_config")
        config_payload = (
            model_contract.get("config") if isinstance(model_contract, dict) else None
        )
        if not isinstance(config_payload, dict):
            raise ValueError("checkpoint has no legacy model config")
        config = IDOnlyConfig(**config_payload)
        model = InferenceIDOnlyPointerPolicy(config)
        model.load_state_dict(payload["state_dict"], strict=True)
        return cls(model, deck)

    def reset(self) -> None:
        self.encoder = None

    def select(self, observation: dict[str, Any]) -> list[int]:
        actor = (observation.get("current") or {}).get("yourIndex")
        if actor not in (0, 1):
            return legal_fallback(observation)
        if self.encoder is None or self.encoder.actor != actor:
            self.encoder = OnlineCausalEncoder(actor, self.deck, self.config)
        try:
            batch = self.encoder.encode(observation)
            with torch.inference_mode():
                decoded = self.model.deterministic_action_tensors(batch)
            if not bool(decoded.legal[0]):
                return legal_fallback(observation)
            length = int(decoded.lengths[0])
            return [int(value) for value in decoded.sequences[0, :length]]
        except (IndexError, RuntimeError, ValueError):
            self.encoder = None
            return legal_fallback(observation)


__all__ = [
    "DecodedActions",
    "InferenceIDOnlyPointerPolicy",
    "PortablePolicy",
]
