"""Model-only portable inference for the multi-memory semantic policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from ..features.prototypes import PrototypeIndex
from ..model.multi_memory import SemanticFoundationPolicy, SemanticModelConfig
from .semantic_online_runtime import OnlineCausalEncoder, _prototype_path


@dataclass(frozen=True)
class DecodedActions:
    sequences: Tensor
    lengths: Tensor
    legal: Tensor


class InferenceSemanticFoundationPolicy(SemanticFoundationPolicy):
    """Add batched legal greedy decoding without adding model parameters."""

    def deterministic_action_tensors(
        self, batch: dict[str, Tensor]
    ) -> DecodedActions:
        state, options = self.encode(batch)
        hidden = torch.tanh(self.decoder_init(state))
        batch_size, option_count, _ = options.shape
        available = batch["option_mask"].clone()
        lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
        legal = torch.ones(batch_size, dtype=torch.bool, device=options.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=options.device)
        maximum_steps = min(
            self.config.max_action_steps,
            option_count,
            int(batch["max_count"].max().item()),
        )
        sequences = torch.full(
            (batch_size, maximum_steps),
            -1,
            dtype=torch.long,
            device=options.device,
        )
        for step in range(maximum_steps):
            scores = self._logits(batch, options, hidden, available, lengths)
            choice = scores.argmax(dim=1)
            chose_stop = choice.eq(option_count)
            selecting = active & ~chose_stop
            legal &= ~(active & chose_stop & lengths.lt(batch["min_count"]))
            active &= ~chose_stop
            if not selecting.any():
                break
            selected = choice.clamp_max(option_count - 1)
            sequences[selecting, step] = selected[selecting]
            updated_hidden, updated_available, updated_lengths = self._consume(
                options, hidden, available, lengths, selected
            )
            hidden = torch.where(selecting.unsqueeze(1), updated_hidden, hidden)
            available = torch.where(
                selecting.unsqueeze(1), updated_available, available
            )
            lengths = torch.where(selecting, updated_lengths, lengths)
            active &= lengths.lt(batch["max_count"])
        legal &= lengths.ge(batch["min_count"]) & lengths.le(batch["max_count"])
        return DecodedActions(sequences, lengths, legal)


class PortableSemanticPolicy:
    requires_source_id = False
    fail_closed_inference_errors = True

    def __init__(
        self, model: InferenceSemanticFoundationPolicy, deck: Sequence[int]
    ) -> None:
        torch.set_num_threads(1)
        self.model = model.eval()
        self.actor = self.model
        self.config = model.config
        self.deck = tuple(int(card_id) for card_id in deck)
        if len(self.deck) != 60 or any(card_id <= 0 for card_id in self.deck):
            raise ValueError("deck must contain exactly 60 positive card IDs")
        self.encoder: OnlineCausalEncoder | None = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, deck: Sequence[int]
    ) -> "PortableSemanticPolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if set(payload) != {"schema_version", "state_dict", "metadata"}:
            raise ValueError("checkpoint is not the 0025 model-only contract")
        metadata = payload["metadata"]
        if not isinstance(metadata, dict) or metadata.get("arm") != "semantic":
            raise ValueError("checkpoint is not the semantic arm")
        model_contract = metadata.get("model_config")
        config_payload = (
            model_contract.get("config") if isinstance(model_contract, dict) else None
        )
        if not isinstance(config_payload, dict):
            raise ValueError("checkpoint has no semantic model config")
        config = SemanticModelConfig(**config_payload)
        model = InferenceSemanticFoundationPolicy(
            config, PrototypeIndex.load(_prototype_path())
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        return cls(model, deck)

    def online_encoder(
        self, actor: int, deck: Sequence[int] | None = None
    ) -> OnlineCausalEncoder:
        return OnlineCausalEncoder(actor, deck or self.deck, self.config)

    def reset(self) -> None:
        self.encoder = None

    def select(self, observation: dict[str, Any]) -> list[int]:
        actor = (observation.get("current") or {}).get("yourIndex")
        if actor not in (0, 1):
            raise ValueError("observation has no valid actor")
        if self.encoder is None or self.encoder.actor != actor:
            self.encoder = self.online_encoder(actor)
        batch = self.encoder.encode(observation)
        with torch.inference_mode():
            result = self.model.deterministic_action_tensors(batch)
        if not bool(result.legal[0]):
            raise RuntimeError("semantic greedy decode violated selection bounds")
        return result.sequences[0, : int(result.lengths[0])].tolist()


__all__ = [
    "DecodedActions",
    "InferenceSemanticFoundationPolicy",
    "PortableSemanticPolicy",
]
