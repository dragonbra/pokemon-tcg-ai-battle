"""Readable top-level assembly of the 0032 audited semantic policy."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..contracts.batch import DecisionBatch
from ..contracts.fields import EXPECTED_BATCH_KEYS
from ..domain.prototypes import PrototypeIndex
from .action_decoder import DecoderState, GreedyActions, ActionDecoder
from .config import ModelConfig
from .option_encoder import OptionEncoder
from .prototype_encoder import OfficialPrototypeEncoder
from .state_encoder import EncodedState, StateEncoder


class SemanticPolicy(nn.Module):
    """Prototype facts -> full state memory -> option retrieval -> ordered action."""

    expected_batch_keys = EXPECTED_BATCH_KEYS

    def __init__(
        self,
        config: ModelConfig,
        prototypes: PrototypeIndex,
        *,
        share_prototype_embeddings: bool = True,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.prototype_encoder = OfficialPrototypeEncoder(config, prototypes)
        self.state_encoder = StateEncoder(config, self.prototype_encoder)
        self.option_encoder = OptionEncoder(config, self.prototype_encoder)
        self.action_decoder = ActionDecoder(config)
        self.share_prototype_embeddings = share_prototype_embeddings

    @staticmethod
    def validate_batch(batch: DecisionBatch | Mapping[str, Tensor]) -> DecisionBatch:
        if isinstance(batch, DecisionBatch):
            return batch
        return DecisionBatch.from_mapping(batch)

    def encode_state(self, batch: DecisionBatch | Mapping[str, Tensor]) -> EncodedState:
        validated = self.validate_batch(batch)
        return self.state_encoder(validated, self.prototype_encoder.encode_all())

    def encode_options(self, batch: DecisionBatch, state: EncodedState) -> Tensor:
        return self.option_encoder(
            batch, state, self.prototype_encoder.encode_all()
        )

    def decode_next(
        self,
        batch: DecisionBatch,
        options: Tensor,
        decoder_state: DecoderState,
    ) -> Tensor:
        return self.action_decoder.logits(batch, options, decoder_state)

    def encode(
        self,
        batch: DecisionBatch | Mapping[str, Tensor],
    ) -> tuple[DecisionBatch, EncodedState, Tensor]:
        validated = self.validate_batch(batch)
        prototype_memory = (
            self.prototype_encoder.encode_all()
            if self.share_prototype_embeddings
            else self.prototype_encoder
        )
        state = self.state_encoder(validated, prototype_memory)
        options = self.option_encoder(validated, state, prototype_memory)
        return validated, state, options

    def forward(
        self,
        batch: DecisionBatch | Mapping[str, Tensor],
        selected_prefix: Tensor | None = None,
    ) -> Tensor:
        batch, state, options = self.encode(batch)
        decoder_state = self.action_decoder.initialize(batch, state.summary)
        decoder_state = self.action_decoder.consume_prefix(
            options,
            decoder_state,
            selected_prefix,
        )
        return self.decode_next(batch, options, decoder_state)

    def teacher_logits(self, batch: DecisionBatch | Mapping[str, Tensor]) -> Tensor:
        batch, state, options = self.encode(batch)
        decoder_state = self.action_decoder.initialize(batch, state.summary)
        option_keys = self.action_decoder.key(options)
        option_bias = self.action_decoder.option_bias(options).squeeze(-1)
        logits = []
        for step in range(batch.targets.shape[1]):
            logits.append(
                self.action_decoder.logits(
                    batch,
                    options,
                    decoder_state,
                    option_keys=option_keys,
                    option_bias=option_bias,
                )
            )
            decoder_state = self.action_decoder.consume(
                options,
                decoder_state,
                batch.targets[:, step],
            )
        return torch.stack(logits, dim=1)

    def deterministic_action_tensors(
        self,
        batch: DecisionBatch | Mapping[str, Tensor],
    ) -> GreedyActions:
        batch, state, options = self.encode(batch)
        return self.action_decoder.greedy(batch, options, state.summary)

    def greedy_action(self, batch: DecisionBatch | Mapping[str, Tensor]) -> list[int]:
        validated = self.validate_batch(batch)
        if validated.batch_size != 1:
            raise ValueError("greedy_action expects one decision")
        result = self.deterministic_action_tensors(validated)
        if not bool(result.legal[0]):
            raise RuntimeError("greedy decode violated selection bounds")
        return result.sequences[0, : int(result.lengths[0])].tolist()


__all__ = ["SemanticPolicy"]
