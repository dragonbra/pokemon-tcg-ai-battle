"""Readable top-level assembly of the 0031 semantic policy."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ..contracts.batch import DecisionBatch
from ..contracts.fields import EXPECTED_BATCH_KEYS
from ..domain.prototypes import PrototypeIndex
from .action_decoder import DecoderState, GreedyActions, ActionDecoder
from .config import ModelConfig
from .option_encoder import OptionEncoder
from .prototype_encoder import OfficialPrototypeEncoder, PrototypeEmbeddings
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
        self._prototype_cache: PrototypeEmbeddings | None = None
        self._prototype_cache_counts: Counter[str] = Counter()

    def _prototype_parameters_frozen(self) -> bool:
        return not any(
            parameter.requires_grad
            for parameter in self.prototype_encoder.parameters()
        )

    def clear_prototype_cache(self, reason: str) -> None:
        """Discard derived runtime tensors without changing checkpoint state."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("prototype cache invalidation requires a reason")
        if self._prototype_cache is not None:
            self._prototype_cache = None
            self._prototype_cache_counts["invalidations"] += 1
            self._prototype_cache_counts[f"invalidation/{reason}"] += 1

    def prepare_prototype_cache(self) -> PrototypeEmbeddings:
        """Build one detached cache for frozen inference/prototype parameters."""
        if self._prototype_cache is not None:
            self._prototype_cache_counts["hits"] += 1
            return self._prototype_cache
        if self.training and not self._prototype_parameters_frozen():
            raise RuntimeError(
                "trainable prototype parameters require live encode_all autograd"
            )
        with torch.no_grad():
            memory = self.prototype_encoder.encode_all()
        self._prototype_cache = PrototypeEmbeddings(
            *(tensor.detach() for tensor in (
                memory.cards,
                memory.attacks,
                memory.skills,
                memory.effects,
            ))
        )
        self._prototype_cache_counts["builds"] += 1
        return self._prototype_cache

    def prototype_memory(
        self,
    ) -> PrototypeEmbeddings | OfficialPrototypeEncoder:
        """Return cached static memory when semantically safe, else live memory."""
        if not self.share_prototype_embeddings:
            return self.prototype_encoder
        if self.training and not self._prototype_parameters_frozen():
            self.clear_prototype_cache("trainable_prototypes")
            return self.prototype_encoder.encode_all()
        return self.prepare_prototype_cache()

    def prototype_cache_stats(self) -> dict[str, int]:
        return {
            name: int(self._prototype_cache_counts[name])
            for name in ("builds", "hits", "invalidations")
        }

    def _apply(self, fn: Any, recurse: bool = True) -> "SemanticPolicy":
        self.clear_prototype_cache("module_apply")
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, *args: Any, **kwargs: Any):
        self.clear_prototype_cache("load_state_dict")
        return super().load_state_dict(*args, **kwargs)

    def train(self, mode: bool = True) -> "SemanticPolicy":
        if mode and not self._prototype_parameters_frozen():
            self.clear_prototype_cache("train_mode")
        return super().train(mode)

    @staticmethod
    def validate_batch(batch: DecisionBatch | Mapping[str, Tensor]) -> DecisionBatch:
        if isinstance(batch, DecisionBatch):
            return batch
        return DecisionBatch.from_mapping(batch)

    def encode_state(self, batch: DecisionBatch | Mapping[str, Tensor]) -> EncodedState:
        validated = self.validate_batch(batch)
        return self.state_encoder(validated, self.prototype_memory())

    def encode_options(self, batch: DecisionBatch, state: EncodedState) -> Tensor:
        return self.option_encoder(
            batch, state, self.prototype_memory()
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
        prototype_memory = self.prototype_memory()
        state = self.state_encoder(validated, prototype_memory)
        options = self.option_encoder(validated, state, prototype_memory)
        return validated, state, options

    def forward(
        self,
        batch: DecisionBatch | Mapping[str, Tensor],
        selected_prefix: Tensor | None = None,
        *,
        teacher_forcing: bool = False,
    ) -> Tensor:
        if teacher_forcing:
            if selected_prefix is not None:
                raise ValueError("teacher forcing does not accept a selected prefix")
            return self.teacher_logits(batch)
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
