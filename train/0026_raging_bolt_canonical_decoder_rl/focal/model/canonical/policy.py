"""Top-level canonical semantic policy data flow."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...features.canonical.schema import ACTOR_KEYS
from ...features.prototypes import PrototypeIndex
from .config import CanonicalModelConfig
from .decoder import GreedyActions, OrderedOptionDecoder
from .options import CanonicalOptionEncoder
from .prototypes import PrototypeBank
from .state import CanonicalStateEncoder


MASK_KEYS = frozenset(
    {
        "card_mask",
        "resource_mask",
        "event_mask",
        "option_mask",
        "option_skill_mask",
        "option_effect_mask",
    }
)
EXPECTED_BATCH_KEYS = ACTOR_KEYS | MASK_KEYS | {"targets"}


class CanonicalSemanticPolicy(nn.Module):
    """Encode every canonical field once, then score ordered legal options."""

    def __init__(self, config: CanonicalModelConfig, prototypes: PrototypeIndex):
        super().__init__()
        config.validate()
        self.config = config
        self.prototypes = PrototypeBank(config, prototypes)
        self.state_encoder = CanonicalStateEncoder(config, self.prototypes)
        self.option_encoder = CanonicalOptionEncoder(config, self.prototypes)
        self.action_decoder = OrderedOptionDecoder(config)

    @staticmethod
    def _validate_batch(batch: dict[str, Tensor]) -> None:
        keys = set(batch)
        if keys != EXPECTED_BATCH_KEYS:
            missing = sorted(EXPECTED_BATCH_KEYS - keys)
            extra = sorted(keys - EXPECTED_BATCH_KEYS)
            raise ValueError(f"canonical batch contract mismatch; missing={missing}, extra={extra}")

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        self._validate_batch(batch)
        state = self.state_encoder(batch)
        options = self.option_encoder(batch, state)
        return state.summary, options

    def forward(self, batch: dict[str, Tensor], selected_prefix: Tensor | None = None) -> Tensor:
        summary, options = self.encode(batch)
        hidden = self.action_decoder.initial_hidden(summary)
        available = batch["option_mask"].clone()
        selected_count = torch.zeros(options.size(0), dtype=torch.long, device=options.device)
        if selected_prefix is not None:
            for step in range(selected_prefix.size(1)):
                hidden, available, selected_count = self.action_decoder.consume(
                    options, hidden, available, selected_count, selected_prefix[:, step]
                )
        return self.action_decoder.logits(
            batch, options, hidden, available, selected_count
        )

    def teacher_logits(self, batch: dict[str, Tensor]) -> Tensor:
        summary, options = self.encode(batch)
        hidden = self.action_decoder.initial_hidden(summary)
        available = batch["option_mask"].clone()
        selected_count = torch.zeros(options.size(0), dtype=torch.long, device=options.device)
        logits = []
        for step in range(batch["targets"].size(1)):
            logits.append(
                self.action_decoder.logits(
                    batch, options, hidden, available, selected_count
                )
            )
            hidden, available, selected_count = self.action_decoder.consume(
                options,
                hidden,
                available,
                selected_count,
                batch["targets"][:, step],
            )
        return torch.stack(logits, dim=1)

    def deterministic_action_tensors(self, batch: dict[str, Tensor]) -> GreedyActions:
        summary, options = self.encode(batch)
        return self.action_decoder.greedy(batch, options, summary)

    def greedy_action(self, batch: dict[str, Tensor]) -> list[int]:
        if batch["option_mask"].size(0) != 1:
            raise ValueError("greedy_action expects one canonical decision")
        result = self.deterministic_action_tensors(batch)
        if not bool(result.legal[0]):
            raise RuntimeError("canonical greedy decode violated selection bounds")
        return result.sequences[0, : int(result.lengths[0])].tolist()


__all__ = ["CanonicalSemanticPolicy", "EXPECTED_BATCH_KEYS"]
