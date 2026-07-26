"""Parameter-identical 0010 control model with parameter-free batched validation helpers."""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor

_legacy = importlib.import_module("train.project_0010_alakazam_sota_model.model")

IDOnlyConfig = _legacy.IDOnlyConfig
IDOnlyCodec = _legacy.IDOnlyCodec
collate_id_only = _legacy.collate_id_only


@dataclass(frozen=True, slots=True)
class BatchedActionTensorResult:
    sequences: Tensor
    lengths: Tensor
    forced_terminal: Tensor
    legal: Tensor


class Faithful0010PointerPolicy(_legacy.IDOnlyPointerPolicy):
    """The original module graph; extra methods only remove repeated validation encoding."""

    def teacher_logits_from_encoding(
        self,
        batch: dict[str, Tensor],
        encoded: tuple[Tensor, Tensor],
    ) -> Tensor:
        state, options = encoded
        batch_size, option_count, _ = options.shape
        keys = self.pointer_key(options)
        hidden = torch.tanh(self.decoder_init(state))
        chosen = torch.zeros(
            (batch_size, option_count), dtype=torch.bool, device=options.device
        )
        outputs: list[Tensor] = []
        rows = torch.arange(batch_size, device=options.device)
        for step in range(batch["targets"].size(1)):
            pointer = (
                self.pointer_query(hidden).unsqueeze(1) * keys
            ).sum(-1) / math.sqrt(self.config.d_model)
            pointer = pointer + self.option_bias(options).squeeze(-1)
            pointer = pointer.masked_fill(
                ~batch["option_mask"] | chosen, torch.finfo(pointer.dtype).min
            )
            stop = self.stop(hidden)
            stop = stop.masked_fill(
                (step < batch["min_count"]).unsqueeze(-1),
                torch.finfo(stop.dtype).min,
            )
            outputs.append(torch.cat((pointer, stop), dim=1))
            target = batch["targets"][:, step]
            valid = (target >= 0) & (target < option_count)
            safe = target.clamp(0, option_count - 1)
            selected = options[rows, safe]
            hidden = torch.where(valid.unsqueeze(-1), self.decoder(selected, hidden), hidden)
            chosen.scatter_(
                1,
                safe.unsqueeze(1),
                chosen.gather(1, safe.unsqueeze(1)) | valid.unsqueeze(1),
            )
        return torch.stack(outputs, dim=1)

    def deterministic_action_tensors(
        self,
        batch: dict[str, Tensor],
        *,
        encoded: tuple[Tensor, Tensor] | None = None,
    ) -> BatchedActionTensorResult:
        state, options = self.encode(batch) if encoded is None else encoded
        batch_size, option_count, _ = options.shape
        maximum_steps = min(int(batch["max_count"].max()), self.config.max_action_steps)
        sequences = torch.full(
            (batch_size, maximum_steps), -1, dtype=torch.long, device=options.device
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
        maximum = batch["max_count"].clamp_max(self.config.max_action_steps)
        hidden = torch.tanh(self.decoder_init(state))
        keys = self.pointer_key(options)
        chosen = torch.zeros(
            (batch_size, option_count), dtype=torch.bool, device=options.device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=options.device)
        legal = torch.ones(batch_size, dtype=torch.bool, device=options.device)
        rows = torch.arange(batch_size, device=options.device)
        for step in range(maximum_steps):
            reached_maximum = ~finished & (lengths >= maximum)
            finished |= reached_maximum
            pointer = (
                self.pointer_query(hidden).unsqueeze(1) * keys
            ).sum(-1) / math.sqrt(self.config.d_model)
            pointer = pointer + self.option_bias(options).squeeze(-1)
            pointer = pointer.masked_fill(
                ~batch["option_mask"] | chosen, torch.finfo(pointer.dtype).min
            )
            best_score, best_option = pointer.max(1)
            stop_score = self.stop(hidden).squeeze(1)
            can_stop = lengths >= batch["min_count"]
            stopping = ~finished & can_stop & (stop_score >= best_score)
            active = ~finished & ~stopping
            selected_is_legal = batch["option_mask"][rows, best_option]
            legal &= ~active | selected_is_legal
            active &= selected_is_legal
            sequences[:, step] = torch.where(active, best_option, -1)
            selected = options[rows, best_option]
            advanced = self.decoder(selected, hidden)
            hidden = torch.where(active.unsqueeze(-1), advanced, hidden)
            chosen.scatter_(
                1,
                best_option.unsqueeze(1),
                chosen.gather(1, best_option.unsqueeze(1)) | active.unsqueeze(1),
            )
            lengths += active
            finished |= stopping | ~selected_is_legal
        forced = lengths >= maximum
        legal &= lengths >= batch["min_count"]
        return BatchedActionTensorResult(sequences, lengths, forced, legal)


def parameter_count(model: Faithful0010PointerPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "BatchedActionTensorResult",
    "Faithful0010PointerPolicy",
    "IDOnlyCodec",
    "IDOnlyConfig",
    "collate_id_only",
    "parameter_count",
]
