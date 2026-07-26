"""Centralized ordered full-action probability implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor


class ActionScorer(Protocol):
    value: Tensor
    def initial(self) -> Any: ...
    def logits(self, hidden: Any, chosen: Tensor) -> tuple[Tensor, Tensor]: ...
    def advance(self, hidden: Any, option_index: int) -> Any: ...


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    sequence: tuple[int, ...]
    log_prob: float
    entropy_sum: float
    mean_step_entropy: float
    value: float
    decision_count: int
    forced_terminal: bool
    terminal_log_prob: float
    terminal_entropy: float


def _validate(option_mask: Tensor, min_count: int, max_count: int) -> tuple[int, int]:
    if option_mask.ndim != 1 or option_mask.dtype is not torch.bool:
        raise ValueError("option_mask must be a one-dimensional bool tensor")
    legal = int(option_mask.sum().item())
    if min_count < 0 or max_count < min_count:
        raise ValueError("invalid min/max action bounds")
    return legal, min(max_count, legal)


def _step_distribution(scorer: ActionScorer, hidden: Any, chosen: Tensor, option_mask: Tensor, allow_stop: bool) -> tuple[Tensor, Tensor]:
    option_logits, stop_logit = scorer.logits(hidden, chosen)
    if option_logits.shape != option_mask.shape:
        raise ValueError("scorer option logits disagree with mask")
    masked = option_logits.masked_fill(~option_mask | chosen, -torch.inf)
    stop = stop_logit.reshape(1) if allow_stop else torch.full((1,), -torch.inf, dtype=masked.dtype, device=masked.device)
    logits = torch.cat((masked, stop))
    if not torch.isfinite(logits).any():
        raise RuntimeError("no legal option available before minCount")
    probabilities = torch.softmax(logits, dim=0)
    log_probabilities = torch.log_softmax(logits, dim=0)
    entropy = -(probabilities[torch.isfinite(log_probabilities)] * log_probabilities[torch.isfinite(log_probabilities)]).sum()
    return log_probabilities, entropy


def _result(sequence: Sequence[int], log_probs: list[float], entropies: list[float], scorer: ActionScorer, forced: bool, terminal_log_prob: float, terminal_entropy: float) -> ActionEvaluation:
    count = len(log_probs)
    total_entropy = sum(entropies)
    value = float(scorer.value.reshape(-1)[0].item())
    return ActionEvaluation(tuple(sequence), sum(log_probs), total_entropy, total_entropy / count if count else 0.0, value, count, forced, terminal_log_prob, terminal_entropy)


def evaluate_action(scorer: ActionScorer, sequence: Sequence[int], *, option_mask: Tensor, min_count: int, max_count: int) -> ActionEvaluation:
    legal, maximum = _validate(option_mask, min_count, max_count)
    values = tuple(sequence)
    if len(set(values)) != len(values):
        raise ValueError("action contains duplicate option indices")
    if not min_count <= len(values) <= maximum:
        raise ValueError("action length violates min/max bounds")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(option_mask) or not bool(option_mask[index]) for index in values):
        raise ValueError("action contains illegal option index")
    hidden = scorer.initial()
    chosen = torch.zeros_like(option_mask)
    log_probs: list[float] = []
    entropies: list[float] = []
    for step, index in enumerate(values):
        logs, entropy = _step_distribution(scorer, hidden, chosen, option_mask, step >= min_count)
        log_probs.append(float(logs[index].item()))
        entropies.append(float(entropy.item()))
        chosen[index] = True
        hidden = scorer.advance(hidden, index)
    if len(values) == maximum:
        return _result(values, log_probs, entropies, scorer, True, 0.0, 0.0)
    logs, entropy = _step_distribution(scorer, hidden, chosen, option_mask, True)
    stop_index = len(option_mask)
    terminal_log = float(logs[stop_index].item())
    terminal_entropy = float(entropy.item())
    log_probs.append(terminal_log)
    entropies.append(terminal_entropy)
    return _result(values, log_probs, entropies, scorer, False, terminal_log, terminal_entropy)


def sample_action(scorer: ActionScorer, *, option_mask: Tensor, min_count: int, max_count: int, deterministic: bool = False, generator: torch.Generator | None = None) -> ActionEvaluation:
    _, maximum = _validate(option_mask, min_count, max_count)
    hidden = scorer.initial()
    chosen = torch.zeros_like(option_mask)
    sequence: list[int] = []
    log_probs: list[float] = []
    entropies: list[float] = []
    for step in range(maximum):
        logs, entropy = _step_distribution(scorer, hidden, chosen, option_mask, step >= min_count)
        if deterministic:
            selected = int(logs.argmax().item())
        else:
            selected = int(torch.multinomial(logs.exp(), 1, generator=generator).item())
        log_probs.append(float(logs[selected].item()))
        entropies.append(float(entropy.item()))
        if selected == len(option_mask):
            return _result(sequence, log_probs, entropies, scorer, False, log_probs[-1], entropies[-1])
        sequence.append(selected)
        chosen[selected] = True
        hidden = scorer.advance(hidden, selected)
    if len(sequence) < min_count:
        raise RuntimeError("no legal option available before minCount")
    return _result(sequence, log_probs, entropies, scorer, True, 0.0, 0.0)


__all__ = ["ActionEvaluation", "ActionScorer", "evaluate_action", "sample_action"]
