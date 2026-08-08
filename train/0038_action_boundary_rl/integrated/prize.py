"""Prize auxiliary objective kept strictly separate from terminal win Value."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from ..rollout.protocol import PolicyTransition

PrizeMode = Literal["off", "directional", "terminal_neutral"]


class PrizeAuxHead(nn.Module):
    """Small sidecar over an otherwise unused latent Value query."""

    query_index = 3

    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, decoded_queries: Tensor) -> Tensor:
        if decoded_queries.ndim != 3 or decoded_queries.shape[1] <= self.query_index:
            raise ValueError("PrizeAuxHead requires latent query index 3")
        return self.network(decoded_queries[:, self.query_index]).squeeze(-1)


@dataclass(frozen=True, slots=True)
class PrizeTargets:
    rewards: Tensor
    advantages: Tensor
    returns: Tensor
    own_turn_discounts: Tensor


def transition_prize_rewards(
    transitions: list[PolicyTransition], *, mode: PrizeMode, scale: float = 1.0 / 24.0
) -> Tensor:
    """Read public prize deltas from boundary metadata without touching V_win reward."""
    if mode not in {"off", "directional", "terminal_neutral"}:
        raise ValueError(f"unknown prize auxiliary mode: {mode}")
    values = torch.tensor([
        (float(item.metadata.get("focal_prizes_taken", 0))
         - float(item.metadata.get("opponent_prizes_taken", 0))) * scale
        for item in transitions
    ], dtype=torch.float32)
    if mode == "off":
        return torch.zeros_like(values)
    if mode == "terminal_neutral" and values.numel():
        terminal = next((i for i in range(len(transitions) - 1, -1, -1)
                         if transitions[i].done), len(transitions) - 1)
        values[terminal] -= values.sum()
    return values


def prize_gae(
    transitions: list[PolicyTransition],
    predictions: Tensor,
    *,
    mode: PrizeMode,
    gamma_prize: float = 0.97,
    lambda_prize: float = 0.97,
    scale: float = 1.0 / 24.0,
) -> PrizeTargets:
    """GAE on an explicit own-turn clock, never on callback/UI count."""
    if not transitions:
        raise ValueError("prize GAE requires transitions")
    if predictions.shape != (len(transitions),):
        raise ValueError("one prize prediction is required per transition")
    if not 0.0 < gamma_prize <= 1.0 or not 0.0 < lambda_prize <= 1.0:
        raise ValueError("prize gamma/lambda must be in (0, 1]")
    rewards = transition_prize_rewards(transitions, mode=mode, scale=scale).to(predictions.device)
    turns = [int(item.metadata.get("own_turn_index", index))
             for index, item in enumerate(transitions)]
    discounts = torch.ones(len(transitions), dtype=predictions.dtype, device=predictions.device)
    advantages = torch.zeros_like(predictions)
    continuation = predictions.new_zeros(())
    for index in range(len(transitions) - 1, -1, -1):
        if index + 1 < len(transitions):
            elapsed = max(0, turns[index + 1] - turns[index])
            discounts[index] = gamma_prize ** elapsed
            bootstrap = predictions[index + 1].detach()
        else:
            discounts[index] = 0.0 if transitions[index].done else gamma_prize
            bootstrap = predictions.new_zeros(())
        delta = rewards[index] + discounts[index] * bootstrap - predictions[index]
        advantages[index] = delta + (
            predictions.new_zeros(()) if transitions[index].done
            else discounts[index] * lambda_prize * continuation
        )
        continuation = advantages[index]
    return PrizeTargets(rewards, advantages, advantages + predictions, discounts)


def normalize_advantage(values: Tensor, *, eps: float = 1e-8) -> Tensor:
    if values.numel() < 2:
        return values - values.mean()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(eps)


def combine_actor_advantages(win: Tensor, prize: Tensor, *, alpha_prize: float) -> Tensor:
    """Normalize each objective separately before forming the actor advantage."""
    if win.shape != prize.shape:
        raise ValueError("win and prize advantages must have identical shape")
    return normalize_advantage(win) + alpha_prize * normalize_advantage(prize)


@dataclass(frozen=True, slots=True)
class GradientAlignment:
    win_norm: float
    prize_norm: float
    cosine: float
    prize_to_win_ratio: float


def gradient_alignment(
    win_loss: Tensor, prize_loss: Tensor, parameters: Iterable[nn.Parameter]
) -> GradientAlignment:
    """Periodic fixed-minibatch diagnostic; never called in every training minibatch."""
    params = tuple(item for item in parameters if item.requires_grad)
    win = torch.autograd.grad(win_loss, params, retain_graph=True, allow_unused=True)
    shaped = torch.autograd.grad(prize_loss, params, retain_graph=True, allow_unused=True)
    win_flat = torch.cat([g.reshape(-1) for g in win if g is not None])
    prize_flat = torch.cat([g.reshape(-1) for g in shaped if g is not None])
    win_norm = win_flat.norm()
    prize_norm = prize_flat.norm()
    cosine = torch.nn.functional.cosine_similarity(win_flat, prize_flat, dim=0)
    return GradientAlignment(
        float(win_norm), float(prize_norm), float(cosine),
        float(prize_norm / win_norm.clamp_min(1e-12)),
    )


__all__ = [
    "GradientAlignment", "PrizeAuxHead", "PrizeMode", "PrizeTargets",
    "combine_actor_advantages", "gradient_alignment", "normalize_advantage",
    "prize_gae", "transition_prize_rewards",
]
