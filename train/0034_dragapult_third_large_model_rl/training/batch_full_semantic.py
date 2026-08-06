"""Convert focal-only trajectories into episode-balanced GAE batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..rollout.protocol import EpisodeTrajectory


@dataclass(frozen=True)
class PreparedBatch:
    features: tuple[dict[str, Tensor], ...]
    sequences: Tensor
    lengths: Tensor
    stopped: Tensor
    rollout_log_prob: Tensor
    old_value: Tensor
    terminal_return: Tensor
    gae_return: Tensor
    advantage: Tensor
    episode_weight: Tensor
    episode_index: Tensor
    turns: Tensor
    candidate_first: Tensor
    opponents: tuple[str, ...]
    source_policy_update: int
    gamma: float
    gae_lambda: float
    credit_clock: str
    loss_weighting: str
    semantic_boundaries: int
    first_decision_credit_mean: float

    @property
    def decisions(self) -> int:
        return len(self.features)


def _episode_gae(
    values: list[float],
    reward: float,
    turns: list[int | None],
    *,
    gamma: float,
    gae_lambda: float,
    credit_clock: str,
) -> tuple[list[float], list[float], int, float]:
    if not 0.0 < gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in (0, 1]")
    if credit_clock not in {"selection", "turn"}:
        raise ValueError(f"unsupported credit clock: {credit_clock}")
    if len(turns) != len(values):
        raise ValueError("turn metadata length does not match values")
    if credit_clock == "turn" and any(
        isinstance(turn, bool) or not isinstance(turn, int) or turn < 0 for turn in turns
    ):
        raise ValueError("turn clock requires nonnegative turn metadata")
    advantages = [0.0] * len(values)
    next_value = 0.0
    next_advantage = 0.0
    boundaries = 0
    for index in range(len(values) - 1, -1, -1):
        immediate = reward if index == len(values) - 1 else 0.0
        delta = immediate + gamma * next_value - values[index]
        if index == len(values) - 1:
            transition_lambda = 1.0
        elif credit_clock == "selection":
            transition_lambda = gae_lambda
            boundaries += 1
        else:
            transition_lambda = gae_lambda if turns[index] != turns[index + 1] else 1.0
            boundaries += int(turns[index] != turns[index + 1])
        current = delta + gamma * transition_lambda * next_advantage
        advantages[index] = current
        next_advantage = current
        next_value = values[index]
    first_credit = gae_lambda**boundaries if values else 0.0
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns, boundaries, first_credit


def prepare_episodes(
    episodes: list[EpisodeTrajectory],
    *,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    credit_clock: str = "selection",
    loss_weighting: str = "episode_equal_decisions",
) -> PreparedBatch:
    if loss_weighting not in {"episode_equal_decisions", "episode_equal_turns"}:
        raise ValueError(f"unsupported loss_weighting: {loss_weighting}")
    valid = [episode for episode in episodes if episode.valid and episode.reward is not None]
    if not valid:
        raise ValueError("no valid terminal episodes to prepare")
    updates = {
        decision.policy_update for episode in valid for decision in episode.decisions
    }
    if len(updates) != 1:
        raise ValueError("PPO batch must contain exactly one source_policy_update")
    features: list[dict[str, Tensor]] = []
    actions: list[tuple[int, ...]] = []
    stopped: list[bool] = []
    log_prob: list[float] = []
    values: list[float] = []
    terminal: list[float] = []
    returns: list[float] = []
    advantages: list[float] = []
    weights: list[float] = []
    episode_indices: list[int] = []
    turns: list[int] = []
    candidate_first: list[bool] = []
    opponents: list[str] = []
    semantic_boundaries = 0
    first_decision_credits: list[float] = []
    for episode_index, episode in enumerate(valid):
        if not episode.decisions:
            continue
        ep_values = [decision.value for decision in episode.decisions]
        ep_advantages, ep_returns, ep_boundaries, first_credit = _episode_gae(
            ep_values,
            float(episode.reward),
            [decision.turn for decision in episode.decisions],
            gamma=gamma,
            gae_lambda=gae_lambda,
            credit_clock=credit_clock,
        )
        semantic_boundaries += ep_boundaries
        first_decision_credits.append(first_credit)
        turn_counts: dict[int, int] = {}
        if loss_weighting == "episode_equal_turns":
            for decision in episode.decisions:
                if (
                    isinstance(decision.turn, bool)
                    or not isinstance(decision.turn, int)
                    or decision.turn < 0
                ):
                    raise ValueError("turn-equal loss_weighting requires turn metadata")
                turn_counts[decision.turn] = turn_counts.get(decision.turn, 0) + 1
        for decision, advantage, gae_return in zip(
            episode.decisions, ep_advantages, ep_returns, strict=True
        ):
            if loss_weighting == "episode_equal_turns":
                weight = 1.0 / (len(turn_counts) * turn_counts[decision.turn])
            else:
                weight = 1.0 / len(episode.decisions)
            features.append(decision.features)
            actions.append(decision.indices)
            stopped.append(decision.stopped)
            log_prob.append(decision.log_prob)
            values.append(decision.value)
            terminal.append(float(episode.reward))
            returns.append(gae_return)
            advantages.append(advantage)
            weights.append(weight)
            episode_indices.append(episode_index)
            turns.append(decision.turn if decision.turn is not None else -1)
            candidate_first.append(episode.job.focal_first)
            opponents.append(episode.job.opponent_id)
    if not features:
        raise ValueError("valid episodes contain no focal decisions")
    width = max(1, max(len(action) for action in actions))
    sequences = torch.full((len(actions), width), -1, dtype=torch.long)
    lengths = torch.tensor([len(action) for action in actions], dtype=torch.long)
    for index, action in enumerate(actions):
        if action:
            sequences[index, : len(action)] = torch.tensor(action, dtype=torch.long)
    raw_advantage = torch.tensor(advantages, dtype=torch.float32)
    episode_weight = torch.tensor(weights, dtype=torch.float32)
    mean = (raw_advantage * episode_weight).sum() / episode_weight.sum()
    variance = (
        (raw_advantage - mean).square() * episode_weight
    ).sum() / episode_weight.sum()
    normalized = (raw_advantage - mean) / variance.sqrt().clamp_min(1e-6)
    return PreparedBatch(
        features=tuple(features),
        sequences=sequences,
        lengths=lengths,
        stopped=torch.tensor(stopped, dtype=torch.bool),
        rollout_log_prob=torch.tensor(log_prob, dtype=torch.float32),
        old_value=torch.tensor(values, dtype=torch.float32),
        terminal_return=torch.tensor(terminal, dtype=torch.float32),
        gae_return=torch.tensor(returns, dtype=torch.float32),
        advantage=normalized,
        episode_weight=episode_weight,
        episode_index=torch.tensor(episode_indices, dtype=torch.long),
        turns=torch.tensor(turns, dtype=torch.long),
        candidate_first=torch.tensor(candidate_first, dtype=torch.bool),
        opponents=tuple(opponents),
        source_policy_update=next(iter(updates)),
        gamma=gamma,
        gae_lambda=gae_lambda,
        credit_clock=credit_clock,
        loss_weighting=loss_weighting,
        semantic_boundaries=semantic_boundaries,
        first_decision_credit_mean=sum(first_decision_credits) / len(first_decision_credits),
    )


def training_batch_metrics(batch: PreparedBatch) -> dict[str, float]:
    weights = batch.episode_weight / batch.episode_weight.sum()

    def moments(values: Tensor) -> tuple[Tensor, Tensor]:
        mean = (values * weights).sum()
        return mean, ((values - mean).square() * weights).sum()

    advantage_mean, advantage_variance = moments(batch.advantage)
    return_mean, return_variance = moments(batch.gae_return)
    value_mean, value_variance = moments(batch.old_value)
    residual = batch.gae_return - batch.old_value
    residual_mean = (residual * weights).sum()
    residual_variance = ((residual - residual_mean).square() * weights).sum()
    explained = torch.where(
        return_variance > 1e-8,
        1.0 - residual_variance / return_variance,
        torch.zeros_like(return_variance),
    )
    return {
        "ppo/advantage_mean": float(advantage_mean),
        "ppo/advantage_std": float(advantage_variance.sqrt()),
        "ppo/return_mean": float(return_mean),
        "ppo/return_std": float(return_variance.sqrt()),
        "ppo/old_value_mean": float(value_mean),
        "ppo/old_value_std": float(value_variance.sqrt()),
        "ppo/explained_variance": float(explained),
        "ppo/semantic_boundaries": float(batch.semantic_boundaries),
        "ppo/semantic_boundaries_per_episode": float(
            batch.semantic_boundaries / max(1, len(set(batch.episode_index.tolist())))
        ),
        "ppo/first_decision_terminal_credit_mean": batch.first_decision_credit_mean,
        "ppo/credit_clock_turn": float(batch.credit_clock == "turn"),
        "ppo/loss_weighting_turn_equal": float(
            batch.loss_weighting == "episode_equal_turns"
        ),
    }


__all__ = ["PreparedBatch", "prepare_episodes", "training_batch_metrics"]
