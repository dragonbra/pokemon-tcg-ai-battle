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

    @property
    def decisions(self) -> int:
        return len(self.features)


def _episode_gae(
    values: list[float], reward: float, *, gamma: float, gae_lambda: float
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(values)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(values) - 1, -1, -1):
        immediate = reward if index == len(values) - 1 else 0.0
        delta = immediate + gamma * next_value - values[index]
        current = delta + gamma * gae_lambda * next_advantage
        advantages[index] = current
        next_advantage = current
        next_value = values[index]
    return advantages, [advantage + value for advantage, value in zip(advantages, values)]


def prepare_episodes(
    episodes: list[EpisodeTrajectory], *, gamma: float = 1.0, gae_lambda: float = 0.95
) -> PreparedBatch:
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
    for episode_index, episode in enumerate(valid):
        if not episode.decisions:
            continue
        ep_values = [decision.value for decision in episode.decisions]
        ep_advantages, ep_returns = _episode_gae(
            ep_values, float(episode.reward), gamma=gamma, gae_lambda=gae_lambda
        )
        weight = 1.0 / len(episode.decisions)
        for decision, advantage, gae_return in zip(
            episode.decisions, ep_advantages, ep_returns, strict=True
        ):
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
            turns.append(episode.turns)
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
    }


__all__ = ["PreparedBatch", "prepare_episodes", "training_batch_metrics"]
