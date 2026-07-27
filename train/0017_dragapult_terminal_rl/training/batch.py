from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..rollout.protocol import Episode


@dataclass(frozen=True)
class PreparedBatch:
    features: tuple[dict[str, Tensor], ...]
    sequences: Tensor
    lengths: Tensor
    stopped: Tensor
    old_log_prob: Tensor
    old_value: Tensor
    terminal_return: Tensor
    gae_return: Tensor
    advantage: Tensor
    episode_weight: Tensor
    episode_index: Tensor
    turns: Tensor
    candidate_first: Tensor
    opponents: tuple[str, ...]

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
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def prepare_episodes(
    episodes: list[Episode], *, gamma: float = 1.0, gae_lambda: float = 0.95
) -> PreparedBatch:
    valid = [episode for episode in episodes if episode.valid and episode.reward is not None]
    if not valid:
        raise ValueError("no valid terminal episodes to prepare")
    features: list[dict[str, Tensor]] = []
    actions: list[tuple[int, ...]] = []
    stopped: list[bool] = []
    old_log_prob: list[float] = []
    old_value: list[float] = []
    terminal_return: list[float] = []
    gae_return: list[float] = []
    advantage: list[float] = []
    episode_weight: list[float] = []
    episode_index: list[int] = []
    turns: list[int] = []
    candidate_first: list[bool] = []
    opponents: list[str] = []
    for ep_index, episode in enumerate(valid):
        if not episode.decisions:
            continue
        values = [decision.old_value for decision in episode.decisions]
        ep_advantage, ep_returns = _episode_gae(
            values, float(episode.reward), gamma=gamma, gae_lambda=gae_lambda
        )
        weight = 1.0 / len(episode.decisions)
        for decision, item_advantage, item_return in zip(
            episode.decisions, ep_advantage, ep_returns, strict=True
        ):
            features.append(decision.features)
            actions.append(decision.action)
            stopped.append(decision.stopped)
            old_log_prob.append(decision.old_log_prob)
            old_value.append(decision.old_value)
            terminal_return.append(float(episode.reward))
            gae_return.append(item_return)
            advantage.append(item_advantage)
            episode_weight.append(weight)
            episode_index.append(ep_index)
            turns.append(decision.turn)
            candidate_first.append(episode.candidate_first)
            opponents.append(episode.opponent)
    if not features:
        raise ValueError("valid episodes contain no candidate decisions")
    width = max(1, max(len(action) for action in actions))
    sequences = torch.full((len(actions), width), -1, dtype=torch.long)
    lengths = torch.tensor([len(action) for action in actions], dtype=torch.long)
    for index, action in enumerate(actions):
        if action:
            sequences[index, : len(action)] = torch.tensor(action, dtype=torch.long)
    raw_advantage = torch.tensor(advantage, dtype=torch.float32)
    weights = torch.tensor(episode_weight, dtype=torch.float32)
    mean = (raw_advantage * weights).sum() / weights.sum()
    variance = ((raw_advantage - mean).square() * weights).sum() / weights.sum()
    normalized_advantage = (raw_advantage - mean) / variance.sqrt().clamp_min(1e-6)
    return PreparedBatch(
        features=tuple(features),
        sequences=sequences,
        lengths=lengths,
        stopped=torch.tensor(stopped, dtype=torch.bool),
        old_log_prob=torch.tensor(old_log_prob, dtype=torch.float32),
        old_value=torch.tensor(old_value, dtype=torch.float32),
        terminal_return=torch.tensor(terminal_return, dtype=torch.float32),
        gae_return=torch.tensor(gae_return, dtype=torch.float32),
        advantage=normalized_advantage,
        episode_weight=weights,
        episode_index=torch.tensor(episode_index, dtype=torch.long),
        turns=torch.tensor(turns, dtype=torch.long),
        candidate_first=torch.tensor(candidate_first, dtype=torch.bool),
        opponents=tuple(opponents),
    )


__all__ = ["PreparedBatch", "prepare_episodes"]
