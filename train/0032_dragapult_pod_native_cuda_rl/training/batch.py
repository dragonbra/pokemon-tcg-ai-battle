"""Immutable complete-Episode PPO batches for resident 0032 rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


FEATURE_NAMES = (
    "global_cat",
    "global_num",
    "entity_cat",
    "entity_num",
    "entity_parent",
    "entity_mask",
    "option_cat",
    "option_num",
    "option_equiv",
    "option_mask",
    "min_count",
    "max_count",
)


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    features: Mapping[str, Tensor]
    sequences: Tensor
    lengths: Tensor
    old_logprob: Tensor
    old_value: Tensor
    gae_return: Tensor
    advantage: Tensor
    episode_weight: Tensor
    episode_id: Tensor
    source_policy_update: int
    completed_episodes: int
    discarded_incomplete_decisions: int

    @property
    def decisions(self) -> int:
        return int(self.lengths.shape[0])


def generalized_advantage_estimate(
    values: Tensor,
    rewards: Tensor,
    dones: Tensor,
    *,
    bootstrap_value: float = 0.0,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    if values.ndim != 1 or rewards.shape != values.shape or dones.shape != values.shape:
        raise ValueError("values/rewards/dones must be aligned rank-one tensors")
    advantages = torch.zeros_like(values)
    next_value = torch.as_tensor(bootstrap_value, dtype=values.dtype, device=values.device)
    next_advantage = torch.zeros_like(next_value)
    for index in range(values.numel() - 1, -1, -1):
        continuation = (~dones[index]).to(values.dtype)
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        next_advantage = delta + gamma * gae_lambda * next_advantage * continuation
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values


def result_reward(result: int) -> float:
    if result == 1:
        return 1.0
    if result == 2:
        return -1.0
    if result == 3:
        return 0.0
    raise ValueError(f"not a terminal official game result: {result}")


def prepare_complete_episodes(
    storage: Mapping[str, Tensor],
    *,
    source_policy_update: int,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> PreparedBatch:
    episode_ids = storage["episode_id"].reshape(-1).long()
    focal = storage["focal_mask"].reshape(-1).bool()
    terminal_ids = storage["terminal_episode_id"].reshape(-1).long()
    terminal_results = storage["terminal_result"].reshape(-1).long()
    completed = {
        int(identity): result_reward(int(result))
        for identity, result in zip(terminal_ids.tolist(), terminal_results.tolist())
        if identity >= 0
    }
    if not completed:
        raise ValueError("rollout window contains no complete Episodes")
    completed_ids = torch.tensor(sorted(completed), dtype=torch.long)
    keep = focal & torch.isin(episode_ids, completed_ids)
    if not keep.any():
        raise ValueError("complete Episodes contain no focal decisions")
    flat_indices = keep.nonzero(as_tuple=False).squeeze(1)
    selected_episode_ids = episode_ids.index_select(0, flat_indices)
    old_value = storage["value"].reshape(-1).index_select(0, flat_indices).float()
    raw_advantage = torch.empty_like(old_value)
    returns = torch.empty_like(old_value)
    weights = torch.empty_like(old_value)
    for episode_id in sorted(completed):
        indices = (selected_episode_ids == episode_id).nonzero(as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            continue
        rewards = torch.zeros(indices.numel(), dtype=torch.float32)
        rewards[-1] = completed[episode_id]
        dones = torch.zeros(indices.numel(), dtype=torch.bool)
        dones[-1] = True
        advantages, episode_returns = generalized_advantage_estimate(
            old_value.index_select(0, indices),
            rewards,
            dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        raw_advantage[indices] = advantages
        returns[indices] = episode_returns
        weights[indices] = 1.0 / indices.numel()
    weighted_mean = (raw_advantage * weights).sum() / weights.sum()
    weighted_variance = ((raw_advantage - weighted_mean).square() * weights).sum() / weights.sum()
    advantage = (raw_advantage - weighted_mean) / weighted_variance.sqrt().clamp_min(1e-6)

    steps, lanes = storage["focal_mask"].shape
    features = {
        name: storage[name].reshape(steps * lanes, *storage[name].shape[2:]).index_select(
            0, flat_indices
        )
        for name in FEATURE_NAMES
    }
    return PreparedBatch(
        features=features,
        sequences=storage["sequences"].reshape(steps * lanes, -1).index_select(0, flat_indices),
        lengths=storage["lengths"].reshape(-1).index_select(0, flat_indices),
        old_logprob=storage["logprob"].reshape(-1).index_select(0, flat_indices).float(),
        old_value=old_value,
        gae_return=returns,
        advantage=advantage,
        episode_weight=weights,
        episode_id=selected_episode_ids,
        source_policy_update=source_policy_update,
        completed_episodes=len(set(selected_episode_ids.tolist())),
        discarded_incomplete_decisions=int(focal.sum().item() - keep.sum().item()),
    )


__all__ = [
    "FEATURE_NAMES",
    "PreparedBatch",
    "generalized_advantage_estimate",
    "prepare_complete_episodes",
    "result_reward",
]
