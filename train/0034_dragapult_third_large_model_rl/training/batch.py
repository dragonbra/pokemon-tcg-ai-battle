"""Immutable complete-Episode PPO batches for resident 0034 rollouts."""

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
    episode_grid = storage["episode_id"].long()
    focal_grid = storage["focal_mask"].bool()
    value_grid = storage["value"].float()
    if episode_grid.ndim != 2 or focal_grid.shape != episode_grid.shape:
        raise ValueError("episode_id and focal_mask must have shape [steps, lanes]")
    if value_grid.shape != episode_grid.shape:
        raise ValueError("value must align with episode_id")
    device = episode_grid.device
    if any(value.device != device for value in storage.values()):
        raise ValueError("rollout storage must remain on one device")

    terminal_ids = storage["terminal_episode_id"].reshape(-1).long()
    terminal_results = storage["terminal_result"].reshape(-1).long()
    terminal = terminal_ids.ge(0)
    completed_ids = terminal_ids[terminal]
    completed_results = terminal_results[terminal]
    if completed_ids.numel() == 0:
        raise ValueError("rollout window contains no complete Episodes")
    if bool(((completed_results < 1) | (completed_results > 3)).any()):
        raise ValueError("terminal result is outside the official outcome contract")
    completed_rewards = torch.where(
        completed_results.eq(1),
        torch.ones_like(completed_results, dtype=torch.float32),
        torch.where(
            completed_results.eq(2),
            -torch.ones_like(completed_results, dtype=torch.float32),
            torch.zeros_like(completed_results, dtype=torch.float32),
        ),
    )
    completed_ids, order = completed_ids.sort()
    completed_rewards = completed_rewards.index_select(0, order)
    if completed_ids.numel() > 1 and bool(completed_ids[1:].eq(completed_ids[:-1]).any()):
        raise ValueError("an Episode has more than one terminal record")

    keep_grid = focal_grid & torch.isin(episode_grid, completed_ids)
    if not bool(keep_grid.any()):
        raise ValueError("complete Episodes contain no focal decisions")

    # Work backwards across time on CUDA.  Opponent rows leave the running
    # learner state untouched, while an Episode id change resets the GAE chain.
    steps, lanes = episode_grid.shape
    advantage_grid = torch.zeros_like(value_grid)
    return_grid = torch.zeros_like(value_grid)
    next_value = torch.zeros(lanes, dtype=value_grid.dtype, device=device)
    next_advantage = torch.zeros_like(next_value)
    next_episode = torch.full((lanes,), -1, dtype=torch.long, device=device)
    next_valid = torch.zeros(lanes, dtype=torch.bool, device=device)
    for step in range(steps - 1, -1, -1):
        current = keep_grid[step]
        identities = episode_grid[step]
        continuation = current & next_valid & identities.eq(next_episode)
        positions = torch.searchsorted(completed_ids, identities).clamp_max(
            completed_ids.numel() - 1
        )
        outcomes = completed_rewards.index_select(0, positions)
        reward = torch.where(
            current & ~continuation,
            outcomes,
            torch.zeros_like(outcomes),
        )
        continuation_value = continuation.to(value_grid.dtype)
        delta = (
            reward
            + gamma * next_value * continuation_value
            - value_grid[step]
        )
        candidate = (
            delta
            + gamma * gae_lambda * next_advantage * continuation_value
        )
        advantage_grid[step] = torch.where(
            current, candidate, advantage_grid[step]
        )
        return_grid[step] = torch.where(
            current, candidate + value_grid[step], return_grid[step]
        )
        next_value = torch.where(current, value_grid[step], next_value)
        next_advantage = torch.where(current, candidate, next_advantage)
        next_episode = torch.where(current, identities, next_episode)
        next_valid |= current

    flat_keep = keep_grid.reshape(-1)
    flat_indices = flat_keep.nonzero(as_tuple=False).squeeze(1)
    selected_episode_ids = episode_grid.reshape(-1).index_select(0, flat_indices)
    old_value = value_grid.reshape(-1).index_select(0, flat_indices)
    raw_advantage = advantage_grid.reshape(-1).index_select(0, flat_indices)
    returns = return_grid.reshape(-1).index_select(0, flat_indices)
    unique_episodes, inverse, counts = torch.unique(
        selected_episode_ids, sorted=True, return_inverse=True, return_counts=True
    )
    weights = counts.index_select(0, inverse).to(value_grid.dtype).reciprocal()
    weighted_mean = (raw_advantage * weights).sum() / weights.sum()
    weighted_variance = ((raw_advantage - weighted_mean).square() * weights).sum() / weights.sum()
    advantage = (raw_advantage - weighted_mean) / weighted_variance.sqrt().clamp_min(1e-6)

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
        completed_episodes=int(unique_episodes.numel()),
        discarded_incomplete_decisions=int(
            (focal_grid.sum() - keep_grid.sum()).item()
        ),
    )


__all__ = [
    "FEATURE_NAMES",
    "PreparedBatch",
    "generalized_advantage_estimate",
    "prepare_complete_episodes",
    "result_reward",
]
