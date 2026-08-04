"""CUDA-resident rollout storage and PPO updates for the 0020 foundation.

The module deliberately accepts the actor-critic protocol implemented by
``train/0020_pluggable_deck_rl/actor_critic.py`` instead of importing the
numeric package name.  This keeps the engine package reusable while allowing
one shared foundation instance to serve every deck-conditioned lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _require_cuda_tensor(name: str, value: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must remain on CUDA, got {value.device}")
    return value


def _clone_cuda_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    output: dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            _require_cuda_tensor(name, value)
            output[name] = value.detach().clone()
        else:
            raise TypeError(f"rollout field {name} must be a CUDA tensor")
    return output


@dataclass
class _RolloutStep:
    batch: dict[str, Any]
    old_logprob: Any
    old_value: Any
    reward: Any
    done: Any


class CudaRolloutBuffer:
    """Append-only fixed-horizon rollout storage with no host tensor copies."""

    def __init__(self, *, max_steps: int, batch_size: int) -> None:
        if max_steps <= 0 or batch_size <= 0:
            raise ValueError("max_steps and batch_size must be positive")
        self.max_steps = int(max_steps)
        self.batch_size = int(batch_size)
        self._steps: list[_RolloutStep] = []
        self._keys: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return len(self._steps)

    def release(self) -> None:
        """Release per-step storage after a flattened rollout has been materialized."""
        self._steps.clear()
        self._keys = None

    @property
    def device(self) -> Any:
        if not self._steps:
            raise ValueError("rollout buffer is empty")
        return self._steps[0].old_value.device

    def append(
        self,
        batch: Mapping[str, Any],
        *,
        old_logprob: Any,
        old_value: Any,
        reward: Any,
        done: Any,
    ) -> None:
        import torch

        if len(self) >= self.max_steps:
            raise RuntimeError("CUDA rollout buffer is full")
        stored_batch = _clone_cuda_batch(batch)
        keys = tuple(sorted(stored_batch))
        if self._keys is None:
            self._keys = keys
        elif keys != self._keys:
            raise ValueError("rollout observation keys changed within one buffer")
        for name, value in stored_batch.items():
            if value.ndim == 0 or value.shape[0] != self.batch_size:
                raise ValueError(f"rollout field {name} has an invalid batch axis")
        tensors = {
            "old_logprob": old_logprob,
            "old_value": old_value,
            "reward": reward,
            "done": done,
        }
        for name, value in tensors.items():
            _require_cuda_tensor(name, value)
            if value.ndim != 1 or value.shape[0] != self.batch_size:
                raise ValueError(f"{name} must be a [batch] CUDA tensor")
        self._steps.append(
            _RolloutStep(
                batch=stored_batch,
                old_logprob=old_logprob.detach().clone(),
                old_value=old_value.detach().clone(),
                reward=reward.detach().clone(),
                done=done.detach().clone().bool(),
            )
        )

    def flatten(self, *, last_value: Any, gamma: float, gae_lambda: float) -> dict[str, Any]:
        import torch

        if not self._steps:
            raise ValueError("cannot flatten an empty rollout buffer")
        _require_cuda_tensor("last_value", last_value)
        if last_value.ndim != 1 or last_value.shape[0] != self.batch_size:
            raise ValueError("last_value must be a [batch] CUDA tensor")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")

        rewards = torch.stack([step.reward for step in self._steps], dim=0)
        dones = torch.stack([step.done for step in self._steps], dim=0)
        values = torch.stack([step.old_value for step in self._steps], dim=0)
        train_masks = torch.stack(
            [
                step.batch.get(
                    "train_mask",
                    torch.ones(self.batch_size, dtype=torch.bool, device=values.device),
                ).bool()
                for step in self._steps
            ],
            dim=0,
        )
        if train_masks.shape != dones.shape:
            raise ValueError("train_mask must be a [time, batch] tensor")

        # Collapse opponent actions out of the learner trajectory.  A learner
        # transition spans its own reward plus all rewards produced by frozen
        # opponent actions until the next learner decision.  Terminal rewards
        # after an opponent action are therefore assigned to the preceding
        # player0 transition, while opponent values never enter GAE.
        advantages = torch.zeros_like(values)
        outcomes = torch.zeros_like(rewards)
        outcome_known = torch.zeros_like(dones)
        running_advantage = torch.zeros_like(last_value)
        next_train_value = torch.where(
            train_masks[-1], last_value, torch.zeros_like(last_value)
        )
        pending_reward = torch.zeros_like(last_value)
        pending_done = torch.zeros_like(dones[0])
        pending_outcome = torch.zeros_like(last_value)
        pending_outcome_known = torch.zeros_like(dones[0])
        for index in range(len(self._steps) - 1, -1, -1):
            is_train = train_masks[index]
            current_done = dones[index]
            # A done on the current learner row ends that transition; rewards
            # after it belong to the next reset game and must be discarded.
            transition_reward = torch.where(
                current_done,
                rewards[index],
                rewards[index] + pending_reward,
            )
            transition_done = current_done | pending_done
            transition_outcome = torch.where(
                current_done, rewards[index], pending_outcome
            )
            transition_outcome_known = current_done | pending_outcome_known
            not_done = (~transition_done).to(values.dtype)
            delta = (
                transition_reward
                + gamma * next_train_value * not_done
                - values[index]
            )
            candidate_advantage = (
                delta + gamma * gae_lambda * not_done * running_advantage
            )
            advantages[index] = torch.where(
                is_train, candidate_advantage, torch.zeros_like(candidate_advantage)
            )
            outcomes[index] = torch.where(
                is_train, transition_outcome, torch.zeros_like(transition_outcome)
            )
            outcome_known[index] = is_train & transition_outcome_known

            running_advantage = torch.where(
                is_train, candidate_advantage, running_advantage
            )
            next_train_value = torch.where(
                is_train, values[index], next_train_value
            )
            # Consume the opponent segment after a learner row.  For a
            # non-learner terminal, reset first so future-game rewards cannot
            # leak backward across the game boundary.
            pending_reward = torch.where(
                is_train,
                torch.zeros_like(pending_reward),
                torch.where(
                    current_done,
                    rewards[index],
                    pending_reward + rewards[index],
                ),
            )
            pending_done = torch.where(
                is_train,
                torch.zeros_like(pending_done),
                pending_done | current_done,
            )
            pending_outcome = torch.where(
                is_train,
                torch.zeros_like(pending_outcome),
                torch.where(current_done, rewards[index], pending_outcome),
            )
            pending_outcome_known = torch.where(
                is_train,
                torch.zeros_like(pending_outcome_known),
                pending_outcome_known | current_done,
            )
        returns = advantages + values

        flat_batch = {
            name: torch.cat([step.batch[name] for step in self._steps], dim=0)
            for name in self._steps[0].batch
        }
        return {
            "batch": flat_batch,
            "old_logprob": torch.cat([step.old_logprob for step in self._steps], dim=0),
            "old_value": torch.cat([step.old_value for step in self._steps], dim=0),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "rewards": rewards.reshape(-1),
            "dones": dones.reshape(-1),
            "train_mask": train_masks.reshape(-1),
            "outcomes": outcomes.reshape(-1),
            "outcome_known": outcome_known.reshape(-1),
        }


def active_deck_lane_mask(
    *,
    completed_by_deck: Any,
    lane_deck_ids: Any,
    target_games_per_deck: int,
) -> Any:
    """Return lanes whose deck has not yet reached its rollout game target."""
    import torch

    if not isinstance(completed_by_deck, torch.Tensor) or completed_by_deck.ndim != 1:
        raise ValueError("completed_by_deck must be a rank-1 tensor")
    if not isinstance(lane_deck_ids, torch.Tensor) or lane_deck_ids.ndim != 1:
        raise ValueError("lane_deck_ids must be a rank-1 tensor")
    if completed_by_deck.device != lane_deck_ids.device:
        raise ValueError("completed_by_deck and lane_deck_ids must share a device")
    if target_games_per_deck < 0:
        raise ValueError("target_games_per_deck must be non-negative")
    if target_games_per_deck == 0:
        return torch.ones_like(lane_deck_ids, dtype=torch.bool)
    collecting_by_deck = completed_by_deck.lt(int(target_games_per_deck))
    return collecting_by_deck.index_select(0, lane_deck_ids.long())


@dataclass(frozen=True)
class CudaPPOStats:
    loss: Any
    policy_loss: Any
    value_loss: Any
    entropy: Any
    approx_kl: Any
    clip_frac: Any
    grad_norm: Any
    ratio_mean: Any
    ratio_std: Any
    explained_variance: Any
    value_mean: Any
    return_mean: Any
    advantage_std: Any
    value_sign_accuracy: Any
    value_vs_outcome_corr: Any
    train_samples: Any
    optimizer_steps: Any

    def as_host(self) -> dict[str, float]:
        """Synchronize only final scalar reporting, never rollout/update tensors."""
        return {
            name: float(getattr(self, name).detach().item())
            for name in self.__dataclass_fields__
        }


def ppo_update_device(
    *,
    model: Any,
    optimizer: Any,
    buffer: CudaRolloutBuffer,
    last_value: Any,
    epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    target_kl: float | None = None,
) -> CudaPPOStats:
    """Run PPO entirely on the buffer/model device and return device scalars."""
    import torch
    if epochs <= 0 or minibatch_size <= 0:
        raise ValueError("epochs and minibatch_size must be positive")
    if not 0.0 < clip_eps < 1.0 or value_coef < 0.0 or entropy_coef < 0.0:
        raise ValueError("invalid PPO coefficients")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")

    flattened = buffer.flatten(last_value=last_value, gamma=gamma, gae_lambda=gae_lambda)
    buffer.release()
    batch = flattened.pop("batch")
    advantages = flattened.pop("advantages")
    returns = flattened.pop("returns")
    old_logprob = flattened.pop("old_logprob")
    old_value = flattened.pop("old_value")
    outcomes = flattened.pop("outcomes")
    outcome_known = flattened.pop("outcome_known").bool()
    flattened.clear()
    train_mask = batch.pop("train_mask", flattened.pop("train_mask", None))
    if train_mask is not None:
        train_indices = train_mask.bool().reshape(-1).nonzero(as_tuple=False).flatten()
        if train_indices.numel() == 0:
            raise RuntimeError("PPO rollout contains no trainable samples")
        batch = {
            name: value.index_select(0, train_indices)
            for name, value in batch.items()
        }
        advantages = advantages.index_select(0, train_indices)
        returns = returns.index_select(0, train_indices)
        old_logprob = old_logprob.index_select(0, train_indices)
        old_value = old_value.index_select(0, train_indices)
        outcomes = outcomes.index_select(0, train_indices)
        outcome_known = outcome_known.index_select(0, train_indices)
    count = int(old_logprob.shape[0])
    raw_advantage_std = advantages.std(unbiased=False)
    return_variance = returns.var(unbiased=False)
    residual_variance = (returns - old_value).var(unbiased=False)
    explained_variance = torch.where(
        return_variance > 1.0e-8,
        1.0 - residual_variance / return_variance.clamp_min(1.0e-8),
        torch.zeros_like(return_variance),
    )
    known_values = old_value[outcome_known]
    known_outcomes = outcomes[outcome_known]
    if known_values.numel() > 0:
        value_sign_accuracy = (
            known_values.sign().eq(known_outcomes.sign()).to(old_value.dtype).mean()
        )
    else:
        value_sign_accuracy = torch.zeros((), dtype=old_value.dtype, device=old_value.device)
    if known_values.numel() > 1:
        centered_value = known_values - known_values.mean()
        centered_outcome = known_outcomes - known_outcomes.mean()
        correlation_denom = (
            centered_value.square().mean().sqrt()
            * centered_outcome.square().mean().sqrt()
        )
        value_vs_outcome_corr = torch.where(
            correlation_denom > 1.0e-8,
            (centered_value * centered_outcome).mean()
            / correlation_denom.clamp_min(1.0e-8),
            torch.zeros_like(correlation_denom),
        )
    else:
        value_vs_outcome_corr = torch.zeros(
            (), dtype=old_value.dtype, device=old_value.device
        )
    if count > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1.0e-8
        )
    metrics: list[dict[str, Any]] = []

    model.train()
    # Foundation dropout is disabled for reproducible actor/critic evaluation;
    # requires_grad remains enabled for either the full model or decoder-only scope.
    model.policy.eval()
    for _ in range(int(epochs)):
        epoch_kls: list[Any] = []
        permutation = torch.randperm(count, device=old_logprob.device)
        for start in range(0, count, int(minibatch_size)):
            indices = permutation[start : start + int(minibatch_size)]
            minibatch = {
                name: value.index_select(0, indices) for name, value in batch.items()
            }
            encoded = model.encode(minibatch)
            logprob, value = model.evaluate_targets_from_encoding(minibatch, encoded)
            entropy = model.target_entropy_from_encoding(minibatch, encoded)
            old = old_logprob.index_select(0, indices).to(dtype=logprob.dtype)
            advantage = advantages.index_select(0, indices).to(dtype=logprob.dtype)
            target_return = returns.index_select(0, indices).to(dtype=value.dtype)
            ratio = torch.exp(logprob - old)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = ((value - target_return) ** 2).mean()
            entropy_loss = entropy
            total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            approx_kl = (old - logprob).detach().mean()
            clip_frac = (
                ((ratio - 1.0).abs() > clip_eps).to(logprob.dtype).detach().mean()
            )
            epoch_kls.append(approx_kl)
            metrics.append(
                {
                    "loss": total_loss.detach(),
                    "policy_loss": policy_loss.detach(),
                    "value_loss": value_loss.detach(),
                    "entropy": entropy.detach(),
                    "approx_kl": approx_kl,
                    "clip_frac": clip_frac,
                    "grad_norm": grad_norm.detach(),
                    "ratio_mean": ratio.detach().mean(),
                    "ratio_std": ratio.detach().std(unbiased=False),
                }
            )
        if (
            target_kl is not None
            and epoch_kls
            and float(torch.stack(epoch_kls).mean().detach().item()) > float(target_kl)
        ):
            break
    if not metrics:
        raise RuntimeError("PPO produced no minibatches")
    aggregated = {
        name: torch.stack([row[name] for row in metrics]).mean()
        for name in metrics[0]
    }
    aggregated["train_samples"] = torch.tensor(
        float(count), dtype=old_logprob.dtype, device=old_logprob.device
    )
    aggregated["optimizer_steps"] = torch.tensor(
        float(len(metrics)), dtype=old_logprob.dtype, device=old_logprob.device
    )
    aggregated["explained_variance"] = explained_variance
    aggregated["value_mean"] = old_value.mean()
    aggregated["return_mean"] = returns.mean()
    aggregated["advantage_std"] = raw_advantage_std
    aggregated["value_sign_accuracy"] = value_sign_accuracy
    aggregated["value_vs_outcome_corr"] = value_vs_outcome_corr
    return CudaPPOStats(**aggregated)


__all__ = [
    "CudaPPOStats",
    "CudaRolloutBuffer",
    "active_deck_lane_mask",
    "ppo_update_device",
]
