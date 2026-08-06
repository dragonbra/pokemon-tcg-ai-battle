"""Decoder/value-only PPO for the POD-native 0033 actor."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor

from ..contract import PodNativeBatch
from ..model import PodNativeActorCritic
from .batch import PreparedBatch


@dataclass(frozen=True, slots=True)
class PPOConfig:
    epochs: int = 2
    minibatch_size: int = 512
    actor_learning_rate: float = 1e-5
    value_learning_rate: float = 1e-4
    clip_ratio: float = 0.10
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    reference_kl_coefficient: float = 0.02
    target_kl: float = 0.02
    max_grad_norm: float = 0.5


class PPOTrainer:
    def __init__(
        self,
        model: PodNativeActorCritic,
        *,
        device: torch.device,
        config: PPOConfig = PPOConfig(),
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        model.requires_grad_(False)
        model.action_decoder.requires_grad_(True)
        model.value_head.requires_grad_(True)
        self.reference_decoder = copy.deepcopy(model.action_decoder).to(device).eval()
        self.reference_decoder.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": model.action_decoder.parameters(), "lr": config.actor_learning_rate},
                {"params": model.value_head.parameters(), "lr": config.value_learning_rate},
            ]
        )

    def _features(self, batch: PreparedBatch, indices: Tensor) -> PodNativeBatch:
        return PodNativeBatch.from_mapping(
            {
                name: value.index_select(0, indices).to(self.device, non_blocking=True)
                for name, value in batch.features.items()
            }
        )

    def update(self, batch: PreparedBatch) -> dict[str, float]:
        if batch.decisions < 1:
            raise ValueError("empty PPO batch")
        cfg = self.config
        totals: dict[str, float] = {}
        minibatches = 0
        early_stop = False
        first_logprob_mae: float | None = None
        self.model.eval()
        for epoch in range(cfg.epochs):
            order = torch.randperm(batch.decisions)
            for start in range(0, batch.decisions, cfg.minibatch_size):
                indices = order[start : start + cfg.minibatch_size]
                features = self._features(batch, indices)
                sequences = batch.sequences.index_select(0, indices).to(self.device)
                lengths = batch.lengths.index_select(0, indices).to(self.device)
                with torch.no_grad():
                    encoded = self.model.encode(features)
                    state_summary = encoded.state_summary.detach()
                    option_tokens = encoded.option_tokens.detach()
                evaluated = self.model.action_decoder.evaluate(
                    features, option_tokens, state_summary, sequences, lengths
                )
                value = self.model.value_head(state_summary).squeeze(-1)
                old_logprob = batch.old_logprob.index_select(0, indices).to(self.device)
                advantage = batch.advantage.index_select(0, indices).to(self.device)
                returns = batch.gae_return.index_select(0, indices).to(self.device)
                weights = batch.episode_weight.index_select(0, indices).to(self.device)
                weights = weights / weights.sum()
                if first_logprob_mae is None:
                    first_logprob_mae = float(
                        ((evaluated.logprob.detach() - old_logprob).abs() * weights).sum()
                    )
                    if first_logprob_mae > 1e-4:
                        raise RuntimeError(
                            f"behavior log-prob mismatch before PPO: {first_logprob_mae}"
                        )
                log_ratio = evaluated.logprob - old_logprob
                ratio = log_ratio.exp()
                approximate_kl = (((ratio - 1.0) - log_ratio) * weights).sum()
                if minibatches and float(approximate_kl.detach()) > cfg.target_kl:
                    early_stop = True
                    break
                unclipped = ratio * advantage
                clipped = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantage
                policy_loss = -(torch.minimum(unclipped, clipped) * weights).sum()
                value_loss = ((value - returns).square() * weights).sum()
                entropy = (evaluated.entropy * weights).sum()
                with torch.no_grad():
                    reference = self.reference_decoder.evaluate(
                        features, option_tokens, state_summary, sequences, lengths
                    )
                reference_kl = (
                    0.5 * (evaluated.logprob - reference.logprob).square() * weights
                ).sum()
                loss = (
                    policy_loss
                    + cfg.value_coefficient * value_loss
                    - cfg.entropy_coefficient * entropy
                    + cfg.reference_kl_coefficient * reference_kl
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    cfg.max_grad_norm,
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("nonfinite PPO gradient")
                self.optimizer.step()
                values = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "reference_kl": reference_kl,
                    "behavior_kl": approximate_kl,
                    "total_loss": loss,
                    "gradient_norm": grad_norm,
                    "clip_fraction": ((ratio - 1.0).abs().gt(cfg.clip_ratio).float() * weights).sum(),
                }
                for name, item in values.items():
                    totals[name] = totals.get(name, 0.0) + float(item.detach())
                minibatches += 1
            if early_stop:
                break
        if minibatches == 0:
            raise RuntimeError("PPO update produced no minibatches")
        return {
            **{f"ppo/{name}": value / minibatches for name, value in totals.items()},
            "ppo/minibatches": float(minibatches),
            "ppo/target_kl_early_stop": float(early_stop),
            "ppo/behavior_logprob_mae_preupdate": float(first_logprob_mae or 0.0),
            "ppo/training_decisions": float(batch.decisions),
            "ppo/completed_episodes": float(batch.completed_episodes),
        }


__all__ = ["PPOConfig", "PPOTrainer"]
