"""PPO over only the canonical ordered-option decoder and value head."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..policy.action_distribution import evaluate_actions_encoded
from ..policy.actor_critic import CanonicalActorCritic, DecoderPolicyHead
from ..policy.batching import collate_feature_batches, move_batch
from .batch import PreparedBatch, training_batch_metrics


@dataclass(frozen=True)
class PPOConfig:
    gae_lambda: float = 0.95
    epochs: int = 4
    batch_size: int = 1024
    actor_learning_rate: float = 1e-5
    value_learning_rate: float = 1e-4
    weight_decay: float = 0.0
    clip_ratio: float = 0.10
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    reference_kl_coefficient: float = 0.02
    target_behavior_kl: float = 0.02
    max_grad_norm: float = 0.5


class PPOTrainer:
    def __init__(
        self,
        model: CanonicalActorCritic,
        *,
        device: torch.device,
        config: PPOConfig = PPOConfig(),
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        model.freeze_representation()
        model.assert_trainable_contract()
        self.initial_representation_sha256 = model.representation_sha256()
        self.reference = DecoderPolicyHead.copy_from(model)
        self.reference_parameters = {
            name: tensor.detach().clone()
            for name, tensor in model.actor.action_decoder.named_parameters()
        }
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.actor.action_decoder.parameters(),
                    "lr": config.actor_learning_rate,
                },
                {"params": model.value_head.parameters(), "lr": config.value_learning_rate},
            ],
            weight_decay=config.weight_decay,
        )

    def _relative_l2(self, baseline: dict[str, torch.Tensor]) -> float:
        difference = torch.zeros((), device=self.device, dtype=torch.float64)
        scale = torch.zeros((), device=self.device, dtype=torch.float64)
        current = dict(self.model.actor.action_decoder.named_parameters())
        for name, source in baseline.items():
            value = current[name].detach().double()
            source = source.to(self.device).double()
            difference += (value - source).square().sum()
            scale += source.square().sum()
        return float((difference.sqrt() / scale.sqrt().clamp_min(1e-12)).item())

    def update(self, batch: PreparedBatch) -> dict[str, float]:
        if batch.source_policy_update < 0:
            raise ValueError("source_policy_update must be nonnegative")
        behavior = DecoderPolicyHead.copy_from(self.model)
        behavior_parameters = {
            name: tensor.detach().clone()
            for name, tensor in self.model.actor.action_decoder.named_parameters()
        }
        accumulators: dict[str, float] = {}
        minibatches = 0
        epochs_completed = 0
        early_stop = False
        rejected_kl = 0.0
        self.model.eval()
        for epoch in range(self.config.epochs):
            order = torch.randperm(batch.decisions)
            epoch_kls: list[float] = []
            for start in range(0, batch.decisions, self.config.batch_size):
                indices = order[start : start + self.config.batch_size]
                features = move_batch(
                    collate_feature_batches([batch.features[int(index)] for index in indices]),
                    self.device,
                )
                sequences = batch.sequences[indices].to(self.device)
                lengths = batch.lengths[indices].to(self.device)
                stopped = batch.stopped[indices].to(self.device)
                rollout_log_prob = batch.rollout_log_prob[indices].to(self.device)
                advantage = batch.advantage[indices].to(self.device)
                returns = batch.gae_return[indices].to(self.device)
                weights = batch.episode_weight[indices].to(self.device)
                weights = weights / weights.sum()
                with torch.no_grad():
                    summary, options = self.model.actor.encode(features)
                    behavior_eval = evaluate_actions_encoded(
                        behavior, features, summary, options, sequences, lengths, stopped
                    )
                    reference_eval = evaluate_actions_encoded(
                        self.reference, features, summary, options, sequences, lengths, stopped
                    )
                evaluated = evaluate_actions_encoded(
                    self.model.head, features, summary, options, sequences, lengths, stopped
                )
                old_log_prob = behavior_eval.log_prob
                log_ratio = evaluated.log_prob - old_log_prob
                ratio = log_ratio.exp()
                approximate_kl = (((ratio - 1.0) - log_ratio) * weights).sum()
                approximate_kl_value = float(approximate_kl.detach())
                if minibatches > 0 and approximate_kl_value > self.config.target_behavior_kl:
                    early_stop = True
                    rejected_kl = approximate_kl_value
                    break
                unclipped = ratio * advantage
                clipped = ratio.clamp(
                    1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                ) * advantage
                policy_loss = -(torch.minimum(unclipped, clipped) * weights).sum()
                value_loss = ((evaluated.value - returns).square() * weights).sum()
                entropy = (evaluated.entropy * weights).sum()
                reference_delta = evaluated.log_prob - reference_eval.log_prob
                reference_kl = (0.5 * reference_delta.square() * weights).sum()
                total_loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.reference_kl_coefficient * reference_kl
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    self.config.max_grad_norm,
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite PPO gradient")
                self.optimizer.step()
                metrics = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "total_loss": total_loss,
                    "behavior_kl": approximate_kl,
                    "reference_kl_surrogate": reference_kl,
                    "rollout_log_prob_mae": (
                        (rollout_log_prob - old_log_prob).abs() * weights
                    ).sum(),
                    "clip_fraction": (
                        ((ratio - 1.0).abs() > self.config.clip_ratio).float() * weights
                    ).sum(),
                    "ratio_mean": (ratio * weights).sum(),
                    "ratio_max": ratio.max(),
                    "gradient_norm": norm,
                }
                for name, value in metrics.items():
                    accumulators[name] = accumulators.get(name, 0.0) + float(value.detach())
                epoch_kls.append(approximate_kl_value)
                minibatches += 1
            epochs_completed = epoch + 1
            if early_stop:
                break
            if epoch_kls and sum(epoch_kls) / len(epoch_kls) > self.config.target_behavior_kl:
                early_stop = True
                break
        if minibatches == 0:
            raise RuntimeError("PPO update produced no minibatches")
        if self.model.representation_sha256() != self.initial_representation_sha256:
            raise RuntimeError("frozen canonical representation changed during PPO")
        result = {
            f"ppo/{name}": value / minibatches for name, value in accumulators.items()
        }
        result.update({
            "ppo/epochs_completed": float(epochs_completed),
            "ppo/target_kl_early_stop": float(early_stop),
            "ppo/rejected_behavior_kl": rejected_kl,
            "ppo/minibatches_completed": float(minibatches),
            "ppo/decisions": float(batch.decisions),
            "ppo/actor_learning_rate": self.config.actor_learning_rate,
            "ppo/value_learning_rate": self.config.value_learning_rate,
            "ppo/actor_relative_l2_vs_behavior": self._relative_l2(behavior_parameters),
            "ppo/actor_relative_l2_vs_reference": self._relative_l2(self.reference_parameters),
        })
        result.update(training_batch_metrics(batch))
        return result


__all__ = ["PPOConfig", "PPOTrainer"]
