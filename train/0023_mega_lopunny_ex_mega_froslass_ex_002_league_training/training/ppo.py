from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from ..decoder import DECODER_COMPONENTS
from ..policy.action_distribution import evaluate_actions
from ..policy.actor_critic import LeagueActorCritic
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


def frozen_reference(model: LeagueActorCritic, device: torch.device) -> LeagueActorCritic:
    # LeaguePolicyPool.RoutedActor keeps the Foundation encoder in an
    # intentionally unregistered shared attribute.  Reusing that object here
    # prevents one full encoder copy per deck (48 references would otherwise
    # defeat the shared-encoder memory contract).  Only the decoder/value are
    # immutable reference state for a deck-local PPO trainer.
    if hasattr(model.actor, "shared_actor"):
        routed_type = type(model.actor)
        actor = routed_type(model.actor.shared_actor)
        for component in DECODER_COMPONENTS:
            setattr(actor, component, copy.deepcopy(getattr(model.actor, component)))
        reference = LeagueActorCritic(actor, copy.deepcopy(model.value_head)).to(device).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference
    reference = copy.deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference


class PPOTrainer:
    """Own the fresh optimizer for one immutable version without ever serializing it."""

    def __init__(
        self,
        model: LeagueActorCritic,
        reference: LeagueActorCritic,
        *,
        device: torch.device,
        config: PPOConfig = PPOConfig(),
    ) -> None:
        self.model = model
        self.reference = reference
        self.device = device
        self.config = config
        model.freeze_encoder()
        model.assert_trainable_contract()
        self._actor_parameter_names = tuple(
            name for name, parameter in model.actor.named_parameters() if parameter.requires_grad
        )
        self._reference_actor_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.actor.named_parameters()
            if name in self._actor_parameter_names
        }
        trainable_actor = [
            parameter for parameter in model.actor.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": trainable_actor, "lr": config.actor_learning_rate},
                {"params": model.value_head.parameters(), "lr": config.value_learning_rate},
            ],
            weight_decay=config.weight_decay,
        )

    def update(self, batch: PreparedBatch) -> dict[str, float]:
        model = self.model
        reference = self.reference
        behavior = frozen_reference(model, self.device)
        device = self.device
        config = self.config
        behavior_actor_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.actor.named_parameters()
            if name in self._actor_parameter_names
        }
        accumulators: dict[str, float] = {}
        minibatches = 0
        epochs_completed = 0
        early_stop = False
        rejected_behavior_kl = 0.0
        # Policy stochasticity comes only from legal categorical sampling, not dropout.
        model.eval()
        reference.eval()
        for epoch in range(config.epochs):
            order = torch.randperm(batch.decisions)
            epoch_kls: list[float] = []
            for start in range(0, batch.decisions, config.batch_size):
                indices = order[start : start + config.batch_size]
                features = move_batch(
                    collate_feature_batches([batch.features[int(i)] for i in indices]),
                    device,
                )
                sequences = batch.sequences[indices].to(device)
                lengths = batch.lengths[indices].to(device)
                stopped = batch.stopped[indices].to(device)
                rollout_log_prob = batch.old_log_prob[indices].to(device)
                advantage = batch.advantage[indices].to(device)
                returns = batch.gae_return[indices].to(device)
                weights = batch.episode_weight[indices].to(device)
                weights = weights / weights.sum()

                evaluated = evaluate_actions(model, features, sequences, lengths, stopped)
                with torch.no_grad():
                    behavior_eval = evaluate_actions(
                        behavior, features, sequences, lengths, stopped
                    )
                    reference_eval = evaluate_actions(
                        reference, features, sequences, lengths, stopped
                    )
                old_log_prob = behavior_eval.log_prob
                rollout_log_prob_mae = (
                    (rollout_log_prob - old_log_prob).abs() * weights
                ).sum()
                log_ratio = evaluated.log_prob - old_log_prob
                ratio = log_ratio.exp()
                approximate_kl = (((ratio - 1.0) - log_ratio) * weights).sum()
                approximate_kl_value = float(approximate_kl.detach())
                if minibatches > 0 and approximate_kl_value > config.target_behavior_kl:
                    early_stop = True
                    rejected_behavior_kl = approximate_kl_value
                    break
                unclipped = ratio * advantage
                clipped = ratio.clamp(
                    1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                ) * advantage
                policy_loss = -(torch.minimum(unclipped, clipped) * weights).sum()
                value_loss = ((evaluated.value - returns).square() * weights).sum()
                entropy = (evaluated.entropy * weights).sum()
                reference_log_delta = evaluated.log_prob - reference_eval.log_prob
                reference_kl_surrogate = (
                    0.5 * reference_log_delta.square() * weights
                ).sum()
                total_loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                    + config.reference_kl_coefficient * reference_kl_surrogate
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    config.max_grad_norm,
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite PPO gradient")
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = (
                        ((ratio - 1.0).abs() > config.clip_ratio).float() * weights
                    ).sum()
                    metrics = {
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "entropy": entropy,
                        "total_loss": total_loss,
                        "behavior_kl": approximate_kl,
                        "reference_kl_surrogate": reference_kl_surrogate,
                        "rollout_log_prob_mae": rollout_log_prob_mae,
                        "clip_fraction": clip_fraction,
                        "ratio_mean": (ratio * weights).sum(),
                        "ratio_max": ratio.max(),
                        "gradient_norm": norm,
                    }
                    for key, value in metrics.items():
                        accumulators[key] = accumulators.get(key, 0.0) + float(value)
                    epoch_kls.append(float(approximate_kl))
                    minibatches += 1
            epochs_completed = epoch + 1
            if early_stop:
                break
            if epoch_kls and sum(epoch_kls) / len(epoch_kls) > config.target_behavior_kl:
                early_stop = True
                break
        if minibatches == 0:
            raise RuntimeError("PPO update produced no minibatches")
        result = {
            f"ppo/{key}": value / minibatches for key, value in accumulators.items()
        }
        result.update(
            {
                "ppo/epochs_completed": float(epochs_completed),
                "ppo/target_kl_early_stop": float(early_stop),
                "ppo/rejected_behavior_kl": rejected_behavior_kl,
                "ppo/minibatches_completed": float(minibatches),
                "ppo/decisions": float(batch.decisions),
                "ppo/actor_learning_rate": config.actor_learning_rate,
                "ppo/value_learning_rate": config.value_learning_rate,
            }
        )
        result.update(training_batch_metrics(batch))
        current_parameters = dict(model.actor.named_parameters())

        def relative_l2(baseline: dict[str, torch.Tensor]) -> float:
            difference = torch.zeros((), device=self.device, dtype=torch.float64)
            scale = torch.zeros((), device=self.device, dtype=torch.float64)
            for name in self._actor_parameter_names:
                current = current_parameters[name].detach().double()
                source = baseline[name].to(self.device).double()
                difference += (current - source).square().sum()
                scale += source.square().sum()
            return float((difference.sqrt() / scale.sqrt().clamp_min(1e-12)).item())

        result.update(
            {
                "ppo/actor_relative_l2_vs_behavior": relative_l2(
                    behavior_actor_parameters
                ),
                "ppo/actor_relative_l2_vs_reference": relative_l2(
                    self._reference_actor_parameters
                ),
            }
        )
        return result


def ppo_update(
    model: LeagueActorCritic,
    reference: LeagueActorCritic,
    batch: PreparedBatch,
    *,
    device: torch.device,
    config: PPOConfig = PPOConfig(),
) -> dict[str, float]:
    """Single-update convenience API; long runs must retain one PPOTrainer."""
    return PPOTrainer(model, reference, device=device, config=config).update(batch)


__all__ = ["PPOConfig", "PPOTrainer", "frozen_reference", "ppo_update"]
