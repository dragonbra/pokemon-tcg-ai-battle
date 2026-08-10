"""0042 PPO with frozen semantic encoders and explicit strategy adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..policy.action_distribution import evaluate_actions_encoded
from ..policy.actor_critic import DecoderPolicyHead, SemanticActorCritic
from ..policy.batching import collate_feature_batches, move_batch
from .batch_full_semantic import PreparedBatch, training_batch_metrics
from ..policy.compound_evaluation import evaluate_parameter_actions
from ..integrated.loss_registry import LossRegistry, LossTerm
from .metric_frequency import is_sparse_diagnostic_update
from ..integrated.diagnostics import gradient_diagnostics


@dataclass(frozen=True, slots=True)
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.95
    credit_clock: str = "turn"
    loss_weighting: str = "episode_equal_decisions"
    epochs: int = 4
    batch_size: int = 1024
    actor_learning_rate: float = 1.0e-5
    value_learning_rate: float = 1.0e-4
    prize_learning_rate: float = 1.0e-4
    meta_anchor_coef: float = 0.10
    weight_decay: float = 0.0
    clip_ratio: float = 0.10
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    reference_kl_coefficient: float = 0.02
    target_behavior_kl: float = 0.02
    max_grad_norm: float = 0.5
    behavior_logprob_mae_limit: float = 1.0e-4
    optimization_mode: str = "fixed_optimizer_budget"
    optimizer_steps_per_update: int = 32
    gradient_accumulation: int = 1

    def validate(self) -> None:
        if self.gamma != 1.0:
            raise ValueError("gamma must remain 1.0 for undiscounted official outcomes")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in (0, 1]")
        if self.credit_clock not in {"selection", "turn"}:
            raise ValueError("credit_clock must be selection or turn")
        if self.loss_weighting not in {
            "episode_equal_decisions",
            "episode_equal_turns",
        }:
            raise ValueError("invalid loss_weighting")
        if self.optimization_mode not in {"fixed_epochs", "fixed_optimizer_budget"}:
            raise ValueError("invalid PPO optimization mode")
        if min(self.epochs, self.batch_size, self.optimizer_steps_per_update,
               self.gradient_accumulation) < 1:
            raise ValueError("PPO capacity settings must be positive")
        if self.meta_anchor_coef <= 0:
            raise ValueError("0042 meta_anchor_coef must be explicitly positive")


class PPOTrainer:
    def __init__(
        self,
        model: SemanticActorCritic,
        *,
        device: torch.device,
        config: PPOConfig = PPOConfig(),
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        config.validate()
        model.freeze_representation()
        model.assert_trainable_contract()
        self.initial_representation_sha256 = model.representation_sha256()
        self.reference = DecoderPolicyHead.copy_from(model)
        self.reference_parameters = {
            name: tensor.detach().clone()
            for name, tensor in model.actor.action_decoder.named_parameters()
        }
        groups = [
                {
                    "name": "action_decoder",
                    "params": model.actor.action_decoder.parameters(),
                    "lr": config.actor_learning_rate,
                },
                {
                    "name": "policy_strategy_adapter",
                    "params": model.policy_strategy_adapter.parameters(),
                    "lr": config.actor_learning_rate,
                },
                {
                    "name": "value_win",
                    "params": [
                        parameter
                        for parameter in model.value_head.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": config.value_learning_rate,
                },
                {
                    "name": "value_adapter",
                    "params": model.value_adapter.parameters(),
                    "lr": config.value_learning_rate,
                },
                {
                    "name": "allocation_head",
                    "params": model.allocation_head.parameters(),
                    "lr": config.actor_learning_rate,
                },
            ]
        if model.prize_aux is not None:
            groups.append({"name": "value_prize", "params": model.prize_aux.parameters(), "lr": config.prize_learning_rate})
        self.optimizer = torch.optim.AdamW(
            groups,
            weight_decay=config.weight_decay,
        )
        self.base_learning_rates = {
            str(group["name"]): float(group["lr"])
            for group in self.optimizer.param_groups
        }

    def optimizer_group_manifest(self) -> list[dict[str, object]]:
        gradient_sources = {
            "action_decoder": "PPO policy + entropy + reference KL + Prize actor advantage",
            "policy_strategy_adapter": "PPO policy through detached strategic context",
            "allocation_head": "Phantom allocation policy + entropy + Prize actor advantage",
            "value_win": "terminal Value loss + q1 Meta anchor",
            "value_adapter": "terminal Value loss only",
            "value_prize": "directional Prize Value loss only",
        }
        return [
            {
                "name": str(group["name"]),
                "base_learning_rate": self.base_learning_rates[str(group["name"])],
                "current_learning_rate": float(group["lr"]),
                "trainable_parameters": sum(
                    int(parameter.numel()) for parameter in group["params"]
                ),
                "tensor_count": len(group["params"]),
                "gradient_source": gradient_sources[str(group["name"])],
            }
            for group in self.optimizer.param_groups
        ]

    def set_group_learning_rates(self, rates: dict[str, float]) -> None:
        names = {str(group["name"]) for group in self.optimizer.param_groups}
        if set(rates) != names:
            raise ValueError("learning-rate update does not cover exact optimizer groups")
        for group in self.optimizer.param_groups:
            value = float(rates[str(group["name"])])
            if value <= 0:
                raise ValueError("optimizer learning rates must remain positive")
            group["lr"] = value

    def snapshot_trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def restore_trainable_state(
        self, state: dict[str, torch.Tensor], *, reset_optimizer: bool = True
    ) -> None:
        current = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if set(state) != set(current):
            raise ValueError("rollback trainable tensor inventory mismatch")
        with torch.no_grad():
            for name, parameter in current.items():
                source = state[name]
                if source.shape != parameter.shape:
                    raise ValueError(f"rollback tensor shape mismatch: {name}")
                parameter.copy_(source.to(parameter.device, parameter.dtype))
        self.optimizer.zero_grad(set_to_none=True)
        if reset_optimizer:
            self.optimizer.state.clear()

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

    def _actor_group_relative_l2(self, baseline: dict[str, torch.Tensor]) -> float:
        if not baseline:
            return 0.0
        difference = torch.zeros((), device=self.device, dtype=torch.float64)
        scale = torch.zeros((), device=self.device, dtype=torch.float64)
        current = dict(self.model.actor.named_parameters())
        for name, source in baseline.items():
            value = current[name].detach().double()
            source = source.to(self.device).double()
            difference += (value - source).square().sum()
            scale += source.square().sum()
        return float((difference.sqrt() / scale.sqrt().clamp_min(1e-12)).item())

    def sparse_gradient_diagnostics(self, batch: PreparedBatch, *, samples: int = 16) -> dict[str, float]:
        """One fixed small graph at scheduled updates; never called per minibatch."""
        count = min(samples, batch.decisions)
        indices = torch.arange(count)
        features = move_batch(
            collate_feature_batches([batch.features[int(index)] for index in indices]), self.device
        )
        validated, state, options = self.model.actor.encode(features)
        value, auxiliary = self.model.value_and_aux_from_encoded(validated, state, options)
        context = self.model.strategy_context(validated, value, auxiliary)
        evaluated = evaluate_actions_encoded(
            self.model.head, validated, self.model.actor_summary(state), options,
            batch.sequences[indices].to(self.device), batch.lengths[indices].to(self.device),
            batch.stopped[indices].to(self.device), value, context,
        )
        parameter_logprob, _ = evaluate_parameter_actions(
            self.model, validated, state, options,
            tuple(batch.macro_actions[int(index)] for index in indices),
        )
        joint = evaluated.log_prob + parameter_logprob
        weights = batch.episode_weight[indices].to(self.device)
        weights = weights / weights.sum()
        win_loss = -(joint * batch.advantage[indices].to(self.device) * weights).sum()
        prize_loss = -(
            joint * batch.prize_advantage[indices].to(self.device) * weights
        ).sum() * self.model.integrated_flags.prize_aux_actor_weight
        trunk = tuple(
            parameter for parameter in self.model.value_head.parameters()
            if parameter.requires_grad
        )
        meta = torch.nn.functional.cross_entropy(
            auxiliary["meta_logits"], batch.opponent_meta_label[indices].to(self.device)
        )
        critic = (
            value - batch.gae_return[indices].to(self.device)
        ).square().mean()
        for name, loss in (("meta", meta), ("critic", critic)):
            gradients = torch.autograd.grad(loss, trunk, retain_graph=True, allow_unused=True)
            present = [item.reshape(-1) for item in gradients if item is not None]
            if name == "meta":
                meta_norm = float(torch.cat(present).norm() if present else torch.zeros(()))
            else:
                critic_norm = float(torch.cat(present).norm() if present else torch.zeros(()))
        shared = [
            parameter for parameter in self.model.policy_strategy_adapter.parameters()
            if parameter.requires_grad
        ]
        metrics = gradient_diagnostics({"win": win_loss, "prize": prize_loss}, shared)
        metrics["gradient/meta/value_trunk_norm"] = meta_norm
        metrics["gradient/critic/value_trunk_norm"] = critic_norm
        return metrics

    def update(self, batch: PreparedBatch, *, update: int = 0) -> dict[str, float]:
        if batch.source_policy_update < 0:
            raise ValueError("source_policy_update must be nonnegative")
        if (
            batch.gamma != self.config.gamma
            or batch.gae_lambda != self.config.gae_lambda
            or batch.credit_clock != self.config.credit_clock
            or batch.loss_weighting != self.config.loss_weighting
        ):
            raise ValueError("prepared batch credit contract does not match PPO config")
        if self.model.representation_sha256() != self.initial_representation_sha256:
            raise RuntimeError("frozen Large Model 0806 representation changed before PPO")
        behavior_parameters = {
            name: tensor.detach().clone()
            for name, tensor in self.model.actor.action_decoder.named_parameters()
        }
        accumulators: dict[str, float] = {}
        minibatches = 0
        epochs_completed = 0
        early_stop = False
        rejected_kl = 0.0
        preupdate_mae_max = 0.0
        optimizer_steps = 0
        accumulation_count = 0
        samples_consumed = 0
        covered = torch.zeros(batch.decisions, dtype=torch.bool)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, batch.decisions, self.config.batch_size):
                indices = torch.arange(start, min(start + self.config.batch_size, batch.decisions))
                features = move_batch(
                    collate_feature_batches([batch.features[int(index)] for index in indices]),
                    self.device,
                )
                sequences = batch.sequences[indices].to(self.device)
                lengths = batch.lengths[indices].to(self.device)
                stopped = batch.stopped[indices].to(self.device)
                rollout_log_prob = batch.rollout_log_prob[indices].to(self.device)
                validated, state, options, current_value, auxiliary, context = (
                    self.model.encode_with_strategy(features)
                )
                replay = evaluate_actions_encoded(
                    self.model.head, validated, self.model.actor_summary(state), options,
                    sequences, lengths, stopped, current_value, context,
                )
                parameter_logprob, _ = evaluate_parameter_actions(
                    self.model, validated, state, options,
                    tuple(batch.macro_actions[int(index)] for index in indices),
                )
                preupdate_mae_max = max(
                    preupdate_mae_max,
                    float((rollout_log_prob - (replay.log_prob + parameter_logprob)).abs().max()),
                )
        if preupdate_mae_max > self.config.behavior_logprob_mae_limit:
            raise RuntimeError(
                f"behavior log-prob parity failed: {preupdate_mae_max}"
            )
        minibatches_per_epoch = (batch.decisions + self.config.batch_size - 1) // self.config.batch_size
        loop_epochs = self.config.epochs
        if self.config.optimization_mode == "fixed_optimizer_budget":
            required_microbatches = self.config.optimizer_steps_per_update * self.config.gradient_accumulation
            loop_epochs = (required_microbatches + minibatches_per_epoch - 1) // minibatches_per_epoch
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(loop_epochs):
            order = torch.randperm(batch.decisions)
            epoch_kls: list[float] = []
            for start in range(0, batch.decisions, self.config.batch_size):
                if (self.config.optimization_mode == "fixed_optimizer_budget"
                        and optimizer_steps >= self.config.optimizer_steps_per_update):
                    break
                indices = order[start : start + self.config.batch_size]
                samples_consumed += int(indices.numel())
                covered[indices.cpu()] = True
                features = move_batch(
                    collate_feature_batches([batch.features[int(index)] for index in indices]),
                    self.device,
                )
                sequences = batch.sequences[indices].to(self.device)
                lengths = batch.lengths[indices].to(self.device)
                stopped = batch.stopped[indices].to(self.device)
                rollout_log_prob = batch.rollout_log_prob[indices].to(self.device)
                advantage = batch.advantage[indices].to(self.device)
                if self.model.integrated_flags.enable_prize_aux:
                    advantage = advantage + (
                        self.model.integrated_flags.prize_aux_actor_weight
                        * batch.prize_advantage[indices].to(self.device)
                    )
                returns = batch.gae_return[indices].to(self.device)
                weights = batch.episode_weight[indices].to(self.device)
                weights = weights / weights.sum()
                validated, state, options = self.model.actor.encode(features)
                current_value, auxiliary = self.model.value_and_aux_from_encoded(
                    validated, state, options
                )
                context = self.model.strategy_context(validated, current_value, auxiliary)
                diagnostic_state = self.model.actor.action_decoder.initialize(
                    validated, self.model.actor_summary(state)
                )
                _, policy_delta = self.model.policy_strategy_adapter(
                    diagnostic_state.hidden, context
                )
                policy_residual_ratio = self.model.policy_strategy_adapter.effective_residual_ratio(
                    diagnostic_state.hidden, policy_delta
                )
                with torch.no_grad():
                    reference_eval = evaluate_actions_encoded(
                        self.reference, validated, self.model.actor_summary(state), options, sequences, lengths, stopped,
                        torch.zeros_like(returns), context,
                    )
                evaluated = evaluate_actions_encoded(
                    self.model.head,
                    validated,
                    self.model.actor_summary(state),
                    options,
                    sequences,
                    lengths,
                    stopped,
                    current_value,
                    context,
                )
                parameter_logprob, parameter_entropy = evaluate_parameter_actions(
                    self.model, validated, state, options,
                    tuple(batch.macro_actions[int(index)] for index in indices),
                )
                old_log_prob = rollout_log_prob
                current_joint_logprob = evaluated.log_prob + parameter_logprob
                log_ratio = current_joint_logprob - old_log_prob
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
                root_entropy = (evaluated.entropy * weights).sum()
                allocation_entropy = (parameter_entropy * weights).sum()
                entropy = root_entropy + allocation_entropy
                reference_delta = evaluated.log_prob - reference_eval.log_prob
                reference_kl = (0.5 * reference_delta.square() * weights).sum()
                prize_value_loss = current_value.new_zeros(())
                if self.model.integrated_flags.enable_prize_aux:
                    prize_value_loss = (
                        (auxiliary["v_prize"] - batch.prize_return[indices].to(self.device)).square()
                        * weights
                    ).sum()
                per_row_meta = torch.nn.functional.cross_entropy(
                    auxiliary["meta_logits"],
                    batch.opponent_meta_label[indices].to(self.device), reduction="none",
                )
                opponent_meta_loss = (per_row_meta * weights).sum()
                registry = LossRegistry()
                registry.register(LossTerm("L_policy_win", policy_loss, 1.0, "actor"))
                registry.register(LossTerm("L_value_win", value_loss,
                                           self.config.value_coefficient, "value_win"))
                registry.register(LossTerm("L_value_prize", prize_value_loss,
                    self.model.integrated_flags.prize_value_loss_weight, "value_prize",
                    self.model.integrated_flags.enable_prize_aux))
                registry.register(LossTerm(
                    "L_meta_anchor", opponent_meta_loss,
                    self.config.meta_anchor_coef, "value_win",
                ))
                registry.register(LossTerm("L_entropy_root", -root_entropy,
                                           self.config.entropy_coefficient, "actor"))
                registry.register(LossTerm("L_entropy_allocation", -allocation_entropy,
                                           self.config.entropy_coefficient, "allocation"))
                registry.register(LossTerm("L_reference_kl", reference_kl,
                                           self.config.reference_kl_coefficient, "actor"))
                total_loss = registry.total()
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("nonfinite full-semantic PPO loss")
                (total_loss / self.config.gradient_accumulation).backward()
                value_gate_grad = (
                    self.model.value_adapter.gate.grad.detach().abs()
                    if self.model.value_adapter.gate.grad is not None
                    else total_loss.new_zeros(())
                )
                policy_gate_grad = (
                    self.model.policy_strategy_adapter.gate.grad.detach().abs()
                    if self.model.policy_strategy_adapter.gate.grad is not None
                    else total_loss.new_zeros(())
                )
                accumulation_count += 1
                norm = total_loss.new_zeros(())
                if accumulation_count == self.config.gradient_accumulation:
                    norm = torch.nn.utils.clip_grad_norm_(
                        [value for value in self.model.parameters() if value.requires_grad],
                        self.config.max_grad_norm,
                    )
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite full-semantic PPO gradient")
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    accumulation_count = 0
                metrics = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "v_prize_loss": prize_value_loss,
                    "opponent_meta_loss": opponent_meta_loss,
                    "opponent_meta_accuracy": (
                        auxiliary["meta_logits"].argmax(dim=-1)
                        .eq(batch.opponent_meta_label[indices].to(self.device)).float()
                        * weights
                    ).sum(),
                    "opponent_meta_entropy": (
                        torch.distributions.Categorical(
                            logits=auxiliary["meta_logits"].float()
                        ).entropy() * weights
                    ).sum(),
                    "value_adapter_gate": self.model.value_adapter.gate.tanh(),
                    "policy_adapter_gate": self.model.policy_strategy_adapter.gate.tanh(),
                    "value_adapter_residual_ratio": (
                        auxiliary["value_adapter_residual_ratio"] * weights
                    ).sum(),
                    "policy_adapter_residual_ratio": (
                        policy_residual_ratio * weights
                    ).sum(),
                    "value_adapter_gate_grad_abs": value_gate_grad,
                    "policy_adapter_gate_grad_abs": policy_gate_grad,
                    "entropy": entropy,
                    "root_entropy": root_entropy,
                    "allocation_entropy": allocation_entropy,
                    "total_loss": total_loss,
                    "behavior_kl": approximate_kl,
                    "reference_kl": reference_kl,
                    "clip_fraction": (
                        ((ratio - 1.0).abs() > self.config.clip_ratio).float() * weights
                    ).sum(),
                    "gradient_norm": norm,
                }
                for name, value in metrics.items():
                    accumulators[name] = accumulators.get(name, 0.0) + float(value.detach())
                epoch_kls.append(approximate_kl_value)
                minibatches += 1
            epochs_completed = epoch + 1
            if early_stop:
                break
            if (self.config.optimization_mode == "fixed_optimizer_budget"
                    and optimizer_steps >= self.config.optimizer_steps_per_update):
                break
            if epoch_kls and sum(epoch_kls) / len(epoch_kls) > self.config.target_behavior_kl:
                early_stop = True
                break
        if accumulation_count:
            if early_stop:
                self.optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
            elif self.config.optimization_mode == "fixed_optimizer_budget":
                raise RuntimeError("fixed optimizer budget ended on a partial accumulation")
            else:
                correction = self.config.gradient_accumulation / accumulation_count
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
                norm = torch.nn.utils.clip_grad_norm_(
                    [value for value in self.model.parameters() if value.requires_grad],
                    self.config.max_grad_norm,
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite accumulated PPO gradient")
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        if minibatches == 0:
            raise RuntimeError("full-semantic PPO produced no minibatches")
        if self.model.representation_sha256() != self.initial_representation_sha256:
            raise RuntimeError("frozen Large Model 0806 representation changed during PPO")
        result = {f"ppo/{name}": value / minibatches for name, value in accumulators.items()}
        result.update(
            {
                "ppo/behavior_logprob_mae_preupdate": preupdate_mae_max,
                "ppo/epochs_completed": float(epochs_completed),
                "ppo/target_kl_early_stop": float(early_stop),
                "ppo/rejected_behavior_kl": rejected_kl,
                "ppo/minibatches_completed": float(minibatches),
                "ppo/optimizer_steps": float(optimizer_steps),
                "ppo/physical_minibatch": float(self.config.batch_size),
                "ppo/gradient_accumulation": float(self.config.gradient_accumulation),
                "ppo/effective_minibatch": float(
                    self.config.batch_size * self.config.gradient_accumulation
                ),
                "ppo/fixed_optimizer_budget": float(
                    self.config.optimization_mode == "fixed_optimizer_budget"
                ),
                "ppo/decisions": float(batch.decisions),
                "ppo/optimizer_samples_consumed": float(samples_consumed),
                "ppo/sample_coverage_ratio": float(covered.float().mean()),
                "ppo/sample_reuse_ratio": float(samples_consumed / max(1, batch.decisions)),
                **{
                    f"optimizer/lr/{group.get('name', index)}": float(group["lr"])
                    for index, group in enumerate(self.optimizer.param_groups)
                },
                "ppo/actor_relative_l2_vs_behavior": self._relative_l2(behavior_parameters),
                "ppo/actor_relative_l2_vs_reference": self._relative_l2(self.reference_parameters),
                "ppo/no_option_lora": 1.0,
                "ppo/gamma": self.config.gamma,
                "ppo/gae_lambda": self.config.gae_lambda,
                "ppo/credit_clock_turn_config": float(
                    self.config.credit_clock == "turn"
                ),
                "ppo/loss_weighting_turn_equal_config": float(
                    self.config.loss_weighting == "episode_equal_turns"
                ),
            }
        )
        result.update(training_batch_metrics(
            batch, include_detailed=is_sparse_diagnostic_update(update)
        ))
        return result


__all__ = ["PPOConfig", "PPOTrainer"]
