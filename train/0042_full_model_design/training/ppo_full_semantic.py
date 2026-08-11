"""0042 PPO with frozen semantic encoders and explicit strategy adapters."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import time

import torch

from ..policy.action_distribution import evaluate_actions_encoded
from ..policy.actor_critic import DecoderPolicyHead, SemanticActorCritic
from .batch_full_semantic import PreparedBatch, training_batch_metrics
from ..policy.compound_evaluation import evaluate_parameter_actions
from ..integrated.loss_registry import LossRegistry, LossTerm
from .metric_frequency import is_sparse_diagnostic_update
from ..integrated.diagnostics import gradient_diagnostics


FROZEN_TARGET_NAMES = (
    "rollout_log_prob",
    "old_value",
    "advantage",
    "gae_return",
)


def index_feature_batch(
    features: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
    feature_widths: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Gather a minibatch from one rollout-collated, device-resident feature store."""
    output: dict[str, torch.Tensor] = {}
    per_device_indices: dict[torch.device, torch.Tensor] = {}
    for name, value in features.items():
        index = per_device_indices.get(value.device)
        if index is None:
            index = indices.to(value.device, non_blocking=True)
            per_device_indices[value.device] = index
        selected = value.index_select(0, index)
        if feature_widths is not None and name in feature_widths:
            width = int(feature_widths[name].index_select(0, indices).max())
            selected = selected[:, :width].contiguous()
        output[name] = selected.to(device, non_blocking=True)
    return output


def complete_epoch_orders(
    decisions: int,
    epochs: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ...]:
    if decisions < 1 or epochs < 1:
        raise ValueError("decisions and epochs must be positive")
    return tuple(torch.randperm(decisions, generator=generator) for _ in range(epochs))


def epoch_minibatches(order: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, ...]:
    if order.ndim != 1 or batch_size < 1:
        raise ValueError("epoch order must be rank one and batch_size positive")
    return tuple(order[start : start + batch_size] for start in range(0, len(order), batch_size))


def behavior_guard_indices(
    decisions: int,
    max_samples: int,
    *,
    source_policy_update: int,
    required_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose one deterministic rollout-wide KL sample shared by all epochs."""
    if min(decisions, max_samples) < 1 or source_policy_update < 0:
        raise ValueError("behavior guard dimensions and source update must be valid")
    if decisions <= max_samples:
        sampled = torch.arange(decisions)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(4_200_420_911 + source_policy_update)
        sampled = torch.randperm(decisions, generator=generator)[:max_samples]
    required = (
        torch.empty(0, dtype=torch.long)
        if required_indices is None
        else required_indices.detach().to(device="cpu", dtype=torch.long)
    )
    if required.numel() and (int(required.min()) < 0 or int(required.max()) >= decisions):
        raise ValueError("required behavior guard index is outside the rollout")
    return torch.cat((sampled, required)).unique().sort().values


def snapshot_frozen_targets(
    targets: object | dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    source = targets if isinstance(targets, dict) else {
        name: getattr(targets, name) for name in FROZEN_TARGET_NAMES
    }
    return {name: source[name].detach().clone() for name in FROZEN_TARGET_NAMES}


def assert_frozen_targets(
    targets: object | dict[str, torch.Tensor],
    snapshot: dict[str, torch.Tensor],
) -> None:
    source = targets if isinstance(targets, dict) else {
        name: getattr(targets, name) for name in FROZEN_TARGET_NAMES
    }
    for name in FROZEN_TARGET_NAMES:
        if not torch.equal(source[name], snapshot[name]):
            raise RuntimeError(f"rollout-frozen PPO target changed during update: {name}")


def sample_usage_metrics(
    usage: torch.Tensor,
    *,
    samples_examined: int,
    samples_optimized: int,
) -> dict[str, float]:
    valid = int(usage.numel())
    unique = int(usage.gt(0).sum())
    return {
        "samples_examined": float(samples_examined),
        "samples_optimized": float(samples_optimized),
        "unique_decisions_optimized": float(unique),
        "coverage_ratio": unique / max(1, valid),
        "optimized_slots_per_valid_decision": samples_optimized / max(1, valid),
        "optimized_slots_per_unique_decision": samples_optimized / max(1, unique),
        "usage_count_0": float(usage.eq(0).sum()),
        "usage_count_1": float(usage.eq(1).sum()),
        "usage_count_2": float(usage.eq(2).sum()),
        "usage_count_3": float(usage.eq(3).sum()),
        "usage_count_4_plus": float(usage.ge(4).sum()),
        "usage_fraction_0": float(usage.eq(0).float().mean()),
        "usage_fraction_1": float(usage.eq(1).float().mean()),
        "usage_fraction_2": float(usage.eq(2).float().mean()),
        "usage_fraction_3": float(usage.eq(3).float().mean()),
        "usage_fraction_4_plus": float(usage.ge(4).float().mean()),
    }


@dataclass(frozen=True, slots=True)
class PPOConfig:
    protocol_version: str = "ppo_protocol_v2"
    gamma: float = 1.0
    gae_lambda: float = 0.95
    credit_clock: str = "turn"
    loss_weighting: str = "episode_equal_decisions"
    epochs: int = 3
    batch_size: int = 2048
    forward_microbatch_size: int = 1024
    decoder_learning_rate: float = 5.0e-6
    policy_adapter_learning_rate: float = 5.0e-6
    allocation_learning_rate: float = 5.0e-6
    value_learning_rate: float = 1.0e-4
    prize_learning_rate: float = 1.0e-4
    meta_anchor_coef: float = 0.10
    weight_decay: float = 0.0
    clip_ratio: float = 0.10
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.003
    reference_kl_coefficient: float = 0.02
    target_behavior_kl: float = 0.015
    hard_behavior_kl_guard: float = 0.025
    max_grad_norm: float = 0.5
    behavior_logprob_mae_limit: float = 1.0e-4
    behavior_guard_samples: int = 4096
    behavior_probe_batch_size: int = 512
    full_behavior_audit_interval: int = 10
    gradient_accumulation: int = 1

    @property
    def actor_learning_rate(self) -> float:
        """Compatibility summary; actor groups are configured independently."""
        return self.decoder_learning_rate

    def actor_group_learning_rates(self) -> dict[str, float]:
        return {
            "action_decoder": self.decoder_learning_rate,
            "policy_strategy_adapter": self.policy_adapter_learning_rate,
            "allocation_head": self.allocation_learning_rate,
        }

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
        if self.protocol_version != "ppo_protocol_v2":
            raise ValueError("0042 formal PPO must use ppo_protocol_v2")
        if min(
            self.epochs,
            self.batch_size,
            self.forward_microbatch_size,
            self.behavior_guard_samples,
            self.behavior_probe_batch_size,
            self.full_behavior_audit_interval,
            self.gradient_accumulation,
        ) < 1:
            raise ValueError("PPO capacity settings must be positive")
        if self.gradient_accumulation != 1:
            raise ValueError("Protocol V2 uses one optimizer step per data minibatch")
        if not 0.0 < self.target_behavior_kl < self.hard_behavior_kl_guard:
            raise ValueError("behavior KL target must be below the hard guard")
        if self.meta_anchor_coef <= 0:
            raise ValueError("0042 meta_anchor_coef must be explicitly positive")
        if min(
            self.decoder_learning_rate,
            self.policy_adapter_learning_rate,
            self.allocation_learning_rate,
            self.value_learning_rate,
            self.prize_learning_rate,
        ) <= 0:
            raise ValueError("all PPO optimizer learning rates must be positive")


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
        if config.meta_anchor_coef != model.integrated_flags.meta_anchor_coef:
            raise ValueError(
                "PPO meta_anchor_coef must match the selected 0042 preset: "
                f"ppo={config.meta_anchor_coef} preset={model.integrated_flags.meta_anchor_coef}"
            )
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
                    "lr": config.decoder_learning_rate,
                },
                {
                    "name": "policy_strategy_adapter",
                    "params": model.policy_strategy_adapter.parameters(),
                    "lr": config.policy_adapter_learning_rate,
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
                    "lr": config.allocation_learning_rate,
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

    def parameter_partition_manifest(self) -> dict[str, dict[str, object]]:
        actor_groups = {"action_decoder", "policy_strategy_adapter", "allocation_head"}
        value_groups = {"value_win", "value_adapter", "value_prize"}
        actor_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            if str(group["name"]) in actor_groups
            for parameter in group["params"]
        }
        value_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            if str(group["name"]) in value_groups
            for parameter in group["params"]
        }
        parameter_sizes = {
            id(parameter): int(parameter.numel())
            for parameter in self.model.parameters() if parameter.requires_grad
        }
        shared = actor_ids.intersection(value_ids)
        actor_only = actor_ids - shared
        value_only = value_ids - shared
        categorized = actor_ids.union(value_ids)
        uncategorized = set(parameter_sizes) - categorized
        if uncategorized:
            raise RuntimeError("trainable parameters are missing from PPO optimizer partitions")
        return {
            "actor_only": {
                "parameters": sum(parameter_sizes[item] for item in actor_only),
                "learning_rates": self.config.actor_group_learning_rates(),
            },
            "value_only": {
                "parameters": sum(parameter_sizes[item] for item in value_only),
                "learning_rate": self.config.value_learning_rate,
            },
            "shared_trainable": {
                "parameters": sum(parameter_sizes[item] for item in shared),
                "learning_rate": self.config.decoder_learning_rate,
            },
        }

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
        relative_side = batch.features["global_cat"][:, 2]
        defined_rows = relative_side.ne(0).nonzero(as_tuple=False).flatten().cpu()
        count = min(samples, int(defined_rows.numel()))
        if count < 1:
            raise ValueError("sparse PPO gradient diagnostic has no post-seat decision")
        indices = defined_rows[:count]
        features = index_feature_batch(
            batch.features, indices, self.device, batch.feature_widths
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
        effective_meta = self.config.meta_anchor_coef * meta_norm
        effective_critic = self.config.value_coefficient * critic_norm
        metrics["gradient/meta/value_trunk_effective_norm"] = effective_meta
        metrics["gradient/critic/value_trunk_effective_norm"] = effective_critic
        metrics["gradient/meta_to_critic_effective_ratio"] = (
            effective_meta / max(effective_critic, 1.0e-12)
        )
        return metrics

    def _behavior_probe(
        self,
        batch: PreparedBatch,
        *,
        probe_indices: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Measure current-vs-behavior policy on a fixed rollout subset."""
        selected_indices = (
            torch.arange(batch.decisions)
            if probe_indices is None
            else probe_indices.detach().to(device="cpu", dtype=torch.long)
        )
        if (
            selected_indices.ndim != 1
            or selected_indices.numel() < 1
            or int(selected_indices.min()) < 0
            or int(selected_indices.max()) >= batch.decisions
            or selected_indices.unique().numel() != selected_indices.numel()
        ):
            raise ValueError("behavior probe indices must be unique valid rollout rows")
        probe_count = int(selected_indices.numel())
        weight_sum = 0.0
        kl_sum = 0.0
        root_kl_sum = macro_kl_sum = 0.0
        root_weight_sum = macro_weight_sum = 0.0
        clip_sum = 0.0
        entropy_sum = 0.0
        mae_max = 0.0
        predictions = torch.empty(probe_count, dtype=torch.float32)
        absolute_errors = torch.empty(probe_count, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            probe_batch_size = min(
                self.config.batch_size, self.config.behavior_probe_batch_size
            )
            for start in range(0, probe_count, probe_batch_size):
                positions = torch.arange(
                    start, min(start + probe_batch_size, probe_count)
                )
                indices = selected_indices.index_select(0, positions)
                features = index_feature_batch(
                    batch.features, indices, self.device, batch.feature_widths
                )
                sequences = batch.sequences[indices].to(self.device)
                lengths = batch.lengths[indices].to(self.device)
                stopped = batch.stopped[indices].to(self.device)
                old_logprob = batch.rollout_log_prob[indices].to(self.device)
                validated, state, options, value, auxiliary, context = (
                    self.model.encode_with_strategy(features)
                )
                evaluated = evaluate_actions_encoded(
                    self.model.head, validated, self.model.actor_summary(state), options,
                    sequences, lengths, stopped, value, context,
                )
                parameter_logprob, parameter_entropy = evaluate_parameter_actions(
                    self.model, validated, state, options,
                    tuple(batch.macro_actions[int(index)] for index in indices),
                )
                current_logprob = evaluated.log_prob + parameter_logprob
                log_ratio = current_logprob - old_logprob
                ratio = log_ratio.exp()
                weights = batch.episode_weight[indices].to(self.device).float()
                per_row_kl = (ratio - 1.0) - log_ratio
                kl_sum += float((per_row_kl * weights).sum())
                micro_macro_mask = torch.tensor(
                    [batch.macro_actions[int(index)] is not None for index in indices],
                    device=self.device,
                    dtype=torch.bool,
                )
                if bool((~micro_macro_mask).any()):
                    root_kl_sum += float(
                        (per_row_kl[~micro_macro_mask] * weights[~micro_macro_mask]).sum()
                    )
                    root_weight_sum += float(weights[~micro_macro_mask].sum())
                if bool(micro_macro_mask.any()):
                    macro_kl_sum += float(
                        (per_row_kl[micro_macro_mask] * weights[micro_macro_mask]).sum()
                    )
                    macro_weight_sum += float(weights[micro_macro_mask].sum())
                clip_sum += float(
                    (((ratio - 1.0).abs() > self.config.clip_ratio).float() * weights).sum()
                )
                entropy_sum += float(
                    ((evaluated.entropy + parameter_entropy) * weights).sum()
                )
                weight_sum += float(weights.sum())
                absolute_error = log_ratio.abs().detach().float().cpu()
                absolute_errors[positions] = absolute_error
                mae_max = max(mae_max, float(absolute_error.max()))
                predictions[positions] = evaluated.value.detach().float().cpu()
        targets = batch.gae_return.index_select(0, selected_indices).detach().float().cpu()
        residual = targets - predictions
        target_variance = float(targets.var(unbiased=False))
        explained_variance = (
            1.0 - float(residual.var(unbiased=False)) / target_variance
            if target_variance > 1.0e-12 else 0.0
        )
        denominator = max(weight_sum, 1.0e-12)
        worst_position = int(absolute_errors.argmax())
        worst_index = int(selected_indices[worst_position])
        macro_mask = torch.tensor([
            batch.macro_actions[int(index)] is not None for index in selected_indices
        ], dtype=torch.bool)
        root_mask = ~macro_mask
        return {
            "behavior_kl": kl_sum / denominator,
            "behavior_root_kl": root_kl_sum / max(root_weight_sum, 1.0e-12),
            "behavior_macro_kl": macro_kl_sum / max(macro_weight_sum, 1.0e-12),
            "behavior_root_weight_fraction": root_weight_sum / denominator,
            "behavior_macro_weight_fraction": macro_weight_sum / denominator,
            "clip_fraction": clip_sum / denominator,
            "entropy": entropy_sum / denominator,
            "behavior_logprob_mae_max": mae_max,
            "behavior_logprob_abs_error_mean": float(absolute_errors.mean()),
            "behavior_logprob_abs_error_p95": float(
                absolute_errors.quantile(0.95)
            ),
            "behavior_logprob_worst_index": float(worst_index),
            "behavior_logprob_worst_is_macro": float(macro_mask[worst_position]),
            "behavior_logprob_root_error_max": float(
                absolute_errors[root_mask].max() if bool(root_mask.any()) else 0.0
            ),
            "behavior_logprob_macro_error_max": float(
                absolute_errors[macro_mask].max() if bool(macro_mask.any()) else 0.0
            ),
            "explained_variance": explained_variance,
            "value_prediction_mean": float(predictions.mean()),
            "value_prediction_std": float(predictions.std(unbiased=False)),
            "sample_count": float(probe_count),
            "sample_fraction": probe_count / batch.decisions,
        }

    def _singleton_behavior_logprob_error(
        self, batch: PreparedBatch, index: int
    ) -> float:
        """Replay one row at its original collection widths for parity diagnosis."""
        indices = torch.tensor([index], dtype=torch.long)
        features = index_feature_batch(
            batch.features, indices, self.device, batch.feature_widths
        )
        self.model.eval()
        with torch.no_grad():
            validated, state, options, value, auxiliary, context = (
                self.model.encode_with_strategy(features)
            )
            evaluated = evaluate_actions_encoded(
                self.model.head,
                validated,
                self.model.actor_summary(state),
                options,
                batch.sequences[indices].to(self.device),
                batch.lengths[indices].to(self.device),
                batch.stopped[indices].to(self.device),
                value,
                context,
            )
            parameter_logprob, _ = evaluate_parameter_actions(
                self.model,
                validated,
                state,
                options,
                (batch.macro_actions[index],),
            )
            current = evaluated.log_prob + parameter_logprob
            old = batch.rollout_log_prob[indices].to(self.device)
            return float((current - old).abs().max())

    def _backward_microbatch(
        self,
        batch: PreparedBatch,
        indices: torch.Tensor,
        *,
        logical_weight_sum: torch.Tensor,
    ) -> dict[str, object]:
        """Accumulate one physical graph toward one logical PPO optimizer step."""
        features = index_feature_batch(
            batch.features, indices, self.device, batch.feature_widths
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
        weights = batch.episode_weight[indices].to(self.device) / logical_weight_sum
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
        policy_residual_ratio = (
            self.model.policy_strategy_adapter.effective_residual_ratio(
                diagnostic_state.hidden, policy_delta
            )
        )
        with torch.no_grad():
            reference_eval = evaluate_actions_encoded(
                self.reference, validated, self.model.actor_summary(state), options,
                sequences, lengths, stopped, torch.zeros_like(returns), context,
            )
        evaluated = evaluate_actions_encoded(
            self.model.head, validated, self.model.actor_summary(state), options,
            sequences, lengths, stopped, current_value, context,
        )
        parameter_logprob, parameter_entropy = evaluate_parameter_actions(
            self.model, validated, state, options,
            tuple(batch.macro_actions[int(index)] for index in indices),
        )
        current_joint_logprob = evaluated.log_prob + parameter_logprob
        log_ratio = current_joint_logprob - rollout_log_prob
        ratio = log_ratio.exp()
        approximate_kl = (((ratio - 1.0) - log_ratio) * weights).sum()
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
        labels = batch.opponent_meta_label[indices].to(self.device)
        per_row_meta = torch.nn.functional.cross_entropy(
            auxiliary["meta_logits"], labels, reduction="none"
        )
        opponent_meta_loss = (per_row_meta * weights).sum()
        registry = LossRegistry()
        registry.register(LossTerm("L_policy_win", policy_loss, 1.0, "actor"))
        registry.register(LossTerm(
            "L_value_win", value_loss, self.config.value_coefficient, "value_win"
        ))
        registry.register(LossTerm(
            "L_value_prize", prize_value_loss,
            self.model.integrated_flags.prize_value_loss_weight, "value_prize",
            self.model.integrated_flags.enable_prize_aux,
        ))
        registry.register(LossTerm(
            "L_meta_anchor", opponent_meta_loss,
            self.config.meta_anchor_coef, "value_win",
        ))
        registry.register(LossTerm(
            "L_entropy_root", -root_entropy,
            self.config.entropy_coefficient, "actor",
        ))
        registry.register(LossTerm(
            "L_entropy_allocation", -allocation_entropy,
            self.config.entropy_coefficient, "allocation",
        ))
        registry.register(LossTerm(
            "L_reference_kl", reference_kl,
            self.config.reference_kl_coefficient, "actor",
        ))
        total_loss = registry.total()
        if not torch.isfinite(total_loss):
            raise FloatingPointError("nonfinite full-semantic PPO loss")
        total_loss.backward()
        value_ratio = auxiliary["value_adapter_residual_ratio"].detach().float()
        policy_ratio = policy_residual_ratio.detach().float()
        detached_ratio = ratio.detach().float()
        scalars = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "v_prize_loss": prize_value_loss.detach(),
            "opponent_meta_loss": opponent_meta_loss.detach(),
            "opponent_meta_accuracy": (
                auxiliary["meta_logits"].argmax(dim=-1).eq(labels).float() * weights
            ).sum().detach(),
            "opponent_meta_entropy": (
                torch.distributions.Categorical(
                    logits=auxiliary["meta_logits"].float()
                ).entropy() * weights
            ).sum().detach(),
            "value_adapter_residual_ratio": (value_ratio * weights).sum(),
            "policy_adapter_residual_ratio": (policy_ratio * weights).sum(),
            "entropy": entropy.detach(),
            "entropy_weighted_contribution": (
                -self.config.entropy_coefficient * entropy
            ).detach(),
            "root_entropy": root_entropy.detach(),
            "allocation_entropy": allocation_entropy.detach(),
            "total_loss": total_loss.detach(),
            "behavior_kl": approximate_kl.detach(),
            "reference_kl": reference_kl.detach(),
            "reference_kl_weighted_contribution": (
                self.config.reference_kl_coefficient * reference_kl
            ).detach(),
            "value_loss_weighted_contribution": (
                self.config.value_coefficient * value_loss
            ).detach(),
            "meta_anchor_raw": opponent_meta_loss.detach(),
            "meta_anchor_weighted_contribution": (
                self.config.meta_anchor_coef * opponent_meta_loss
            ).detach(),
            "clip_fraction": (
                ((ratio - 1.0).abs() > self.config.clip_ratio).float() * weights
            ).sum().detach(),
        }
        return {
            "scalars": scalars,
            "ratio": detached_ratio,
            "value_ratio": value_ratio,
            "policy_ratio": policy_ratio,
            "value_prediction_sum": evaluated.value.detach().sum(),
            "value_target_sum": returns.detach().sum(),
        }

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
            raise RuntimeError("frozen Policy-0809 representation changed before PPO")
        behavior_parameters = {
            name: tensor.detach().clone()
            for name, tensor in self.model.actor.action_decoder.named_parameters()
        }
        frozen_targets = snapshot_frozen_targets(batch)
        macro_indices = torch.tensor([
            index for index, value in enumerate(batch.macro_actions) if value is not None
        ], dtype=torch.long)
        guard_indices = behavior_guard_indices(
            batch.decisions,
            self.config.behavior_guard_samples,
            source_policy_update=batch.source_policy_update,
            required_indices=macro_indices,
        )
        full_preupdate_audit = (
            update <= 1 or update % self.config.full_behavior_audit_interval == 0
        )
        preupdate = self._behavior_probe(
            batch,
            probe_indices=None if full_preupdate_audit else guard_indices,
        )
        preupdate_mae_max = preupdate["behavior_logprob_mae_max"]
        optimizer_steps = 0
        minibatches = 0
        epochs_completed = 0
        samples_examined = 0
        samples_optimized = 0
        usage = torch.zeros(batch.decisions, dtype=torch.int16)
        overall_sums: dict[str, float] = {}
        overall_samples = 0
        ratio_sum = ratio_square_sum = 0.0
        ratio_count = 0
        ratio_min = float("inf")
        ratio_max = float("-inf")
        if preupdate_mae_max > self.config.behavior_logprob_mae_limit:
            worst_index = int(preupdate["behavior_logprob_worst_index"])
            singleton_error = self._singleton_behavior_logprob_error(
                batch, worst_index
            )
            raise RuntimeError(
                "behavior log-prob parity failed: "
                f"max={preupdate_mae_max} "
                f"mean={preupdate['behavior_logprob_abs_error_mean']} "
                f"p95={preupdate['behavior_logprob_abs_error_p95']} "
                f"worst_index={worst_index} "
                f"worst_is_macro={bool(preupdate['behavior_logprob_worst_is_macro'])} "
                f"root_max={preupdate['behavior_logprob_root_error_max']} "
                f"macro_max={preupdate['behavior_logprob_macro_error_max']} "
                f"singleton_original_width_error={singleton_error}"
            )
        minibatches_per_epoch = math.ceil(batch.decisions / self.config.batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        progress_started = time.perf_counter()
        expected_optimizer_steps = self.config.epochs * minibatches_per_epoch
        early_stop = False
        hard_guard_triggered = False
        stop_kl = 0.0
        result: dict[str, float] = {}
        orders = complete_epoch_orders(batch.decisions, self.config.epochs)
        for epoch, order in enumerate(orders, start=1):
            epoch_sums: dict[str, float] = {}
            epoch_samples = 0
            epoch_steps = 0
            progression_thresholds = {
                max(1, math.ceil(minibatches_per_epoch * fraction / 4)): fraction * 25
                for fraction in range(1, 5)
            }
            for indices in epoch_minibatches(order, self.config.batch_size):
                count = int(indices.numel())
                samples_examined += count
                logical_weight_sum = batch.episode_weight[indices].to(
                    self.device
                ).sum()
                micro_results = []
                for micro_indices in epoch_minibatches(
                    indices, self.config.forward_microbatch_size
                ):
                    micro_results.append(self._backward_microbatch(
                        batch,
                        micro_indices,
                        logical_weight_sum=logical_weight_sum,
                    ))
                detached_ratios = torch.cat([
                    item["ratio"] for item in micro_results
                ])
                ratio_sum += float(detached_ratios.sum())
                ratio_square_sum += float(detached_ratios.square().sum())
                ratio_count += detached_ratios.numel()
                ratio_min = min(ratio_min, float(detached_ratios.min()))
                ratio_max = max(ratio_max, float(detached_ratios.max()))
                value_gate_grad = (
                    self.model.value_adapter.gate.grad.detach().abs()
                    if self.model.value_adapter.gate.grad is not None
                    else logical_weight_sum.new_zeros(())
                )
                policy_gate_grad = (
                    self.model.policy_strategy_adapter.gate.grad.detach().abs()
                    if self.model.policy_strategy_adapter.gate.grad is not None
                    else logical_weight_sum.new_zeros(())
                )
                def grad_l2(parameters) -> torch.Tensor:
                    present = [
                        parameter.grad.detach().float().square().sum()
                        for parameter in parameters if parameter.grad is not None
                    ]
                    return (
                        torch.stack(present).sum().sqrt()
                        if present else logical_weight_sum.new_zeros(())
                    )

                value_mlp_grad = grad_l2(self.model.value_adapter.mlp.parameters())
                value_embedding_grad = grad_l2(self.model.value_adapter.own_embedding.parameters())
                policy_mlp_grad = grad_l2(self.model.policy_strategy_adapter.mlp.parameters())
                policy_embedding_grad = grad_l2(
                    self.model.policy_strategy_adapter.own_embedding.parameters()
                )
                group_gradient_norms = {
                    str(group["name"]): grad_l2(group["params"])
                    for group in self.optimizer.param_groups
                }
                norm = torch.nn.utils.clip_grad_norm_(
                    [value for value in self.model.parameters() if value.requires_grad],
                    self.config.max_grad_norm,
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite full-semantic PPO gradient")
                clip_scale = min(
                    1.0,
                    self.config.max_grad_norm / max(float(norm), 1.0e-12),
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                epoch_steps += 1
                minibatches += 1
                samples_optimized += count
                usage[indices.cpu()] += 1
                value_ratio = torch.cat([
                    item["value_ratio"] for item in micro_results
                ])
                policy_ratio = torch.cat([
                    item["policy_ratio"] for item in micro_results
                ])
                scalar_names = tuple(micro_results[0]["scalars"])
                logical_scalars = {
                    name: torch.stack([
                        item["scalars"][name] for item in micro_results
                    ]).sum()
                    for name in scalar_names
                }
                metrics = {
                    **logical_scalars,
                    "value_adapter_gate_raw": self.model.value_adapter.gate,
                    "value_adapter_gate_effective": self.model.value_adapter.gate.tanh(),
                    "policy_adapter_gate_raw": self.model.policy_strategy_adapter.gate,
                    "policy_adapter_gate_effective": self.model.policy_strategy_adapter.gate.tanh(),
                    "value_adapter_gate": self.model.value_adapter.gate.tanh(),
                    "policy_adapter_gate": self.model.policy_strategy_adapter.gate.tanh(),
                    "value_adapter_residual_ratio_p50": value_ratio.quantile(0.50),
                    "value_adapter_residual_ratio_p95": value_ratio.quantile(0.95),
                    "value_adapter_residual_ratio_max": value_ratio.max(),
                    "policy_adapter_residual_ratio_p50": policy_ratio.quantile(0.50),
                    "policy_adapter_residual_ratio_p95": policy_ratio.quantile(0.95),
                    "policy_adapter_residual_ratio_max": policy_ratio.max(),
                    "value_adapter_gate_grad_abs": value_gate_grad,
                    "policy_adapter_gate_grad_abs": policy_gate_grad,
                    "value_adapter_mlp_grad_l2": value_mlp_grad,
                    "value_adapter_own_embedding_grad_l2": value_embedding_grad,
                    "policy_adapter_mlp_grad_l2": policy_mlp_grad,
                    "policy_adapter_own_embedding_grad_l2": policy_embedding_grad,
                    "value_prediction_mean": torch.stack([
                        item["value_prediction_sum"] for item in micro_results
                    ]).sum() / count,
                    "value_target_mean": torch.stack([
                        item["value_target_sum"] for item in micro_results
                    ]).sum() / count,
                    "gradient_norm": norm,
                    "gradient_clip_scale": torch.tensor(
                        clip_scale, device=self.device
                    ),
                }
                for group in self.optimizer.param_groups:
                    group_name = str(group["name"])
                    group_norm = group_gradient_norms[group_name]
                    metrics[f"gradient_group_{group_name}_preclip_l2"] = group_norm
                    metrics[f"gradient_group_{group_name}_postclip_l2"] = (
                        group_norm * clip_scale
                    )
                    metrics[f"learning_rate_{group_name}"] = torch.tensor(
                        float(group["lr"]), device=self.device
                    )
                metrics["physical_microbatches"] = torch.tensor(
                    float(len(micro_results)), device=self.device
                )
                for name, value in metrics.items():
                    numeric = float(value.detach())
                    epoch_sums[name] = epoch_sums.get(name, 0.0) + numeric * count
                    overall_sums[name] = overall_sums.get(name, 0.0) + numeric * count
                epoch_samples += count
                overall_samples += count
                elapsed = max(time.perf_counter() - progress_started, 1.0e-9)
                width = 20
                completed = min(width, int(width * optimizer_steps / expected_optimizer_steps))
                print(
                    f"\r[0042 PPO][update {update:04d} epoch {epoch}/{self.config.epochs}] "
                    f"[{'#' * completed}{'.' * (width - completed)}] "
                    f"step {epoch_steps}/{minibatches_per_epoch} "
                    f"global {optimizer_steps}/{expected_optimizer_steps} "
                    f"{optimizer_steps / elapsed:.2f} iter/s",
                    end="\n" if epoch_steps == minibatches_per_epoch else "",
                    flush=True,
                    file=sys.stderr,
                )
                if epoch_steps in progression_thresholds:
                    progress = int(progression_thresholds[epoch_steps])
                    for name in (
                        "behavior_kl", "clip_fraction", "entropy", "policy_loss",
                        "value_loss", "gradient_norm",
                    ):
                        result[f"ppo/epoch_{epoch}/progress_{progress}/{name}"] = (
                            epoch_sums[name] / epoch_samples
                        )
            epochs_completed = epoch
            expected_usage = torch.full_like(usage, epoch)
            if not torch.equal(usage, expected_usage):
                raise RuntimeError(f"PPO epoch {epoch} did not cover every decision exactly once")
            assert_frozen_targets(batch, frozen_targets)
            probe = self._behavior_probe(batch, probe_indices=guard_indices)
            for name, total in epoch_sums.items():
                result[f"ppo/epoch_{epoch}/{name}"] = total / epoch_samples
            for name, value in probe.items():
                result[f"ppo/epoch_{epoch}/end_{name}"] = value
            result[f"ppo/epoch_{epoch}/optimizer_steps"] = float(epoch_steps)
            result[f"ppo/epoch_{epoch}/samples_optimized"] = float(epoch_samples)
            result[f"ppo/epoch_{epoch}/coverage_ratio"] = 1.0
            result[f"ppo/epoch_{epoch}/cumulative_reuse"] = float(epoch)
            stop_kl = probe["behavior_kl"]
            hard_guard_triggered = stop_kl >= self.config.hard_behavior_kl_guard
            if stop_kl >= self.config.target_behavior_kl:
                early_stop = epoch < self.config.epochs
                break
        if minibatches == 0:
            raise RuntimeError("full-semantic PPO produced no minibatches")
        assert_frozen_targets(batch, frozen_targets)
        if self.model.representation_sha256() != self.initial_representation_sha256:
            raise RuntimeError("frozen pretrained representation changed during PPO")
        for name, value in overall_sums.items():
            result[f"ppo/{name}"] = value / max(1, overall_samples)
        ratio_mean = ratio_sum / max(1, ratio_count)
        optimizer_elapsed = max(time.perf_counter() - progress_started, 1.0e-9)
        result.update({
            "ppo/ratio_mean": ratio_mean,
            "ppo/ratio_std": max(0.0, ratio_square_sum / max(1, ratio_count) - ratio_mean ** 2) ** 0.5,
            "ppo/ratio_min": ratio_min if ratio_count else 0.0,
            "ppo/ratio_max": ratio_max if ratio_count else 0.0,
            "ppo/meta_anchor_coef": self.config.meta_anchor_coef,
            "ppo/reference_kl_coefficient": self.config.reference_kl_coefficient,
            "ppo/value_loss_coefficient": self.config.value_coefficient,
            "ppo/entropy_coefficient": self.config.entropy_coefficient,
            "ppo/optimizer_iterations_per_second": optimizer_steps / optimizer_elapsed,
            "system/ppo_peak_allocated_bytes": float(
                torch.cuda.max_memory_allocated(self.device)
            ),
            "system/ppo_peak_reserved_bytes": float(
                torch.cuda.max_memory_reserved(self.device)
            ),
            "ppo/behavior_probe_batch_size": float(
                min(self.config.batch_size, self.config.behavior_probe_batch_size)
            ),
        })
        usage_metrics = sample_usage_metrics(
            usage,
            samples_examined=samples_examined,
            samples_optimized=samples_optimized,
        )
        result.update(
            {
                "ppo/behavior_logprob_mae_preupdate": preupdate_mae_max,
                "ppo/behavior_logprob_abs_error_mean_preupdate": preupdate[
                    "behavior_logprob_abs_error_mean"
                ],
                "ppo/behavior_logprob_abs_error_p95_preupdate": preupdate[
                    "behavior_logprob_abs_error_p95"
                ],
                "ppo/behavior_logprob_root_error_max_preupdate": preupdate[
                    "behavior_logprob_root_error_max"
                ],
                "ppo/behavior_logprob_macro_error_max_preupdate": preupdate[
                    "behavior_logprob_macro_error_max"
                ],
                "ppo/behavior_logprob_audit_samples_preupdate": preupdate["sample_count"],
                "ppo/behavior_logprob_audit_fraction_preupdate": preupdate["sample_fraction"],
                "ppo/behavior_logprob_full_audit_preupdate": float(full_preupdate_audit),
                "ppo/epochs_completed": float(epochs_completed),
                "ppo/target_kl_early_stop": float(early_stop),
                "ppo/hard_behavior_kl_guard_triggered": float(hard_guard_triggered),
                "ppo/epoch_end_behavior_kl": stop_kl,
                "ppo/target_behavior_kl": self.config.target_behavior_kl,
                "ppo/hard_behavior_kl_guard": self.config.hard_behavior_kl_guard,
                "ppo/minibatches_completed": float(minibatches),
                "ppo/optimizer_steps": float(optimizer_steps),
                "ppo/physical_minibatch": float(
                    min(self.config.batch_size, self.config.forward_microbatch_size)
                ),
                "ppo/gradient_accumulation": float(math.ceil(
                    self.config.batch_size
                    / min(self.config.batch_size, self.config.forward_microbatch_size)
                )),
                "ppo/effective_minibatch": float(self.config.batch_size),
                "ppo/fixed_optimizer_budget": 0.0,
                "ppo/decisions": float(batch.decisions),
                **{f"ppo/{name}": value for name, value in usage_metrics.items()},
                "ppo/optimizer_samples_consumed": float(samples_optimized),
                "ppo/sample_coverage_ratio": usage_metrics["coverage_ratio"],
                "ppo/sample_reuse_ratio": usage_metrics[
                    "optimized_slots_per_valid_decision"
                ],
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
                "ppo/protocol_v2": 1.0,
                "ppo/drop_last": 0.0,
                "ppo/sampling_without_replacement": 1.0,
            }
        )
        result.update(training_batch_metrics(
            batch, include_detailed=is_sparse_diagnostic_update(update)
        ))
        return result


__all__ = [
    "PPOConfig",
    "PPOTrainer",
    "assert_frozen_targets",
    "complete_epoch_orders",
    "epoch_minibatches",
    "index_feature_batch",
    "sample_usage_metrics",
    "snapshot_frozen_targets",
]
