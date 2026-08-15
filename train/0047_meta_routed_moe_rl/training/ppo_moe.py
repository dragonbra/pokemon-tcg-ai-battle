"""Compact PPO implementation over the exact 0047 routed effective policy."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time

import torch

from ..integrated.loss_registry import LossRegistry, LossTerm
from ..policy.moe_actor_critic import MetaRoutedMoEActorCritic
from ..policy.moe_distribution import evaluate_moe_actions
from .batch_full_semantic import PreparedBatch, training_batch_metrics


def index_feature_batch(features, indices, device, feature_widths=None):
    output = {}
    for name, value in features.items():
        selected = value.index_select(0, indices.to(value.device))
        if feature_widths is not None and name in feature_widths:
            width = int(feature_widths[name].index_select(0, indices).max())
            selected = selected[:, :width].contiguous()
        output[name] = selected.to(device, non_blocking=True)
    return output


def set_batch_own_archetype_ids(model, batch, indices):
    model.set_runtime_own_archetype_ids(
        batch.own_archetype_id.index_select(0, indices).to(model.device)
    )


@dataclass(frozen=True, slots=True)
class PPOConfig:
    protocol_version: str = "ppo_protocol_v2_effective_moe"
    gamma: float = 1.0
    gae_lambda: float = 0.95
    credit_clock: str = "turn"
    loss_weighting: str = "episode_equal_decisions"
    epochs: int = 3
    batch_size: int = 2048
    forward_microbatch_size: int = 512
    decoder_learning_rate: float = 1.0e-5
    allocation_learning_rate: float = 1.0e-5
    option_lora_learning_rate: float = 2.0e-5
    router_learning_rate: float = 2.0e-6
    value_learning_rate: float = 2.0e-5
    prize_learning_rate: float = 2.0e-5
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

    def validate(self) -> None:
        if self.gamma != 1.0 or not 0 < self.gae_lambda <= 1:
            raise ValueError("0047 PPO credit settings changed")
        if self.epochs < 1 or min(
            self.batch_size, self.forward_microbatch_size,
            self.behavior_guard_samples,
        ) < 1:
            raise ValueError("0047 PPO batch settings must be positive")
        if not 0 < self.target_behavior_kl < self.hard_behavior_kl_guard:
            raise ValueError("0047 behavior KL guards are invalid")
        if abs(self.router_learning_rate / self.decoder_learning_rate - 0.2) > 1e-9:
            raise ValueError("0047 Router LR must be 0.2x decoder LR")


class PPOTrainer:
    def __init__(self, model: MetaRoutedMoEActorCritic, *, device: torch.device,
                 config: PPOConfig = PPOConfig()) -> None:
        config.validate()
        self.model, self.device, self.config = model, device, config
        model.freeze_contract()
        model.assert_trainable_contract()
        self.initial_representation_sha256 = model.representation_sha256()
        self.reference_model = copy.deepcopy(model).to(device).eval()
        self.reference_model.requires_grad_(False)
        self.reference_parameters = {
            name: value.detach().clone()
            for name, value in model.experts.named_parameters()
        }
        groups = [
            {"name": "expert_decoders", "params": [p for e in model.experts for p in e.action_decoder.parameters()], "lr": config.decoder_learning_rate},
            {"name": "expert_allocations", "params": [p for e in model.experts for p in e.allocation_head.parameters()], "lr": config.allocation_learning_rate},
            {"name": "expert_option_lora", "params": [p for e in model.experts for p in e.policy_option_lora.parameters()], "lr": config.option_lora_learning_rate},
            {"name": "router", "params": [model.router_logits], "lr": config.router_learning_rate},
            {"name": "value_win", "params": [p for p in model.value_head.parameters() if p.requires_grad], "lr": config.value_learning_rate},
            {"name": "value_adapter", "params": list(model.value_adapter.parameters()), "lr": config.value_learning_rate},
            {"name": "value_prize", "params": list(model.prize_aux.parameters()), "lr": config.prize_learning_rate},
        ]
        self.optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)

    def set_reference_model(self, reference_model: MetaRoutedMoEActorCritic) -> None:
        self.reference_model = copy.deepcopy(reference_model).to(self.device).eval()
        self.reference_model.requires_grad_(False)
        self.reference_parameters = {
            name: value.detach().clone()
            for name, value in reference_model.experts.named_parameters()
        }

    def optimizer_group_manifest(self):
        return [
            {
                "name": str(group["name"]), "learning_rate": float(group["lr"]),
                "parameters": sum(p.numel() for p in group["params"]),
                "tensor_count": len(group["params"]),
            }
            for group in self.optimizer.param_groups
        ]

    def _snapshot(self):
        return {
            name: value.detach().clone()
            for name, value in self.model.named_parameters() if value.requires_grad
        }

    def _restore(self, snapshot):
        current = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in snapshot.items():
                current[name].copy_(value.to(current[name].device))
        self.optimizer.state.clear()
        self.optimizer.zero_grad(set_to_none=True)

    def _evaluate(self, model, batch, indices):
        features = index_feature_batch(
            batch.features, indices, self.device, batch.feature_widths
        )
        set_batch_own_archetype_ids(model, batch, indices)
        return evaluate_moe_actions(
            model, features,
            batch.sequences[indices].to(self.device),
            batch.lengths[indices].to(self.device),
            batch.stopped[indices].to(self.device),
            batch.routing_meta_id[indices].to(self.device),
            tuple(batch.macro_actions[int(index)] for index in indices),
        )

    def _all_logprobs(self, model, batch, *, include_value=False):
        logprob = torch.empty(batch.decisions, dtype=torch.float32)
        values = torch.empty_like(logprob) if include_value else None
        entropy = torch.empty_like(logprob) if include_value else None
        with torch.no_grad():
            for start in range(0, batch.decisions, self.config.forward_microbatch_size):
                indices = torch.arange(start, min(start + self.config.forward_microbatch_size, batch.decisions))
                evaluated = self._evaluate(model, batch, indices)
                logprob[indices] = evaluated.joint_log_prob.float().cpu()
                if include_value:
                    values[indices] = evaluated.value.float().cpu()
                    entropy[indices] = (evaluated.root_entropy + evaluated.allocation_entropy).float().cpu()
        return logprob, values, entropy

    def _relative_l2(self, baseline):
        current = dict(self.model.experts.named_parameters())
        numerator = denominator = torch.zeros((), dtype=torch.float64, device=self.device)
        for name, source in baseline.items():
            value = current[name].detach().double()
            numerator = numerator + (value - source.to(self.device).double()).square().sum()
            denominator = denominator + source.to(self.device).double().square().sum()
        return float(numerator.sqrt() / denominator.sqrt().clamp_min(1e-12))

    def _group_update(self, before, needle):
        current = dict(self.model.named_parameters())
        terms = [
            (current[name].detach() - source.to(current[name].device)).float().square().sum()
            for name, source in before.items() if needle in name
        ]
        return float(torch.stack(terms).sum().sqrt()) if terms else 0.0

    def _behavior_metrics(self, batch):
        current, values, entropy = self._all_logprobs(self.model, batch, include_value=True)
        old = batch.rollout_log_prob.float().cpu()
        delta = current - old
        ratio = delta.exp()
        weights = batch.episode_weight.float().cpu()
        weights = weights / weights.sum()
        returns = batch.gae_return.float().cpu()
        residual = returns - values
        variance = returns.var(unbiased=False)
        return {
            "behavior_kl": float((((ratio - 1) - delta) * weights).sum()),
            "behavior_logprob_mae_max": float(delta.abs().max()),
            "behavior_logprob_abs_error_mean": float(delta.abs().mean()),
            "clip_fraction": float(((ratio - 1).abs().gt(self.config.clip_ratio).float() * weights).sum()),
            "entropy": float((entropy * weights).sum()),
            "explained_variance": float(1 - residual.var(unbiased=False) / variance) if variance > 1e-12 else 0.0,
        }

    def update(self, batch: PreparedBatch, *, update: int = 0) -> dict[str, float]:
        started = time.perf_counter()
        print(
            f"[0047 PPO][update {update:04d}] precheck decisions={batch.decisions} "
            f"epochs={self.config.epochs} minibatch={self.config.batch_size} "
            f"microbatch={self.config.forward_microbatch_size}",
            flush=True,
        )
        if self.model.representation_sha256() != self.initial_representation_sha256:
            raise RuntimeError("frozen Policy-0814 backbone changed before PPO")
        pre = self._behavior_metrics(batch)
        if pre["behavior_logprob_mae_max"] > self.config.behavior_logprob_mae_limit:
            raise RuntimeError(
                "0047 effective-policy old/new logprob parity failed: "
                f"{pre['behavior_logprob_mae_max']}"
            )
        reference_logprob, _, _ = self._all_logprobs(self.reference_model, batch)
        before = self._snapshot()
        behavior_parameters = {
            name: value.detach().clone()
            for name, value in self.model.experts.named_parameters()
        }
        sums: dict[str, float] = {}
        optimized = 0
        epochs_completed = 0
        early_stop = False
        hard_guard = False
        for epoch in range(self.config.epochs):
            epoch_started = time.perf_counter()
            order = torch.randperm(batch.decisions)
            epoch_kl_sum = 0.0
            epoch_weight_sum = 0.0
            for start in range(0, batch.decisions, self.config.batch_size):
                logical = order[start:start + self.config.batch_size]
                logical_weight = batch.episode_weight[logical].to(self.device).sum()
                self.optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, logical.numel(), self.config.forward_microbatch_size):
                    indices = logical[micro_start:micro_start + self.config.forward_microbatch_size]
                    evaluated = self._evaluate(self.model, batch, indices)
                    weights = batch.episode_weight[indices].to(self.device) / logical_weight
                    old = batch.rollout_log_prob[indices].to(self.device)
                    advantage = batch.advantage[indices].to(self.device)
                    if self.model.integrated_flags.enable_prize_aux:
                        advantage = advantage + self.model.integrated_flags.prize_aux_actor_weight * batch.prize_advantage[indices].to(self.device)
                    log_ratio = evaluated.joint_log_prob - old
                    ratio = log_ratio.exp()
                    unclipped = ratio * advantage
                    clipped = ratio.clamp(1-self.config.clip_ratio, 1+self.config.clip_ratio) * advantage
                    policy_loss = -(torch.minimum(unclipped, clipped) * weights).sum()
                    value_loss = ((evaluated.value - batch.gae_return[indices].to(self.device)).square() * weights).sum()
                    entropy = ((evaluated.root_entropy + evaluated.allocation_entropy) * weights).sum()
                    reference_delta = evaluated.joint_log_prob - reference_logprob[indices].to(self.device)
                    reference_kl = (0.5 * reference_delta.square() * weights).sum()
                    # Critic auxiliary heads reuse the shared Value path only.
                    features = index_feature_batch(batch.features, indices, self.device, batch.feature_widths)
                    validated, state, _prefix, value_options = self.model.encode_shared(features)
                    _value, auxiliary = self.model.value_and_aux_from_encoded(validated, state, value_options)
                    labels = batch.opponent_meta_label[indices].to(self.device)
                    meta_loss = (torch.nn.functional.cross_entropy(auxiliary["meta_logits"], labels, reduction="none") * weights).sum()
                    prize_loss = ((auxiliary["v_prize"] - batch.prize_return[indices].to(self.device)).square() * weights).sum()
                    registry = LossRegistry()
                    registry.register(LossTerm("L_policy_win", policy_loss, 1.0, "actor"))
                    registry.register(LossTerm("L_value_win", value_loss, self.config.value_coefficient, "value_win"))
                    registry.register(LossTerm("L_value_prize", prize_loss, self.model.integrated_flags.prize_value_loss_weight, "value_prize"))
                    registry.register(LossTerm("L_meta_anchor", meta_loss, self.config.meta_anchor_coef, "value_win"))
                    registry.register(LossTerm("L_entropy", -entropy, self.config.entropy_coefficient, "actor"))
                    registry.register(LossTerm("L_reference_kl", reference_kl, self.config.reference_kl_coefficient, "actor"))
                    loss = registry.total()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite 0047 PPO loss")
                    loss.backward()
                    approx_kl = (((ratio - 1) - log_ratio) * weights).sum()
                    metrics = {
                        "policy_loss": policy_loss, "value_loss": value_loss,
                        "prize_value_loss": prize_loss, "opponent_meta_loss": meta_loss,
                        "entropy": entropy, "reference_kl": reference_kl,
                        "behavior_kl": approx_kl,
                    }
                    for name, value in metrics.items():
                        sums[name] = sums.get(name, 0.0) + float(value.detach()) * indices.numel()
                    optimized += int(indices.numel())
                    epoch_kl_sum += float(approx_kl.detach()) * indices.numel()
                    epoch_weight_sum += indices.numel()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite 0047 PPO gradient")
                self.optimizer.step()
                completed = min(start + logical.numel(), batch.decisions)
                print(
                    f"[0047 PPO][update {update:04d}][epoch {epoch + 1}/{self.config.epochs}] "
                    f"decisions {completed}/{batch.decisions} "
                    f"elapsed={time.perf_counter() - epoch_started:.1f}s",
                    flush=True,
                )
            epochs_completed += 1
            epoch_kl = epoch_kl_sum / max(1.0, epoch_weight_sum)
            if epoch_kl > self.config.hard_behavior_kl_guard:
                self._restore(before)
                hard_guard = True
                break
            if epoch_kl > self.config.target_behavior_kl:
                early_stop = True
                break
        print(
            f"[0047 PPO][update {update:04d}] optimization_complete "
            f"epochs={epochs_completed} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        post = self._behavior_metrics(batch)
        result = {
            **{f"ppo/{name}": value / max(1, optimized) for name, value in sums.items()},
            "ppo/preupdate_behavior_kl": pre["behavior_kl"],
            "ppo/preupdate_behavior_logprob_mae_max": pre["behavior_logprob_mae_max"],
            "ppo/behavior_kl": post["behavior_kl"],
            "ppo/behavior_logprob_mae_max": post["behavior_logprob_mae_max"],
            "ppo/entropy": post["entropy"],
            "ppo/explained_variance": post["explained_variance"],
            "ppo/epochs_completed": float(epochs_completed),
            "ppo/target_kl_early_stop": float(early_stop),
            "ppo/hard_behavior_kl_guard_triggered": float(hard_guard),
            "ppo/optimizer_samples_consumed": float(optimized),
            "ppo/actor_relative_l2_vs_behavior": self._relative_l2(behavior_parameters),
            "ppo/actor_relative_l2_vs_reference": self._relative_l2(self.reference_parameters),
            "ppo/action_decoder_update_norm": self._group_update(before, ".action_decoder."),
            "ppo/option_lora_update_norm": self._group_update(before, ".policy_option_lora."),
            "ppo/allocation_head_update_norm": self._group_update(before, ".allocation_head."),
            "ppo/router_update_norm": self._group_update(before, "router_logits"),
            "ppo/router_soft_phase": float(self.model.soft_routing),
            "ppo/decisions": float(batch.decisions),
        }
        result.update(training_batch_metrics(batch, include_detailed=(update <= 1 or update % 10 == 0)))
        return result


__all__ = ["PPOConfig", "PPOTrainer", "index_feature_batch"]
