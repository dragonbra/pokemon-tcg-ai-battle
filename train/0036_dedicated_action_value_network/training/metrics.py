"""Episode-weighted calibration and auxiliary diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..model.value_network import ValueOutputs


@dataclass
class ValueMetrics:
    bins: int = 15
    collect_auc: bool = True
    weight: float = 0.0
    log_loss: float = 0.0
    brier: float = 0.0
    correct: float = 0.0
    archetype_correct: float = 0.0
    diff_abs_error: float = 0.0
    diff_exact: float = 0.0
    bin_weight: list[float] = field(default_factory=lambda: [0.0] * 15)
    bin_probability: list[float] = field(default_factory=lambda: [0.0] * 15)
    bin_target: list[float] = field(default_factory=lambda: [0.0] * 15)
    probabilities: list[float] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    sample_weights: list[float] = field(default_factory=list)
    archetype_weight: list[float] = field(default_factory=lambda: [0.0] * 15)
    archetype_class_correct: list[float] = field(default_factory=lambda: [0.0] * 15)

    def update(self, outputs: ValueOutputs, value_target: Tensor, archetype_target: Tensor,
               final_diff_target: Tensor, episode_weight: Tensor) -> None:
        probability = outputs.win_probability.detach().float().cpu()
        target = value_target.detach().float().cpu()
        weights = episode_weight.detach().float().cpu()
        log_loss = torch.nn.functional.binary_cross_entropy(probability.clamp(1e-7, 1 - 1e-7), target, reduction="none")
        diff_values = torch.arange(-6, 7, dtype=torch.float32)
        expected_diff = (outputs.final_diff_logits.detach().float().cpu().softmax(-1) * diff_values).sum(-1)
        true_diff = final_diff_target.detach().cpu().float() - 6.0
        archetype = outputs.archetype_logits.detach().cpu().argmax(-1)
        diff_class = outputs.final_diff_logits.detach().cpu().argmax(-1)
        self.weight += float(weights.sum())
        self.log_loss += float((log_loss * weights).sum())
        self.brier += float(((probability - target).square() * weights).sum())
        self.correct += float(((probability >= 0.5) == target.bool()).float().mul(weights).sum())
        self.archetype_correct += float((archetype == archetype_target.cpu()).float().mul(weights).sum())
        self.diff_abs_error += float((expected_diff - true_diff).abs().mul(weights).sum())
        self.diff_exact += float((diff_class == final_diff_target.cpu()).float().mul(weights).sum())
        indices = (probability * self.bins).long().clamp_max(self.bins - 1)
        for index in range(self.bins):
            selected = indices == index
            if selected.any():
                local_weights = weights[selected]
                self.bin_weight[index] += float(local_weights.sum())
                self.bin_probability[index] += float((probability[selected] * local_weights).sum())
                self.bin_target[index] += float((target[selected] * local_weights).sum())
        class_target = archetype_target.detach().cpu()
        for class_id in range(15):
            selected = class_target == class_id
            if selected.any():
                local_weights = weights[selected]
                self.archetype_weight[class_id] += float(local_weights.sum())
                self.archetype_class_correct[class_id] += float(
                    (archetype[selected] == class_id).float().mul(local_weights).sum()
                )
        if self.collect_auc:
            self.probabilities.extend(probability.tolist())
            self.targets.extend(target.long().tolist())
            self.sample_weights.extend(weights.tolist())

    def _auc(self) -> float:
        rows = sorted(zip(self.probabilities, self.targets, self.sample_weights), key=lambda row: row[0])
        positive = sum(weight for _, target, weight in rows if target == 1)
        negative = sum(weight for _, target, weight in rows if target == 0)
        if positive == 0 or negative == 0:
            return 0.5
        rank_negative = 0.0
        area = 0.0
        index = 0
        while index < len(rows):
            end = index
            while end < len(rows) and rows[end][0] == rows[index][0]:
                end += 1
            group_negative = sum(weight for _, target, weight in rows[index:end] if target == 0)
            group_positive = sum(weight for _, target, weight in rows[index:end] if target == 1)
            area += group_positive * (rank_negative + 0.5 * group_negative)
            rank_negative += group_negative
            index = end
        return area / (positive * negative)

    def compute(self) -> dict[str, float]:
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("0036 metrics contain no positive weight")
        ece = 0.0
        for weight, probability, target in zip(self.bin_weight, self.bin_probability, self.bin_target):
            if weight:
                ece += weight / self.weight * abs(probability / weight - target / weight)
        result = {
            "value/loss": self.log_loss / self.weight,
            "value/brier": self.brier / self.weight,
            "value/ece_15": ece,
            "value/accuracy": self.correct / self.weight,
            "aux/archetype_accuracy": self.archetype_correct / self.weight,
            "aux/archetype_macro_accuracy": sum(
                correct / weight
                for correct, weight in zip(self.archetype_class_correct, self.archetype_weight)
                if weight
            ) / max(1, sum(weight > 0 for weight in self.archetype_weight)),
            "aux/final_diff_mae": self.diff_abs_error / self.weight,
            "aux/final_diff_exact": self.diff_exact / self.weight,
            "system/episode_weight": self.weight,
        }
        if self.collect_auc:
            result["value/auroc"] = self._auc()
        for class_id, (correct, weight) in enumerate(
            zip(self.archetype_class_correct, self.archetype_weight)
        ):
            if weight:
                result[f"aux/archetype_accuracy_class_{class_id:02d}"] = correct / weight
        return result


__all__ = ["ValueMetrics"]
