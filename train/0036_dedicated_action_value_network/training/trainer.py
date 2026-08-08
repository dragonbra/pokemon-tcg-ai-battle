"""One-pass-per-epoch frozen-Encoder Value training."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from rl_environment.logging import TrainingLogger

from ..model.source import SourceIdentity
from ..model.value_network import FrozenEncoderValueNetwork
from ..model.value_network import ValueOutputs
from .checkpoints import save_value_checkpoint
from .materialized import MaterializedValueDataset
from .metrics import ValueMetrics
from .objective import LossWeights, value_objective


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 1024
    validation_batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    seed: int = 20260808
    amp_dtype: str = "bfloat16"
    checkpoint_retention: str = "all"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.validation_batch_size < 1:
            raise ValueError("0036 epoch/batch sizes must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("0036 optimizer values are invalid")
        if self.amp_dtype not in {"bfloat16", "float32"} or self.checkpoint_retention != "all":
            raise ValueError("0036 requires bfloat16/float32 and all-checkpoint retention")


class ValueTrainer:
    def __init__(self, model: FrozenEncoderValueNetwork, dataset: MaterializedValueDataset,
                 source: SourceIdentity, run_root: Path, config: TrainerConfig,
                 loss_weights: LossWeights, device: str | torch.device = "cuda") -> None:
        config.validate()
        loss_weights.validate()
        self.model = model.to(device)
        self.model.assert_frozen_encoder()
        self.dataset = dataset
        self.source = source
        self.run_root = run_root
        self.config = config
        self.loss_weights = loss_weights
        self.device = torch.device(device)
        parameters = [parameter for parameter in model.value_head.parameters() if parameter.requires_grad]
        kwargs = {"fused": True} if self.device.type == "cuda" else {}
        self.optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                           weight_decay=config.weight_decay, **kwargs)
        self.trainable_parameters = sum(parameter.numel() for parameter in parameters)

    def _autocast(self):
        if self.device.type == "cuda" and self.config.amp_dtype == "bfloat16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _run_split(self, split: str, epoch: int, *, optimize: bool) -> tuple[dict[str, float], int, float]:
        self.model.train(optimize)
        metrics = ValueMetrics(collect_auc=not optimize)
        exact_metrics = ValueMetrics(collect_auc=True) if not optimize else None
        exact_decisions = 0
        objective_sums = {"total": 0.0, "value": 0.0, "archetype": 0.0, "final_diff": 0.0}
        objective_weight = 0.0
        decisions = 0
        started = time.perf_counter()
        batch_size = self.config.batch_size if optimize else self.config.validation_batch_size
        source = iter(self.dataset.batches(
            split, batch_size, epoch=epoch,
            shuffle=optimize, seed=self.config.seed,
        ))
        data_wait_seconds = 0.0
        batch_index = 0
        while True:
            wait_started = time.perf_counter()
            try:
                batch = next(source)
            except StopIteration:
                break
            data_wait_seconds += time.perf_counter() - wait_started
            batch_index += 1
            batch = batch.to(self.device, non_blocking=True)
            if optimize:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(optimize), self._autocast():
                outputs = self.model(batch.features)
                loss = value_objective(
                    outputs,
                    value_target=batch.value_target,
                    archetype_target=batch.archetype_target,
                    final_diff_target=batch.final_diff_target,
                    episode_weight=batch.episode_weight,
                    weights=self.loss_weights,
                )
            if optimize:
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.value_head.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            batch_weight = float(batch.episode_weight.sum())
            objective_weight += batch_weight
            objective_values = torch.stack(
                [loss.total, loss.value, loss.archetype, loss.final_diff]
            ).detach().float().cpu().tolist()
            for name, value in zip(objective_sums, objective_values):
                objective_sums[name] += value * batch_weight
            metrics.update(outputs, batch.value_target, batch.archetype_target,
                           batch.final_diff_target, batch.episode_weight)
            if exact_metrics is not None and batch.is_exact_007.any():
                selected = batch.is_exact_007
                exact_metrics.update(
                    ValueOutputs(
                        outputs.value_logit[selected],
                        outputs.archetype_logits[selected],
                        outputs.final_diff_logits[selected],
                    ),
                    batch.value_target[selected],
                    batch.archetype_target[selected],
                    batch.final_diff_target[selected],
                    batch.episode_weight[selected],
                )
                exact_decisions += int(selected.sum())
            decisions += batch.size
            if batch_index % 100 == 0:
                current_elapsed = time.perf_counter() - started
                print(
                    {
                        "event": "0036_value_progress",
                        "epoch": epoch,
                        "split": split,
                        "batches": batch_index,
                        "decisions": decisions,
                        "seconds": round(current_elapsed, 3),
                        "decisions_per_second": round(decisions / current_elapsed, 3),
                    },
                    flush=True,
                )
        elapsed = time.perf_counter() - started
        result = metrics.compute()
        for name, value in objective_sums.items():
            result[f"objective/{name}"] = value / objective_weight
        if exact_metrics is not None:
            if exact_decisions == 0:
                raise ValueError("0036 focal validation contains no exact-007 decisions")
            result.update({f"exact007/{key}": value for key, value in exact_metrics.compute().items()})
            result["exact007/system/decisions"] = float(exact_decisions)
        result["system/decisions"] = float(decisions)
        result["system/decisions_per_second"] = decisions / elapsed
        result["system/data_wait_seconds"] = data_wait_seconds
        result["system/data_wait_fraction"] = data_wait_seconds / elapsed
        return result, decisions, elapsed

    def train(self) -> dict[str, object]:
        artifact = self.run_root / "artifact"
        checkpoint = self.run_root / "checkpoint" / "value"
        logger = TrainingLogger(artifact / "training_metrics.jsonl", self.run_root / "tensorboard")
        logger.initialize_wandb({"trainer/epoch": 0, "value/lifecycle/initialized": 1})
        best_loss = float("inf")
        best_all_dragapult_loss = float("inf")
        best_epoch = 0
        try:
            for epoch in range(1, self.config.epochs + 1):
                optimization, train_decisions, train_seconds = self._run_split("train", epoch, optimize=True)
                validation, validation_decisions, validation_seconds = self._run_split("validation", epoch, optimize=False)
                def namespaced(stage: str, values: dict[str, float]) -> dict[str, float]:
                    mapped = {}
                    for key, value in values.items():
                        if key.startswith("exact007/value/"):
                            name = f"value/{stage}_exact007/{key.removeprefix('exact007/value/')}"
                        elif key.startswith("exact007/"):
                            name = f"value/{stage}_exact007/{key.removeprefix('exact007/')}"
                        elif key.startswith("value/"):
                            name = f"value/{stage}/{key.removeprefix('value/')}"
                        elif key.startswith("system/"):
                            name = f"system/{stage}/{key.removeprefix('system/')}"
                        else:
                            name = f"value/{stage}/{key}"
                        mapped[name] = value
                    return mapped

                record = {"trainer/epoch": epoch,
                          **namespaced("optimization", optimization),
                          **namespaced("validation", validation),
                          "system/train_decisions": train_decisions,
                          "system/validation_decisions": validation_decisions,
                          "system/train_seconds": train_seconds,
                          "system/validation_seconds": validation_seconds,
                          "system/trainable_parameter_count": self.trainable_parameters,
                          "value/baseline/constant_half_bce": 0.6931471805599453,
                          "value/baseline/constant_half_brier": 0.25}
                logger.log(epoch, record)
                metadata = {
                    "project_id": "0036_dedicated_action_value_network",
                    "version": self.run_root.name,
                    "epoch": epoch,
                    "architecture": self.model.architecture,
                    "value_config": self.model.value_config,
                    "source_checkpoint_sha256": self.source.checkpoint_sha256,
                    "dataset_catalog_sha256": self.dataset.manifest["catalog_sha256"],
                    "validation_value_loss": validation["value/loss"],
                    "validation_exact007_value_loss": validation["exact007/value/loss"],
                    "checkpoint_retention": "all",
                    "optimizer_state_saved": False,
                    "resumable_training_state_saved": False,
                }
                save_value_checkpoint(checkpoint / f"epoch_{epoch:04d}.pt", self.model.value_head, metadata)
                selection_key = (validation["exact007/value/loss"], validation["value/loss"])
                if selection_key < (best_loss, best_all_dragapult_loss):
                    best_loss, best_all_dragapult_loss = selection_key
                    best_epoch = epoch
                    save_value_checkpoint(checkpoint / "best_validation_value_loss.pt", self.model.value_head, metadata)
        except BaseException:
            logger.close(exit_code=1)
            raise
        else:
            logger.close(exit_code=0)
        return {"status": "complete", "best_epoch": best_epoch,
                "best_validation_exact007_value_loss": best_loss,
                "best_validation_dragapult_value_loss": best_all_dragapult_loss,
                "trainable_parameter_count": self.trainable_parameters}


__all__ = ["TrainerConfig", "ValueTrainer"]
