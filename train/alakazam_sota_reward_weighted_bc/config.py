from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from train.alakazam_sota_model.model import IDOnlyConfig


@dataclass(frozen=True)
class RewardComponent:
    name: str
    field: str
    weight: float
    enabled: bool = True
    default: float | None = None
    clip_min: float | None = None
    clip_max: float | None = None


@dataclass(frozen=True)
class RewardConfig:
    components: tuple[RewardComponent, ...] = ()
    clip_min: float | None = -2.0
    clip_max: float | None = 2.0


@dataclass(frozen=True)
class WeightingConfig:
    mode: str = "exponential"
    baseline: str = "dataset_mean"
    temperature: float = 1.0
    linear_scale: float = 1.0
    min_weight: float = 0.20
    max_weight: float = 4.0

    def validate(self) -> None:
        if self.mode not in {"uniform", "linear", "exponential"}:
            raise ValueError(f"unsupported weighting mode: {self.mode}")
        if self.baseline not in {"zero", "dataset_mean"}:
            raise ValueError(f"unsupported reward baseline: {self.baseline}")
        if self.temperature <= 0 or self.linear_scale < 0:
            raise ValueError("weighting temperature must be positive and scale non-negative")
        if self.min_weight <= 0 or self.max_weight < self.min_weight:
            raise ValueError("invalid imitation weight bounds")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str = "alakazam_sota_reward_weighted_bc_v1"
    model: IDOnlyConfig = field(default_factory=IDOnlyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    batch_size: int = 96
    epochs: int = 20
    learning_rate: float = 4e-4
    weight_decay: float = 0.02
    grad_clip: float = 1.0
    seed: int = 20260725
    shuffle_buffer_size: int = 4096
    train_eval_interval: int = 1
    minimum_epochs_before_stop: int = 8
    early_stopping_patience: int = 5
    log_every_batches: int = 100
    amp: bool = True
    device: str = "cuda"
    max_model_mib: float = 50.0
    storage_path: str = "."
    min_free_gib: float = 20.0

    def validate(self) -> None:
        self.model.validate()
        self.weighting.validate()
        if self.schema_version != "alakazam_sota_reward_weighted_bc_v1":
            raise ValueError(f"unsupported config schema: {self.schema_version}")
        if not self.reward.components:
            raise ValueError("reward config must declare at least one component")
        names = [component.name for component in self.reward.components]
        if len(names) != len(set(names)):
            raise ValueError("reward component names must be unique")
        if not any(component.enabled for component in self.reward.components):
            raise ValueError("at least one reward component must be enabled")
        if self.batch_size <= 0 or self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("batch size, epochs, and learning rate must be positive")
        if self.weight_decay < 0 or self.grad_clip <= 0 or self.shuffle_buffer_size <= 0:
            raise ValueError("invalid optimizer or streaming settings")
        if self.minimum_epochs_before_stop < 3:
            raise ValueError("minimum_epochs_before_stop must be at least 3")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.minimum_epochs_before_stop > self.epochs:
            raise ValueError("minimum_epochs_before_stop cannot exceed epochs")
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ValueError(f"unsupported device: {self.device}")
        if self.max_model_mib <= 0 or self.min_free_gib < 0:
            raise ValueError("invalid model or storage limit")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        payload = dict(value)
        model = IDOnlyConfig(**payload.pop("model", {}))
        reward_payload = payload.pop("reward", {})
        components = tuple(
            RewardComponent(**component)
            for component in reward_payload.pop("components", [])
        )
        reward = RewardConfig(components=components, **reward_payload)
        weighting = WeightingConfig(**payload.pop("weighting", {}))
        config = cls(model=model, reward=reward, weighting=weighting, **payload)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("experiment config must be a JSON object")
        return cls.from_dict(value)
