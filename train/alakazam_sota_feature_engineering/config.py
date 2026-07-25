from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .model import FeatureModelConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """One immutable 0012 training-version contract."""

    schema_version: str = "alakazam_sota_feature_engineering_v1"
    experiment_name: str = "baseline_control"
    model: FeatureModelConfig = field(default_factory=FeatureModelConfig)
    batch_size: int = 96
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 0.02
    grad_clip: float = 1.0
    seed: int = 20260725
    shuffle_buffer_size: int = 4096
    early_stopping_patience: int = 5
    permutation_augmentation: bool = False
    permutation_validation: bool = True
    transition_delta_weight: float = 0.10
    transition_turn_weight: float = 0.05
    log_every_batches: int = 100
    amp: bool = True
    device: str = "cuda"
    max_model_mib: float = 50.0
    storage_path: str = "."
    min_free_gib: float = 20.0

    def validate(self) -> None:
        self.model.validate()
        if self.schema_version != "alakazam_sota_feature_engineering_v1":
            raise ValueError(f"unsupported config schema: {self.schema_version}")
        if self.batch_size <= 0 or self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("batch size, epochs, and learning rate must be positive")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight decay must be non-negative and grad clip positive")
        if self.shuffle_buffer_size <= 0 or self.early_stopping_patience <= 0:
            raise ValueError("shuffle buffer and early-stopping patience must be positive")
        if self.log_every_batches <= 0:
            raise ValueError("log_every_batches must be positive")
        if self.transition_delta_weight < 0 or self.transition_turn_weight < 0:
            raise ValueError("transition loss weights must be non-negative")
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ValueError(f"unsupported device: {self.device}")
        if self.max_model_mib <= 0 or self.min_free_gib < 0:
            raise ValueError("model limit must be positive and free-space threshold non-negative")
        if self.permutation_augmentation and not self.model.remove_option_position:
            raise ValueError(
                "permutation augmentation requires remove_option_position=true to avoid "
                "training a new randomized position shortcut"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        payload = dict(value)
        model = FeatureModelConfig(**payload.pop("model", {}))
        config = cls(model=model, **payload)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("experiment config must be a JSON object")
        return cls.from_dict(value)
