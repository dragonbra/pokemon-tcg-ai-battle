from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .model import IDOnlyConfig


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings matching the reference Notebook by default."""

    schema_version: str = "alakazam_sota_id_only_bc_v1"
    model: IDOnlyConfig = field(default_factory=IDOnlyConfig)
    batch_size: int = 96
    epochs: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 0.02
    grad_clip: float = 1.0
    seed: int = 20260723
    shuffle_buffer_size: int = 4096
    train_eval_interval: int = 1
    early_stopping_patience: int = 0
    log_every_batches: int = 100
    amp: bool = True
    device: str = "cuda"
    max_model_mib: float = 50.0
    storage_path: str = "."
    min_free_gib: float = 20.0

    def validate(self) -> None:
        self.model.validate()
        if self.schema_version != "alakazam_sota_id_only_bc_v1":
            raise ValueError(f"unsupported config schema: {self.schema_version}")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.grad_clip <= 0 or self.shuffle_buffer_size <= 0:
            raise ValueError("grad_clip and shuffle_buffer_size must be positive")
        if (
            self.train_eval_interval < 0
            or self.early_stopping_patience < 0
            or self.log_every_batches <= 0
        ):
            raise ValueError("evaluation interval must be non-negative and logging positive")
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ValueError(f"unsupported device: {self.device}")
        if self.max_model_mib <= 0 or self.min_free_gib < 0:
            raise ValueError("model limit must be positive and free-space threshold non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingConfig":
        payload = dict(value)
        model = IDOnlyConfig(**payload.pop("model", {}))
        config = cls(model=model, **payload)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("training config must be a JSON object")
        return cls.from_dict(value)
