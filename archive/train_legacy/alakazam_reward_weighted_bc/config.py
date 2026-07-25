from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RewardComponentConfig:
    """One auditable scalar field contributing to the offline reward target."""

    name: str
    field: str
    weight: float = 1.0
    enabled: bool = True
    default: float | None = None
    clip_min: float | None = None
    clip_max: float | None = None

    def validate(self) -> None:
        if not self.name or not self.field:
            raise ValueError("reward component name and field must be non-empty")
        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min > self.clip_max:
                raise ValueError(f"invalid clip range for reward component {self.name}")


@dataclass(frozen=True)
class RewardConfig:
    components: tuple[RewardComponentConfig, ...] = ()
    clip_min: float | None = None
    clip_max: float | None = None

    def validate(self) -> None:
        names = [component.name for component in self.components if component.enabled]
        if len(names) != len(set(names)):
            raise ValueError("enabled reward component names must be unique")
        for component in self.components:
            component.validate()
        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min > self.clip_max:
                raise ValueError("invalid combined reward clip range")


@dataclass(frozen=True)
class WeightingConfig:
    """Convert detached reward advantages into non-negative imitation weights."""

    mode: str = "uniform"
    baseline: str = "zero"
    temperature: float = 1.0
    linear_scale: float = 1.0
    min_weight: float = 0.1
    max_weight: float = 10.0

    def validate(self) -> None:
        if self.mode not in {"uniform", "linear", "exponential"}:
            raise ValueError(f"unsupported weighting mode: {self.mode}")
        if self.baseline not in {"zero", "dataset_mean", "value"}:
            raise ValueError(f"unsupported reward baseline: {self.baseline}")
        if self.temperature <= 0:
            raise ValueError("weighting temperature must be positive")
        if not 0 <= self.min_weight <= self.max_weight:
            raise ValueError("weight range must satisfy 0 <= min_weight <= max_weight")


@dataclass(frozen=True)
class LossConfig:
    policy: float = 1.0
    count: float = 1.0
    value: float = 0.0

    def validate(self) -> None:
        if min(self.policy, self.count, self.value) < 0:
            raise ValueError("loss coefficients must be non-negative")
        if self.policy + self.count + self.value <= 0:
            raise ValueError("at least one loss coefficient must be positive")


@dataclass(frozen=True)
class CardCategoryEmbeddingConfig:
    enabled: bool = False
    scale: float = 1.0

    def validate(self) -> None:
        if self.scale < 0:
            raise ValueError("card category embedding scale must be non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str = "alakazam_reward_weighted_bc_v1"
    seed: int = 7
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    streaming: bool = True
    shuffle_buffer_size: int = 8192
    progress_batches: int = 100
    train_eval_interval: int = 0
    device: str = "auto"
    storage_path: str = "/mnt/c"
    min_free_gib: float = 20.0
    evaluate_test: bool = False
    reset_value_output: bool = False
    value_warmup_epochs: int = 0
    freeze_backbone_during_value_warmup: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    card_category_embedding: CardCategoryEmbeddingConfig = field(
        default_factory=CardCategoryEmbeddingConfig
    )

    def validate(self) -> None:
        if self.schema_version != "alakazam_reward_weighted_bc_v1":
            raise ValueError(f"unsupported experiment config: {self.schema_version}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not 0 <= self.value_warmup_epochs < self.epochs:
            raise ValueError("value_warmup_epochs must be in [0, epochs)")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("invalid optimizer configuration")
        if self.shuffle_buffer_size < 1 or self.progress_batches < 1:
            raise ValueError("streaming buffer and progress interval must be positive")
        if self.train_eval_interval < 0:
            raise ValueError("train_eval_interval must be non-negative")
        if self.min_free_gib < 0:
            raise ValueError("min_free_gib must be non-negative")
        self.reward.validate()
        self.weighting.validate()
        self.loss.validate()
        self.card_category_embedding.validate()
        if self.weighting.mode != "uniform" and not any(
            component.enabled for component in self.reward.components
        ):
            raise ValueError("reward weighting requires at least one enabled reward component")
        if self.loss.value > 0 and not any(
            component.enabled for component in self.reward.components
        ):
            raise ValueError("value loss requires at least one enabled reward component")
        if self.value_warmup_epochs and self.loss.value <= 0:
            raise ValueError("value warm-up requires a positive value loss coefficient")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"config field {name} must be an object")
    return dict(value)


def load_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    root = _mapping(payload, "root")
    reward_payload = _mapping(root.pop("reward", {}), "reward")
    raw_components = reward_payload.pop("components", [])
    if not isinstance(raw_components, list):
        raise ValueError("reward.components must be a list")
    reward = RewardConfig(
        components=tuple(
            RewardComponentConfig(**_mapping(component, "reward.components[]"))
            for component in raw_components
        ),
        **reward_payload,
    )
    weighting = WeightingConfig(**_mapping(root.pop("weighting", {}), "weighting"))
    loss = LossConfig(**_mapping(root.pop("loss", {}), "loss"))
    card_category_embedding = CardCategoryEmbeddingConfig(
        **_mapping(root.pop("card_category_embedding", {}), "card_category_embedding")
    )
    config = ExperimentConfig(
        reward=reward,
        weighting=weighting,
        loss=loss,
        card_category_embedding=card_category_embedding,
        **root,
    )
    config.validate()
    return config
