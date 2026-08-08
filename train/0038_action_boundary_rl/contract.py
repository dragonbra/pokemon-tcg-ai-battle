"""Strict tensor contract for CUDA PolicyCodecV1 actor batches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch
from torch import Tensor


CODEC_VERSION = "pure_policy_codec_v1"
ENTITY_CAPACITY = 128
OPTION_CAPACITY = 128

FIELD_SHAPES: dict[str, tuple[int | None, ...]] = {
    "global_cat": (None, 8),
    "global_num": (None, 16),
    "entity_cat": (None, ENTITY_CAPACITY, 6),
    "entity_num": (None, ENTITY_CAPACITY, 10),
    "entity_parent": (None, ENTITY_CAPACITY),
    "entity_mask": (None, ENTITY_CAPACITY),
    "option_cat": (None, OPTION_CAPACITY, 12),
    "option_num": (None, OPTION_CAPACITY, 4),
    "option_equiv": (None, OPTION_CAPACITY),
    "option_mask": (None, OPTION_CAPACITY),
    "min_count": (None,),
    "max_count": (None,),
}

LONG_FIELDS = frozenset(
    {
        "global_cat",
        "entity_cat",
        "entity_parent",
        "option_cat",
        "option_equiv",
        "min_count",
        "max_count",
    }
)
FLOAT_FIELDS = frozenset({"global_num", "entity_num", "option_num"})
BOOL_FIELDS = frozenset({"entity_mask", "option_mask"})


class PodNativeBatch(Mapping[str, Tensor]):
    """Validated mapping whose CUDA path never synchronizes values to the host."""

    def __init__(self, tensors: Mapping[str, Tensor]) -> None:
        self._tensors = dict(tensors)

    @classmethod
    def from_mapping(cls, tensors: Mapping[str, Tensor]) -> "PodNativeBatch":
        keys = set(tensors)
        expected = set(FIELD_SHAPES)
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing or extra:
            raise ValueError(f"POD batch contract mismatch; missing={missing}, extra={extra}")
        if not all(isinstance(value, Tensor) for value in tensors.values()):
            raise TypeError("every POD batch value must be a torch.Tensor")

        batch_size: int | None = None
        device: torch.device | None = None
        for name, expected_shape in FIELD_SHAPES.items():
            value = tensors[name]
            if value.ndim != len(expected_shape):
                raise ValueError(
                    f"{name} rank mismatch: expected {len(expected_shape)}, got {value.ndim}"
                )
            if any(
                expected_dim is not None and actual_dim != expected_dim
                for actual_dim, expected_dim in zip(value.shape, expected_shape)
            ):
                raise ValueError(
                    f"{name} shape mismatch: expected {expected_shape}, got {tuple(value.shape)}"
                )
            if batch_size is None:
                batch_size = int(value.shape[0])
                device = value.device
            elif value.shape[0] != batch_size:
                raise ValueError(f"{name} has inconsistent batch size {value.shape[0]}")
            if value.device != device:
                raise ValueError(f"{name} is on {value.device}, expected {device}")

        for name in LONG_FIELDS:
            if tensors[name].dtype is not torch.long:
                raise ValueError(f"{name} must have dtype torch.long")
        for name in FLOAT_FIELDS:
            if not tensors[name].dtype.is_floating_point:
                raise ValueError(f"{name} must have a floating dtype")
        for name in BOOL_FIELDS:
            if tensors[name].dtype is not torch.bool:
                raise ValueError(f"{name} must have dtype torch.bool")

        if batch_size is None or batch_size < 1:
            raise ValueError("POD batch must contain at least one row")
        if device is not None and device.type == "cpu":
            cls._validate_cpu_values(tensors)
        return cls(tensors)

    @staticmethod
    def _validate_cpu_values(tensors: Mapping[str, Tensor]) -> None:
        if torch.any(tensors["min_count"] < 0):
            raise ValueError("min_count cannot be negative")
        if torch.any(tensors["max_count"] < tensors["min_count"]):
            raise ValueError("max_count cannot be less than min_count")
        option_counts = tensors["option_mask"].sum(dim=1)
        if torch.any(option_counts < 1):
            raise ValueError("every row must expose at least one legal option")
        if torch.any(tensors["max_count"] > option_counts):
            raise ValueError("max_count cannot exceed the legal option count")
        valid_parents = tensors["entity_parent"][tensors["entity_mask"]]
        if torch.any(valid_parents < -1) or torch.any(valid_parents >= ENTITY_CAPACITY):
            raise ValueError("entity_parent contains an out-of-range index")

    @property
    def batch_size(self) -> int:
        return int(self._tensors["global_cat"].shape[0])

    @property
    def option_count(self) -> int:
        return OPTION_CAPACITY

    @property
    def device(self) -> torch.device:
        return self._tensors["global_cat"].device

    def as_dict(self) -> dict[str, Tensor]:
        return dict(self._tensors)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "PodNativeBatch":
        return PodNativeBatch.from_mapping(
            {
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self._tensors.items()
            }
        )

    def __getitem__(self, key: str) -> Tensor:
        return self._tensors[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def __getattr__(self, name: str) -> Tensor:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise AttributeError(name) from error


__all__ = [
    "BOOL_FIELDS",
    "CODEC_VERSION",
    "ENTITY_CAPACITY",
    "FIELD_SHAPES",
    "FLOAT_FIELDS",
    "LONG_FIELDS",
    "OPTION_CAPACITY",
    "PodNativeBatch",
]
