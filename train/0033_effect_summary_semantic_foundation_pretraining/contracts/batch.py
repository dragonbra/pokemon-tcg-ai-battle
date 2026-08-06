"""Validated, named tensor batch passed to the 0032 policy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch
from torch import Tensor

from .fields import EXPECTED_BATCH_KEYS, WIDTHS
from .fields import (
    CARD_CAT_VOCABS,
    EVENT_CAT_VOCABS,
    GLOBAL_CAT_VOCABS,
    OPTION_CAT_VOCABS,
    RESOURCE_CAT_VOCABS,
)


_FIELD_WIDTHS = {
    "global_cat": WIDTHS.global_cat,
    "global_num": WIDTHS.global_num,
    "card_cat": WIDTHS.card_cat,
    "card_num": WIDTHS.card_num,
    "resource_cat": WIDTHS.resource_cat,
    "resource_num": WIDTHS.resource_num,
    "event_cat": WIDTHS.event_cat,
    "event_num": WIDTHS.event_num,
    "option_cat": WIDTHS.option_cat,
    "option_num": WIDTHS.option_num,
    "option_state": WIDTHS.option_state,
    "global_state": WIDTHS.global_state,
    "card_state": WIDTHS.card_state,
    "resource_state": WIDTHS.resource_state,
    "event_state": WIDTHS.event_state,
}
_MASK_RELATIONS = {
    "card_mask": "card_cat",
    "resource_mask": "resource_cat",
    "event_mask": "event_cat",
    "option_mask": "option_cat",
}
_SEQUENCE_RELATIONS = {
    "card_parent": "card_cat",
    "event_source": "event_cat",
    "event_target": "event_cat",
    "option_source": "option_cat",
    "option_target": "option_cat",
}
_LONG_TENSORS = frozenset(
    {
        "global_cat",
        "card_cat",
        "card_parent",
        "resource_cat",
        "event_cat",
        "event_source",
        "event_target",
        "option_cat",
        "option_state",
        "global_state",
        "card_state",
        "resource_state",
        "event_state",
        "option_source",
        "option_target",
        "min_count",
        "max_count",
        "targets",
    }
)

_CATEGORICAL_VOCABS = {
    "global_cat": GLOBAL_CAT_VOCABS,
    "card_cat": CARD_CAT_VOCABS,
    "resource_cat": RESOURCE_CAT_VOCABS,
    "event_cat": EVENT_CAT_VOCABS,
    "option_cat": OPTION_CAT_VOCABS,
}


class DecisionBatch(Mapping[str, Tensor]):
    """Read-only mapping with attribute access and a strict actor contract."""

    def __init__(self, tensors: Mapping[str, Tensor]):
        self._tensors = dict(tensors)

    @classmethod
    def from_mapping(cls, tensors: Mapping[str, Tensor]) -> "DecisionBatch":
        keys = set(tensors)
        missing = sorted(EXPECTED_BATCH_KEYS - keys)
        extra = sorted(keys - EXPECTED_BATCH_KEYS)
        if missing or extra:
            raise ValueError(f"decision batch contract mismatch; missing={missing}, extra={extra}")
        if not all(isinstance(value, Tensor) for value in tensors.values()):
            raise TypeError("every decision batch value must be a torch.Tensor")

        batch_sizes = {name: int(value.shape[0]) for name, value in tensors.items() if value.ndim}
        if len(set(batch_sizes.values())) != 1 or len(batch_sizes) != len(tensors):
            raise ValueError(f"decision batch dimensions disagree: {batch_sizes}")

        for name, width in _FIELD_WIDTHS.items():
            value = tensors[name]
            expected_rank = 2 if name.startswith("global_") else 3
            if value.ndim != expected_rank or value.shape[-1] != width:
                raise ValueError(
                    f"{name} width/rank mismatch; expected rank={expected_rank}, width={width}, "
                    f"got shape={tuple(value.shape)}"
                )

        for mask_name, values_name in _MASK_RELATIONS.items():
            mask = tensors[mask_name]
            values = tensors[values_name]
            if mask.dtype is not torch.bool:
                raise ValueError(f"{mask_name} must have dtype torch.bool")
            if mask.ndim != 2 or mask.shape != values.shape[:2]:
                raise ValueError(
                    f"{mask_name} shape {tuple(mask.shape)} does not match "
                    f"{values_name} sequence {tuple(values.shape[:2])}"
                )

        for relation_name, values_name in _SEQUENCE_RELATIONS.items():
            relation = tensors[relation_name]
            values = tensors[values_name]
            if relation.ndim != 2 or relation.shape != values.shape[:2]:
                raise ValueError(
                    f"{relation_name} shape {tuple(relation.shape)} does not match "
                    f"{values_name} sequence {tuple(values.shape[:2])}"
                )

        batch_size = next(iter(batch_sizes.values()))
        for name in ("min_count", "max_count"):
            if tensors[name].shape != (batch_size,):
                raise ValueError(f"{name} must have shape [batch]")
        if tensors["targets"].ndim != 2:
            raise ValueError("targets must have shape [batch, action_steps]")
        # Value checks happen on canonical CPU batches. Performing them again
        # after transfer would serialize every GPU forward on the host.
        if tensors["min_count"].device.type == "cpu":
            if torch.any(tensors["min_count"] < 0):
                raise ValueError("min_count cannot be negative")
            if torch.any(tensors["max_count"] < tensors["min_count"]):
                raise ValueError("max_count cannot be less than min_count")
            valid_options = tensors["option_mask"].sum(dim=1)
            if torch.any(tensors["max_count"] > valid_options):
                raise ValueError("max_count cannot exceed the number of legal options")

            for name, vocabularies in _CATEGORICAL_VOCABS.items():
                values = tensors[name]
                for column, vocabulary in enumerate(vocabularies):
                    field = values[..., column]
                    if torch.any(field < 0) or torch.any(field >= vocabulary):
                        raise ValueError(
                            f"{name} field {column} exceeds vocabulary {vocabulary}"
                        )
            required_identity_columns = (
                ("card_cat", "card_mask", 0),
                ("resource_cat", "resource_mask", 0),
                ("event_cat", "event_mask", 0),
                ("option_cat", "option_mask", 0),
            )
            for values_name, mask_name, column in required_identity_columns:
                if torch.any(tensors[mask_name] & tensors[values_name][..., column].eq(0)):
                    raise ValueError(f"{values_name} field {column} cannot be padding on a real token")
            for name in ("global_num", "card_num", "resource_num", "event_num", "option_num"):
                if not torch.isfinite(tensors[name]).all():
                    raise ValueError(f"{name} contains a non-finite value")

            if torch.any((tensors["global_state"] < 1) | (tensors["global_state"] > 3)):
                raise ValueError("global_state must explicitly encode present/unknown/not-applicable")
            for prefix in ("card", "resource", "event", "option"):
                states = tensors[f"{prefix}_state"]
                mask = tensors[f"{prefix}_mask"].unsqueeze(-1)
                if torch.any(mask & ((states < 1) | (states > 3))):
                    raise ValueError(f"{prefix}_state has padding/invalid state on a real token")
                if torch.any((~mask) & states.ne(0)):
                    raise ValueError(f"{prefix}_state must be zero on padding")

            targets = tensors["targets"]
            termination = tensors["option_mask"].shape[1]
            relation_masks = {
                "card_parent": tensors["card_mask"],
                "event_source": tensors["event_mask"],
                "event_target": tensors["event_mask"],
                "option_source": tensors["option_mask"],
                "option_target": tensors["option_mask"],
            }
            for name, relation_mask in relation_masks.items():
                if torch.any((~relation_mask) & tensors[name].ne(0)):
                    raise ValueError(f"{name} must be zero on padding")
            for row_index in range(batch_size):
                row = targets[row_index]
                card_count = int(tensors["card_mask"][row_index].sum())
                legal_count = int(valid_options[row_index])
                for name in ("card_parent", "event_source", "event_target", "option_source", "option_target"):
                    relation = tensors[name][row_index]
                    if torch.any(relation < 0) or torch.any(relation > card_count):
                        raise ValueError(f"{name} contains an invalid one-based card relation")
                invalid = (row != -100) & (row != termination) & ((row < 0) | (row >= legal_count))
                if torch.any(invalid):
                    raise ValueError("targets contain an illegal or padded option index")

        for name in _LONG_TENSORS:
            if tensors[name].dtype is not torch.long:
                raise ValueError(f"{name} must have dtype torch.long")
        for name in ("global_num", "card_num", "resource_num", "event_num", "option_num"):
            if not tensors[name].dtype.is_floating_point:
                raise ValueError(f"{name} must use a floating dtype")
        return cls(tensors)

    @property
    def batch_size(self) -> int:
        return int(self._tensors["global_cat"].shape[0])

    @property
    def option_count(self) -> int:
        return int(self._tensors["option_mask"].shape[1])

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "DecisionBatch":
        return DecisionBatch.from_mapping(
            {
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self._tensors.items()
            }
        )

    def as_dict(self) -> dict[str, Tensor]:
        return dict(self._tensors)

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


__all__ = ["DecisionBatch"]
