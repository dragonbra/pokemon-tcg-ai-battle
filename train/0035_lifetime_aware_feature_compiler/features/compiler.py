"""Authoritative stateless canonical feature compiler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..domain.prototypes import PrototypeIndex
from ..knowledge.state import CausalSnapshot
from .layers import (
    ZONE,
    assemble_canonical_record,
    compile_card_layer,
    compile_event_layer,
    compile_global_layer,
    compile_option_layer,
    compile_resource_layer,
)


def compile_canonical_layers(
    row: Mapping[str, Any], snapshot: CausalSnapshot, prototypes: PrototypeIndex
) -> dict[str, Any]:
    """Compile all immutable layers and assemble the legacy mutable record."""
    cards = compile_card_layer(row, snapshot, prototypes)
    resources = compile_resource_layer(row, snapshot)
    events = compile_event_layer(row, snapshot, cards)
    options = compile_option_layer(row, snapshot, prototypes, cards)
    globals_ = compile_global_layer(row, snapshot)
    return assemble_canonical_record(row, cards, resources, events, options, globals_)


def compile_canonical_row(
    row: Mapping[str, Any], snapshot: CausalSnapshot, prototypes: PrototypeIndex
) -> dict[str, Any]:
    """Preserve the public full-rebuild API used by training and deployment."""
    return compile_canonical_layers(row, snapshot, prototypes)


__all__ = ["ZONE", "compile_canonical_layers", "compile_canonical_row"]
