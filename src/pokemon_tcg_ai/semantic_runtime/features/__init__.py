"""Canonical actor features compiled directly from official observations."""

from .compiler import compile_canonical_row
from ..contracts.fields import ACTOR_KEYS, SCHEMA_VERSION

__all__ = ["ACTOR_KEYS", "SCHEMA_VERSION", "compile_canonical_row"]
