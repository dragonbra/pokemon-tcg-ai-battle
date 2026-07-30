"""Resident GPU inference for compatible Arena BC opponents."""

from .catalog import OpponentInferenceKind, classify_opponent
from .resident_pool import OpponentRequest, ResidentOpponentPool

__all__ = [
    "OpponentInferenceKind",
    "OpponentRequest",
    "ResidentOpponentPool",
    "classify_opponent",
]
