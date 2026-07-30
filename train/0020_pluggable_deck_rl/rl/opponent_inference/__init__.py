"""Resident GPU inference for compatible Arena BC opponents."""

from .catalog import OpponentInferenceKind, classify_opponent
from .resident_pool import OpponentRequest, ResidentOpponentPool
from .shared_foundation_pool import SharedFoundationOpponentPool

__all__ = [
    "OpponentInferenceKind",
    "OpponentRequest",
    "ResidentOpponentPool",
    "SharedFoundationOpponentPool",
    "classify_opponent",
]
