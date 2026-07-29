"""Deck-agnostic CUDA engine prototype and CPU reference utilities."""

from .memory import MemoryEstimate, estimate_memory
from .native import create_engine, pack_reset_specs
from .policy_pool import FixedRoutePlanner, PolicyPoolManifest, PolicySpec
from .reference import BatchedReferenceEngine, ReferenceResetSpec
from .schema import Action, Instruction, RulePack

__all__ = [
    "Action",
    "BatchedReferenceEngine",
    "FixedRoutePlanner",
    "Instruction",
    "MemoryEstimate",
    "PolicyPoolManifest",
    "PolicySpec",
    "ReferenceResetSpec",
    "RulePack",
    "create_engine",
    "estimate_memory",
    "pack_reset_specs",
]
