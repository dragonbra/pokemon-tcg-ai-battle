"""Deck-agnostic CUDA engine prototype and CPU reference utilities."""

from .memory import MemoryEstimate, estimate_memory
from .native import create_engine, pack_reset_specs
from .policy_pool import MAX_POLICIES, FixedRoutePlanner, PolicyPoolManifest, PolicySpec
from .reference import BatchedReferenceEngine, ReferenceResetSpec
from .schema import Action, Instruction, RulePack
from .semantic0031_bridge import (
    KNOWN_0031_ARCHIVE_SHA256,
    Semantic0031DeviceAdapter,
    inspect_semantic0031_archive,
    load_semantic0031_package,
    semantic0031_v2_ready_batch,
)

__all__ = [
    "Action",
    "BatchedReferenceEngine",
    "FixedRoutePlanner",
    "Instruction",
    "MemoryEstimate",
    "MAX_POLICIES",
    "KNOWN_0031_ARCHIVE_SHA256",
    "PolicyPoolManifest",
    "PolicySpec",
    "ReferenceResetSpec",
    "RulePack",
    "Semantic0031DeviceAdapter",
    "create_engine",
    "estimate_memory",
    "inspect_semantic0031_archive",
    "load_semantic0031_package",
    "pack_reset_specs",
    "semantic0031_v2_ready_batch",
]
