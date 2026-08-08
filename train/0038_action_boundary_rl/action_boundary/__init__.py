"""Action-boundary contracts that sit above the unchanged official protocol."""

from .decision_gate import DecisionClass, DecisionGate, DecisionGateResult
from .dragapult import (
    DragapultDamageAllocation,
    StableTargetIdentity,
    enumerate_allocations,
)
from .macro_protocol import MacroProtocolError, PendingMacroTransaction
from .macro_planner import MacroPlanner, PlannedMacro
from .official_protocol import OfficialProtocolExecutor

__all__ = [
    "DecisionClass",
    "DecisionGate",
    "DecisionGateResult",
    "DragapultDamageAllocation",
    "MacroProtocolError",
    "MacroPlanner",
    "OfficialProtocolExecutor",
    "PendingMacroTransaction",
    "PlannedMacro",
    "StableTargetIdentity",
    "enumerate_allocations",
]
