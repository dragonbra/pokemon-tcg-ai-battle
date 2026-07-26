"""Actor-visible model feature contracts for the semantic goal policy."""

from .action_contract import (
    ActionTermination,
    OptionSemanticIdentity,
    OrderedAction,
    remap_action,
    validate_ordered_action,
)

__all__ = [
    "ActionTermination",
    "OptionSemanticIdentity",
    "OrderedAction",
    "remap_action",
    "validate_ordered_action",
]
