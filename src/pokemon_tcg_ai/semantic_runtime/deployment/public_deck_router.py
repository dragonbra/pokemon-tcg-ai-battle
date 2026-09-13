"""Packaged CPU runtime for Public Deck Router V2."""

from __future__ import annotations

import torch

from .public_deck_memory import (
    DEFAULT_UPDATE, PUBLIC_POLICY_ID, PublicDeckMemory, ROUTED_UPDATES,
    RULE_MANIFEST,
)
from .public_meta_router import PublicRoutedCompoundPolicy


class PublicDeckRoutedCompoundPolicy(PublicRoutedCompoundPolicy):
    policy_id = PUBLIC_POLICY_ID
    default_update = DEFAULT_UPDATE
    routed_updates = ROUTED_UPDATES
    rule_manifest = RULE_MANIFEST
    memory_type = PublicDeckMemory
    compact_schema = "0045_public_deck_router_v2_compact_heads_v1"

    @classmethod
    def _shared_actor_state(cls, actor: torch.nn.Module) -> dict[str, torch.Tensor]:
        effective: dict[str, torch.Tensor] = {}
        raw_prefixes: list[str] = []
        for module_name, module in actor.named_modules():
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is None:
                continue
            for tensor_name in parametrizations.keys():
                name = f"{module_name}.{tensor_name}" if module_name else tensor_name
                raw = (
                    f"{module_name}.parametrizations.{tensor_name}"
                    if module_name else f"parametrizations.{tensor_name}"
                )
                raw_prefixes.append(raw)
                effective[name] = getattr(module, tensor_name).detach()
        for name, value in actor.state_dict().items():
            if name.startswith("action_decoder.") or any(
                name.startswith(prefix + ".") for prefix in raw_prefixes
            ):
                continue
            effective[name] = value
        return effective


__all__ = ["PublicDeckRoutedCompoundPolicy"]
