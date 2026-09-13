"""Complete-Actor CPU/deployment runtime for Public Deck Router V3."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Sequence

import torch

from . import compound_inference as _compound
from .public_deck_memory_v3 import (
    PUBLIC_POLICY_ID,
    PublicDeckMemory,
    ROUTED_UPDATES,
    route_manifest,
)
from .public_meta_router import PublicRoutedCompoundPolicy


class PublicDeckV3RoutedCompoundPolicy(PublicRoutedCompoundPolicy):
    """Switch each route's complete effective Actor, not only its decoder head."""

    policy_id = PUBLIC_POLICY_ID
    routed_updates = ROUTED_UPDATES
    compact_schema = "0045_public_deck_router_v3_complete_actor_v1"

    def __init__(
        self,
        policies: Mapping[int, _compound.PortableCompoundSemanticPolicy],
        *,
        default_update: int,
    ) -> None:
        if set(policies) != set(self.routed_updates):
            raise ValueError("CPU public deck V3 policy inventory mismatch")
        self.default_update = int(default_update)
        self.rule_manifest = route_manifest(self.default_update)
        default = policies[self.default_update]
        _compound.PortableCompoundSemanticPolicy.__init__(
            self,
            default.actor,
            default.value_head,
            default.allocation_head,
            default.value_adapter,
            default.policy_strategy_adapter,
            default.deck,
            default.metadata,
            default.policy_option_lora,
            default.meta_actor_residual,
        )
        if self.policy_strategy_adapter is not None or self.meta_actor_residual is not None:
            raise RuntimeError("0045 V3 router requires Critic-free minimal Actors")
        self.policies = dict(policies)
        self.public_memory = PublicDeckMemory(
            1, torch.device("cpu"), default_update=self.default_update
        )
        self.active_update = self.default_update
        self._activate(self.default_update)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: Mapping[int, Path],
        deck: Sequence[int],
        *,
        default_update: int,
    ) -> "PublicDeckV3RoutedCompoundPolicy":
        if set(checkpoints) != set(cls.routed_updates):
            raise ValueError("CPU public deck V3 checkpoint inventory mismatch")
        route_manifest(default_update)
        policies: dict[int, _compound.PortableCompoundSemanticPolicy] = {}
        for update in cls.routed_updates:
            payload = torch.load(
                checkpoints[update], map_location="cpu", weights_only=True
            )
            if (
                payload.get("schema_version") != "0045_minimal_lora_candidate_v1"
                or payload.get("metadata", {}).get("checkpoint_update") != update
            ):
                raise RuntimeError(f"U{update} portable candidate identity mismatch")
            floating = [
                value
                for field in (
                    "actor_state_dict",
                    "policy_option_lora_state_dict",
                    "allocation_head_state_dict",
                )
                for value in payload[field].values()
                if torch.is_floating_point(value)
            ]
            if not floating or any(value.dtype != torch.float16 for value in floating):
                raise RuntimeError(f"U{update} portable Actor is not FP16 stored")
            policy = _compound.PortableCompoundSemanticPolicy.from_checkpoint(
                checkpoints[update], deck
            )
            if policy.policy_option_lora is None:
                raise RuntimeError(f"U{update} portable candidate lacks Option LoRA")
            policies[update] = policy
        return cls(policies, default_update=default_update)

    @staticmethod
    def _zero_adapter_b_matrices(module: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if name.endswith(".b") or name.endswith("_b"):
                    parameter.zero_()

    @classmethod
    def from_compact_checkpoint(
        cls, default_checkpoint: Path, routed_deltas_checkpoint: Path,
        deck: Sequence[int], *, default_update: int,
    ) -> "PublicDeckV3RoutedCompoundPolicy":
        rules = route_manifest(default_update)
        default = _compound.PortableCompoundSemanticPolicy.from_checkpoint(
            default_checkpoint, deck
        )
        if int(default.metadata.get("checkpoint_update", -1)) != default_update:
            raise RuntimeError("V3 compact default checkpoint update mismatch")
        payload = torch.load(
            routed_deltas_checkpoint, map_location="cpu", weights_only=True
        )
        if (
            set(payload) != {"schema_version", "routed_policy_deltas", "metadata"}
            or payload.get("schema_version") != cls.compact_schema
        ):
            raise RuntimeError("V3 compact policy schema mismatch")
        metadata = payload.get("metadata") or {}
        if (
            metadata.get("policy_id") != cls.policy_id
            or metadata.get("default_checkpoint_update") != default_update
            or tuple(metadata.get("routed_updates", ())) != cls.routed_updates
            or metadata.get("rules") != rules
            or metadata.get("storage_dtype") != "fp16"
            or metadata.get("runtime_dtype") != "fp32"
        ):
            raise RuntimeError("V3 compact policy metadata mismatch")
        routed = payload.get("routed_policy_deltas")
        expected = {str(update) for update in cls.routed_updates if update != default_update}
        if not isinstance(routed, Mapping) or set(routed) != expected:
            raise RuntimeError("V3 compact route inventory mismatch")
        policies = {default_update: default}
        for update in cls.routed_updates:
            if update == default_update:
                continue
            stored = routed[str(update)]
            if set(stored) != {
                "actor_routed_state_dict", "policy_option_lora_state_dict",
                "allocation_head_state_dict",
            }:
                raise RuntimeError(f"U{update} compact route fields mismatch")
            floating = [
                value for state in stored.values() for value in state.values()
                if torch.is_floating_point(value)
            ]
            if not floating or any(value.dtype != torch.float16 for value in floating):
                raise RuntimeError(f"U{update} compact route is not FP16 stored")
            policy = copy.deepcopy(default)
            cls._zero_adapter_b_matrices(policy.actor)
            actor_result = policy.actor.load_state_dict({
                name: value.float() if torch.is_floating_point(value) else value
                for name, value in stored["actor_routed_state_dict"].items()
            }, strict=False)
            if actor_result.unexpected_keys:
                raise RuntimeError(f"U{update} compact Actor has unexpected tensors")
            cls._zero_adapter_b_matrices(policy.policy_option_lora)
            option_result = policy.policy_option_lora.load_state_dict({
                name: value.float() if torch.is_floating_point(value) else value
                for name, value in stored["policy_option_lora_state_dict"].items()
            }, strict=False)
            if option_result.unexpected_keys:
                raise RuntimeError(f"U{update} compact Option LoRA has unexpected tensors")
            policy.allocation_head.load_state_dict({
                name: value.float() if torch.is_floating_point(value) else value
                for name, value in stored["allocation_head_state_dict"].items()
            }, strict=True)
            policy.metadata = dict(policy.metadata)
            policy.metadata["checkpoint_update"] = update
            policies[update] = policy
        result = cls(policies, default_update=default_update)
        result.deployment_identity = dict(metadata)
        return result

    def _activate(self, update: int) -> None:
        policy = self.policies[int(update)]
        self.actor = policy.actor
        self.policy_option_lora = policy.policy_option_lora
        self.allocation_head = policy.allocation_head
        self.planner = _compound.MacroPlanner(self.allocation_head)
        self.active_update = int(update)

    def reset(self) -> None:
        _compound.PortableCompoundSemanticPolicy.reset(self)
        self.public_memory = PublicDeckMemory(
            1, torch.device("cpu"), default_update=self.default_update
        )
        self._activate(self.default_update)


__all__ = ["PublicDeckV3RoutedCompoundPolicy"]
