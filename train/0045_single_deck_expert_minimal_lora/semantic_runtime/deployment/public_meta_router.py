"""CPU/official-environment Agent wrapper for public-information expert routing."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .public_meta_memory import (
    DEFAULT_UPDATE,
    PUBLIC_POLICY_ID,
    PublicMetaMemory,
    ROUTED_UPDATES,
    RULE_MANIFEST,
)
from . import compound_inference as _compound


@dataclass(frozen=True, slots=True)
class CpuPolicyHead:
    checkpoint_update: int
    action_decoder: torch.nn.Module
    policy_option_lora: torch.nn.Module
    allocation_head: torch.nn.Module


class PublicRoutedCompoundPolicy(_compound.PortableCompoundSemanticPolicy):
    """One shared semantic backbone with Agent-owned per-battle public routing."""

    policy_id = PUBLIC_POLICY_ID

    def __init__(
        self,
        default: _compound.PortableCompoundSemanticPolicy,
        heads: Mapping[int, CpuPolicyHead],
    ) -> None:
        if set(heads) != set(ROUTED_UPDATES):
            raise ValueError("CPU public router requires U40/U90/U200/U282 heads")
        super().__init__(
            default.actor, default.value_head, default.allocation_head,
            default.value_adapter, default.policy_strategy_adapter,
            default.deck, default.metadata, default.policy_option_lora,
            default.meta_actor_residual,
        )
        if self.policy_strategy_adapter is not None or self.meta_actor_residual is not None:
            raise RuntimeError("0045 public router requires Critic-free minimal Actor")
        self.heads = dict(heads)
        self.public_memory = PublicMetaMemory(1, torch.device("cpu"))
        self.active_update = DEFAULT_UPDATE
        self._activate(DEFAULT_UPDATE)

    @classmethod
    def from_checkpoints(
        cls, checkpoints: Mapping[int, Path], deck: Sequence[int]
    ) -> "PublicRoutedCompoundPolicy":
        if set(checkpoints) != set(ROUTED_UPDATES):
            raise ValueError("CPU public router checkpoint inventory mismatch")
        policies = {
            update: _compound.PortableCompoundSemanticPolicy.from_checkpoint(path, deck)
            for update, path in checkpoints.items()
        }
        heads = {
            update: CpuPolicyHead(
                update, policy.actor.action_decoder,
                policy.policy_option_lora, policy.allocation_head,
            )
            for update, policy in policies.items()
        }
        if any(head.policy_option_lora is None for head in heads.values()):
            raise RuntimeError("CPU public router source lacks Policy Option LoRA")
        return cls(policies[DEFAULT_UPDATE], heads)

    @staticmethod
    def _tensor_digest(rows: Mapping[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(rows.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8")); digest.update(b"\0")
            digest.update(str(tensor.dtype).encode("ascii")); digest.update(b"\0")
            digest.update(
                json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
            ); digest.update(b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def _head_hash(cls, head: CpuPolicyHead) -> str:
        rows: dict[str, torch.Tensor] = {}
        for index, module in enumerate((
            head.action_decoder, head.policy_option_lora, head.allocation_head,
        )):
            rows.update({
                f"module{index}.{name}": value
                for name, value in module.state_dict().items()
            })
        return cls._tensor_digest(rows)

    @classmethod
    def from_compact_checkpoint(
        cls, default_checkpoint: Path, routed_heads_checkpoint: Path,
        deck: Sequence[int],
    ) -> "PublicRoutedCompoundPolicy":
        """Load one complete U282 candidate plus three small FP16 head deltas."""
        default = _compound.PortableCompoundSemanticPolicy.from_checkpoint(
            default_checkpoint, deck
        )
        if int(default.metadata.get("checkpoint_update", -1)) != DEFAULT_UPDATE:
            raise RuntimeError("compact public router default is not U282")
        payload = torch.load(
            routed_heads_checkpoint, map_location="cpu", weights_only=True
        )
        if set(payload) != {
            "schema_version", "routed_head_state_dicts", "metadata",
        } or payload.get("schema_version") != "0045_public_meta_router_compact_heads_v1":
            raise RuntimeError("compact public-router head schema mismatch")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping) or (
            metadata.get("policy_id") != PUBLIC_POLICY_ID
            or metadata.get("default_checkpoint_update") != DEFAULT_UPDATE
            or tuple(metadata.get("routed_updates", ())) != ROUTED_UPDATES
            or metadata.get("rules") != RULE_MANIFEST
            or metadata.get("storage_dtype") != "fp16"
            or metadata.get("runtime_dtype") != "fp32"
            or metadata.get("deployment_contract")
            != "kaggle_fp16_storage_fp32_runtime_v1"
        ):
            raise RuntimeError("compact public-router metadata identity mismatch")
        rules_hash = hashlib.sha256(json.dumps(
            RULE_MANIFEST, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if metadata.get("rules_sha256") != rules_hash:
            raise RuntimeError("compact public-router rule hash mismatch")
        routed = payload.get("routed_head_state_dicts")
        expected_delta_updates = {str(update) for update in ROUTED_UPDATES if update != DEFAULT_UPDATE}
        if not isinstance(routed, Mapping) or set(routed) != expected_delta_updates:
            raise RuntimeError("compact public-router delta inventory mismatch")

        default_head = CpuPolicyHead(
            DEFAULT_UPDATE, default.actor.action_decoder,
            default.policy_option_lora, default.allocation_head,
        )
        heads: dict[int, CpuPolicyHead] = {DEFAULT_UPDATE: default_head}
        for update in ROUTED_UPDATES:
            if update == DEFAULT_UPDATE:
                continue
            stored = routed[str(update)]
            if not isinstance(stored, Mapping) or set(stored) != {
                "action_decoder_state_dict", "policy_option_lora_state_dict",
                "allocation_head_state_dict",
            }:
                raise RuntimeError(f"U{update} compact head fields mismatch")
            floating = [
                value for state in stored.values() for value in state.values()
                if torch.is_floating_point(value)
            ]
            if not floating or any(value.dtype != torch.float16 for value in floating):
                raise RuntimeError(f"U{update} compact head is not FP16 stored")
            decoder = copy.deepcopy(default.actor.action_decoder)
            option_lora = copy.deepcopy(default.policy_option_lora)
            allocation = copy.deepcopy(default.allocation_head)
            decoder.load_state_dict({
                name: value.float() if torch.is_floating_point(value) else value
                for name, value in stored["action_decoder_state_dict"].items()
            }, strict=True)
            option_lora.load_state_dict({
                name: value.float() if torch.is_floating_point(value) else value
                for name, value in stored["policy_option_lora_state_dict"].items()
            }, strict=True)
            allocation.load_state_dict({
                name: value.float() if torch.is_floating_point(value) else value
                for name, value in stored["allocation_head_state_dict"].items()
            }, strict=True)
            heads[update] = CpuPolicyHead(update, decoder, option_lora, allocation)

        expected_heads = metadata.get("head_effective_sha256")
        if not isinstance(expected_heads, Mapping) or set(expected_heads) != {
            str(update) for update in ROUTED_UPDATES
        }:
            raise RuntimeError("compact public-router head identity inventory mismatch")
        observed_heads = {
            str(update): cls._head_hash(head) for update, head in heads.items()
        }
        if observed_heads != dict(expected_heads):
            raise RuntimeError("compact public-router effective head identity mismatch")
        shared = {
            name: value for name, value in default.actor.state_dict().items()
            if not name.startswith("action_decoder.")
        }
        if cls._tensor_digest(shared) != metadata.get("shared_effective_sha256"):
            raise RuntimeError("compact public-router shared Actor identity mismatch")
        source_payload = metadata.get("source_identity_payload")
        if not isinstance(source_payload, Mapping) or (
            source_payload.get("shared_effective_sha256")
            != metadata.get("shared_effective_sha256")
            or source_payload.get("head_effective_sha256") != expected_heads
            or source_payload.get("candidate_effective_sha256")
            != metadata.get("candidate_effective_sha256")
        ):
            raise RuntimeError("compact public-router source identity payload mismatch")
        source_composite = hashlib.sha256(json.dumps(
            source_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if source_composite != metadata.get("source_composite_sha256"):
            raise RuntimeError("compact public-router source composite mismatch")
        identity_payload = {
            "policy_id": PUBLIC_POLICY_ID,
            "rules": RULE_MANIFEST,
            "source_composite": source_composite,
        }
        composite = hashlib.sha256(json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if composite != metadata.get("composite_effective_sha256"):
            raise RuntimeError("compact public-router deployment composite mismatch")
        policy = cls(default, heads)
        runtime_floating = [
            parameter for module in (
                policy.actor, policy.value_head, policy.value_adapter,
                *(head.action_decoder for head in heads.values()),
                *(head.policy_option_lora for head in heads.values()),
                *(head.allocation_head for head in heads.values()),
            ) for parameter in module.parameters() if torch.is_floating_point(parameter)
        ]
        if any(parameter.dtype != torch.float32 for parameter in runtime_floating):
            raise RuntimeError("compact public-router runtime is not wholly FP32")
        policy.deployment_identity = dict(metadata)
        return policy

    def _activate(self, update: int) -> None:
        head = self.heads[int(update)]
        self.actor.action_decoder = head.action_decoder
        self.policy_option_lora = head.policy_option_lora
        self.allocation_head = head.allocation_head
        self.planner = _compound.MacroPlanner(self.allocation_head)
        self.active_update = int(update)

    def reset(self) -> None:
        super().reset()
        self.public_memory = PublicMetaMemory(1, torch.device("cpu"))
        self._activate(DEFAULT_UPDATE)

    def select(self, observation: dict[str, Any]) -> list[int]:
        current = observation.get("current") or {}
        actor = current.get("yourIndex")
        if actor not in (0, 1):
            raise ValueError("observation has no valid actor")
        encoder = self._ensure_encoder(int(actor))

        if self.pending is not None:
            self._observe_only(observation, int(actor))
            try:
                action = self.pending.next_primitive(
                    observation, battle_id=self.battle_id
                )
            except _compound.MacroProtocolError:
                self.pending = None
                raise
            else:
                if self.pending.complete:
                    self.pending = None
                return action

        gate = self.gate.classify(observation)
        if gate.classification is _compound.DecisionClass.MASK_ERROR:
            raise RuntimeError(f"DecisionGate MASK_ERROR: {gate.reason}")
        if gate.classification in {
            _compound.DecisionClass.FORCED,
            _compound.DecisionClass.LEGAL_EMPTY_PASS,
        }:
            self._observe_only(observation, int(actor))
            return list(gate.forced_action or ())
        if gate.classification is not _compound.DecisionClass.STRATEGIC:
            raise RuntimeError(f"unsupported decision class: {gate.classification}")

        batch = _compound.DecisionBatch.from_mapping(encoder.encode(observation))
        public_view = self.actor.validate_batch(batch)
        update = int(self.public_memory.observe(
            public_view, torch.tensor([0], dtype=torch.long)
        )[0])
        self._activate(update)
        with torch.inference_mode():
            validated, state, _, options = self._encode_options(batch)
            sequences, lengths, legal = self._greedy_strategy(
                validated, state, options, None
            )
        if not bool(legal[0]):
            raise RuntimeError("public-router greedy decode violated selection bounds")
        action = sequences[0, : int(lengths[0])].tolist()

        root_options = (observation.get("select") or {}).get("option") or []
        phantom_root = next((
            index for index in action
            if 0 <= index < len(root_options)
            and isinstance(root_options[index], Mapping)
            and root_options[index].get("attackId")
            == _compound.PHANTOM_DIVE_ATTACK_ID
        ), None)
        players = current.get("players") or []
        raw_bench = (
            list(players[1 - int(actor)].get("bench") or [])
            if len(players) == 2 and isinstance(players[1 - int(actor)], Mapping)
            else []
        )
        if (
            phantom_root is not None and 1 <= len(raw_bench) <= 8
            and not self._confused(current, int(actor))
        ):
            identities = sorted(
                _compound.StableTargetIdentity(
                    1 - int(actor), int(raw["serial"]), int(raw["id"]), slot
                )
                for slot, raw in enumerate(raw_bench)
            )
            raw_by_serial = {
                int(raw["serial"]): {**raw, "benchSlot": slot}
                for slot, raw in enumerate(raw_bench)
            }
            target_rows = []
            for target in identities:
                card_mask = (
                    validated.card_mask[0]
                    & validated.card_cat[0, :, 2].eq(2)
                    & validated.card_cat[0, :, 3].eq(6)
                    & validated.card_cat[0, :, 4].eq(
                        target.initial_bench_slot + 1
                    )
                )
                positions = card_mask.nonzero(as_tuple=False).flatten()
                if positions.numel() != 1:
                    raise RuntimeError("Phantom target does not map to one state token")
                target_rows.append(state.cards[0, int(positions[0])])
            planned = self.planner.plan(
                state_summary=state.summary[0],
                root_option=options[0, phantom_root],
                target_embeddings=torch.stack(target_rows),
                target_identities=identities,
                visible_targets=[
                    _compound.with_public_prize(
                        raw_by_serial[target.serial], self.prizes
                    )
                    for target in identities
                ],
                greedy=True,
            )
            self.pending = _compound.PendingMacroTransaction(
                self.battle_id, int(actor), planned.allocation
            )
        return action


__all__ = ["CpuPolicyHead", "PublicRoutedCompoundPolicy"]
