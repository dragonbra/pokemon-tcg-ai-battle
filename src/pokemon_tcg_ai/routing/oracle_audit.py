"""Explicitly cheating 29-way opponent-Meta router for diagnostic evaluation only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ..evaluation.candidate import CandidateAudit, materialize as materialize_candidate


ORACLE_POLICY_ID = "Experimental-Oracle-MetaRouter-V1"
DEFAULT_UPDATE = 282
ROUTED_UPDATES = (40, 90, 200, 282)
ROUTED_ACTOR_PREFIX = "action_decoder."


def route_update_for_meta(meta_id: int) -> int:
    """Literal oracle route requested by the experiment owner."""
    if not isinstance(meta_id, int) or not 0 <= meta_id < 29:
        raise ValueError("oracle route requires a valid 29-way Meta ID")
    if meta_id in (6, 8, 10, 27):
        return 200
    elif meta_id in (1, 2, 17):
        return 40
    elif meta_id == 3:
        return 90
    return DEFAULT_UPDATE


def route_manifest() -> dict[str, int]:
    return {f"{meta_id:02d}": route_update_for_meta(meta_id) for meta_id in range(29)}


def partition_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[Mapping[str, Any]]]:
    partitions: dict[int, list[Mapping[str, Any]]] = {
        update: [] for update in ROUTED_UPDATES
    }
    seen: set[str] = set()
    for row in rows:
        game_id = str(row["game_id"])
        if game_id in seen:
            raise RuntimeError(f"duplicate game ID in oracle schedule: {game_id}")
        seen.add(game_id)
        update = route_update_for_meta(int(row["opponent_meta_archetype_id"]))
        partitions[update].append(row)
    return partitions


def scatter_by_game_id(
    schedule_rows: Sequence[Mapping[str, Any]], values_by_game_id: Mapping[str, Any]
) -> list[Any]:
    expected = [str(row["game_id"]) for row in schedule_rows]
    if len(set(expected)) != len(expected) or set(expected) != set(values_by_game_id):
        raise RuntimeError("oracle game ID inventory mismatch during scatter")
    return [values_by_game_id[game_id] for game_id in expected]


def _tensor_digest(rows: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(rows.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0"); digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_shared_actor_states(
    actor_states: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    if set(actor_states) != set(ROUTED_UPDATES):
        # Unit-sized audit fixtures are allowed as long as they contain the default.
        if DEFAULT_UPDATE not in actor_states or len(actor_states) < 2:
            raise ValueError("shared Actor audit requires the default and another route")
    reference = actor_states[DEFAULT_UPDATE]
    names = set(reference)
    if any(set(state) != names for state in actor_states.values()):
        raise RuntimeError("shared Actor tensor inventory mismatch")
    excluded = sorted(name for name in names if name.startswith(ROUTED_ACTOR_PREFIX))
    shared = sorted(names - set(excluded))
    for update, state in actor_states.items():
        for name in shared:
            if not torch.equal(reference[name], state[name]):
                raise RuntimeError(
                    f"shared frozen Actor tensor mismatch: U{update} {name}"
                )
    return {
        "status": "PASS",
        "reference_update": DEFAULT_UPDATE,
        "shared_tensor_count": len(shared),
        "shared_tensor_names": shared,
        "shared_effective_sha256": _tensor_digest({name: reference[name] for name in shared}),
        "excluded_routed_actor_tensor_count": len(excluded),
        "excluded_routed_actor_tensor_names": excluded,
        "compared_updates": sorted(actor_states),
        "comparison": "torch.equal_after_fp16_storage_fp32_runtime",
    }


def _module_state_hash(*modules: nn.Module) -> str:
    rows: dict[str, torch.Tensor] = {}
    for index, module in enumerate(modules):
        rows.update({f"module{index}.{name}": value for name, value in module.state_dict().items()})
    return _tensor_digest(rows)


def _parameter_pointers(*modules: nn.Module) -> set[int]:
    return {
        parameter.untyped_storage().data_ptr()
        for module in modules for parameter in module.parameters()
    }


@dataclass(frozen=True, slots=True)
class OracleHeadBundle:
    checkpoint_update: int
    action_decoder: nn.Module
    policy_option_lora: nn.Module
    allocation_head: nn.Module
    component_sha256: str
    candidate_audit: CandidateAudit


@dataclass(slots=True)
class MetaOracleRuntime:
    model: nn.Module
    heads: Mapping[int, OracleHeadBundle]
    audit: Mapping[str, Any]
    active_update: int = DEFAULT_UPDATE

    def activate(self, update: int) -> None:
        if update not in self.heads:
            raise KeyError(f"oracle has no routed head U{update}")
        bundle = self.heads[update]
        self.model.actor.action_decoder = bundle.action_decoder
        self.model.policy_option_lora = bundle.policy_option_lora
        self.model.allocation_head = bundle.allocation_head
        self.active_update = update


def materialize_oracle(
    *, checkpoints: Mapping[int, Path], base_portable: Path,
    deck: Sequence[int], deck_id: str, own_archetype_id: int,
    output_root: Path, device: torch.device,
) -> MetaOracleRuntime:
    if set(checkpoints) != set(ROUTED_UPDATES):
        raise ValueError(f"oracle checkpoints must be exactly {ROUTED_UPDATES}")
    models: dict[int, nn.Module] = {}
    candidate_audits: dict[int, CandidateAudit] = {}
    for update in ROUTED_UPDATES:
        model, audit = materialize_candidate(
            checkpoint=checkpoints[update], base_portable=base_portable,
            deck=deck, deck_id=deck_id, own_archetype_id=own_archetype_id,
            output=output_root / f"update-{update:06d}/model.bin",
            device=device,
        )
        if audit.checkpoint_update != update or audit.status != "PASS":
            raise RuntimeError(f"U{update} candidate materialization identity failed")
        models[update] = model
        candidate_audits[update] = audit

    shared_audit = audit_shared_actor_states({
        update: model.actor.state_dict() for update, model in models.items()
    })
    heads: dict[int, OracleHeadBundle] = {}
    head_pointers: dict[int, set[int]] = {}
    for update, model in models.items():
        modules = (
            model.actor.action_decoder,
            model.policy_option_lora,
            model.allocation_head,
        )
        heads[update] = OracleHeadBundle(
            checkpoint_update=update,
            action_decoder=modules[0],
            policy_option_lora=modules[1],
            allocation_head=modules[2],
            component_sha256=_module_state_hash(*modules),
            candidate_audit=candidate_audits[update],
        )
        head_pointers[update] = _parameter_pointers(*modules)
    for left_index, left in enumerate(ROUTED_UPDATES):
        for right in ROUTED_UPDATES[left_index + 1:]:
            if head_pointers[left] & head_pointers[right]:
                raise RuntimeError(f"oracle routed heads U{left}/U{right} alias storage")

    shared_model = models[DEFAULT_UPDATE]
    shared_actor_pointers = {
        parameter.untyped_storage().data_ptr()
        for name, parameter in shared_model.actor.named_parameters()
        if not name.startswith(ROUTED_ACTOR_PREFIX)
    }
    if any(shared_actor_pointers & pointers for pointers in head_pointers.values()):
        raise RuntimeError("oracle routed head aliases shared frozen Actor storage")

    composite_payload = {
        "policy_id": ORACLE_POLICY_ID,
        "route": route_manifest(),
        "shared_effective_sha256": shared_audit["shared_effective_sha256"],
        "head_effective_sha256": {
            str(update): heads[update].component_sha256 for update in ROUTED_UPDATES
        },
        "candidate_effective_sha256": {
            str(update): candidate_audits[update].effective_candidate_sha256
            for update in ROUTED_UPDATES
        },
    }
    composite_sha256 = hashlib.sha256(
        json.dumps(composite_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    audit = {
        "schema_version": "0045_meta_oracle_v1_identity_audit",
        "status": "PASS",
        "policy_id": ORACLE_POLICY_ID,
        "evidence_class": "diagnostic_oracle_not_promote_not_kaggle",
        "cheat_feature": "opponent_exact_deck_to_own_archetypes_v2_meta_id",
        "default_checkpoint_update": DEFAULT_UPDATE,
        "route": route_manifest(),
        "shared_actor": shared_audit,
        "routed_modules": [
            "policy_option_lora", "actor.action_decoder", "allocation_head"
        ],
        "head_effective_sha256": composite_payload["head_effective_sha256"],
        "candidate_materializations": {
            str(update): asdict(candidate_audits[update]) for update in ROUTED_UPDATES
        },
        "shared_vs_head_storage_aliases": 0,
        "cross_head_storage_aliases": 0,
        "critic_source_update": DEFAULT_UPDATE,
        "critic_outputs_consumed_by_actor": False,
        "composite_effective_sha256": composite_sha256,
    }
    runtime = MetaOracleRuntime(model=shared_model, heads=heads, audit=audit)
    runtime.activate(DEFAULT_UPDATE)
    return runtime


__all__ = [
    "DEFAULT_UPDATE", "MetaOracleRuntime", "ORACLE_POLICY_ID", "OracleHeadBundle",
    "ROUTED_UPDATES", "audit_shared_actor_states", "materialize_oracle",
    "partition_rows", "route_manifest", "route_update_for_meta", "scatter_by_game_id",
]
