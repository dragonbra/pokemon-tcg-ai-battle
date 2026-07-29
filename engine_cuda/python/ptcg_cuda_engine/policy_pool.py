from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


SUPPORTED_DTYPES = {"fp32", "fp16", "bf16", "int8"}


@dataclass(frozen=True)
class PolicySpec:
    policy_id: int
    name: str
    deck: str
    checkpoint: str
    adapter: str
    codec: str
    frozen: bool
    dtype: str = "bf16"
    parameter_mib: float | None = None
    provenance: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicySpec":
        allowed = {
            "policy_id",
            "name",
            "deck",
            "checkpoint",
            "adapter",
            "codec",
            "frozen",
            "dtype",
            "parameter_mib",
            "provenance",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown policy fields: {sorted(unknown)}")
        spec = cls(
            policy_id=int(raw["policy_id"]),
            name=str(raw["name"]),
            deck=str(raw["deck"]),
            checkpoint=str(raw["checkpoint"]),
            adapter=str(raw["adapter"]),
            codec=str(raw["codec"]),
            frozen=bool(raw["frozen"]),
            dtype=str(raw.get("dtype", "bf16")).lower(),
            parameter_mib=(
                None if raw.get("parameter_mib") is None else float(raw["parameter_mib"])
            ),
            provenance=(None if raw.get("provenance") is None else str(raw["provenance"])),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not 0 <= self.policy_id < 32:
            raise ValueError("policy_id must be in [0, 31]")
        for name, value in (
            ("name", self.name),
            ("deck", self.deck),
            ("checkpoint", self.checkpoint),
            ("adapter", self.adapter),
            ("codec", self.codec),
        ):
            if not value:
                raise ValueError(f"policy {name} must not be empty")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported policy dtype: {self.dtype}")
        if self.parameter_mib is not None and self.parameter_mib <= 0:
            raise ValueError("parameter_mib must be positive")


@dataclass(frozen=True)
class PolicyPoolManifest:
    name: str
    max_policies: int
    policies: tuple[PolicySpec, ...]
    routing: Mapping[str, Any]
    version: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyPoolManifest":
        allowed = {"version", "name", "max_policies", "routing", "policies", "notes"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        rows = raw.get("policies")
        if not isinstance(rows, list):
            raise ValueError("manifest policies must be a list")
        manifest = cls(
            version=int(raw.get("version", 1)),
            name=str(raw.get("name", "unnamed_pool")),
            max_policies=int(raw.get("max_policies", 32)),
            policies=tuple(PolicySpec.from_dict(item) for item in rows),
            routing=dict(raw.get("routing") or {}),
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "PolicyPoolManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("policy-pool manifest root must be an object")
        return cls.from_dict(raw)

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported policy-pool manifest version: {self.version}")
        if not 1 <= self.max_policies <= 32:
            raise ValueError("max_policies must be in [1, 32]")
        if not self.policies:
            raise ValueError("policy pool must not be empty")
        if len(self.policies) > self.max_policies:
            raise ValueError("policy count exceeds max_policies")
        ids = [policy.policy_id for policy in self.policies]
        if len(ids) != len(set(ids)):
            raise ValueError("policy IDs must be unique")
        if ids != list(range(len(ids))):
            raise ValueError("policy IDs must be contiguous from zero")
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        trainable = [policy for policy in self.policies if not policy.frozen]
        if len(trainable) > 1:
            raise ValueError("the rollout pool supports at most one trainable learner")
        for policy in self.policies:
            policy.validate()

    @property
    def frozen_policy_count(self) -> int:
        return sum(policy.frozen for policy in self.policies)

    @property
    def learner(self) -> PolicySpec | None:
        return next((policy for policy in self.policies if not policy.frozen), None)

    @property
    def codecs(self) -> tuple[str, ...]:
        return tuple(sorted({policy.codec for policy in self.policies}))

    @property
    def adapter_families(self) -> tuple[str, ...]:
        return tuple(sorted({policy.adapter for policy in self.policies}))


@dataclass(frozen=True)
class RoutePlan:
    routes: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]
    overflow_envs: tuple[int, ...]


class FixedRoutePlanner:
    """CPU reference for the fixed-capacity device policy router."""

    def __init__(self, policy_count: int, capacity: int):
        if not 1 <= policy_count <= 32:
            raise ValueError("policy_count must be in [1, 32]")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.policy_count = policy_count
        self.capacity = capacity

    def route(self, policy_ids: list[int], ready: list[bool] | None = None) -> RoutePlan:
        if ready is None:
            ready = [True] * len(policy_ids)
        if len(ready) != len(policy_ids):
            raise ValueError("ready mask length does not match policy IDs")
        routes = [[-1] * self.capacity for _ in range(self.policy_count)]
        counts = [0] * self.policy_count
        overflow: list[int] = []
        for env, (policy_id, is_ready) in enumerate(zip(policy_ids, ready)):
            if not is_ready:
                continue
            if not 0 <= policy_id < self.policy_count:
                raise ValueError(f"invalid policy ID {policy_id} at environment {env}")
            slot = counts[policy_id]
            counts[policy_id] += 1
            if slot >= self.capacity:
                overflow.append(env)
            else:
                routes[policy_id][slot] = env
        return RoutePlan(
            routes=tuple(tuple(row) for row in routes),
            counts=tuple(counts),
            overflow_envs=tuple(overflow),
        )


def route_torch(policy_ids: Any, ready_mask: Any, policy_count: int, capacity: int) -> tuple[Any, Any, Any]:
    """Build fixed routes entirely with device tensor operations.

    The function deliberately returns device counts without reading them on the
    host. Invalid or overflow rows map to a sink element that is dropped.
    """

    import torch

    if policy_ids.ndim != 1 or ready_mask.ndim != 1 or policy_ids.shape != ready_mask.shape:
        raise ValueError("policy_ids and ready_mask must be equal one-dimensional tensors")
    if not 1 <= policy_count <= 32 or capacity <= 0:
        raise ValueError("bad policy_count or capacity")
    device = policy_ids.device
    batch_size = policy_ids.shape[0]
    env_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    policy_axis = torch.arange(policy_count, dtype=torch.long, device=device)
    membership = ready_mask.bool().view(-1, 1) & policy_ids.long().view(-1, 1).eq(
        policy_axis.view(1, -1)
    )
    slots = membership.long().cumsum(dim=0) - 1
    policy_grid = policy_axis.view(1, -1).expand(batch_size, -1)
    valid = membership & slots.lt(capacity)
    sink = policy_count * capacity
    targets = torch.where(valid, policy_grid * capacity + slots, sink)
    sources = torch.where(
        valid,
        env_ids.view(-1, 1).expand(-1, policy_count),
        torch.full_like(targets, -1),
    )
    flat_routes = torch.full((sink + 1,), -1, dtype=torch.long, device=device)
    flat_routes.scatter_(0, targets.reshape(-1), sources.reshape(-1))
    routes = flat_routes[:sink].view(policy_count, capacity)
    counts = membership.long().sum(dim=0)
    return routes, routes.ge(0), counts


def _gather_device_batch(
    batch: Mapping[str, Any], route: Any, full_batch_size: int
) -> dict[str, Any]:
    safe_route = route.clamp_min(0)
    gathered: dict[str, Any] = {}
    for name, value in batch.items():
        if (
            hasattr(value, "ndim")
            and value.ndim > 0
            and value.shape[0] == full_batch_size
        ):
            gathered[name] = value.index_select(0, safe_route)
        else:
            gathered[name] = value
    return gathered


def normalize_actions_device(
    proposed: Any,
    proposed_lengths: Any,
    option_mask: Any,
    min_count: Any,
    max_count: Any,
    *,
    max_select: int,
) -> tuple[Any, Any]:
    """Normalize a padded multi-select action without leaving the device."""

    import torch

    if proposed.ndim != 2 or option_mask.ndim != 2:
        raise ValueError("proposed and option_mask must be rank two")
    if proposed.shape[0] != option_mask.shape[0]:
        raise ValueError("proposed and option_mask batch sizes differ")
    if max_select <= 0:
        raise ValueError("max_select must be positive")
    batch_size, option_count = option_mask.shape
    device = proposed.device
    output = torch.full((batch_size, max_select + 1), -1, dtype=torch.long, device=device)
    selected = torch.zeros((batch_size, option_count), dtype=torch.bool, device=device)
    accepted = torch.zeros(batch_size, dtype=torch.long, device=device)
    row_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    min_count = min_count.long().view(-1).clamp(min=0, max=max_select)
    max_count = max_count.long().view(-1).clamp(min=0, max=max_select)

    for step in range(proposed.shape[1]):
        candidate = proposed[:, step].long()
        in_range = candidate.ge(0) & candidate.lt(option_count) & step < proposed_lengths.view(-1)
        safe = candidate.clamp(min=0, max=max(option_count - 1, 0))
        legal = in_range & option_mask.bool().gather(1, safe.view(-1, 1)).view(-1)
        novel = ~selected.gather(1, safe.view(-1, 1)).view(-1)
        take = legal & novel & accepted.lt(max_count) & accepted.lt(max_select)
        destination = torch.where(take, accepted, torch.full_like(accepted, max_select))
        value = torch.where(take, safe, torch.full_like(safe, -1))
        output[row_ids, destination] = value
        selected[row_ids, safe] |= take
        accepted += take.long()

    for _ in range(max_select):
        available = option_mask.bool() & ~selected
        candidate = available.long().argmax(dim=1)
        take = accepted.lt(min_count) & accepted.lt(max_count) & available.any(dim=1)
        destination = torch.where(take, accepted, torch.full_like(accepted, max_select))
        value = torch.where(take, candidate, torch.full_like(candidate, -1))
        output[row_ids, destination] = value
        selected[row_ids, candidate] |= take
        accepted += take.long()

    return output[:, :max_select], accepted


class DevicePolicyAdapter(Protocol):
    def act_device(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        """Return padded option indices and lengths on the input device."""


@dataclass(frozen=True)
class DeviceActionBatch:
    indices: Any
    lengths: Any
    routes: Any
    route_mask: Any
    route_counts: Any


class GPUResidentPolicyPool:
    """Heterogeneous, fixed-capacity policy dispatcher.

    All adapters are constructed and moved to the target GPU before rollout.
    The Python loop launches one fixed-shape cohort per policy, but it performs
    no route-count read or tensor transfer. This is compatible with a later
    CUDA Graph capture and with models that do not share an architecture.
    """

    def __init__(
        self,
        manifest: PolicyPoolManifest,
        adapters: Mapping[int, DevicePolicyAdapter],
        *,
        capacity: int,
        max_select: int = 80,
    ):
        manifest.validate()
        if set(adapters) != {policy.policy_id for policy in manifest.policies}:
            raise ValueError("adapters must contain exactly one entry per manifest policy")
        if capacity <= 0 or max_select <= 0:
            raise ValueError("capacity and max_select must be positive")
        self.manifest = manifest
        self.adapters = dict(adapters)
        self.capacity = capacity
        self.max_select = max_select

    def act(
        self,
        encoded_by_codec: Mapping[str, Mapping[str, Any]],
        acting_policy_ids: Any,
        ready_mask: Any,
    ) -> DeviceActionBatch:
        import torch

        routes, route_mask, counts = route_torch(
            acting_policy_ids,
            ready_mask,
            len(self.manifest.policies),
            self.capacity,
        )
        batch_size = acting_policy_ids.shape[0]
        output = torch.full(
            (batch_size + 1, self.max_select),
            -1,
            dtype=torch.long,
            device=acting_policy_ids.device,
        )
        lengths = torch.zeros(
            batch_size + 1,
            dtype=torch.long,
            device=acting_policy_ids.device,
        )
        sink = torch.full((self.capacity,), batch_size, dtype=torch.long, device=acting_policy_ids.device)

        with torch.inference_mode():
            for policy in self.manifest.policies:
                if policy.codec not in encoded_by_codec:
                    raise KeyError(f"missing device codec batch: {policy.codec}")
                route = routes[policy.policy_id]
                valid = route_mask[policy.policy_id]
                cohort = _gather_device_batch(
                    encoded_by_codec[policy.codec], route, batch_size
                )
                cohort["_route_mask"] = valid
                proposed, proposed_lengths = self.adapters[policy.policy_id].act_device(cohort)
                normalized, normalized_lengths = normalize_actions_device(
                    proposed,
                    proposed_lengths,
                    cohort["option_mask"],
                    cohort["min_count"],
                    cohort["max_count"],
                    max_select=self.max_select,
                )
                destinations = torch.where(valid, route, sink)
                values = torch.where(valid.view(-1, 1), normalized, torch.full_like(normalized, -1))
                output.scatter_(
                    0,
                    destinations.view(-1, 1).expand(-1, self.max_select),
                    values,
                )
                lengths.scatter_(
                    0,
                    destinations,
                    torch.where(valid, normalized_lengths, torch.zeros_like(normalized_lengths)),
                )

        return DeviceActionBatch(
            indices=output[:batch_size],
            lengths=lengths[:batch_size],
            routes=routes,
            route_mask=route_mask,
            route_counts=counts,
        )
