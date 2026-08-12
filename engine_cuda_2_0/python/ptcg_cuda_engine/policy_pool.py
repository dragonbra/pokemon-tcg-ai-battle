from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


SUPPORTED_DTYPES = {"fp32", "fp16", "bf16", "int8"}
MAX_POLICIES = 64


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
        if not 0 <= self.policy_id < MAX_POLICIES:
            raise ValueError(f"policy_id must be in [0, {MAX_POLICIES - 1}]")
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
            max_policies=int(raw.get("max_policies", MAX_POLICIES)),
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
        if not 1 <= self.max_policies <= MAX_POLICIES:
            raise ValueError(f"max_policies must be in [1, {MAX_POLICIES}]")
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
        if not 1 <= policy_count <= MAX_POLICIES:
            raise ValueError(f"policy_count must be in [1, {MAX_POLICIES}]")
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
    if not 1 <= policy_count <= MAX_POLICIES or capacity <= 0:
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


def _static_shared_adapter_info(adapter: Any) -> tuple[Any, Any, Mapping[str, Any]] | None:
    key = getattr(adapter, "shared_batch_key", None)
    fields = getattr(adapter, "shared_static_fields", None)
    if key is None or fields is None:
        return None
    return key, adapter, fields


def _padded_static_fields_for_shared_group(
    adapters: list[Any], capacity: int, device: Any
) -> dict[str, Any]:
    import torch

    infos = [_static_shared_adapter_info(adapter) for adapter in adapters]
    if any(info is None for info in infos):
        raise ValueError("shared adapters must expose static fields")
    fields_by_adapter = [dict(info[2]) for info in infos if info is not None]
    names = set(fields_by_adapter[0])
    if any(set(fields) != names for fields in fields_by_adapter):
        raise ValueError("shared adapters have incompatible static fields")
    merged: dict[str, Any] = {}
    for name in sorted(names):
        values = [fields[name] for fields in fields_by_adapter]
        if any(value.device != device for value in values):
            raise ValueError(f"static field {name} is on the wrong device")
        ranks = {int(value.ndim) for value in values}
        if len(ranks) != 1:
            raise ValueError(f"static field {name} rank mismatch")
        rank = ranks.pop()
        if rank == 0:
            merged[name] = torch.cat([value.expand(capacity) for value in values], dim=0)
            continue
        if any(value.shape[0] != 1 for value in values):
            raise ValueError(f"static field {name} must have singleton batch axes")
        tail = tuple(max(int(value.shape[axis]) for value in values) for axis in range(1, rank))
        rows = []
        for value in values:
            expanded = value.expand(capacity, *value.shape[1:])
            if tuple(value.shape[1:]) == tail:
                rows.append(expanded)
                continue
            padded = torch.zeros((capacity, *tail), dtype=value.dtype, device=value.device)
            slices = (slice(None),) + tuple(slice(0, int(size)) for size in value.shape[1:])
            padded[slices] = expanded
            rows.append(padded)
        merged[name] = torch.cat(rows, dim=0)
    return merged


def _lane_static_fields_for_shared_group(
    policies: list[PolicySpec], adapters: list[Any], acting_policy_ids: Any, device: Any
) -> dict[str, Any]:
    import torch

    infos = [_static_shared_adapter_info(adapter) for adapter in adapters]
    if any(info is None for info in infos):
        raise ValueError("shared adapters must expose static fields")
    fields_by_adapter = [dict(info[2]) for info in infos if info is not None]
    names = set(fields_by_adapter[0])
    if any(set(fields) != names for fields in fields_by_adapter):
        raise ValueError("shared adapters have incompatible static fields")
    selector = torch.zeros_like(acting_policy_ids, dtype=torch.long)
    for local_index, policy in enumerate(policies):
        selector = torch.where(
            acting_policy_ids.long().eq(int(policy.policy_id)),
            torch.full_like(selector, int(local_index)),
            selector,
        )
    merged: dict[str, Any] = {}
    for name in sorted(names):
        values = [fields[name] for fields in fields_by_adapter]
        if any(value.device != device for value in values):
            raise ValueError(f"static field {name} is on the wrong device")
        ranks = {int(value.ndim) for value in values}
        if len(ranks) != 1:
            raise ValueError(f"static field {name} rank mismatch")
        dtypes = {value.dtype for value in values}
        if len(dtypes) != 1:
            raise ValueError(f"static field {name} dtype mismatch")
        rank = ranks.pop()
        if rank == 0:
            stacked = torch.stack(values, dim=0)
            merged[name] = stacked.index_select(0, selector)
            continue
        if any(value.shape[0] != 1 for value in values):
            raise ValueError(f"static field {name} must have singleton batch axes")
        tail = tuple(
            max(int(value.shape[axis]) for value in values)
            for axis in range(1, rank)
        )
        rows = []
        for value in values:
            padded = torch.zeros(tail, dtype=value.dtype, device=value.device)
            slices = tuple(slice(0, int(size)) for size in value.shape[1:])
            padded[slices] = value[0]
            rows.append(padded)
        stacked = torch.stack(rows, dim=0)
        merged[name] = stacked.index_select(0, selector)
    return merged


def _shared_policy_groups(policies: tuple[PolicySpec, ...], adapters: Mapping[int, Any]) -> list[list[PolicySpec]]:
    buckets: dict[tuple[str, Any], list[PolicySpec]] = {}
    for policy in policies:
        info = _static_shared_adapter_info(adapters[policy.policy_id])
        if info is None:
            continue
        key, _inner, _fields = info
        buckets.setdefault((policy.codec, key), []).append(policy)
    return [group for group in buckets.values() if len(group) > 1]


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

    available = option_mask.bool() & ~selected
    fill_needed = (min_count - accepted).clamp_min(0)
    fill_room = torch.minimum(max_count - accepted, max_select - accepted).clamp_min(0)
    fill_needed = torch.minimum(fill_needed, fill_room)
    if option_count:
        option_ids = torch.arange(option_count, dtype=torch.long, device=device).view(1, -1)
        fill_rank = available.long().cumsum(dim=1) - 1
        fill_take = available & fill_rank.lt(fill_needed.view(-1, 1))
        fill_dest = accepted.view(-1, 1) + fill_rank
        sink = torch.full_like(fill_dest, max_select)
        scatter_dest = torch.where(fill_take, fill_dest, sink)
        scatter_values = torch.where(
            fill_take,
            option_ids.expand(batch_size, -1),
            torch.full_like(fill_dest, -1),
        )
        output.scatter_(1, scatter_dest, scatter_values)
        accepted += fill_take.long().sum(dim=1)

    return output[:, :max_select], accepted


def extend_normalized_actions_device(
    proposed: Any,
    proposed_lengths: Any,
    option_mask: Any,
    min_count: Any,
    max_count: Any,
    *,
    max_select: int,
) -> tuple[Any, Any]:
    """Extend a trusted normalized action batch to the execution contract.

    Real resident greedy adapters already return legal, de-duplicated option
    indices.  The full normalizer remains available for untrusted adapters, but
    the hot path can skip the per-proposed-step legality scan and only fill
    additional options when the official execution contract requires more
    selections than the legacy model-facing horizon.
    """

    import torch

    if proposed.ndim != 2 or option_mask.ndim != 2:
        raise ValueError("proposed and option_mask must be rank two")
    if proposed.shape[0] != option_mask.shape[0]:
        raise ValueError("proposed and option_mask batch sizes differ")
    if max_select <= 0:
        raise ValueError("max_select must be positive")
    batch_size, option_count = option_mask.shape
    device = proposed.device
    output = torch.full(
        (batch_size, max_select + 1),
        -1,
        dtype=torch.long,
        device=device,
    )
    selected = torch.zeros((batch_size, option_count), dtype=torch.bool, device=device)
    min_count = min_count.long().view(-1).clamp(min=0, max=max_select)
    max_count = max_count.long().view(-1).clamp(min=0, max=max_select)
    accepted = torch.minimum(
        proposed_lengths.long().view(-1).clamp(min=0, max=max_select),
        max_count,
    )

    copy_width = min(int(proposed.shape[1]), max_select)
    if copy_width:
        steps = torch.arange(copy_width, dtype=torch.long, device=device).view(1, -1)
        copy_mask = steps.lt(accepted.view(-1, 1))
        values = proposed[:, :copy_width].long()
        output[:, :copy_width] = torch.where(
            copy_mask,
            values,
            torch.full_like(values, -1),
        )
        if option_count:
            safe_values = values.clamp(min=0, max=option_count - 1)
            in_range = values.ge(0) & values.lt(option_count) & copy_mask
            selected.scatter_(
                1,
                safe_values,
                selected.gather(1, safe_values) | in_range,
            )

    available = option_mask.bool() & ~selected
    fill_needed = (min_count - accepted).clamp_min(0)
    fill_room = torch.minimum(max_count - accepted, max_select - accepted).clamp_min(0)
    fill_needed = torch.minimum(fill_needed, fill_room)
    if option_count:
        option_ids = torch.arange(option_count, dtype=torch.long, device=device).view(1, -1)
        fill_rank = available.long().cumsum(dim=1) - 1
        fill_take = available & fill_rank.lt(fill_needed.view(-1, 1))
        fill_dest = accepted.view(-1, 1) + fill_rank
        sink = torch.full_like(fill_dest, max_select)
        scatter_dest = torch.where(fill_take, fill_dest, sink)
        scatter_values = torch.where(
            fill_take,
            option_ids.expand(batch_size, -1),
            torch.full_like(fill_dest, -1),
        )
        output.scatter_(1, scatter_dest, scatter_values)
        accepted += fill_take.long().sum(dim=1)

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
        *,
        engine: Any | None = None,
        policy_profile: dict[str, Any] | None = None,
    ) -> DeviceActionBatch:
        import torch
        import time

        def add_profile(bucket: str, policy_name: str, value_ms: float) -> None:
            if policy_profile is None:
                return
            rows = policy_profile.setdefault(bucket, {})
            rows[policy_name] = float(rows.get(policy_name, 0.0)) + float(value_ms)

        def add_profile_call(policy_name: str) -> None:
            if policy_profile is None:
                return
            rows = policy_profile.setdefault("calls", {})
            rows[policy_name] = int(rows.get(policy_name, 0)) + 1

        def start_profile() -> float:
            if policy_profile is None:
                return 0.0
            torch.cuda.synchronize(acting_policy_ids.device)
            return time.perf_counter()

        def stop_profile(bucket: str, policy_name: str, started: float) -> None:
            if policy_profile is None:
                return
            torch.cuda.synchronize(acting_policy_ids.device)
            add_profile(bucket, policy_name, (time.perf_counter() - started) * 1000.0)

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
            shared_groups = _shared_policy_groups(self.manifest.policies, self.adapters)
            shared_policy_ids = {
                policy.policy_id
                for group in shared_groups
                for policy in group
            }
            for group in shared_groups:
                codec = group[0].codec
                adapters = [self.adapters[policy.policy_id] for policy in group]
                info = _static_shared_adapter_info(adapters[0])
                if info is None:
                    raise ValueError("shared policy group lost its shared adapter info")
                _key, adapter, _fields = info
                semantic0031_v2_method = getattr(adapter, "act_device_semantic0031_v2", None)
                if semantic0031_v2_method is None and codec not in encoded_by_codec:
                    raise KeyError(f"missing device codec batch: {codec}")
                if semantic0031_v2_method is not None:
                    if engine is None:
                        raise ValueError("semantic0031 v2 shared policy group requires engine")
                    lane_parts = []
                    for policy in group:
                        route = routes[policy.policy_id]
                        valid = route_mask[policy.policy_id]
                        lane_parts.append(route[valid])
                    lane_indices = torch.cat(lane_parts, dim=0).long()
                    if lane_indices.numel() == 0:
                        continue
                    cohort = engine.encode_semantic0031_v2_lanes(lane_indices.contiguous())
                    cohort["_route_mask"] = torch.ones_like(lane_indices, dtype=torch.bool)
                    group_name = f"shared:{codec}:{len(group)}"
                    started = start_profile()
                    proposed, proposed_lengths = semantic0031_v2_method(cohort)
                    stop_profile("adapter_wall_ms", group_name, started)
                    action_min_count = cohort.get("action_min_count", cohort["min_count"])
                    action_max_count = cohort.get("action_max_count", cohort["max_count"])
                    started = start_profile()
                    if getattr(adapter, "outputs_normalized", False):
                        normalized, normalized_lengths = extend_normalized_actions_device(
                            proposed,
                            proposed_lengths,
                            cohort["option_mask"],
                            action_min_count,
                            action_max_count,
                            max_select=self.max_select,
                        )
                    else:
                        normalized, normalized_lengths = normalize_actions_device(
                            proposed,
                            proposed_lengths,
                            cohort["option_mask"],
                            action_min_count,
                            action_max_count,
                            max_select=self.max_select,
                        )
                    stop_profile("postprocess_wall_ms", group_name, started)
                    started = start_profile()
                    output.scatter_(
                        0,
                        lane_indices.view(-1, 1).expand(-1, self.max_select),
                        normalized,
                    )
                    lengths.scatter_(
                        0,
                        lane_indices,
                        normalized_lengths,
                    )
                    stop_profile("scatter_wall_ms", group_name, started)
                    add_profile_call(group_name)
                    continue

                super_route = torch.arange(
                    batch_size,
                    dtype=torch.long,
                    device=acting_policy_ids.device,
                )
                member = torch.zeros(
                    batch_size,
                    dtype=torch.bool,
                    device=acting_policy_ids.device,
                )
                routed = torch.zeros(
                    batch_size + 1,
                    dtype=torch.bool,
                    device=acting_policy_ids.device,
                )
                for policy in group:
                    member |= acting_policy_ids.long().eq(int(policy.policy_id))
                    route = routes[policy.policy_id]
                    valid = route_mask[policy.policy_id]
                    destinations = torch.where(valid, route, sink)
                    routed.scatter_(0, destinations, valid)
                super_valid = ready_mask.bool() & member & routed[:batch_size]
                cohort = _gather_device_batch(
                    encoded_by_codec[codec], super_route, batch_size
                )
                cohort["_route_mask"] = super_valid
                cohort.update(
                    _lane_static_fields_for_shared_group(
                        group, adapters, acting_policy_ids, acting_policy_ids.device
                    )
                )
                group_name = f"shared:{codec}:{len(group)}"
                method = getattr(adapter, "act_device_shared_static", None)
                started = start_profile()
                if method is not None:
                    proposed, proposed_lengths = method(cohort)
                else:
                    proposed, proposed_lengths = adapter.act_device(cohort)
                stop_profile("adapter_wall_ms", group_name, started)
                action_min_count = cohort.get("action_min_count", cohort["min_count"])
                action_max_count = cohort.get("action_max_count", cohort["max_count"])
                started = start_profile()
                if getattr(adapter, "outputs_normalized", False):
                    normalized, normalized_lengths = extend_normalized_actions_device(
                        proposed,
                        proposed_lengths,
                        cohort["option_mask"],
                        action_min_count,
                        action_max_count,
                        max_select=self.max_select,
                    )
                else:
                    normalized, normalized_lengths = normalize_actions_device(
                        proposed,
                        proposed_lengths,
                        cohort["option_mask"],
                        action_min_count,
                        action_max_count,
                        max_select=self.max_select,
                    )
                stop_profile("postprocess_wall_ms", group_name, started)
                started = start_profile()
                destinations = torch.where(
                    super_valid,
                    super_route,
                    torch.full_like(super_route, batch_size),
                )
                values = torch.where(
                    super_valid.view(-1, 1),
                    normalized,
                    torch.full_like(normalized, -1),
                )
                output.scatter_(
                    0,
                    destinations.view(-1, 1).expand(-1, self.max_select),
                    values,
                )
                lengths.scatter_(
                    0,
                    destinations,
                    torch.where(
                        super_valid,
                        normalized_lengths,
                        torch.zeros_like(normalized_lengths),
                    ),
                )
                stop_profile("scatter_wall_ms", group_name, started)
                add_profile_call(group_name)

            for policy in self.manifest.policies:
                if policy.policy_id in shared_policy_ids:
                    continue
                adapter = self.adapters[policy.policy_id]
                semantic0031_v2_method = getattr(adapter, "act_device_semantic0031_v2", None)
                if semantic0031_v2_method is None and policy.codec not in encoded_by_codec:
                    raise KeyError(f"missing device codec batch: {policy.codec}")
                route = routes[policy.policy_id]
                valid = route_mask[policy.policy_id]
                if semantic0031_v2_method is not None:
                    if engine is None:
                        raise ValueError("semantic0031 v2 policy requires engine")
                    cohort_route = route[valid].long()
                    if cohort_route.numel() == 0:
                        continue
                    cohort = engine.encode_semantic0031_v2_lanes(
                        cohort_route.contiguous()
                    )
                    cohort_valid = torch.ones_like(cohort_route, dtype=torch.bool)
                else:
                    cohort = _gather_device_batch(
                        encoded_by_codec[policy.codec], route, batch_size
                    )
                    cohort_route = route
                    cohort_valid = valid
                cohort["_route_mask"] = cohort_valid
                started = start_profile()
                if semantic0031_v2_method is not None:
                    proposed, proposed_lengths = semantic0031_v2_method(cohort)
                else:
                    proposed, proposed_lengths = adapter.act_device(cohort)
                stop_profile("adapter_wall_ms", policy.name, started)
                action_min_count = cohort.get("action_min_count", cohort["min_count"])
                action_max_count = cohort.get("action_max_count", cohort["max_count"])
                started = start_profile()
                if getattr(adapter, "outputs_normalized", False):
                    normalized, normalized_lengths = extend_normalized_actions_device(
                        proposed,
                        proposed_lengths,
                        cohort["option_mask"],
                        action_min_count,
                        action_max_count,
                        max_select=self.max_select,
                    )
                else:
                    normalized, normalized_lengths = normalize_actions_device(
                        proposed,
                        proposed_lengths,
                        cohort["option_mask"],
                        action_min_count,
                        action_max_count,
                        max_select=self.max_select,
                    )
                stop_profile("postprocess_wall_ms", policy.name, started)
                started = start_profile()
                destinations = torch.where(
                    cohort_valid,
                    cohort_route,
                    torch.full_like(cohort_route, batch_size),
                )
                values = torch.where(
                    cohort_valid.view(-1, 1),
                    normalized,
                    torch.full_like(normalized, -1),
                )
                output.scatter_(
                    0,
                    destinations.view(-1, 1).expand(-1, self.max_select),
                    values,
                )
                lengths.scatter_(
                    0,
                    destinations,
                    torch.where(
                        cohort_valid,
                        normalized_lengths,
                        torch.zeros_like(normalized_lengths),
                    ),
                )
                stop_profile("scatter_wall_ms", policy.name, started)
                add_profile_call(policy.name)

        return DeviceActionBatch(
            indices=output[:batch_size],
            lengths=lengths[:batch_size],
            routes=routes,
            route_mask=route_mask,
            route_counts=counts,
        )
