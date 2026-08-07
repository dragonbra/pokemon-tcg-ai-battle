"""Deck-routed persistent policy inference for isolated official-engine workers."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import queue
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any


@dataclass
class _AbilityRepeatGuard:
    limit: int
    turn_actor: tuple[int, int] | None = None
    counts: dict[tuple[tuple[str, Any], ...], int] = field(default_factory=dict)


def _apply_ability_repeat_guard(
    observation: dict[str, Any], action: Any, guard: _AbilityRepeatGuard
) -> Any:
    if guard.limit <= 0 or not isinstance(action, list) or len(action) != 1:
        return action
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    if select.get("type") != 0:
        return action
    turn = current.get("turn")
    actor = current.get("yourIndex")
    if type(turn) is not int or actor not in (0, 1):
        return action
    turn_actor = (turn, actor)
    if guard.turn_actor != turn_actor:
        guard.turn_actor = turn_actor
        guard.counts.clear()
    options = select.get("option") or []
    selected = action[0]
    if type(selected) is not int or not 0 <= selected < len(options):
        return action
    option = options[selected]
    if not isinstance(option, dict) or option.get("type") != 10:
        return action
    identity = tuple(
        sorted(
            (str(key), value)
            for key, value in option.items()
            if isinstance(value, (int, str, bool, type(None)))
        )
    )
    guard.counts[identity] = guard.counts.get(identity, 0) + 1
    if guard.counts[identity] <= guard.limit:
        return action
    return next(
        ([index] for index, item in enumerate(options) if item.get("type") == 14),
        action,
    )


def _forced_action(observation: dict[str, Any]) -> list[int] | None:
    """Return the only legal sequence when the engine requires one singleton option."""
    select = observation.get("select")
    if not isinstance(select, dict):
        return None
    options = select.get("option")
    if (
        isinstance(options, list)
        and len(options) == 1
        and select.get("minCount") == 1
        and select.get("maxCount") == 1
    ):
        return [0]
    return None


def _install_static_loader_cache(encoder_type: type) -> None:
    """Cache immutable prototype indexes without modifying packaged policy assets."""
    module = sys.modules.get(encoder_type.__module__)
    prototype_type = getattr(module, "PrototypeIndex", None) if module is not None else None
    if prototype_type is None or getattr(prototype_type, "_evaluation_load_cached", False):
        return
    descriptor = inspect.getattr_static(prototype_type, "load", None)
    if not isinstance(descriptor, classmethod):
        return
    original = prototype_type.load
    cache: dict[tuple[Any, ...], Any] = {}
    cache_lock = threading.Lock()

    @classmethod
    def cached(_cls, *args: Any, **kwargs: Any) -> Any:
        key = (*args, *sorted(kwargs.items()))
        try:
            hash(key)
        except TypeError:
            return original(*args, **kwargs)
        with cache_lock:
            if key not in cache:
                cache[key] = original(*args, **kwargs)
            return cache[key]

    prototype_type.load = cached
    prototype_type._evaluation_load_cached = True


def _install_raw_record_batching(encoder_type: type, policy: Any):
    """Defer canonical tensor collation so a resident batch is materialized once."""
    if getattr(policy, "requires_source_id", True):
        return None
    module = sys.modules.get(encoder_type.__module__)
    collator = getattr(module, "collate_canonical_records", None) if module else None
    if collator is None or getattr(module, "_evaluation_raw_record_batching", False):
        return None

    def keep_single_record(records: Any, **kwargs: Any) -> Any:
        if kwargs or not isinstance(records, (list, tuple)) or len(records) != 1:
            return collator(records, **kwargs)
        return records[0]

    module.collate_canonical_records = keep_single_record
    module._evaluation_raw_record_batching = True
    return collator


def _collate_raw_records_equivalent(
    collator: Any, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Collate once while preserving the old per-observation target sentinel."""
    batch = collator(records)
    targets = batch.get("targets")
    if targets is None:
        return batch
    for index, record in enumerate(records):
        actor = record["actor"]
        action = record["target"]["ordered_action"]
        targets[index, len(action)] = len(actor["option_cat"])
    return batch


class PolicyServer:
    def __init__(
        self,
        candidate_root: Path,
        device: str,
        batch_size: int,
        batch_wait_ms: float,
        ability_repeat_limit: int,
        inference_dtype: str = "fp32",
        profile: bool = False,
        forced_action_shortcut: bool = False,
    ) -> None:
        self._candidate_root = candidate_root.resolve()
        if str(self._candidate_root) not in sys.path:
            sys.path.insert(0, str(self._candidate_root))
        entrypoint = self._candidate_root / "main.py"
        spec = importlib.util.spec_from_file_location("evaluation_policy_server", entrypoint)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load candidate entrypoint: {entrypoint}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._agent = module.agent
        self._policy = getattr(module, "POLICY", getattr(module, "_POLICY", None))
        if self._policy is None:
            raise RuntimeError("candidate must expose POLICY for shared inference")
        import torch
        from strategy.inference import legal_fallback
        from strategy.online_runtime import OnlineCausalEncoder

        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        dtype = _resolve_inference_dtype(torch, inference_dtype, self._device)
        self._policy.model = self._policy.model.to(
            device=self._device,
            dtype=dtype,
        ).eval()
        self._policy.runtime_dtype = dtype
        self._torch = torch
        self._encoder_type = OnlineCausalEncoder
        self._legal_fallback = legal_fallback
        self._batch_size = batch_size
        self._batch_wait_seconds = batch_wait_ms / 1000.0
        self._requests: queue.Queue[_InferenceRequest] = queue.Queue()
        self._default_deck = _normalize_request_deck(None, tuple(self._policy.deck))
        self._encoders: dict[str, tuple[tuple[int, ...], Any]] = {}
        self._ability_guards: dict[str, _AbilityRepeatGuard] = {}
        self._ability_repeat_limit = ability_repeat_limit
        self._profile = _InferenceProfile(enabled=profile)
        _install_static_loader_cache(self._encoder_type)
        self._raw_record_collator = _install_raw_record_batching(
            self._encoder_type, self._policy
        )
        self._forced_action_shortcut = forced_action_shortcut
        self._thread = threading.Thread(target=self._dispatch, daemon=True)
        self._thread.start()

    def call(
        self,
        session_id: str,
        observation: dict[str, Any],
        deck: Any = None,
    ) -> Any:
        request = _InferenceRequest(
            session_id,
            observation,
            _normalize_request_deck(deck, self._default_deck),
        )
        self._requests.put(request)
        request.ready.wait()
        if request.error is not None:
            raise request.error
        return request.action

    def close_session(self, session_id: str) -> None:
        self._encoders.pop(session_id, None)
        self._ability_guards.pop(session_id, None)

    def profile_snapshot(self) -> dict[str, Any]:
        return self._profile.snapshot()

    def reset_profile(self) -> None:
        self._profile.reset()

    def _dispatch(self) -> None:
        while True:
            first = self._requests.get()
            requests = [first]
            deadline = time.monotonic() + self._batch_wait_seconds
            while len(requests) < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    requests.append(self._requests.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                self._infer(requests)
            except BaseException as exc:
                for request in requests:
                    request.error = exc
                    request.ready.set()

    def _infer(self, requests: list[_InferenceRequest]) -> None:
        inference_started_ns = time.perf_counter_ns()
        model_requests = [
            request for request in requests if request.observation.get("select") is not None
        ]
        for request in requests:
            if request not in model_requests:
                request.action = list(request.deck)
                self._encoders.pop(request.session_id, None)
                self._ability_guards.pop(request.session_id, None)
                request.ready.set()
        if not model_requests:
            return

        encoded: list[dict[str, Any]] = []
        active_requests: list[_InferenceRequest] = []
        for request in model_requests:
            observation = request.observation
            current = observation.get("current") or {}
            actor = current.get("yourIndex")
            select = observation.get("select") or {}
            options = select.get("option") or []
            minimum = select.get("minCount", 0)
            if actor not in (0, 1) or len(options) > self._policy.config.max_options or (
                isinstance(minimum, int)
                and not isinstance(minimum, bool)
                and minimum > self._policy.config.max_action_steps
            ):
                request.action = self._legal_fallback(observation)
                request.ready.set()
                continue
            cached = self._encoders.get(request.session_id)
            encoder = cached[1] if cached is not None and cached[0] == request.deck else None
            if encoder is None or encoder.actor != actor:
                encoder_started_ns = time.perf_counter_ns()
                encoder = self._encoder_type(actor, request.deck, self._policy.config)
                self._profile.add_time(
                    "encoder_init_seconds", time.perf_counter_ns() - encoder_started_ns
                )
                self._profile.increment("encoder_initializations")
                self._encoders[request.session_id] = (request.deck, encoder)
            forced = _forced_action(observation) if self._forced_action_shortcut else None
            if forced is not None:
                consume_started_ns = time.perf_counter_ns()
                encoder.knowledge.consume(observation)
                self._profile.add_time(
                    "forced_consume_seconds", time.perf_counter_ns() - consume_started_ns
                )
                self._profile.increment("forced_action_shortcuts")
                request.action = forced
                request.ready.set()
                self._profile.finish_request(request)
                continue
            try:
                encode_started_ns = time.perf_counter_ns()
                row = encoder.encode(observation)
            except (IndexError, RuntimeError, ValueError):
                self._encoders.pop(request.session_id, None)
                if getattr(self._policy, "fail_closed_inference_errors", False):
                    raise
                request.action = self._legal_fallback(observation)
                request.ready.set()
                self._profile.finish_request(request)
                continue
            self._profile.add_time(
                "feature_encode_seconds", time.perf_counter_ns() - encode_started_ns
            )
            if self._raw_record_collator is None:
                _inject_source_id_if_required(self._torch, row, self._policy)
            encoded.append(row)
            active_requests.append(request)
        if not encoded:
            return

        collate_started_ns = time.perf_counter_ns()
        if self._raw_record_collator is None:
            batch = _stack_batches(self._torch, encoded, self._device)
        else:
            batch = {
                name: value.to(
                    self._device,
                    non_blocking=self._device.type == "cuda",
                )
                for name, value in _collate_raw_records_equivalent(
                    self._raw_record_collator, encoded
                ).items()
            }
        if self._profile.enabled and self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)
        self._profile.add_time(
            "collate_h2d_seconds", time.perf_counter_ns() - collate_started_ns
        )
        # Shared inference bypasses the candidate's single-row ``select`` path.
        # Align floating inputs with the loaded model while preserving categorical
        # indices and masks as integer tensors.
        model_dtype = next(
            (parameter.dtype for parameter in self._policy.model.parameters()
             if parameter.dtype.is_floating_point),
            None,
        )
        if model_dtype is not None:
            batch = {
                name: value.to(dtype=model_dtype)
                if value.dtype.is_floating_point else value
                for name, value in batch.items()
            }
        model_started_ns = time.perf_counter_ns()
        cuda_start = cuda_end = None
        if self._profile.enabled and self._device.type == "cuda":
            cuda_start = self._torch.cuda.Event(enable_timing=True)
            cuda_end = self._torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        with self._torch.inference_mode():
            result = self._policy.model.deterministic_action_tensors(batch)
        if cuda_end is not None:
            cuda_end.record()
            cuda_end.synchronize()
            self._profile.add_seconds("gpu_model_seconds", cuda_start.elapsed_time(cuda_end) / 1000.0)
        self._profile.add_time("model_wall_seconds", time.perf_counter_ns() - model_started_ns)
        decode_started_ns = time.perf_counter_ns()
        sequences = result.sequences.cpu().tolist()
        lengths = result.lengths.cpu().tolist()
        legal = result.legal.cpu().tolist()
        for index, request in enumerate(active_requests):
            if not legal[index] and getattr(
                self._policy, "fail_closed_inference_errors", False
            ):
                raise RuntimeError("canonical policy produced an illegal action sequence")
            action = [
                int(value) for value in sequences[index][: lengths[index]]
            ] if legal[index] else self._legal_fallback(request.observation)
            guard = self._ability_guards.setdefault(
                request.session_id,
                _AbilityRepeatGuard(limit=self._ability_repeat_limit),
            )
            request.action = _apply_ability_repeat_guard(
                request.observation, action, guard
            )
            request.ready.set()
            self._profile.finish_request(request)
        self._profile.add_time("d2h_decode_seconds", time.perf_counter_ns() - decode_started_ns)
        self._profile.record_batch(len(active_requests))
        self._profile.add_time(
            "inference_dispatch_seconds", time.perf_counter_ns() - inference_started_ns
        )


def _stack_batches(torch: Any, rows: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    """Pad one-observation online batches and move one combined batch to the device."""
    if not rows:
        raise ValueError("cannot stack an empty inference batch")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise ValueError("online inference rows have different tensor fields")
    batch: dict[str, Any] = {}
    for key in sorted(keys):
        tensors = [row[key] for row in rows]
        dimensions = tensors[0].ndim
        if any(tensor.size(0) != 1 or tensor.ndim != dimensions for tensor in tensors):
            raise ValueError(f"{key} is not a compatible one-observation tensor")
        target_shape = [len(tensors)] + [
            max(tensor.size(axis) for tensor in tensors)
            for axis in range(1, dimensions)
        ]
        combined = torch.full(
            target_shape,
            -100 if key == "targets" else 0,
            dtype=tensors[0].dtype,
        )
        for index, tensor in enumerate(tensors):
            slices = (index, *(slice(0, tensor.size(axis)) for axis in range(1, dimensions)))
            combined[slices] = tensor[0]
        batch[key] = combined.to(device, non_blocking=device.type == "cuda")
    return batch


def _inject_source_id_if_required(torch: Any, row: dict[str, Any], policy: Any) -> None:
    """Preserve legacy persona routing without exposing it to canonical actors."""
    if getattr(policy, "requires_source_id", True):
        row["source_id"] = torch.zeros(1, dtype=torch.long)


def _normalize_request_deck(value: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    deck = fallback if value is None else value
    if (
        not isinstance(deck, (list, tuple))
        or len(deck) != 60
        or not all(type(card_id) is int and card_id > 0 for card_id in deck)
    ):
        raise ValueError("inference request deck must contain exactly 60 positive integer card IDs")
    return tuple(deck)


def _resolve_inference_dtype(torch: Any, value: str, device: Any) -> Any:
    dtypes = {"fp32": torch.float32, "fp16": torch.float16}
    if value not in dtypes:
        raise ValueError(f"unsupported inference dtype: {value}")
    if value == "fp16" and device.type != "cuda":
        raise ValueError("fp16 inference requires a CUDA device")
    return dtypes[value]


@dataclass
class _InferenceRequest:
    session_id: str
    observation: dict[str, Any]
    deck: tuple[int, ...]
    action: Any = None
    error: BaseException | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    enqueued_ns: int = field(default_factory=time.perf_counter_ns)


@dataclass
class _InferenceProfile:
    enabled: bool = False
    counters: Counter[str] = field(default_factory=Counter)
    seconds: Counter[str] = field(default_factory=Counter)
    batch_sizes: Counter[int] = field(default_factory=Counter)
    request_latencies_ms: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, name: str, value: int = 1) -> None:
        if self.enabled:
            with self.lock:
                self.counters[name] += value

    def add_time(self, name: str, nanoseconds: int) -> None:
        self.add_seconds(name, nanoseconds / 1e9)

    def add_seconds(self, name: str, value: float) -> None:
        if self.enabled:
            with self.lock:
                self.seconds[name] += float(value)

    def record_batch(self, size: int) -> None:
        if self.enabled:
            with self.lock:
                self.batch_sizes[int(size)] += 1

    def finish_request(self, request: _InferenceRequest) -> None:
        if self.enabled:
            latency = (time.perf_counter_ns() - request.enqueued_ns) / 1e6
            with self.lock:
                self.counters["completed_requests"] += 1
                self.request_latencies_ms.append(latency)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            latencies = sorted(self.request_latencies_ms)
            requests = int(self.counters.get("completed_requests", 0))
            batches = sum(self.batch_sizes.values())
            weighted = sum(size * count for size, count in self.batch_sizes.items())
            return {
                "enabled": self.enabled,
                "counters": dict(sorted(self.counters.items())),
                "seconds": dict(sorted(self.seconds.items())),
                "batch_histogram": {
                    str(size): count for size, count in sorted(self.batch_sizes.items())
                },
                "batches": batches,
                "mean_batch_size": weighted / batches if batches else 0.0,
                "request_latency_ms": {
                    "count": requests,
                    "total": sum(latencies),
                    "p50": _percentile(latencies, 0.50),
                    "p95": _percentile(latencies, 0.95),
                    "p99": _percentile(latencies, 0.99),
                },
            }

    def reset(self) -> None:
        with self.lock:
            self.counters.clear()
            self.seconds.clear()
            self.batch_sizes.clear()
            self.request_latencies_ms.clear()


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int((len(values) - 1) * quantile)))
    return float(values[index])


def serve(
    candidate_root: Path,
    socket_path: Path,
    device: str,
    batch_size: int,
    batch_wait_ms: float,
    ability_repeat_limit: int,
    inference_dtype: str = "fp32",
    profile: bool = False,
    forced_action_shortcut: bool = False,
) -> None:
    socket_path.unlink(missing_ok=True)
    server = PolicyServer(
        candidate_root,
        device,
        batch_size,
        batch_wait_ms,
        ability_repeat_limit,
        inference_dtype,
        profile,
        forced_action_shortcut,
    )
    listener = Listener(str(socket_path), family="AF_UNIX")
    try:
        while True:
            connection = listener.accept()
            thread = threading.Thread(
                target=_handle_connection,
                args=(server, connection),
                daemon=True,
            )
            thread.start()
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)


def _handle_connection(server: PolicyServer, connection: Any) -> None:
    session_id = uuid.uuid4().hex
    try:
        while True:
            request = connection.recv()
            if isinstance(request, dict) and request.get("command") == "profile":
                connection.send({"ok": True, "profile": server.profile_snapshot()})
                continue
            if isinstance(request, dict) and request.get("command") == "reset_profile":
                server.reset_profile()
                connection.send({"ok": True})
                continue
            if not isinstance(request, dict) or not isinstance(request.get("observation"), dict):
                raise RuntimeError("invalid inference request")
            action = server.call(session_id, request["observation"], request.get("deck"))
            connection.send({"ok": True, "action": action})
    except (EOFError, OSError):
        pass
    except BaseException as exc:
        try:
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass
    finally:
        server.close_session(session_id)
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="local persistent policy inference server")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--ability-repeat-limit", type=int, default=0)
    parser.add_argument("--inference-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--forced-action-shortcut", action="store_true")
    args = parser.parse_args(argv)
    serve(
        args.candidate,
        args.socket,
        args.device,
        args.batch_size,
        args.batch_wait_ms,
        args.ability_repeat_limit,
        args.inference_dtype,
        args.profile,
        args.forced_action_shortcut,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
