"""Deck-routed persistent policy inference for isolated official-engine workers."""

from __future__ import annotations

import argparse
import hashlib
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

from evaluation.runner.compiler_pool import CompilerPool


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


def _canonical_record_batching(encoder_type: type, policy: Any) -> tuple[Any, bool]:
    """Prefer an explicit record API and retain the legacy shim as fallback."""
    if getattr(policy, "requires_source_id", True):
        return None, False
    module = sys.modules.get(encoder_type.__module__)
    collator = getattr(module, "collate_canonical_records", None) if module else None
    if callable(getattr(encoder_type, "encode_record", None)) and callable(collator):
        return collator, True
    return _install_raw_record_batching(encoder_type, policy), False


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
        compiler_workers: int = 1,
        async_h2d: bool = False,
        resident_tensor_cache: bool = False,
    ) -> None:
        if compiler_workers < 1:
            raise ValueError("compiler_workers must be at least one")
        if compiler_workers > 1 and forced_action_shortcut:
            raise ValueError("parallel compiler workers do not support forced-action shortcut")
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
        if async_h2d and self._device.type != "cuda":
            raise ValueError("async_h2d requires a CUDA inference device")
        if async_h2d and resident_tensor_cache:
            raise ValueError("resident_tensor_cache cannot use the async H2D path")
        dtype = _resolve_inference_dtype(torch, inference_dtype, self._device)
        self._policy.model = self._policy.model.to(
            device=self._device,
            dtype=dtype,
        ).eval()
        self._policy.runtime_dtype = dtype
        self._model_dtype = next(
            (
                parameter.dtype
                for parameter in self._policy.model.parameters()
                if parameter.dtype.is_floating_point
            ),
            None,
        )
        self._torch = torch
        self._encoder_type = OnlineCausalEncoder
        self._legal_fallback = legal_fallback
        self._batch_size = batch_size
        self._batch_wait_seconds = batch_wait_ms / 1000.0
        self._requests: queue.Queue[_InferenceRequest] = queue.Queue()
        self._async_h2d = bool(async_h2d and self._device.type == "cuda")
        self._resident_tensor_cache_enabled = bool(resident_tensor_cache)
        self._async_coalesce_seconds = max(self._batch_wait_seconds, 0.025)
        self._async_min_batch_size = min(self._batch_size, max(2, self._batch_size // 2))
        self._default_deck = _normalize_request_deck(None, tuple(self._policy.deck))
        self._encoders: dict[str, tuple[tuple[int, ...], Any]] = {}
        self._ability_guards: dict[str, _AbilityRepeatGuard] = {}
        self._ability_repeat_limit = ability_repeat_limit
        self._profile = _InferenceProfile(enabled=profile)
        _install_static_loader_cache(self._encoder_type)
        self._raw_record_collator, self._encode_records_directly = _canonical_record_batching(
            self._encoder_type, self._policy
        )
        self._resident_tensor_store = None
        self._resident_tensor_key_type = None
        self._session_tensor_keys: dict[str, Any] = {}
        if self._resident_tensor_cache_enabled:
            if self._raw_record_collator is None or getattr(
                self._policy, "requires_source_id", True
            ):
                raise ValueError(
                    "resident tensor cache requires canonical raw-record inference"
                )
            from strategy.deployment.event_embedding_cache import EventEmbeddingStore
            from strategy.deployment.session_tensor_cache import SessionTensorKey

            self._resident_tensor_key_type = SessionTensorKey
            self._resident_tensor_store = EventEmbeddingStore(self._policy.model)
        self._forced_action_shortcut = forced_action_shortcut
        self._compiler_pool: CompilerPool | None = None
        if compiler_workers > 1:
            if self._raw_record_collator is None or getattr(
                self._policy, "requires_source_id", True
            ):
                raise ValueError(
                    "parallel compiler workers require a canonical persona-free policy"
                )
            config_payload = (
                self._policy.config.to_dict()
                if callable(getattr(self._policy.config, "to_dict", None))
                else dict(vars(self._policy.config))
            )
            self._compiler_pool = CompilerPool(
                self._candidate_root,
                config_payload,
                compiler_workers,
            )
        self._closed = threading.Event()
        self._pipeline_stop = object()
        self._prepared_batches: queue.Queue[Any] | None = None
        self._transferred_batches: queue.Queue[Any] | None = None
        self._transfer_stream = None
        self._transfer_thread: threading.Thread | None = None
        self._compute_thread: threading.Thread | None = None
        self._pinned_pool: _PinnedBatchPool | None = None
        if self._async_h2d:
            self._prepared_batches = queue.Queue(maxsize=1)
            self._transferred_batches = queue.Queue(maxsize=1)
            self._transfer_stream = self._torch.cuda.Stream(device=self._device)
            self._pinned_pool = _PinnedBatchPool(
                self._torch,
                slots=3,
                minimum_pinned_bytes=64 * 1024,
            )
            self._transfer_thread = threading.Thread(
                target=self._transfer_loop,
                name="evaluation-inference-h2d",
                daemon=True,
            )
            self._compute_thread = threading.Thread(
                target=self._compute_loop,
                name="evaluation-inference-compute",
                daemon=True,
            )
            self._transfer_thread.start()
            self._compute_thread.start()
        self._thread = threading.Thread(target=self._dispatch, daemon=True)
        self._thread.start()

    def call(
        self,
        session_id: str,
        observation: dict[str, Any],
        deck: Any = None,
        record: dict[str, Any] | None = None,
        *,
        client_send_ns: int | None = None,
        ingress_received_ns: int | None = None,
    ) -> Any:
        if (
            self._profile.enabled
            and type(client_send_ns) is int
            and type(ingress_received_ns) is int
            and ingress_received_ns >= client_send_ns
        ):
            self._profile.record_latency_ms(
                "ipc_ingress",
                (ingress_received_ns - client_send_ns) / 1e6,
            )
        request = _InferenceRequest(
            session_id,
            observation,
            _normalize_request_deck(deck, self._default_deck),
            record=record,
        )
        self._requests.put(request)
        request.ready.wait()
        returned_ns = time.perf_counter_ns()
        if request.completed_ns is not None:
            self._profile.record_latency_ms(
                "handler_wakeup",
                (returned_ns - request.completed_ns) / 1e6,
            )
        if request.error is not None:
            raise request.error
        return request.action

    def record_response_send(self, nanoseconds: int) -> None:
        self._profile.add_concurrent_time("ipc_egress_send_seconds", nanoseconds)

    def close_session(self, session_id: str) -> None:
        self._encoders.pop(session_id, None)
        self._ability_guards.pop(session_id, None)
        tensor_key = self._session_tensor_keys.pop(session_id, None)
        if tensor_key is not None and self._resident_tensor_store is not None:
            self._resident_tensor_store.close_session(tensor_key)
        if self._compiler_pool is not None:
            self._compiler_pool.close_session(session_id)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._prepared_batches is not None:
            self._prepared_batches.put(self._pipeline_stop)
        if self._transfer_thread is not None:
            self._transfer_thread.join(timeout=10)
        if self._compute_thread is not None:
            self._compute_thread.join(timeout=10)
        if self._compiler_pool is not None:
            self._compiler_pool.close()

    def compiler_contract(self) -> dict[str, Any]:
        supported = self._raw_record_collator is not None and not getattr(
            self._policy, "requires_source_id", True
        )
        config = (
            self._policy.config.to_dict()
            if callable(getattr(self._policy.config, "to_dict", None))
            else dict(vars(self._policy.config))
        )
        return {
            "supported": supported,
            "config": config,
            "record_contract": "canonical_raw_record_v1",
            "async_h2d": self._async_h2d,
            "resident_tensor_cache": self._resident_tensor_cache_enabled,
        }

    def profile_snapshot(self) -> dict[str, Any]:
        payload = self._profile.snapshot()
        payload["compiler"] = (
            self._compiler_pool.profile_snapshot()
            if self._compiler_pool is not None
            else {"workers": 1, "mode": "serial_in_process"}
        )
        if self._resident_tensor_store is not None:
            payload["resident_tensor_cache"] = self._resident_tensor_store.stats()
        return payload

    def reset_profile(self) -> None:
        self._profile.reset()
        if self._compiler_pool is not None:
            self._compiler_pool.reset_profile()
        if self._resident_tensor_store is not None:
            self._resident_tensor_store.reset_stats()

    @staticmethod
    def _deck_sha256(deck: tuple[int, ...]) -> str:
        encoded = ",".join(str(identity) for identity in sorted(deck)).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def _collate_resident(
        self,
        active_requests: list[_InferenceRequest],
        encoded: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._resident_tensor_store is None:
            raise RuntimeError("resident tensor cache is not initialized")
        if len(active_requests) != len(encoded):
            raise RuntimeError("resident tensor request/record count mismatch")
        keys: list[Any] = []
        source_events: list[tuple[int, ...]] = []
        update_started_ns = time.perf_counter_ns()
        for request, record in zip(active_requests, encoded, strict=True):
            current = request.observation.get("current") or {}
            actor = current.get("yourIndex")
            key = self._resident_tensor_key_type(
                request.session_id,
                actor,
                self._deck_sha256(request.deck),
            )
            previous_key = self._session_tensor_keys.get(request.session_id)
            if previous_key is not None and previous_key != key:
                raise RuntimeError("resident tensor session identity changed")
            identities = record.get("_runtime_event_source_ids")
            if not isinstance(identities, tuple):
                raise RuntimeError("resident event cache has no source-event identities")
            self._session_tensor_keys[request.session_id] = key
            source_events.append(identities)
            keys.append(key)
        event_batch, event_static_components = self._resident_tensor_store.update_batch(
            keys, encoded, source_events
        )
        self._profile.add_time(
            "resident_delta_apply_seconds", time.perf_counter_ns() - update_started_ns
        )
        batch_started_ns = time.perf_counter_ns()
        dynamic_batch = self._raw_record_collator(encoded, omit_event=True)
        targets = dynamic_batch.get("targets")
        if targets is not None:
            for index, record in enumerate(encoded):
                actor = record["actor"]
                action = record["target"]["ordered_action"]
                targets[index, len(action)] = len(actor["option_cat"])
        if self._model_dtype is not None:
            dynamic_batch = {
                name: value.to(dtype=self._model_dtype)
                if value.dtype.is_floating_point else value
                for name, value in dynamic_batch.items()
            }
        dynamic_batch = {
            name: value.to(
                self._device,
                non_blocking=self._device.type == "cuda",
            )
            for name, value in dynamic_batch.items()
        }
        batch = {**dynamic_batch, **event_batch}
        batch["_runtime_event_static_components"] = event_static_components
        self._profile.add_time(
            "resident_batch_assembly_seconds", time.perf_counter_ns() - batch_started_ns
        )
        return batch

    def _dispatch(self) -> None:
        while True:
            idle_started_ns = time.perf_counter_ns()
            first = self._requests.get()
            first_dequeued_ns = time.perf_counter_ns()
            self._profile.add_time(
                "dispatch_idle_wait_seconds", first_dequeued_ns - idle_started_ns
            )
            requests = [first]
            deadline = time.monotonic() + (
                self._async_coalesce_seconds
                if self._async_h2d
                else self._batch_wait_seconds
            )
            while len(requests) < self._batch_size:
                if self._async_h2d and len(requests) >= self._async_min_batch_size:
                    while len(requests) < self._batch_size:
                        try:
                            requests.append(self._requests.get_nowait())
                        except queue.Empty:
                            break
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    requests.append(self._requests.get(timeout=remaining))
                except queue.Empty:
                    break
            batch_ready_ns = time.perf_counter_ns()
            self._profile.add_time(
                "batch_coalesce_seconds", batch_ready_ns - first_dequeued_ns
            )
            self._profile.increment(
                "batches_full" if len(requests) >= self._batch_size else "batches_deadline"
            )
            self._profile.record_latencies_ms(
                "queue_wait",
                [
                    (batch_ready_ns - request.enqueued_ns) / 1e6
                    for request in requests
                ],
            )
            self._profile.record_batch_ready(batch_ready_ns)
            bookkeeping_finished_ns = time.perf_counter_ns()
            self._profile.add_time(
                "dispatch_profile_bookkeeping_seconds",
                bookkeeping_finished_ns - batch_ready_ns,
            )
            try:
                if self._async_h2d:
                    self._prepare_async(requests)
                else:
                    self._infer(requests)
            except BaseException as exc:
                self._fail_requests(requests, exc)
            finally:
                self._profile.add_time(
                    "batch_cycle_seconds",
                    time.perf_counter_ns() - batch_ready_ns,
                )

    def _prepare_records(
        self, requests: list[_InferenceRequest]
    ) -> tuple[int, list[dict[str, Any]], list[_InferenceRequest]] | None:
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
        parallel_requests: list[dict[str, Any]] = []
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
            if request.record is not None:
                if self._resident_tensor_store is not None:
                    raise RuntimeError(
                        "resident event cache requires in-process causal encoder metadata"
                    )
                if self._raw_record_collator is None or getattr(
                    self._policy, "requires_source_id", True
                ):
                    raise RuntimeError(
                        "compiled inference request requires canonical persona-free policy"
                    )
                if not isinstance(request.record.get("actor"), dict) or not isinstance(
                    request.record.get("target"), dict
                ):
                    raise RuntimeError("compiled inference request has invalid canonical record")
                encoded.append(request.record)
                active_requests.append(request)
                continue
            if self._compiler_pool is not None:
                if self._resident_tensor_store is not None:
                    raise RuntimeError(
                        "resident event cache cannot use worker-local compiler state"
                    )
                parallel_requests.append(
                    {
                        "session_id": request.session_id,
                        "actor": actor,
                        "deck": request.deck,
                        "observation": observation,
                    }
                )
                active_requests.append(request)
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
                row = (
                    encoder.encode_record(observation)
                    if self._encode_records_directly
                    else encoder.encode(observation)
                )
                if self._resident_tensor_store is not None:
                    source_ids = getattr(encoder, "last_event_source_ids", None)
                    if not isinstance(source_ids, tuple):
                        raise RuntimeError(
                            "causal encoder does not expose source-event identities"
                        )
                    row["_runtime_event_source_ids"] = source_ids
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
        if parallel_requests:
            encode_started_ns = time.perf_counter_ns()
            encoded.extend(self._compiler_pool.encode(parallel_requests))
            self._profile.add_time(
                "feature_encode_seconds", time.perf_counter_ns() - encode_started_ns
            )
        if not encoded:
            return None
        self._profile.add_time(
            "prepare_records_seconds", time.perf_counter_ns() - inference_started_ns
        )
        return inference_started_ns, encoded, active_requests

    def _collate_cpu(self, encoded: list[dict[str, Any]]) -> dict[str, Any]:
        if self._raw_record_collator is None:
            batch = _stack_batches(self._torch, encoded, self._torch.device("cpu"))
        else:
            batch = _collate_raw_records_equivalent(
                self._raw_record_collator, encoded
            )
        if self._model_dtype is not None:
            batch = {
                name: value.to(dtype=self._model_dtype)
                if value.dtype.is_floating_point
                else value
                for name, value in batch.items()
            }
        return batch

    def _prepare_async(self, requests: list[_InferenceRequest]) -> None:
        prepared = self._prepare_records(requests)
        if prepared is None:
            return
        inference_started_ns, encoded, active_requests = prepared
        collate_started_ns = time.perf_counter_ns()
        assert self._pinned_pool is not None
        slot, host_batch = self._pinned_pool.copy(self._collate_cpu(encoded))
        self._profile.add_time(
            "collate_pin_seconds", time.perf_counter_ns() - collate_started_ns
        )
        item = _PreparedInference(
            requests=active_requests,
            host_batch=host_batch,
            inference_started_ns=inference_started_ns,
            pinned_slot=slot,
        )
        assert self._prepared_batches is not None
        queue_started_ns = time.perf_counter_ns()
        self._prepared_batches.put(item)
        self._profile.add_time(
            "prepared_queue_wait_seconds", time.perf_counter_ns() - queue_started_ns
        )

    def _transfer_loop(self) -> None:
        assert self._prepared_batches is not None
        assert self._transferred_batches is not None
        assert self._transfer_stream is not None
        while True:
            item = self._prepared_batches.get()
            if item is self._pipeline_stop:
                self._transferred_batches.put(self._pipeline_stop)
                return
            try:
                enqueue_started_ns = time.perf_counter_ns()
                h2d_start = self._torch.cuda.Event(enable_timing=True)
                h2d_end = self._torch.cuda.Event(enable_timing=True)
                with self._torch.cuda.stream(self._transfer_stream):
                    h2d_start.record(self._transfer_stream)
                    device_batch = {
                        name: value.to(self._device, non_blocking=True)
                        for name, value in item.host_batch.items()
                    }
                    h2d_end.record(self._transfer_stream)
                self._profile.add_time(
                    "h2d_enqueue_seconds", time.perf_counter_ns() - enqueue_started_ns
                )
                self._transferred_batches.put(
                    _TransferredInference(
                        prepared=item,
                        device_batch=device_batch,
                        h2d_start=h2d_start,
                        h2d_end=h2d_end,
                    )
                )
            except BaseException as exc:
                self._profile.increment("async_stage_errors")
                self._fail_requests(item.requests, exc)
                assert self._pinned_pool is not None
                self._pinned_pool.release(item.pinned_slot)

    def _compute_loop(self) -> None:
        assert self._transferred_batches is not None
        while True:
            item = self._transferred_batches.get()
            if item is self._pipeline_stop:
                return
            try:
                compute_stream = self._torch.cuda.current_stream(self._device)
                compute_stream.wait_event(item.h2d_end)
                self._run_model(
                    item.prepared.requests,
                    item.device_batch,
                    item.prepared.inference_started_ns,
                    h2d_events=(item.h2d_start, item.h2d_end),
                )
            except BaseException as exc:
                self._profile.increment("async_stage_errors")
                self._fail_requests(item.prepared.requests, exc)
            finally:
                assert self._pinned_pool is not None
                self._pinned_pool.release(item.prepared.pinned_slot)

    def _infer(self, requests: list[_InferenceRequest]) -> None:
        prepared = self._prepare_records(requests)
        if prepared is None:
            return
        inference_started_ns, encoded, active_requests = prepared

        combined_started_ns = time.perf_counter_ns()
        collate_started_ns = combined_started_ns
        if self._resident_tensor_store is not None:
            batch = self._collate_resident(active_requests, encoded)
        elif self._raw_record_collator is None:
            batch = _stack_batches(
                self._torch, encoded, self._torch.device("cpu")
            )
        else:
            batch = _collate_raw_records_equivalent(
                self._raw_record_collator, encoded
            )
        collate_finished_ns = time.perf_counter_ns()
        self._profile.add_time(
            "collate_cpu_seconds", collate_finished_ns - collate_started_ns
        )
        if self._resident_tensor_store is not None:
            self._profile.add_time(
                "collate_h2d_seconds", collate_finished_ns - combined_started_ns
            )
            self._run_model(active_requests, batch, inference_started_ns)
            return
        h2d_started_ns = time.perf_counter_ns()
        h2d_start = h2d_end = None
        if self._profile.enabled and self._device.type == "cuda":
            h2d_start = self._torch.cuda.Event(enable_timing=True)
            h2d_end = self._torch.cuda.Event(enable_timing=True)
            h2d_start.record()
        batch = {
            name: value.to(
                self._device,
                non_blocking=self._device.type == "cuda",
            )
            for name, value in batch.items()
        }
        if h2d_end is not None:
            h2d_end.record()
            h2d_end.synchronize()
            self._profile.add_seconds(
                "h2d_gpu_seconds", h2d_start.elapsed_time(h2d_end) / 1000.0
            )
        h2d_finished_ns = time.perf_counter_ns()
        self._profile.add_time(
            "h2d_wall_seconds", h2d_finished_ns - h2d_started_ns
        )
        self._profile.add_time(
            "collate_h2d_seconds", h2d_finished_ns - combined_started_ns
        )
        # Shared inference bypasses the candidate's single-row ``select`` path.
        # Align floating inputs with the loaded model while preserving categorical
        # indices and masks as integer tensors.
        if self._model_dtype is not None:
            dtype_started_ns = time.perf_counter_ns()
            batch = {
                name: value.to(dtype=self._model_dtype)
                if value.dtype.is_floating_point else value
                for name, value in batch.items()
            }
            self._profile.add_time(
                "dtype_cast_wall_seconds", time.perf_counter_ns() - dtype_started_ns
            )
        self._run_model(active_requests, batch, inference_started_ns)

    def _run_model(
        self,
        active_requests: list[_InferenceRequest],
        batch: dict[str, Any],
        inference_started_ns: int,
        *,
        h2d_events: tuple[Any, Any] | None = None,
    ) -> None:
        model_started_ns = time.perf_counter_ns()
        cuda_start = cuda_end = None
        if self._profile.enabled and self._device.type == "cuda":
            cuda_start = self._torch.cuda.Event(enable_timing=True)
            cuda_end = self._torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        with self._torch.inference_mode():
            event_static_components = batch.pop(
                "_runtime_event_static_components", None
            )
            if event_static_components is None:
                result = self._policy.model.deterministic_action_tensors(batch)
            else:
                result = self._policy.model.deterministic_action_tensors(
                    batch, event_static_components
                )
        if cuda_end is not None:
            cuda_end.record()
            cuda_end.synchronize()
            self._profile.add_seconds("gpu_model_seconds", cuda_start.elapsed_time(cuda_end) / 1000.0)
        if h2d_events is not None:
            self._profile.add_seconds(
                "h2d_gpu_seconds",
                h2d_events[0].elapsed_time(h2d_events[1]) / 1000.0,
            )
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
            request.completed_ns = time.perf_counter_ns()
            request.ready.set()
            self._profile.finish_request(request)
        self._profile.add_time("d2h_decode_seconds", time.perf_counter_ns() - decode_started_ns)
        self._profile.record_batch(len(active_requests))
        self._profile.add_time(
            "inference_dispatch_seconds", time.perf_counter_ns() - inference_started_ns
        )

    @staticmethod
    def _fail_requests(
        requests: list[_InferenceRequest], exc: BaseException
    ) -> None:
        for request in requests:
            if request.ready.is_set():
                continue
            request.error = exc
            request.ready.set()


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


class _PinnedBatchPool:
    """Bounded grow-only pinned buffers whose views live through async H2D."""

    def __init__(
        self,
        torch: Any,
        *,
        slots: int,
        minimum_pinned_bytes: int = 0,
    ) -> None:
        if slots < 1:
            raise ValueError("pinned batch pool requires at least one slot")
        self._torch = torch
        self._minimum_pinned_bytes = minimum_pinned_bytes
        self._available: queue.Queue[int] = queue.Queue(maxsize=slots)
        self._buffers: list[dict[str, Any]] = [{} for _ in range(slots)]
        for index in range(slots):
            self._available.put(index)

    def copy(self, batch: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        slot = self._available.get()
        try:
            buffers = self._buffers[slot]
            output: dict[str, Any] = {}
            for name, value in batch.items():
                if value.device.type != "cpu":
                    raise ValueError("only CPU tensors can enter the pinned H2D pipeline")
                if value.numel() * value.element_size() < self._minimum_pinned_bytes:
                    output[name] = value
                    continue
                buffer = buffers.get(name)
                capacity = tuple(value.shape)
                if buffer is not None:
                    if buffer.dtype != value.dtype or buffer.ndim != value.ndim:
                        buffer = None
                    else:
                        capacity = tuple(
                            max(buffer.size(axis), value.size(axis))
                            for axis in range(value.ndim)
                        )
                        if tuple(buffer.shape) != capacity:
                            buffer = None
                if buffer is None:
                    buffer = self._torch.empty(
                        capacity,
                        dtype=value.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    buffers[name] = buffer
                view = buffer[
                    tuple(slice(0, value.size(axis)) for axis in range(value.ndim))
                ]
                view.copy_(value)
                output[name] = view
            return slot, output
        except BaseException:
            self._available.put(slot)
            raise

    def release(self, slot: int) -> None:
        self._available.put(slot)


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
    record: dict[str, Any] | None = None
    action: Any = None
    error: BaseException | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    enqueued_ns: int = field(default_factory=time.perf_counter_ns)
    completed_ns: int | None = None


@dataclass
class _PreparedInference:
    requests: list[_InferenceRequest]
    host_batch: dict[str, Any]
    inference_started_ns: int
    pinned_slot: int


@dataclass
class _TransferredInference:
    prepared: _PreparedInference
    device_batch: dict[str, Any]
    h2d_start: Any
    h2d_end: Any


@dataclass
class _InferenceProfile:
    enabled: bool = False
    counters: Counter[str] = field(default_factory=Counter)
    seconds: Counter[str] = field(default_factory=Counter)
    batch_sizes: Counter[int] = field(default_factory=Counter)
    request_latencies_ms: list[float] = field(default_factory=list)
    latency_samples_ms: dict[str, list[float]] = field(default_factory=dict)
    last_batch_ready_ns: int | None = None
    counters_lock: threading.Lock = field(default_factory=threading.Lock)
    seconds_lock: threading.Lock = field(default_factory=threading.Lock)
    concurrent_seconds_lock: threading.Lock = field(default_factory=threading.Lock)
    latency_lock: threading.Lock = field(default_factory=threading.Lock)
    request_lock: threading.Lock = field(default_factory=threading.Lock)
    concurrent_seconds: Counter[str] = field(default_factory=Counter)
    completed_requests: int = 0

    def increment(self, name: str, value: int = 1) -> None:
        if self.enabled:
            with self.counters_lock:
                self.counters[name] += value

    def add_time(self, name: str, nanoseconds: int) -> None:
        self.add_seconds(name, nanoseconds / 1e9)

    def add_seconds(self, name: str, value: float) -> None:
        if self.enabled:
            with self.seconds_lock:
                self.seconds[name] += float(value)

    def add_concurrent_time(self, name: str, nanoseconds: int) -> None:
        if self.enabled:
            with self.concurrent_seconds_lock:
                self.concurrent_seconds[name] += nanoseconds / 1e9

    def record_batch(self, size: int) -> None:
        if self.enabled:
            with self.counters_lock:
                self.batch_sizes[int(size)] += 1

    def record_latency_ms(self, name: str, value: float) -> None:
        self.record_latencies_ms(name, [value])

    def record_latencies_ms(self, name: str, values: list[float]) -> None:
        if self.enabled and values:
            with self.latency_lock:
                self.latency_samples_ms.setdefault(name, []).extend(
                    float(value) for value in values
                )

    def record_batch_ready(self, ready_ns: int) -> None:
        if self.enabled:
            with self.latency_lock:
                if self.last_batch_ready_ns is not None:
                    self.latency_samples_ms.setdefault(
                        "batch_ready_interval", []
                    ).append((ready_ns - self.last_batch_ready_ns) / 1e6)
                self.last_batch_ready_ns = ready_ns

    def finish_request(self, request: _InferenceRequest) -> None:
        if self.enabled:
            latency = (time.perf_counter_ns() - request.enqueued_ns) / 1e6
            with self.request_lock:
                self.completed_requests += 1
                self.request_latencies_ms.append(latency)

    def snapshot(self) -> dict[str, Any]:
        with self.counters_lock:
            counters = dict(self.counters)
            batch_sizes = dict(self.batch_sizes)
        with self.seconds_lock:
            seconds = Counter(self.seconds)
        with self.concurrent_seconds_lock:
            seconds.update(self.concurrent_seconds)
        with self.latency_lock:
            latency_samples = {
                name: list(values)
                for name, values in self.latency_samples_ms.items()
            }
        with self.request_lock:
            latencies = sorted(self.request_latencies_ms)
            requests = self.completed_requests
        counters["completed_requests"] = requests
        batches = sum(batch_sizes.values())
        weighted = sum(size * count for size, count in batch_sizes.items())
        latency_payload = {
            name: _latency_summary(values)
            for name, values in sorted(latency_samples.items())
        }
        return {
            "enabled": self.enabled,
            "counters": dict(sorted(counters.items())),
            "seconds": dict(sorted(seconds.items())),
            "batch_histogram": {
                str(size): count for size, count in sorted(batch_sizes.items())
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
            "latency_ms": latency_payload,
        }

    def reset(self) -> None:
        with self.counters_lock:
            self.counters.clear()
            self.batch_sizes.clear()
        with self.seconds_lock:
            self.seconds.clear()
        with self.concurrent_seconds_lock:
            self.concurrent_seconds.clear()
        with self.latency_lock:
            self.latency_samples_ms.clear()
            self.last_batch_ready_ns = None
        with self.request_lock:
            self.request_latencies_ms.clear()
            self.completed_requests = 0


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "mean": sum(ordered) / len(ordered) if ordered else 0.0,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
    }


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
    compiler_workers: int = 1,
    async_h2d: bool = False,
    resident_tensor_cache: bool = False,
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
        compiler_workers,
        async_h2d,
        resident_tensor_cache,
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
        server.close()
        socket_path.unlink(missing_ok=True)


def _handle_connection(server: PolicyServer, connection: Any) -> None:
    implicit_session_id = uuid.uuid4().hex
    try:
        while True:
            request = connection.recv()
            ingress_received_ns = time.perf_counter_ns()
            if isinstance(request, dict) and request.get("command") == "profile":
                connection.send({"ok": True, "profile": server.profile_snapshot()})
                continue
            if isinstance(request, dict) and request.get("command") == "reset_profile":
                server.reset_profile()
                connection.send({"ok": True})
                continue
            if isinstance(request, dict) and request.get("command") == "compiler_contract":
                connection.send({"ok": True, "contract": server.compiler_contract()})
                continue
            if isinstance(request, dict) and request.get("command") == "close_session":
                session_id = _explicit_session_id(request)
                server.close_session(session_id)
                connection.send({"ok": True})
                continue
            if not isinstance(request, dict) or not isinstance(request.get("observation"), dict):
                raise RuntimeError("invalid inference request")
            session_id = (
                _explicit_session_id(request)
                if "session_id" in request
                else implicit_session_id
            )
            record = request.get("record")
            if record is not None and not isinstance(record, dict):
                raise RuntimeError("compiled inference record must be an object")
            action = server.call(
                session_id,
                request["observation"],
                request.get("deck"),
                record,
                client_send_ns=request.get("_profile_client_send_ns"),
                ingress_received_ns=ingress_received_ns,
            )
            response_started_ns = time.perf_counter_ns()
            connection.send({"ok": True, "action": action})
            server.record_response_send(time.perf_counter_ns() - response_started_ns)
    except (EOFError, OSError):
        pass
    except BaseException as exc:
        try:
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass
    finally:
        server.close_session(implicit_session_id)
        connection.close()


def _explicit_session_id(request: dict[str, Any]) -> str:
    session_id = request.get("session_id")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
        raise RuntimeError("explicit inference session_id must contain 1..256 characters")
    return session_id


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
    parser.add_argument("--compiler-workers", type=int, default=1)
    parser.add_argument("--async-h2d", action="store_true")
    parser.add_argument("--resident-tensor-cache", action="store_true")
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
        args.compiler_workers,
        args.async_h2d,
        args.resident_tensor_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
