"""Session-affine CPU process pool for canonical online feature compilation."""

from __future__ import annotations

import hashlib
import inspect
import multiprocessing as mp
import sys
import threading
import time
from collections import Counter
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any


def shard_for_session(session_id: str, workers: int) -> int:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("compiler session_id must be a non-empty string")
    if workers < 1:
        raise ValueError("compiler workers must be at least one")
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % workers


class CompilerPool:
    """Own causal encoders in stable subprocess shards and return raw records."""

    def __init__(
        self,
        candidate_root: Path,
        config_payload: dict[str, Any],
        workers: int,
    ) -> None:
        if workers < 2:
            raise ValueError("parallel compiler pool requires at least two workers")
        self.workers = workers
        self._context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._closed = False
        self._profile = Counter()
        self._locks = [threading.Lock() for _ in range(workers)]
        for index in range(workers):
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_compiler_worker,
                args=(child, candidate_root.resolve(), dict(config_payload), index),
                name=f"evaluation-feature-compiler-{index}",
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        self._request_all({"command": "ready"})

    def encode(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("compiler pool is closed")
        if not requests:
            return []
        groups: dict[int, list[dict[str, Any]]] = {}
        for request_index, request in enumerate(requests):
            session_id = request.get("session_id")
            shard = shard_for_session(session_id, self.workers)
            groups.setdefault(shard, []).append(
                {**request, "request_index": request_index}
            )
        started_ns = time.perf_counter_ns()
        pending: dict[Connection, int] = {}
        locked_shards = sorted(groups)
        for shard in locked_shards:
            self._locks[shard].acquire()
        restored: list[dict[str, Any] | None] = [None] * len(requests)
        try:
            for shard, rows in groups.items():
                connection = self._connections[shard]
                connection.send({"command": "encode", "requests": rows})
                pending[connection] = shard
            while pending:
                ready = wait(list(pending))
                if not ready:
                    raise RuntimeError("compiler shards stopped responding")
                for connection in ready:
                    shard = pending.pop(connection)
                    reply = self._receive(connection, shard)
                    results = reply.get("results")
                    if not isinstance(results, list):
                        raise RuntimeError(f"compiler shard {shard} returned no result list")
                    for row in results:
                        if not isinstance(row, dict) or type(row.get("request_index")) is not int:
                            raise RuntimeError(f"compiler shard {shard} returned a malformed result")
                        index = row["request_index"]
                        if not 0 <= index < len(restored) or restored[index] is not None:
                            raise RuntimeError(f"compiler shard {shard} returned an invalid index")
                        record = row.get("record")
                        if not isinstance(record, dict):
                            raise RuntimeError(f"compiler shard {shard} returned no canonical record")
                        restored[index] = record
                    self._merge_profile(reply.get("profile"))
        finally:
            for shard in reversed(locked_shards):
                self._locks[shard].release()
        if any(record is None for record in restored):
            raise RuntimeError("compiler shard result count mismatch")
        self._profile["ipc_round_trip_seconds"] += (
            time.perf_counter_ns() - started_ns
        ) / 1e9
        self._profile["dispatcher_batches"] += 1
        self._profile["dispatcher_requests"] += len(requests)
        return [record for record in restored if record is not None]

    def close_session(self, session_id: str) -> None:
        if self._closed:
            return
        shard = shard_for_session(session_id, self.workers)
        connection = self._connections[shard]
        with self._locks[shard]:
            connection.send({"command": "close_session", "session_id": session_id})
            self._receive(connection, shard)

    def reset_profile(self) -> None:
        if self._closed:
            return
        self._profile.clear()
        self._request_all({"command": "reset_profile"})

    def profile_snapshot(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "dispatcher_batches": int(self._profile["dispatcher_batches"]),
            "dispatcher_requests": int(self._profile["dispatcher_requests"]),
            "ipc_round_trip_seconds": float(self._profile["ipc_round_trip_seconds"]),
            "child_encode_seconds": float(self._profile["child_encode_seconds"]),
            "encoder_initializations": int(self._profile["encoder_initializations"]),
            "shard_batches": int(self._profile["shard_batches"]),
            "shard_requests": int(self._profile["shard_requests"]),
            "max_shard_batch": int(self._profile["max_shard_batch"]),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send({"command": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for connection in self._connections:
            try:
                if connection.poll(5):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

    def _request_all(self, payload: dict[str, Any]) -> None:
        for lock in self._locks:
            lock.acquire()
        try:
            for connection in self._connections:
                connection.send(payload)
            pending = set(self._connections)
            while pending:
                for connection in wait(list(pending)):
                    shard = self._connections.index(connection)
                    self._receive(connection, shard)
                    pending.remove(connection)
        finally:
            for lock in reversed(self._locks):
                lock.release()

    @staticmethod
    def _receive(connection: Connection, shard: int) -> dict[str, Any]:
        try:
            reply = connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"compiler shard {shard} crashed: {exc}") from exc
        if not isinstance(reply, dict) or not reply.get("ok"):
            detail = reply.get("error") if isinstance(reply, dict) else reply
            raise RuntimeError(f"compiler shard {shard} failed: {detail}")
        return reply

    def _merge_profile(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        for name in (
            "child_encode_seconds",
            "encoder_initializations",
            "shard_batches",
            "shard_requests",
        ):
            value = payload.get(name, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._profile[name] += value
        maximum = payload.get("max_shard_batch", 0)
        if type(maximum) is int:
            self._profile["max_shard_batch"] = max(
                self._profile["max_shard_batch"], maximum
            )

    def __enter__(self) -> "CompilerPool":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _install_static_loader_cache(encoder_type: type) -> None:
    module = sys.modules.get(encoder_type.__module__)
    prototype_type = getattr(module, "PrototypeIndex", None) if module else None
    if prototype_type is None or getattr(prototype_type, "_compiler_pool_cached", False):
        return
    descriptor = inspect.getattr_static(prototype_type, "load", None)
    if not isinstance(descriptor, classmethod):
        return
    original = prototype_type.load
    cache: dict[tuple[Any, ...], Any] = {}

    @classmethod
    def cached(_cls, *args: Any, **kwargs: Any) -> Any:
        key = (*args, *sorted(kwargs.items()))
        if key not in cache:
            cache[key] = original(*args, **kwargs)
        return cache[key]

    prototype_type.load = cached
    prototype_type._compiler_pool_cached = True


def _raw_record_encoder(encoder_type: type) -> None:
    if callable(getattr(encoder_type, "encode_record", None)):
        return
    module = sys.modules.get(encoder_type.__module__)
    collator = getattr(module, "collate_canonical_records", None) if module else None
    if not callable(collator):
        raise RuntimeError("candidate encoder has no raw canonical record interface")

    def keep_single_record(records: Any, **kwargs: Any) -> Any:
        if kwargs or not isinstance(records, (list, tuple)) or len(records) != 1:
            return collator(records, **kwargs)
        return records[0]

    module.collate_canonical_records = keep_single_record


def _compiler_worker(
    connection: Connection,
    candidate_root: Path,
    config_payload: dict[str, Any],
    shard_index: int,
) -> None:
    try:
        root = str(candidate_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from strategy.model.config import ModelConfig
        from strategy.online_runtime import OnlineCausalEncoder

        config = ModelConfig(**config_payload)
        _install_static_loader_cache(OnlineCausalEncoder)
        _raw_record_encoder(OnlineCausalEncoder)
        encoders: dict[str, tuple[int, tuple[int, ...], Any]] = {}
        profile = Counter()
        while True:
            message = connection.recv()
            command = message.get("command") if isinstance(message, dict) else None
            if command == "ready":
                connection.send({"ok": True, "shard": shard_index})
                continue
            if command == "reset_profile":
                profile.clear()
                connection.send({"ok": True})
                continue
            if command == "close_session":
                encoders.pop(str(message.get("session_id")), None)
                connection.send({"ok": True})
                continue
            if command == "close":
                connection.send({"ok": True})
                return
            if command != "encode" or not isinstance(message.get("requests"), list):
                raise RuntimeError("invalid compiler worker command")
            rows = message["requests"]
            results = []
            batch_started_ns = time.perf_counter_ns()
            initializations = 0
            for request in rows:
                session_id = request["session_id"]
                actor = int(request["actor"])
                deck = tuple(int(card_id) for card_id in request["deck"])
                cached = encoders.get(session_id)
                encoder = (
                    cached[2]
                    if cached is not None and cached[:2] == (actor, deck)
                    else None
                )
                if encoder is None:
                    encoder = OnlineCausalEncoder(actor, deck, config)
                    encoders[session_id] = (actor, deck, encoder)
                    initializations += 1
                record = (
                    encoder.encode_record(request["observation"])
                    if callable(getattr(encoder, "encode_record", None))
                    else encoder.encode(request["observation"])
                )
                results.append(
                    {"request_index": request["request_index"], "record": record}
                )
            elapsed = (time.perf_counter_ns() - batch_started_ns) / 1e9
            profile["child_encode_seconds"] += elapsed
            profile["encoder_initializations"] += initializations
            profile["shard_batches"] += 1
            profile["shard_requests"] += len(rows)
            profile["max_shard_batch"] = max(profile["max_shard_batch"], len(rows))
            connection.send(
                {
                    "ok": True,
                    "results": results,
                    "profile": {
                        "child_encode_seconds": elapsed,
                        "encoder_initializations": initializations,
                        "shard_batches": 1,
                        "shard_requests": len(rows),
                        "max_shard_batch": len(rows),
                    },
                }
            )
    except BaseException as exc:
        try:
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


__all__ = ["CompilerPool", "shard_for_session"]
