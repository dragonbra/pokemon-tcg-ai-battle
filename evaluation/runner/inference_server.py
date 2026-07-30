"""Small local policy inference server for isolated evaluation workers.

The server is intentionally candidate-specific. It keeps the BC policy model in one
long-lived process while each client connection represents one isolated game session.
The official engine remains in the client worker process.
"""

from __future__ import annotations

import argparse
import importlib.util
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any


class PolicyServer:
    def __init__(
        self,
        candidate_root: Path,
        device: str,
        batch_size: int,
        batch_wait_ms: float,
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

        self._policy.model = self._policy.model.to(device).eval()
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        self._torch = torch
        self._encoder_type = OnlineCausalEncoder
        self._legal_fallback = legal_fallback
        self._batch_size = batch_size
        self._batch_wait_seconds = batch_wait_ms / 1000.0
        self._requests: queue.Queue[_InferenceRequest] = queue.Queue()
        self._encoders: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._dispatch, daemon=True)
        self._thread.start()

    def call(self, session_id: str, observation: dict[str, Any]) -> Any:
        request = _InferenceRequest(session_id, observation)
        self._requests.put(request)
        request.ready.wait()
        if request.error is not None:
            raise request.error
        return request.action

    def close_session(self, session_id: str) -> None:
        self._encoders.pop(session_id, None)

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
        model_requests = [
            request for request in requests if request.observation.get("select") is not None
        ]
        for request in requests:
            if request not in model_requests:
                request.action = self._agent(request.observation)
                self._encoders.pop(request.session_id, None)
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
            encoder = self._encoders.get(request.session_id)
            if encoder is None or encoder.actor != actor:
                encoder = self._encoder_type(actor, self._policy.deck, self._policy.config)
                self._encoders[request.session_id] = encoder
            try:
                row = encoder.encode(observation)
            except (IndexError, RuntimeError, ValueError):
                self._encoders.pop(request.session_id, None)
                request.action = self._legal_fallback(observation)
                request.ready.set()
                continue
            row["source_id"] = self._torch.zeros(1, dtype=self._torch.long)
            encoded.append(row)
            active_requests.append(request)
        if not encoded:
            return

        batch = _stack_batches(self._torch, encoded, self._device)
        with self._torch.inference_mode():
            result = self._policy.model.deterministic_action_tensors(batch)
        sequences = result.sequences.cpu().tolist()
        lengths = result.lengths.cpu().tolist()
        legal = result.legal.cpu().tolist()
        for index, request in enumerate(active_requests):
            request.action = (
                [int(value) for value in sequences[index][: lengths[index]]]
                if legal[index]
                else self._legal_fallback(request.observation)
            )
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


@dataclass
class _InferenceRequest:
    session_id: str
    observation: dict[str, Any]
    action: Any = None
    error: BaseException | None = None
    ready: threading.Event = field(default_factory=threading.Event)


def serve(
    candidate_root: Path,
    socket_path: Path,
    device: str,
    batch_size: int,
    batch_wait_ms: float,
) -> None:
    socket_path.unlink(missing_ok=True)
    server = PolicyServer(candidate_root, device, batch_size, batch_wait_ms)
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
            if not isinstance(request, dict) or not isinstance(request.get("observation"), dict):
                raise RuntimeError("invalid inference request")
            action = server.call(session_id, request["observation"])
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
    args = parser.parse_args(argv)
    serve(args.candidate, args.socket, args.device, args.batch_size, args.batch_wait_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
