"""Small local policy inference server for isolated evaluation workers.

The server is intentionally candidate-specific. It keeps the BC policy model in one
long-lived process while each client connection represents one isolated game session.
The official engine remains in the client worker process.
"""

from __future__ import annotations

import argparse
import copy
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
        self._policy = getattr(module, "_POLICY", None)
        if self._policy is None:
            raise RuntimeError("candidate must expose a _POLICY for session history isolation")
        import torch
        from strategy.features import encode_observation

        self._policy.model = self._policy.model.to(device).eval()
        self._policy.device = torch.device(device)
        self._torch = torch
        self._encode_observation = encode_observation
        self._batch_size = batch_size
        self._batch_wait_seconds = batch_wait_ms / 1000.0
        self._requests: queue.Queue[_InferenceRequest] = queue.Queue()
        self._histories: dict[str, list[dict[str, int]]] = {}
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
        self._histories.pop(session_id, None)

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
                self._policy.history = []
                request.action = self._agent(request.observation)
                self._histories[request.session_id] = []
                request.ready.set()
        if not model_requests:
            return

        encoded = []
        for request in model_requests:
            observation = copy.deepcopy(request.observation)
            observation["rl_history"] = list(self._histories.get(request.session_id, []))
            observation["rl_deck"] = list(self._policy.deck)
            observation["rl_expert_id"] = self._policy.expert_id
            observation["rl_card_metadata"] = self._policy.card_metadata
            encoded.append(self._encode_observation(observation, self._policy.feature_config))
        dtypes = {
            "state_numeric": self._torch.float32,
            "state_card_ids": self._torch.long,
            "action_type_ids": self._torch.long,
            "action_card_ids": self._torch.long,
            "action_target_ids": self._torch.long,
            "action_numeric": self._torch.float32,
            "action_mask": self._torch.bool,
            "deck_card_ids": self._torch.long,
            "deck_card_numeric": self._torch.float32,
            "entity_card_ids": self._torch.long,
            "entity_numeric": self._torch.float32,
            "history_card_ids": self._torch.long,
            "history_numeric": self._torch.float32,
            "expert_ids": self._torch.long,
        }
        batch = {
            key: self._torch.tensor([item[key] for item in encoded], dtype=dtype).to(
                self._policy.device
            )
            for key, dtype in dtypes.items()
            if key in encoded[0]
        }
        with self._torch.no_grad():
            _, logits, count_logits = self._policy.model.forward_with_count(**batch)
        for index, request in enumerate(model_requests):
            select = request.observation["select"]
            minimum = max(0, _int(select.get("minCount"), 0))
            maximum = min(
                self._policy.model.max_selection_count,
                _int(select.get("maxCount"), minimum),
            )
            count_mask = self._torch.zeros_like(count_logits[index], dtype=self._torch.bool)
            count_mask[minimum : maximum + 1] = True
            masked_counts = count_logits[index].masked_fill(
                ~count_mask,
                self._torch.finfo(count_logits.dtype).min,
            )
            count = int(masked_counts.argmax())
            count = min(count, sum(encoded[index]["action_mask"]))
            action = (
                sorted(
                    int(value)
                    for value in self._torch.topk(logits[index], k=count).indices.tolist()
                )
                if count
                else []
            )
            is_main_selection = (
                _int(select.get("type")) == 0 and _int(select.get("context")) == 0
            )
            if is_main_selection and len(action) == 1:
                options = select.get("option") or []
                if 0 <= action[0] < len(options) and isinstance(options[action[0]], dict):
                    option = options[action[0]]
                    history = self._histories.setdefault(request.session_id, [])
                    history.append(
                        {
                            "type": _int(option.get("type"), 0),
                            "cardId": _int(option.get("cardId"), 0),
                            "attackId": _int(option.get("attackId"), 0),
                        }
                    )
                    del history[:-32]
            request.action = action
            request.ready.set()


@dataclass
class _InferenceRequest:
    session_id: str
    observation: dict[str, Any]
    action: Any = None
    error: BaseException | None = None
    ready: threading.Event = field(default_factory=threading.Event)


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
