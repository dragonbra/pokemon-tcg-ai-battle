"""Serialized GPU-resident inference server for 0045 compound packages."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

from evaluation.runner.inference_server import _handle_connection


@dataclass
class _SessionState:
    encoder: Any = None
    pending: Any = None
    session_serial: int = 0
    battle_id: str = "kaggle-session-0"
    public_memory: Any = None
    active_route: Any = None


class CompoundPackageServer:
    """Keep one strict-loaded FP32 policy on GPU with isolated battle state."""

    def __init__(self, package_root: Path, device: str = "cuda:0") -> None:
        import torch

        self._torch = torch
        self._device = torch.device(device)
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("0045 compound package inference requires CUDA")
        package_root = package_root.resolve()
        sys.path.insert(0, str(package_root))
        spec = importlib.util.spec_from_file_location(
            "evaluation_0045_compound_package", package_root / "main.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load package entrypoint")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._policy = getattr(module, "POLICY", None)
        if self._policy is None or not hasattr(self._policy, "actor"):
            raise RuntimeError("package does not expose a compound POLICY")
        self._move_policy_to_gpu()
        self._install_gpu_batch_transfer()
        self._sessions: dict[str, _SessionState] = {}
        self._lock = threading.Lock()
        self._requests = 0
        self._model_seconds = 0.0

    def _move_policy_to_gpu(self) -> None:
        modules: list[Any] = []
        for name in (
            "actor", "value_head", "allocation_head", "value_adapter",
            "policy_strategy_adapter", "policy_option_lora", "meta_actor_residual",
        ):
            module = getattr(self._policy, name, None)
            if module is not None:
                modules.append(module)
        for head in getattr(self._policy, "heads", {}).values():
            modules.extend((head.action_decoder, head.policy_option_lora, head.allocation_head))
        seen: set[int] = set()
        for module in modules:
            if id(module) in seen:
                continue
            seen.add(id(module))
            module.to(device=self._device, dtype=self._torch.float32).eval()
            module.requires_grad_(False)
        if hasattr(self._policy, "public_memory"):
            self._policy.public_memory = type(self._policy.public_memory)(1, self._device)
        route = getattr(
            self._policy, "active_update", getattr(self._policy, "active_head", None)
        )
        if route is not None:
            self._policy._activate(route)
        floating = [
            parameter for module in modules for parameter in module.parameters()
            if parameter.is_floating_point()
        ]
        if not floating or any(
            parameter.device != self._device or parameter.dtype != self._torch.float32
            for parameter in floating
        ):
            raise RuntimeError("compound package did not materialize as CUDA FP32")

    def _install_gpu_batch_transfer(self) -> None:
        # Resolve the package-local class through the policy method globals. This
        # avoids importing a second copy of the strategy package.
        batch_type = self._policy.select.__globals__["_compound"].DecisionBatch \
            if "_compound" in self._policy.select.__globals__ \
            else self._policy.select.__globals__["DecisionBatch"]
        original = batch_type.from_mapping.__func__
        device = self._device

        @classmethod
        def from_mapping(cls, tensors):
            return original(cls, {
                name: value.to(device, non_blocking=False)
                for name, value in tensors.items()
            })

        batch_type.from_mapping = from_mapping

    def _new_state(self) -> _SessionState:
        state = _SessionState()
        if hasattr(self._policy, "public_memory"):
            state.public_memory = type(self._policy.public_memory)(1, self._device)
            state.active_route = getattr(
                self._policy, "default_update", getattr(self._policy, "active_head", None)
            )
        return state

    def _restore(self, state: _SessionState) -> None:
        self._policy.encoder = state.encoder
        self._policy.pending = state.pending
        self._policy.session_serial = state.session_serial
        self._policy.battle_id = state.battle_id
        if state.public_memory is not None:
            self._policy.public_memory = state.public_memory
            self._policy._activate(state.active_route)

    def _save(self, state: _SessionState) -> None:
        state.encoder = self._policy.encoder
        state.pending = self._policy.pending
        state.session_serial = self._policy.session_serial
        state.battle_id = self._policy.battle_id
        if state.public_memory is not None:
            state.public_memory = self._policy.public_memory
            state.active_route = getattr(
                self._policy, "active_update", getattr(self._policy, "active_head", None)
            )

    def call(self, session_id: str, observation: dict[str, Any], deck=None, record=None, **_) -> Any:
        del deck, record
        started = time.perf_counter()
        with self._lock:
            state = self._sessions.setdefault(session_id, self._new_state())
            self._restore(state)
            action = self._policy.select(observation)
            self._save(state)
            self._requests += 1
            self._model_seconds += time.perf_counter() - started
            return action

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def close(self) -> None:
        self._sessions.clear()

    def compiler_contract(self) -> dict[str, Any]:
        return {"supported": False, "mode": "serialized_compound_cuda_fp32"}

    def profile_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "requests": self._requests,
            "model_wall_seconds": self._model_seconds,
            "device": str(self._device),
            "runtime_dtype": "fp32",
        }

    def reset_profile(self) -> None:
        self._requests = 0
        self._model_seconds = 0.0

    def record_response_send(self, nanoseconds: int) -> None:
        del nanoseconds


def serve(package_root: Path, socket_path: Path, device: str) -> None:
    socket_path.unlink(missing_ok=True)
    server = CompoundPackageServer(package_root, device)
    listener = Listener(str(socket_path), family="AF_UNIX")
    try:
        while True:
            connection = listener.accept()
            threading.Thread(
                target=_handle_connection, args=(server, connection), daemon=True
            ).start()
    finally:
        listener.close()
        server.close()
        socket_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    serve(args.package_root, args.socket, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
