"""U270 GPU policy service with an evaluation-only branch-Value command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from multiprocessing.connection import Listener
from pathlib import Path
import sys
import threading
from typing import Any
import uuid

import torch

from evaluation.runner.inference_server import (
    _AbilityRepeatGuard,
    _apply_ability_repeat_guard,
)
from ..policy.batching import collate_feature_batches


@dataclass
class _Session:
    encoder: Any = None
    pending: Any = None
    session_serial: int = 0
    battle_id: str = ""


class ValueSearchGPUService:
    def __init__(self, package_root: Path, device: str, repeat_limit: int) -> None:
        self.root = package_root.resolve()
        sys.path.insert(0, str(self.root))
        spec = importlib.util.spec_from_file_location(
            "u270_value_search_package", self.root / "main.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load U270 package entrypoint")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.policy = getattr(module, "POLICY", None)
        if self.policy is None or not hasattr(self.policy, "value_head"):
            raise RuntimeError("package does not expose the compound policy + critic")
        self.deck = tuple(int(card_id) for card_id in module.read_deck_csv())
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Value Search service requires an available CUDA device")
        for name in (
            "actor", "value_head", "allocation_head", "meta_head", "meta_conditioner"
        ):
            setattr(
                self.policy,
                name,
                getattr(self.policy, name).to(self.device, dtype=torch.float32).eval(),
            )
        original_encode = self.policy.actor.encode

        def encode_on_device(batch: Any):
            return original_encode(batch.to(self.device, non_blocking=False))

        self.policy.actor.encode = encode_on_device
        self.sessions: dict[str, _Session] = {}
        self.guards: dict[str, _AbilityRepeatGuard] = {}
        self.repeat_limit = int(repeat_limit)
        self.lock = threading.Lock()

    def _validate_deck(self, deck: Any) -> None:
        if deck is not None and tuple(int(card_id) for card_id in deck) != self.deck:
            raise RuntimeError("Value Search session deck differs from packaged exact deck")

    def select(
        self, session_id: str, observation: dict[str, Any], deck: Any
    ) -> list[int]:
        self._validate_deck(deck)
        if observation.get("select") is None:
            self.sessions.pop(session_id, None)
            self.guards.pop(session_id, None)
            return list(self.deck)
        state = self.sessions.setdefault(
            session_id, _Session(battle_id=f"value-search-session-{session_id}")
        )
        self.policy.encoder = state.encoder
        self.policy.pending = state.pending
        self.policy.session_serial = state.session_serial
        self.policy.battle_id = state.battle_id
        try:
            action = self.policy.select(observation)
        finally:
            state.encoder = self.policy.encoder
            state.pending = self.policy.pending
            state.session_serial = self.policy.session_serial
            state.battle_id = self.policy.battle_id
        guard = self.guards.setdefault(
            session_id, _AbilityRepeatGuard(limit=self.repeat_limit)
        )
        return _apply_ability_repeat_guard(observation, action, guard)

    def values(
        self, session_id: str, observations: list[dict[str, Any]], deck: Any
    ) -> list[float]:
        self._validate_deck(deck)
        state = self.sessions.get(session_id)
        if state is None or state.encoder is None:
            raise RuntimeError("branch Value requested before a root policy decision")
        if not observations:
            raise RuntimeError("branch Value request is empty")
        root_index = state.encoder.knowledge._decision_index
        batches = [state.encoder.fork().encode(observation) for observation in observations]
        if state.encoder.knowledge._decision_index != root_index:
            raise RuntimeError("branch encoding polluted the live causal root")
        merged = collate_feature_batches(batches)
        from strategy.contracts.batch import DecisionBatch

        with torch.inference_mode():
            validated, encoded_state, encoded_options = self.policy.actor.encode(
                DecisionBatch.from_mapping(merged)
            )
            values = self.policy.value_from_encoded(
                validated, encoded_state, encoded_options
            )
        return [float(value) for value in values.detach().cpu().tolist()]

    def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.guards.pop(session_id, None)


def _handle(service: ValueSearchGPUService, connection: Any) -> None:
    implicit_session_id = uuid.uuid4().hex
    try:
        while True:
            request = connection.recv()
            try:
                if not isinstance(request, dict):
                    raise RuntimeError("invalid Value Search inference request")
                session_id = str(request.get("session_id") or implicit_session_id)
                command = request.get("command")
                with service.lock:
                    if command == "close_session":
                        service.close_session(session_id)
                        response = {"ok": True}
                    elif command == "value_search_v0_values":
                        observations = request.get("observations")
                        if not isinstance(observations, list) or not all(
                            isinstance(item, dict) for item in observations
                        ):
                            raise RuntimeError("invalid branch observation batch")
                        response = {
                            "ok": True,
                            "values": service.values(
                                session_id, observations, request.get("deck")
                            ),
                        }
                    else:
                        observation = request.get("observation")
                        if not isinstance(observation, dict):
                            raise RuntimeError("invalid policy observation")
                        response = {
                            "ok": True,
                            "action": service.select(
                                session_id, observation, request.get("deck")
                            ),
                        }
            except BaseException as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            connection.send(response)
    except (EOFError, OSError):
        pass
    except BaseException as exc:
        try:
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass
    finally:
        with service.lock:
            service.close_session(implicit_session_id)
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ability-repeat-limit", type=int, default=20)
    args = parser.parse_args()
    args.socket.unlink(missing_ok=True)
    service = ValueSearchGPUService(
        args.candidate, args.device, args.ability_repeat_limit
    )
    listener = Listener(str(args.socket), family="AF_UNIX")
    try:
        while True:
            connection = listener.accept()
            threading.Thread(
                target=_handle, args=(service, connection), daemon=True
            ).start()
    finally:
        listener.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
