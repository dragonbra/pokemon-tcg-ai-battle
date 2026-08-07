"""Multi-environment official-engine worker for resident policy inference."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import sys
import threading
import time
import queue
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from multiprocessing.connection import Client
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from evaluation.runner.models import GameRequest, GameResult
from evaluation.runner.worker import (
    _error_result,
    _exception_text,
    _load_game_api_with_runtime,
    _normalize_winner,
    _package_from_payload,
    _serialized_result,
    _state_summary,
)
from evaluation.runtime.loader import assert_cg_compatible


WORKER_COMPILER_BACKENDS = frozenset({"policy_stateless", "0035_incremental"})


def _install_package_boundary(name: str, path: Path) -> None:
    """Expose submodules without executing a tensor-heavy package __init__."""
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path.resolve())]
    sys.modules[name] = module


class _PointerBattle:
    """One battle pointer backed by an unchanged official cg runtime."""

    def __init__(self, library: Any, library_lock: threading.Lock) -> None:
        self._library = library
        self._library_lock = library_lock
        self._pointer: Any = None
        self._finished = False

    def _observation(self) -> dict[str, Any]:
        if self._pointer is None:
            raise RuntimeError("battle has not started")
        with self._library_lock:
            serial = self._library.GetBattleData(self._pointer)
            payload = bytes(serial.json).decode()
            search_input = ctypes.string_at(serial.data, serial.count).decode("ascii")
        observation = json.loads(payload)
        observation["search_begin_input"] = search_input
        return observation

    def start(
        self, deck0: list[int], deck1: list[int], *, engine_seed: int | None = None
    ) -> dict[str, Any] | None:
        if self._pointer is not None:
            raise RuntimeError("battle has already started")
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("The deck must contain 60 cards.")
        cards = deck0 + deck1
        argument = (ctypes.c_int * len(cards))(*cards)
        with self._library_lock:
            if engine_seed is None:
                start_data = self._library.BattleStart(argument)
            else:
                start_data = self._library.BattleStartSeeded(argument, engine_seed)
        self._pointer = start_data.battlePtr
        if self._pointer is None or self._pointer == 0:
            return None
        return self._observation()

    def select(self, action: list[int]) -> dict[str, Any]:
        if not isinstance(action, list) or not all(type(index) is int for index in action):
            raise ValueError("select_list is not list[int]")
        argument = (ctypes.c_int * len(action))(*action)
        with self._library_lock:
            error = self._library.Select(self._pointer, argument, len(action))
        if error == 30:
            raise ValueError("battle_ptr broken.")
        if error != 0:
            raise IndexError()
        return self._observation()

    def finish(self) -> None:
        if self._finished or self._pointer is None:
            return
        self._finished = True
        with self._library_lock:
            self._library.BattleFinish(self._pointer)


class _RemotePolicyAgent:
    def __init__(self, socket_path: str, deck: list[int], role: str) -> None:
        self._connection = Client(socket_path, family="AF_UNIX")
        self._deck = tuple(int(card_id) for card_id in deck)
        self._role = role

    def __call__(self, observation: dict[str, Any]) -> Any:
        self._connection.send({"observation": observation, "deck": self._deck})
        response = self._connection.recv()
        if not isinstance(response, dict) or not response.get("ok"):
            detail = response.get("error") if isinstance(response, dict) else response
            raise RuntimeError(f"shared {self._role} inference failed: {detail}")
        return response.get("action")

    def close(self) -> None:
        self._connection.close()


class _RemotePolicyChannelPool:
    """Bounded reusable inference connections with explicit causal sessions."""

    def __init__(
        self,
        socket_path: str,
        role: str,
        *,
        channel_count: int,
        profile: bool = False,
        connection_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if channel_count < 1:
            raise ValueError("channel_count must be at least one")
        factory = connection_factory or (
            lambda path: Client(path, family="AF_UNIX")
        )
        self._role = role
        self._profile = profile
        self._connections = [factory(socket_path) for _ in range(channel_count)]
        self._available: queue.Queue[Any] = queue.Queue(maxsize=channel_count)
        for connection in self._connections:
            self._available.put(connection)
        self._closed = False

    def call(
        self,
        session_id: str,
        observation: dict[str, Any],
        deck: list[int],
        timing: Counter[str] | None = None,
    ) -> Any:
        return self._exchange(
            {
                "session_id": session_id,
                "observation": observation,
                "deck": tuple(int(card_id) for card_id in deck),
            },
            expect_action=True,
            timing=timing,
        )

    def close_session(self, session_id: str) -> None:
        self._exchange(
            {"command": "close_session", "session_id": session_id},
            expect_action=False,
        )

    def compiler_contract(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"shared {self._role} inference channel pool is closed")
        connection = self._available.get()
        try:
            connection.send({"command": "compiler_contract"})
            response = connection.recv()
        finally:
            self._available.put(connection)
        if not isinstance(response, dict) or not response.get("ok"):
            detail = response.get("error") if isinstance(response, dict) else response
            raise RuntimeError(f"shared {self._role} compiler contract failed: {detail}")
        contract = response.get("contract")
        if not isinstance(contract, dict):
            raise RuntimeError(f"shared {self._role} compiler contract is malformed")
        return contract

    def call_record(
        self,
        session_id: str,
        observation: dict[str, Any],
        deck: list[int],
        record: dict[str, Any],
        timing: Counter[str] | None = None,
    ) -> Any:
        return self._exchange(
            {
                "session_id": session_id,
                "observation": observation,
                "deck": tuple(int(card_id) for card_id in deck),
                "record": record,
            },
            expect_action=True,
            timing=timing,
        )

    def _exchange(
        self,
        payload: dict[str, Any],
        *,
        expect_action: bool,
        timing: Counter[str] | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError(f"shared {self._role} inference channel pool is closed")
        roundtrip_started_ns = time.perf_counter_ns()
        connection = self._available.get()
        acquired_ns = time.perf_counter_ns()
        try:
            send_started_ns = time.perf_counter_ns()
            if self._profile:
                payload["_profile_client_send_ns"] = send_started_ns
            connection.send(payload)
            send_finished_ns = time.perf_counter_ns()
            response = connection.recv()
            received_ns = time.perf_counter_ns()
        finally:
            self._available.put(connection)
        if self._profile and timing is not None:
            timing["ipc_calls"] += 1
            timing["ipc_connection_wait_seconds"] += (
                acquired_ns - roundtrip_started_ns
            ) / 1e9
            timing["ipc_send_seconds"] += (
                send_finished_ns - send_started_ns
            ) / 1e9
            timing["ipc_response_wait_seconds"] += (
                received_ns - send_finished_ns
            ) / 1e9
            timing["ipc_roundtrip_seconds"] += (
                received_ns - roundtrip_started_ns
            ) / 1e9
        if not isinstance(response, dict) or not response.get("ok"):
            detail = response.get("error") if isinstance(response, dict) else response
            raise RuntimeError(f"shared {self._role} inference failed: {detail}")
        return response.get("action") if expect_action else None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass


class _PooledRemotePolicyAgent:
    def __init__(
        self,
        pool: _RemotePolicyChannelPool,
        session_id: str,
        deck: list[int],
    ) -> None:
        self._pool = pool
        self._session_id = session_id
        self._deck = deck
        self._closed = False
        self.ipc_timing: Counter[str] = Counter()

    def __call__(self, observation: dict[str, Any]) -> Any:
        return self._pool.call(
            self._session_id, observation, self._deck, self.ipc_timing
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.close_session(self._session_id)


class _WorkerLocalCompilerFactory:
    """Load a torch-free raw compiler once inside an engine worker process."""

    def __init__(
        self,
        policy_root: Path,
        config_payload: dict[str, Any],
        *,
        backend: str = "policy_stateless",
    ) -> None:
        if backend not in WORKER_COMPILER_BACKENDS:
            raise ValueError(f"unsupported worker compiler backend: {backend}")
        root = str(policy_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        # Importing strategy.model.config executes strategy.model.__init__, which
        # imports the torch policy. strategy.contracts.__init__ similarly imports
        # the tensor DecisionBatch. Install only the contracts package boundary so
        # compiler.py can resolve contracts.fields without executing that init.
        incremental_type = None
        if backend == "policy_stateless":
            _install_package_boundary(
                "strategy.contracts",
                policy_root.resolve() / "strategy" / "contracts",
            )
            from strategy.domain.prototypes import PrototypeIndex
            from strategy.features.compiler import compile_canonical_row
            from strategy.knowledge.state import CausalKnowledge

            assets = policy_root.resolve() / "strategy" / "assets"
        else:
            project_name = "train.0035_lifetime_aware_feature_compiler"
            project_root = (
                Path(__file__).resolve().parents[2]
                / "train/0035_lifetime_aware_feature_compiler"
            )
            _install_package_boundary(
                f"{project_name}.contracts", project_root / "contracts"
            )
            _install_package_boundary(
                f"{project_name}.features", project_root / "features"
            )
            prototype_module = importlib.import_module(
                f"{project_name}.domain.prototypes"
            )
            compiler_module = importlib.import_module(
                f"{project_name}.features.compiler"
            )
            incremental_module = importlib.import_module(
                f"{project_name}.features.incremental"
            )
            knowledge_module = importlib.import_module(
                f"{project_name}.knowledge.state"
            )
            PrototypeIndex = prototype_module.PrototypeIndex
            compile_canonical_row = compiler_module.compile_canonical_row
            CausalKnowledge = knowledge_module.CausalKnowledge
            incremental_type = incremental_module.IncrementalCanonicalCompiler
            assets = project_root / "assets"
        self._prototypes = PrototypeIndex.load(
            assets / "official_public_prototypes_v1.json",
            assets / "official_full_engine_prototypes_v2.json",
        )
        self._compile = compile_canonical_row
        self._knowledge_type = CausalKnowledge
        self._incremental_type = incremental_type
        self.backend = backend
        self._max_options = int(config_payload["max_options"])
        self._max_action_steps = int(config_payload["max_action_steps"])
        if "torch" in sys.modules:
            raise RuntimeError("worker-local raw compiler imported torch")

    def create(self, actor: int, deck: list[int]) -> Any:
        return _WorkerLocalRawEncoder(
            actor=actor,
            deck=deck,
            prototypes=self._prototypes,
            compile_record=self._compile,
            knowledge_type=self._knowledge_type,
            incremental_compiler_type=self._incremental_type,
            max_options=self._max_options,
            max_action_steps=self._max_action_steps,
        )


class _WorkerLocalRawEncoder:
    """Stateful causal compiler that emits records without collating tensors."""

    def __init__(
        self,
        *,
        actor: int,
        deck: list[int],
        prototypes: Any,
        compile_record: Callable[[Mapping[str, Any], Any, Any], dict[str, Any]],
        knowledge_type: type,
        incremental_compiler_type: type | None,
        max_options: int,
        max_action_steps: int,
    ) -> None:
        if actor not in (0, 1):
            raise ValueError("actor must be 0 or 1")
        self._deck = tuple(int(card_id) for card_id in deck)
        if len(self._deck) != 60 or any(card_id <= 0 for card_id in self._deck):
            raise ValueError("registered deck must contain exactly 60 positive card IDs")
        self._actor = actor
        self._prototypes = prototypes
        self._compile_record = compile_record
        self._knowledge = knowledge_type(actor, self._deck)
        self._incremental_compiler = (
            incremental_compiler_type(prototypes)
            if incremental_compiler_type is not None
            else None
        )
        self._max_options = max_options
        self._max_action_steps = max_action_steps

    def encode_record(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        current = observation.get("current")
        select = observation.get("select")
        if not isinstance(current, Mapping) or current.get("yourIndex") != self._actor:
            raise ValueError("worker-local online actor mismatch")
        if not isinstance(select, Mapping):
            raise ValueError("worker-local online observation has no select payload")
        options = select.get("option")
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
            raise ValueError("worker-local online observation has no legal options")
        minimum = select.get("minCount", 0)
        maximum = select.get("maxCount", len(options))
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 0 <= minimum <= maximum <= len(options)
        ):
            raise ValueError("worker-local online selection bounds are invalid")
        if len(options) > self._max_options or minimum > self._max_action_steps:
            raise RuntimeError("observation exceeds worker-local inference bounds")
        row = {
            "actor_observation": dict(observation),
            "ordered_action": list(range(minimum)),
            "action_termination": "online_structural_target",
            "identity": None,
            "split": "online",
            "deck_manifest": {"counts": sorted(Counter(self._deck).items())},
        }
        snapshot = self._knowledge.consume(observation)
        if self._incremental_compiler is not None:
            return self._incremental_compiler.compile(row, snapshot)
        return self._compile_record(row, snapshot, self._prototypes)


class _LocalCompiledPolicyAgent:
    def __init__(
        self,
        pool: _RemotePolicyChannelPool,
        session_id: str,
        deck: list[int],
        encoder: Any,
    ) -> None:
        self._pool = pool
        self._session_id = session_id
        self._deck = deck
        self._encoder = encoder
        self._closed = False
        self.compiler_calls = 0
        self.compiler_seconds = 0.0
        self.ipc_timing: Counter[str] = Counter()

    def __call__(self, observation: dict[str, Any]) -> Any:
        if observation.get("select") is None:
            return list(self._deck)
        started_ns = time.perf_counter_ns()
        record = (
            self._encoder.encode_record(observation)
            if callable(getattr(self._encoder, "encode_record", None))
            else self._encoder.encode(observation)
        )
        self.compiler_seconds += (time.perf_counter_ns() - started_ns) / 1e9
        self.compiler_calls += 1
        current = observation.get("current") or {}
        control_observation = {
            "current": {
                key: current.get(key)
                for key in ("turn", "yourIndex")
            },
            "select": observation.get("select"),
        }
        return self._pool.call_record(
            self._session_id,
            control_observation,
            self._deck,
            record,
            self.ipc_timing,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.close_session(self._session_id)


AgentFactory = Callable[
    [GameRequest, str], tuple[Callable[[dict[str, Any]], Any], Callable[[], None]]
]


def _run_pointer_game(
    request: GameRequest,
    trace_path: Path,
    *,
    library: Any,
    library_lock: threading.Lock,
    agent_factory: AgentFactory,
) -> GameResult:
    worker_started_ns = time.perf_counter_ns()
    engine_start_ns = 0
    engine_select_ns = 0
    agent_ns = 0
    agent_calls = 0
    engine_select_calls = 0
    candidate_physical_index = 0
    physical_packages = (request.candidate, request.opponent)
    trace: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "run_id": request.run_id,
        "game_id": request.game_id,
        "candidate": request.candidate.name,
        "opponent": request.opponent.name,
        "candidate_first": request.candidate_first,
        "trace": trace,
    }
    selection_count = 0
    result: GameResult | None = None
    battle = _PointerBattle(library, library_lock)
    candidate_agent = opponent_agent = None
    candidate_close = opponent_close = lambda: None
    start_attempted = False
    phase = "validation"

    try:
        if request.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if request.visualize:
            raise ValueError("engine pool does not support visualization")
        phase = "load"
        candidate_agent, candidate_close = agent_factory(request, "candidate")
        opponent_agent, opponent_close = agent_factory(request, "opponent")
        phase = "start"
        started_ns = time.perf_counter_ns()
        start_attempted = True
        try:
            observation = battle.start(
                physical_packages[0].deck,
                physical_packages[1].deck,
                engine_seed=request.seed if request.engine_library is not None else None,
            )
        finally:
            engine_start_ns += time.perf_counter_ns() - started_ns
        if observation is None:
            result = _error_result(
                request,
                trace_path,
                candidate_physical_index=candidate_physical_index,
                steps=0,
                status="start_error",
                error_kind="start_error",
                error="battle_start returned no observation",
            )
        else:
            phase = "game"
            for step in range(request.max_steps):
                summary = _state_summary(observation)
                current = observation.get("current") or {}
                physical_winner = current.get("result")
                if isinstance(physical_winner, int) and physical_winner >= 0:
                    trace.append({"step": step, "state": summary, "observation": observation})
                    result = GameResult(
                        game_id=request.game_id,
                        opponent=request.opponent.name,
                        candidate_first=request.candidate_first,
                        candidate_physical_index=candidate_physical_index,
                        finished=True,
                        winner=_normalize_winner(physical_winner, candidate_physical_index),
                        status="finished",
                        error_kind=None,
                        error=None,
                        steps=selection_count,
                        trace_path=trace_path,
                    )
                    break
                engine_turn = current.get("turn")
                if (
                    request.engine_turn_draw_limit > 0
                    and type(engine_turn) is int
                    and engine_turn >= request.engine_turn_draw_limit
                ):
                    trace.append({"step": step, "state": summary, "observation": observation})
                    result = GameResult(
                        game_id=request.game_id,
                        opponent=request.opponent.name,
                        candidate_first=request.candidate_first,
                        candidate_physical_index=candidate_physical_index,
                        finished=True,
                        winner=None,
                        status="finished",
                        error_kind=None,
                        error=(
                            "Arena turn-limit draw at engine turn "
                            f"{request.engine_turn_draw_limit}"
                        ),
                        steps=selection_count,
                        trace_path=trace_path,
                    )
                    break
                current_player = int(current.get("yourIndex", 0))
                select = observation.get("select") or {}
                forced_first_player = select.get("context") == 41
                selected_agent = (
                    candidate_agent
                    if current_player == candidate_physical_index
                    else opponent_agent
                )
                try:
                    if forced_first_player:
                        action = [0] if request.candidate_first else [1]
                    else:
                        agent_started_ns = time.perf_counter_ns()
                        try:
                            action = selected_agent(observation)
                        finally:
                            agent_ns += time.perf_counter_ns() - agent_started_ns
                            agent_calls += 1
                except BaseException as exc:
                    error_side = (
                        "candidate_error"
                        if current_player == candidate_physical_index
                        else "opponent_error"
                    )
                    trace.append(
                        {
                            "step": step,
                            "state": summary,
                            "observation": observation,
                            "error": _exception_text(exc),
                        }
                    )
                    result = _error_result(
                        request,
                        trace_path,
                        candidate_physical_index=candidate_physical_index,
                        steps=selection_count,
                        status=error_side,
                        error_kind=error_side,
                        error=_exception_text(exc),
                    )
                    break
                selection_count += 1
                trace_entry = {
                    "step": step,
                    "state": summary,
                    "observation": observation,
                    "action": action,
                }
                if forced_first_player:
                    trace_entry["forced_by_harness"] = "first_player"
                trace.append(trace_entry)
                selected_ns = time.perf_counter_ns()
                try:
                    observation = battle.select(action)
                except IndexError as exc:
                    error_side = (
                        "candidate_error"
                        if current_player == candidate_physical_index
                        else "opponent_error"
                    )
                    result = GameResult(
                        game_id=request.game_id,
                        opponent=request.opponent.name,
                        candidate_first=request.candidate_first,
                        candidate_physical_index=candidate_physical_index,
                        finished=True,
                        winner=1 if error_side == "candidate_error" else 0,
                        status="finished",
                        error_kind=error_side,
                        error=_exception_text(exc),
                        steps=selection_count,
                        trace_path=trace_path,
                    )
                    break
                finally:
                    engine_select_ns += time.perf_counter_ns() - selected_ns
                    engine_select_calls += 1
                if forced_first_player:
                    actual_first = (observation.get("current") or {}).get("firstPlayer")
                    expected_first = 0 if request.candidate_first else 1
                    if actual_first != expected_first:
                        raise RuntimeError(
                            "official engine did not honor forced first player: "
                            f"expected {expected_first}, got {actual_first}"
                        )
                physical_winner = (observation.get("current") or {}).get("result")
                if isinstance(physical_winner, int) and physical_winner >= 0:
                    trace.append(
                        {
                            "step": step + 1,
                            "state": _state_summary(observation),
                            "observation": observation,
                        }
                    )
                    result = GameResult(
                        game_id=request.game_id,
                        opponent=request.opponent.name,
                        candidate_first=request.candidate_first,
                        candidate_physical_index=candidate_physical_index,
                        finished=True,
                        winner=_normalize_winner(physical_winner, candidate_physical_index),
                        status="finished",
                        error_kind=None,
                        error=None,
                        steps=selection_count,
                        trace_path=trace_path,
                    )
                    break
            else:
                result = _error_result(
                    request,
                    trace_path,
                    candidate_physical_index=candidate_physical_index,
                    steps=selection_count,
                    status="unfinished",
                    error_kind="step_limit",
                    error=f"step limit reached ({request.max_steps})",
                )
    except BaseException as exc:
        error_kind = "worker_error" if phase == "validation" else f"{phase}_error"
        result = _error_result(
            request,
            trace_path,
            candidate_physical_index=candidate_physical_index,
            steps=selection_count,
            status=error_kind,
            error_kind=error_kind,
            error=str(exc) if phase == "validation" else _exception_text(exc),
        )
    finally:
        if start_attempted:
            try:
                battle.finish()
            except BaseException as exc:
                if result is None or result.finished:
                    result = _error_result(
                        request,
                        trace_path,
                        candidate_physical_index=candidate_physical_index,
                        steps=selection_count,
                        status="finish_error",
                        error_kind="finish_error",
                        error=_exception_text(exc),
                    )
        for close in (candidate_close, opponent_close):
            try:
                close()
            except BaseException:
                pass
        if result is None:
            result = _error_result(
                request,
                trace_path,
                candidate_physical_index=candidate_physical_index,
                steps=selection_count,
                status="worker_error",
                error_kind="worker_error",
                error="pool worker did not produce a result",
            )
        result = replace(
            result,
            performance={
                "worker_wall_seconds": (time.perf_counter_ns() - worker_started_ns) / 1e9,
                "engine_start_seconds": engine_start_ns / 1e9,
                "engine_select_seconds": engine_select_ns / 1e9,
                "agent_seconds": agent_ns / 1e9,
                "agent_calls": agent_calls,
                "engine_select_calls": engine_select_calls,
                "compiler_seconds": sum(
                    float(getattr(agent, "compiler_seconds", 0.0))
                    for agent in (candidate_agent, opponent_agent)
                    if agent is not None
                ),
                "compiler_calls": sum(
                    int(getattr(agent, "compiler_calls", 0))
                    for agent in (candidate_agent, opponent_agent)
                    if agent is not None
                ),
                **{
                    field: sum(
                        float(getattr(agent, "ipc_timing", {}).get(field, 0.0))
                        for agent in (candidate_agent, opponent_agent)
                        if agent is not None
                    )
                    for field in (
                        "ipc_connection_wait_seconds",
                        "ipc_send_seconds",
                        "ipc_response_wait_seconds",
                        "ipc_roundtrip_seconds",
                    )
                },
                "ipc_calls": sum(
                    int(getattr(agent, "ipc_timing", {}).get("ipc_calls", 0))
                    for agent in (candidate_agent, opponent_agent)
                    if agent is not None
                ),
            },
        )
        payload["result"] = _serialized_result(result)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def run_pool_games(
    jobs: list[tuple[GameRequest, Path]],
    *,
    library: Any,
    pool_size: int,
    agent_factory: AgentFactory,
    validate_compatibility: bool = True,
) -> list[GameResult]:
    if pool_size < 1:
        raise ValueError("pool_size must be at least one")
    if validate_compatibility:
        for request, _trace_path in jobs:
            assert_cg_compatible(request.candidate, request.opponent)
    library_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [
            executor.submit(
                _run_pointer_game,
                request,
                trace_path,
                library=library,
                library_lock=library_lock,
                agent_factory=agent_factory,
            )
            for request, trace_path in jobs
        ]
        return [future.result() for future in futures]


def _request_from_payload(payload: dict[str, Any]) -> GameRequest:
    return GameRequest(
        run_id=str(payload["run_id"]),
        game_id=str(payload["game_id"]),
        candidate=_package_from_payload(payload["candidate"]),
        opponent=_package_from_payload(payload["opponent"]),
        candidate_first=bool(payload["candidate_first"]),
        max_steps=int(payload["max_steps"]),
        visualize=bool(payload["visualize"]),
        engine_turn_draw_limit=int(payload.get("engine_turn_draw_limit", 0)),
        seed=int(payload.get("seed", 0)),
        policy_seed=int(payload.get("policy_seed", payload.get("seed", 0))),
        search_seed=int(payload.get("search_seed", payload.get("seed", 0))),
        engine_library=(
            Path(payload["engine_library"])
            if payload.get("engine_library") is not None
            else None
        ),
        arbitrary_legal_actions=bool(payload.get("arbitrary_legal_actions", False)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    jobs = [
        (_request_from_payload(row["request"]), Path(row["trace_path"]))
        for row in payload["jobs"]
    ]
    if not jobs:
        raise ValueError("engine pool request has no jobs")
    engine_library = jobs[0][0].engine_library
    if engine_library is not None:
        if any(request.engine_library != engine_library for request, _ in jobs):
            raise ValueError("engine pool jobs use different engine libraries")
        os.environ["PTCG_CG_LIBRARY"] = str(engine_library.resolve())
    game_module = _load_game_api_with_runtime(jobs[0][0].candidate.root)
    if engine_library is not None:
        game_module.lib.BattleStartSeeded.restype = game_module.lib.BattleStart.restype
        game_module.lib.BattleStartSeeded.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_ulonglong,
        ]
    candidate_socket = str(payload["candidate_inference_socket"])
    opponent_socket = str(payload["opponent_inference_socket"])
    inference_channels = int(
        payload.get("inference_channels_per_role", min(int(payload["pool_size"]), 8))
    )
    candidate_pool = _RemotePolicyChannelPool(
        candidate_socket,
        "candidate",
        channel_count=inference_channels,
        profile=bool(payload.get("inference_profile", False)),
    )
    opponent_pool = _RemotePolicyChannelPool(
        opponent_socket,
        "opponent",
        channel_count=inference_channels,
        profile=bool(payload.get("inference_profile", False)),
    )
    local_compiler_factory = None
    if bool(payload.get("worker_local_compiler", False)):
        policy_roots = {request.candidate.root.resolve() for request, _ in jobs} | {
            request.opponent.root.resolve() for request, _ in jobs
        }
        if len(policy_roots) != 1:
            raise ValueError("worker-local compiler requires one identical policy root")
        candidate_contract = candidate_pool.compiler_contract()
        opponent_contract = opponent_pool.compiler_contract()
        if candidate_contract != opponent_contract or not candidate_contract.get("supported"):
            raise ValueError("worker-local compiler contracts are incompatible")
        config_payload = candidate_contract.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError("worker-local compiler contract has no model config")
        local_compiler_factory = _WorkerLocalCompilerFactory(
            next(iter(policy_roots)),
            config_payload,
            backend=str(payload.get("worker_compiler_backend", "policy_stateless")),
        )

    def agent_factory(request: GameRequest, role: str):
        package = request.candidate if role == "candidate" else request.opponent
        pool = candidate_pool if role == "candidate" else opponent_pool
        session_id = f"{request.run_id}:{request.game_id}:{role}"
        if local_compiler_factory is None:
            agent = _PooledRemotePolicyAgent(pool, session_id, package.deck)
        else:
            actor = 0 if role == "candidate" else 1
            agent = _LocalCompiledPolicyAgent(
                pool,
                session_id,
                package.deck,
                local_compiler_factory.create(actor, package.deck),
            )
        return agent, agent.close

    try:
        results = run_pool_games(
            jobs,
            library=game_module.lib,
            pool_size=int(payload["pool_size"]),
            agent_factory=agent_factory,
        )
    finally:
        candidate_pool.close()
        opponent_pool.close()
    serialized = []
    for result in results:
        row = asdict(result)
        row["trace_path"] = str(result.trace_path)
        serialized.append(row)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps({"results": serialized}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
