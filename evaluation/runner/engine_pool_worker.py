"""Multi-environment official-engine worker for resident policy inference."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from multiprocessing.connection import Client
from pathlib import Path
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

    def start(self, deck0: list[int], deck1: list[int]) -> dict[str, Any] | None:
        if self._pointer is not None:
            raise RuntimeError("battle has already started")
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("The deck must contain 60 cards.")
        cards = deck0 + deck1
        argument = (ctypes.c_int * len(cards))(*cards)
        with self._library_lock:
            start_data = self._library.BattleStart(argument)
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
    candidate_physical_index = 0 if request.candidate_first else 1
    physical_packages = (
        (request.candidate, request.opponent)
        if request.candidate_first
        else (request.opponent, request.candidate)
    )
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
                selected_agent = (
                    candidate_agent
                    if current_player == candidate_physical_index
                    else opponent_agent
                )
                agent_started_ns = time.perf_counter_ns()
                try:
                    action = selected_agent(observation)
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
                finally:
                    agent_ns += time.perf_counter_ns() - agent_started_ns
                    agent_calls += 1
                selection_count += 1
                trace.append(
                    {
                        "step": step,
                        "state": summary,
                        "observation": observation,
                        "action": action,
                    }
                )
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
    game_module = _load_game_api_with_runtime(jobs[0][0].candidate.root)
    candidate_socket = str(payload["candidate_inference_socket"])
    opponent_socket = str(payload["opponent_inference_socket"])

    def agent_factory(request: GameRequest, role: str):
        package = request.candidate if role == "candidate" else request.opponent
        socket = candidate_socket if role == "candidate" else opponent_socket
        agent = _RemotePolicyAgent(socket, package.deck, role)
        return agent, agent.close

    results = run_pool_games(
        jobs,
        library=game_module.lib,
        pool_size=int(payload["pool_size"]),
        agent_factory=agent_factory,
    )
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
