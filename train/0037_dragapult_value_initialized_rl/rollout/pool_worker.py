"""Persistent multi-battle official-engine worker for 0037 RL rollout."""

from __future__ import annotations

import ctypes
import json
import os
import queue
import signal
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from typing import Any

from evaluation.runtime.seeded import load_seeded_library

from .protocol import RolloutJob
from .worker_compiler import WorkerLocalCompiler


class _PointerBattle:
    """One independent battle pointer in the unchanged seeded runtime."""

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

    def start(self, deck0: tuple[int, ...], deck1: tuple[int, ...], seed: int):
        cards = deck0 + deck1
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("official battle decks must contain exactly 60 cards")
        argument = (ctypes.c_int * len(cards))(*cards)
        with self._library_lock:
            started = self._library.BattleStartSeeded(argument, seed)
        self._pointer = started.battle_ptr
        if self._pointer is None or self._pointer == 0:
            return None
        return self._observation()

    def select(self, action: list[int]) -> dict[str, Any]:
        if not isinstance(action, list) or not all(type(index) is int for index in action):
            raise ValueError("official action must be list[int]")
        argument = (ctypes.c_int * len(action))(*action)
        with self._library_lock:
            error = int(self._library.Select(self._pointer, argument, len(action)))
        if error == 30:
            raise ValueError("battle pointer is broken")
        if error != 0:
            raise IndexError(f"official engine rejected action: {error}")
        return self._observation()

    def finish(self) -> None:
        if self._finished or self._pointer is None:
            return
        self._finished = True
        with self._library_lock:
            self._library.BattleFinish(self._pointer)


class _ChannelPool:
    """Lease one synchronous duplex inference channel per active request."""

    def __init__(self, connections: list[Connection]) -> None:
        if not connections:
            raise ValueError("inference channel pool cannot be empty")
        self._connections = connections
        self._available: queue.Queue[Connection] = queue.Queue(len(connections))
        for connection in connections:
            self._available.put(connection)

    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        connection = self._available.get()
        try:
            connection.send(payload)
            response = connection.recv()
        finally:
            self._available.put(connection)
        if not isinstance(response, dict) or response.get("kind") != "action":
            raise RuntimeError("collector returned an invalid action response")
        return response

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass


def _result(
    job: RolloutJob,
    *,
    valid: bool,
    reward: float | None,
    status: str,
    error: str | None,
    selections: int,
    final_turn: int | None,
) -> dict[str, Any]:
    return {
        "kind": "result",
        "game_id": job.game_id,
        "valid": valid,
        "reward": reward,
        "status": status,
        "error": error,
        "engine_selections": selections,
        "final_turn": final_turn,
    }


def _option_identity(option: dict[str, Any]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(
        (key, value)
        for key, value in option.items()
        if isinstance(value, (int, str, bool, type(None)))
    ))


def _run_job(
    job: RolloutJob,
    *,
    library: Any,
    library_lock: threading.Lock,
    focal_channels: _ChannelPool,
    opponent_channels: _ChannelPool,
) -> dict[str, Any]:
    battle = _PointerBattle(library, library_lock)
    selections = 0
    final_turn = None
    focal_compiler = WorkerLocalCompiler(0, job.focal_deck)
    opponent_compiler = WorkerLocalCompiler(1, job.opponent_deck)
    repeated_actions: dict[
        tuple[int, int, int, object, tuple[tuple[tuple[str, object], ...], ...], tuple[int, ...]],
        int,
    ] = defaultdict(int)
    turn_actor: tuple[int, int] | None = None
    try:
        observation = battle.start(job.focal_deck, job.opponent_deck, job.seed)
        if observation is None:
            return _result(
                job, valid=False, reward=None, status="start_error",
                error="BattleStartSeeded returned no observation", selections=0,
                final_turn=None,
            )
        for _ in range(job.max_steps):
            current = observation.get("current") or {}
            final_turn = int(current.get("turn", 0))
            winner = current.get("result")
            if isinstance(winner, int) and winner >= 0:
                reward = 0.0 if winner not in (0, 1) else (1.0 if winner == 0 else -1.0)
                return _result(
                    job, valid=True, reward=reward, status="finished", error=None,
                    selections=selections, final_turn=final_turn,
                )
            if (
                job.full_round_draw_limit > 0
                and (final_turn + 1) // 2 >= job.full_round_draw_limit
            ):
                return _result(
                    job, valid=True, reward=0.0, status="turn_limit_draw", error=None,
                    selections=selections, final_turn=final_turn,
                )
            actor = int(current.get("yourIndex", -1))
            if actor not in (0, 1):
                raise RuntimeError(f"official observation has invalid actor: {actor}")
            if turn_actor != (final_turn, actor):
                turn_actor = (final_turn, actor)
                repeated_actions.clear()
            selection = observation.get("select") or {}
            if selection.get("context") == 41:
                action = [0] if job.focal_first else [1]
                selections += 1
                observation = battle.select(action)
                actual_first = (observation.get("current") or {}).get("firstPlayer")
                expected_first = 0 if job.focal_first else 1
                if actual_first != expected_first:
                    raise RuntimeError(
                        "official engine did not honor forced first player: "
                        f"expected {expected_first}, got {actual_first}"
                    )
                continue
            compile_started = time.perf_counter()
            record = (
                focal_compiler if actor == 0 else opponent_compiler
            ).compile(observation)
            compile_seconds = time.perf_counter() - compile_started
            role = "focal" if actor == 0 else "opponent"
            channels = focal_channels if actor == 0 else opponent_channels
            response = channels.exchange({
                "kind": "decision",
                "session_id": job.game_id,
                "game_id": job.game_id,
                "role": role,
                "actor": actor,
                "selection_index": selections,
                "record": record,
                "turn": final_turn,
                "select": {
                    "type": selection.get("type"),
                    "context": selection.get("context"),
                    "minCount": selection.get("minCount"),
                    "maxCount": selection.get("maxCount"),
                    "option": selection.get("option"),
                },
                "compile_seconds": compile_seconds,
            })
            action = response.get("action")
            options = selection.get("option") or []
            if (
                job.ability_repeat_limit > 0
                and isinstance(action, list)
                and all(isinstance(value, int) for value in action)
            ):
                option_fingerprints = tuple(
                    _option_identity(option)
                    for option in options
                    if isinstance(option, dict)
                )
                context = selection.get("context")
                try:
                    hash(context)
                except TypeError:
                    context = repr(context)
                key = (
                    final_turn,
                    actor,
                    int(selection.get("type", -1)),
                    context,
                    option_fingerprints,
                    tuple(action),
                )
                repeated_actions[key] += 1
                if repeated_actions[key] > job.ability_repeat_limit:
                    return _result(
                        job,
                        valid=True,
                        reward=-1.0 if actor == 0 else 1.0,
                        status=(
                            "ability_repeat_forfeit"
                            if selection.get("type") == 0
                            else "repeated_selection_forfeit"
                        ),
                        error=None,
                        selections=selections,
                        final_turn=final_turn,
                    )
            selections += 1
            observation = battle.select(action)
        return _result(
            job, valid=False, reward=None, status="step_limit",
            error=f"step limit reached ({job.max_steps})", selections=selections,
            final_turn=final_turn,
        )
    except (EOFError, BrokenPipeError):
        return _result(
            job, valid=False, reward=None, status="collector_closed",
            error="collector inference channel closed", selections=selections,
            final_turn=final_turn,
        )
    except BaseException as error:
        return _result(
            job, valid=False, reward=None, status="engine_error",
            error=f"{type(error).__name__}: {error}", selections=selections,
            final_turn=final_turn,
        )
    finally:
        try:
            battle.finish()
        except BaseException:
            pass


def run_engine_pool(
    result_connection: Connection,
    focal_connections: list[Connection],
    opponent_connections: list[Connection],
    jobs: list[RolloutJob],
    engines_per_worker: int,
) -> None:
    """Run a stable worker shard and stream per-game results to the parent."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if engines_per_worker < 1:
        raise ValueError("engines_per_worker must be at least one")
    if not jobs:
        result_connection.close()
        return
    libraries = {job.engine_library.resolve() for job in jobs if job.engine_library is not None}
    if len(libraries) != 1 or any(job.engine_library is None for job in jobs):
        raise ValueError("pooled RL requires one explicit seeded engine library")
    library = load_seeded_library(next(iter(libraries)))
    library_lock = threading.Lock()
    result_lock = threading.Lock()
    focal_channels = _ChannelPool(focal_connections)
    opponent_channels = _ChannelPool(opponent_connections)

    def execute(job: RolloutJob) -> None:
        message = _run_job(
            job,
            library=library,
            library_lock=library_lock,
            focal_channels=focal_channels,
            opponent_channels=opponent_channels,
        )
        with result_lock:
            result_connection.send(message)

    try:
        with ThreadPoolExecutor(max_workers=engines_per_worker) as executor:
            futures = [executor.submit(execute, job) for job in jobs]
            for future in futures:
                future.result()
        result_connection.send({"kind": "worker_done"})
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        focal_channels.close()
        opponent_channels.close()
        result_connection.close()


__all__ = ["_ChannelPool", "_PointerBattle", "run_engine_pool"]
