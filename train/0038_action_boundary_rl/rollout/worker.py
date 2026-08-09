"""One isolated official-engine process; policy inference stays in the parent."""

from __future__ import annotations

import os
import random
import signal
import ctypes
import time
from collections import defaultdict
from multiprocessing.connection import Connection
from typing import Any

from evaluation.packages.loader import _clear_cg_modules
from evaluation.runner.worker import _load_game_api_with_runtime

from .protocol import RolloutJob
from .worker_compiler import WorkerLocalCompiler
from ..action_boundary.decision_gate import DecisionClass, DecisionGate
from ..action_boundary.macro_protocol import PendingMacroTransaction
from ..action_boundary.official_protocol import OfficialProtocolExecutor


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


def run_engine_episode(connection: Connection, job: RolloutJob) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    random.seed(job.policy_seed or job.seed)
    agent_selects_first_player = job.focal_won_toss is not None
    focal_index = (
        0 if bool(job.focal_won_toss) else 1
    ) if agent_selects_first_player else 0
    decks = (
        (job.focal_deck, job.opponent_deck)
        if focal_index == 0
        else (job.opponent_deck, job.focal_deck)
    )
    game = None
    selections = 0
    final_turn = None
    repeated_actions: dict[
        tuple[int, int, int, object, tuple[tuple[tuple[str, object], ...], ...], tuple[int, ...]],
        int,
    ] = defaultdict(int)
    turn_actor: tuple[int, int] | None = None
    focal_compiler = WorkerLocalCompiler(focal_index, job.focal_deck)
    opponent_compiler = WorkerLocalCompiler(1 - focal_index, job.opponent_deck)
    decision_gate = DecisionGate()
    protocol_executor = OfficialProtocolExecutor()

    def option_identity(option: dict[str, Any]) -> tuple[tuple[str, object], ...]:
        return tuple(
            sorted(
                (key, value)
                for key, value in option.items()
                if isinstance(value, (int, str, bool, type(None)))
            )
        )
    try:
        _clear_cg_modules()
        if job.engine_library is not None:
            os.environ["PTCG_CG_LIBRARY"] = str(job.engine_library.resolve())
        game = _load_game_api_with_runtime(job.runtime_root)
        if job.engine_library is not None:
            configure = getattr(getattr(game, "lib", None), "ConfigureSeeds", None)
            if configure is None:
                raise RuntimeError("seeded engine runtime has no ConfigureSeeds symbol")
            configure.restype = ctypes.c_int
            configure.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong]
            error = int(configure(job.seed, job.search_seed))
            if error != 0:
                raise RuntimeError(f"seeded engine configuration failed: {error}")
        observation, _ = game.battle_start(list(decks[0]), list(decks[1]))
        if observation is None:
            connection.send(_result(
                job, valid=False, reward=None, status="start_error",
                error="battle_start returned no observation", selections=0, final_turn=None,
            ))
            return
        for _ in range(job.max_steps):
            current = observation.get("current") or {}
            final_turn = int(current.get("turn", 0))
            winner = current.get("result")
            if isinstance(winner, int) and winner >= 0:
                reward = 0.0 if winner not in (0, 1) else (1.0 if winner == focal_index else -1.0)
                connection.send(_result(
                    job, valid=True, reward=reward, status="finished", error=None,
                    selections=selections, final_turn=final_turn,
                ))
                return
            if (
                job.full_round_draw_limit > 0
                and isinstance(final_turn, int)
                and (final_turn + 1) // 2 >= job.full_round_draw_limit
            ):
                connection.send(_result(
                    job, valid=True, reward=0.0, status="turn_limit_draw", error=None,
                    selections=selections, final_turn=final_turn,
                ))
                return
            actor = int(current.get("yourIndex", -1))
            if actor not in (0, 1):
                raise RuntimeError(f"engine returned invalid current player: {actor}")
            if turn_actor != (final_turn, actor):
                turn_actor = (final_turn, actor)
                repeated_actions.clear()
            selection = observation.get("select") or {}
            if selection.get("context") == 41 and not agent_selects_first_player:
                action = [0] if job.focal_first else [1]
                selections += 1
                observation = game.battle_select(action)
                actual_first = (observation.get("current") or {}).get("firstPlayer")
                expected_first = 0 if job.focal_first else 1
                if actual_first != expected_first:
                    raise RuntimeError(
                        "official engine did not honor forced first player: "
                        f"expected {expected_first}, got {actual_first}"
                    )
                continue
            if protocol_executor.has_pending(job.game_id):
                compiler = focal_compiler if actor == focal_index else opponent_compiler
                compiler.observe_only(observation)
                selections += 1
                observation = game.battle_select(protocol_executor.select(job.game_id, observation))
                continue
            gate = decision_gate.classify(observation)
            if gate.classification is DecisionClass.MASK_ERROR:
                raise RuntimeError(f"DecisionGate MASK_ERROR: {gate.reason}")
            if (
                job.action_boundary_mode == "enabled"
                and gate.classification in {DecisionClass.FORCED, DecisionClass.LEGAL_EMPTY_PASS}
            ):
                compiler = focal_compiler if actor == focal_index else opponent_compiler
                compiler.observe_only(observation)
                selections += 1
                observation = game.battle_select(list(gate.forced_action or ()))
                continue
            compile_started = time.perf_counter()
            record = (
                focal_compiler if actor == focal_index else opponent_compiler
            ).compile(observation)
            compile_seconds = time.perf_counter() - compile_started
            connection.send({
                "kind": "decision",
                "role": "focal" if actor == focal_index else "opponent",
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
                "visible_opponent_bench": list(
                    ((current.get("players") or [{}, {}])[1 - actor] or {}).get("bench") or []
                ),
            })
            response = connection.recv()
            if not isinstance(response, dict) or response.get("kind") != "action":
                raise RuntimeError("collector returned an invalid action response")
            action = response.get("action")
            macro = response.get("macro_transaction")
            if macro is not None:
                transaction = PendingMacroTransaction.from_wire(macro)
                if transaction.battle_id != job.game_id or transaction.actor != actor:
                    raise RuntimeError("collector returned a macro for the wrong session/actor")
                protocol_executor.begin(transaction)
            selection = observation.get("select") or {}
            options = selection.get("option") or []
            if (
                job.ability_repeat_limit > 0
                and isinstance(action, list)
                and all(isinstance(value, int) for value in action)
            ):
                option_fingerprints = tuple(
                    option_identity(option)
                    for option in options
                    if isinstance(option, dict)
                )
                context = selection.get("context")
                try:
                    hash(context)
                except TypeError:
                    context = repr(context)
                key = (
                    int(final_turn),
                    actor,
                    int(selection.get("type", -1)),
                    context,
                    option_fingerprints,
                    tuple(action),
                )
                repeated_actions[key] += 1
                if repeated_actions[key] >= job.ability_repeat_limit:
                    focal_reward = -1.0 if actor == focal_index else 1.0
                    status = (
                        "ability_repeat_forfeit"
                        if selection.get("type") == 0
                        else "repeated_selection_forfeit"
                    )
                    connection.send(_result(
                        job, valid=True, reward=focal_reward,
                        status=status, error=None,
                        selections=selections, final_turn=final_turn,
                    ))
                    return
            selections += 1
            observation = game.battle_select(response.get("action"))
            if selection.get("context") == 41:
                action = response.get("action")
                if (
                    not isinstance(action, list)
                    or len(action) != 1
                    or action[0] not in (0, 1)
                ):
                    raise RuntimeError("Agent first-player choice is not one Yes/No action")
                actual_first = (observation.get("current") or {}).get("firstPlayer")
                expected_first = actor if action[0] == 0 else 1 - actor
                if actual_first != expected_first:
                    raise RuntimeError(
                        "official engine did not honor Agent first-player choice: "
                        f"expected {expected_first}, got {actual_first}"
                    )
        connection.send(_result(
            job, valid=False, reward=None, status="step_limit",
            error=f"step limit reached ({job.max_steps})", selections=selections,
            final_turn=final_turn,
        ))
    except (EOFError, BrokenPipeError):
        return
    except BaseException as error:
        try:
            connection.send(_result(
                job, valid=False, reward=None, status="engine_error",
                error=f"{type(error).__name__}: {error}", selections=selections,
                final_turn=final_turn,
            ))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        protocol_executor.clear(job.game_id)
        if game is not None:
            try:
                game.battle_finish()
            except BaseException:
                pass
        _clear_cg_modules()
        connection.close()


__all__ = ["run_engine_episode"]
