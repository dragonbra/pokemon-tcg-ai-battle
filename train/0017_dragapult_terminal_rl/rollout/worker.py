from __future__ import annotations

import os
import random
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from evaluation.packages.loader import _clear_cg_modules
from evaluation.runner.worker import (
    _isolated_agent,
    _load_agent_module,
    _load_game_api_with_runtime,
)

from ..constants import TARGET_DECK
from .protocol import RolloutJob


def _terminal_reward(physical_winner: int | None, candidate_index: int) -> float:
    if physical_winner not in (0, 1):
        return 0.0
    return 1.0 if physical_winner == candidate_index else -1.0


def _result_message(
    job: RolloutJob,
    *,
    valid: bool,
    reward: float | None,
    winner: int | None,
    status: str,
    error: str | None,
    selections: int,
    final_turn: int | None,
) -> dict[str, Any]:
    return {
        "kind": "result",
        "episode_id": job.episode_id,
        "valid": valid,
        "reward": reward,
        "winner": winner,
        "status": status,
        "error": error,
        "engine_selections": selections,
        "final_turn": final_turn,
        "complete_rounds": None if final_turn is None else (final_turn + 1) // 2,
    }


def run_engine_episode(connection: Connection, job: RolloutJob) -> None:
    """Run one isolated official-engine game and RPC candidate decisions to the parent."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    random.seed(job.seed)
    candidate_index = 0 if job.candidate_first else 1
    decks = (
        (list(TARGET_DECK), job.opponent.deck)
        if job.candidate_first
        else (job.opponent.deck, list(TARGET_DECK))
    )
    game_module = None
    started = False
    selections = 0
    final_turn: int | None = None
    parent_cwd = Path.cwd()
    parent_path = list(sys.path)
    try:
        _clear_cg_modules()
        game_module = _load_game_api_with_runtime(job.opponent.root)
        opponent_module = _load_agent_module(job.opponent, "rl_opponent", ())
        opponent_agent = _isolated_agent(job.opponent, opponent_module.agent, ())
        started = True
        observation, _ = game_module.battle_start(decks[0], decks[1])
        if observation is None:
            connection.send(
                _result_message(
                    job,
                    valid=False,
                    reward=None,
                    winner=None,
                    status="start_error",
                    error="battle_start returned no observation",
                    selections=0,
                    final_turn=None,
                )
            )
            return
        for _ in range(job.max_steps):
            current = observation.get("current") or {}
            raw_winner = current.get("result")
            final_turn = int(current.get("turn", 0))
            if isinstance(raw_winner, int) and raw_winner >= 0:
                connection.send(
                    _result_message(
                        job,
                        valid=True,
                        reward=_terminal_reward(raw_winner, candidate_index),
                        winner=raw_winner if raw_winner in (0, 1) else None,
                        status="finished",
                        error=None,
                        selections=selections,
                        final_turn=final_turn,
                    )
                )
                return
            actor = int(current.get("yourIndex", -1))
            if actor == candidate_index:
                connection.send({"kind": "decision", "observation": observation})
                response = connection.recv()
                if not isinstance(response, dict) or response.get("kind") != "action":
                    raise RuntimeError("collector returned an invalid action response")
                action = response.get("action")
            elif actor in (0, 1):
                action = opponent_agent(observation)
            else:
                raise RuntimeError(f"engine returned invalid current player: {actor}")
            selections += 1
            observation = game_module.battle_select(action)
        connection.send(
            _result_message(
                job,
                valid=False,
                reward=None,
                winner=None,
                status="step_limit",
                error=f"step limit reached ({job.max_steps})",
                selections=selections,
                final_turn=final_turn,
            )
        )
    except (EOFError, BrokenPipeError):
        return
    except BaseException as error:
        try:
            connection.send(
                _result_message(
                    job,
                    valid=False,
                    reward=None,
                    winner=None,
                    status="engine_error",
                    error=f"{type(error).__name__}: {error}",
                    selections=selections,
                    final_turn=final_turn,
                )
            )
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if started and game_module is not None:
            try:
                game_module.battle_finish()
            except BaseException:
                pass
        sys.path[:] = parent_path
        os.chdir(parent_cwd)
        _clear_cg_modules()
        connection.close()


__all__ = ["run_engine_episode"]
