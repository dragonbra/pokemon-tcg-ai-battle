"""Isolated official-engine process; all policy decisions stay in the GPU parent."""

from __future__ import annotations

import os
import random
import signal
from multiprocessing.connection import Connection
from typing import Any

from evaluation.packages.loader import _clear_cg_modules
from evaluation.runner.worker import _load_game_api_with_runtime

from .protocol import RolloutJob


def _result(job: RolloutJob, *, valid: bool, reward: float | None, status: str,
            error: str | None, selections: int, final_turn: int | None) -> dict[str, Any]:
    return {
        "kind": "result", "game_id": job.game_id, "valid": valid,
        "reward": reward, "status": status, "error": error,
        "engine_selections": selections, "final_turn": final_turn,
    }


def run_engine_episode(connection: Connection, job: RolloutJob) -> None:
    # The parent converts terminal Ctrl-C into an update-boundary stop request.
    # Keep SIGTERM available for collector cleanup of a stuck worker.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    random.seed(job.seed)
    focal_index = 0 if job.focal_first else 1
    decks = (job.focal_deck, job.opponent_deck) if job.focal_first else (job.opponent_deck, job.focal_deck)
    game = None
    selections = 0
    final_turn = None
    try:
        _clear_cg_modules()
        game = _load_game_api_with_runtime(job.runtime_root)
        observation, _ = game.battle_start(list(decks[0]), list(decks[1]))
        if observation is None:
            connection.send(_result(job, valid=False, reward=None, status="start_error", error="battle_start returned no observation", selections=0, final_turn=None))
            return
        for _ in range(job.max_steps):
            current = observation.get("current") or {}
            final_turn = int(current.get("turn", 0))
            winner = current.get("result")
            if isinstance(winner, int) and winner >= 0:
                reward = 0.0 if winner not in (0, 1) else (1.0 if winner == focal_index else -1.0)
                connection.send(_result(job, valid=True, reward=reward, status="finished", error=None, selections=selections, final_turn=final_turn))
                return
            actor = int(current.get("yourIndex", -1))
            if actor not in (0, 1):
                raise RuntimeError(f"engine returned invalid current player: {actor}")
            connection.send({"kind": "decision", "role": "focal" if actor == focal_index else "opponent", "actor": actor, "observation": observation})
            response = connection.recv()
            if not isinstance(response, dict) or response.get("kind") != "action":
                raise RuntimeError("collector returned an invalid action response")
            selections += 1
            observation = game.battle_select(response.get("action"))
        connection.send(_result(job, valid=False, reward=None, status="step_limit", error=f"step limit reached ({job.max_steps})", selections=selections, final_turn=final_turn))
    except (EOFError, BrokenPipeError):
        return
    except BaseException as error:
        try:
            connection.send(_result(job, valid=False, reward=None, status="engine_error", error=f"{type(error).__name__}: {error}", selections=selections, final_turn=final_turn))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if game is not None:
            try:
                game.battle_finish()
            except BaseException:
                pass
        _clear_cg_modules()
        connection.close()


__all__ = ["run_engine_episode"]
