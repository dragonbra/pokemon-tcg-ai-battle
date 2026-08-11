from __future__ import annotations

import importlib
from pathlib import Path
import unittest


worker = importlib.import_module(
    "train.0042_full_model_design.rollout.worker"
)
protocol = importlib.import_module(
    "train.0042_full_model_design.rollout.protocol"
)


class _Connection:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.action_requests = 0

    def send(self, message: dict) -> None:
        self.messages.append(message)

    def recv(self) -> dict:
        self.action_requests += 1
        return {"kind": "action", "action": [0]}

    def close(self) -> None:
        pass


class _Game:
    def __init__(self, observation: dict) -> None:
        self.observation = observation

    def battle_start(self, *_args):
        return self.observation, None

    def battle_select(self, *_args):
        return self.observation

    def battle_finish(self) -> None:
        pass


class RolloutTerminationTest(unittest.TestCase):
    def _run(self, observation: dict) -> tuple[dict, int]:
        original = worker._load_game_api_with_runtime
        original_compiler = worker.WorkerLocalCompiler
        worker._load_game_api_with_runtime = lambda _root: _Game(observation)
        worker.WorkerLocalCompiler = lambda *_args: type(
            "Compiler", (), {"compile": lambda _self, _observation: {}}
        )()
        try:
            connection = _Connection()
            job = protocol.RolloutJob(
                "test", "deck", True, 1, 0, (1,) * 60, (2,) * 60, Path("."),
                "Policy-0809",
            )
            worker.run_engine_episode(connection, job)
            return connection.messages[-1], connection.action_requests
        finally:
            worker._load_game_api_with_runtime = original
            worker.WorkerLocalCompiler = original_compiler

    def test_repeated_selection_chain_forfeits_to_opponent(self) -> None:
        observation = {
            "current": {"turn": 0, "yourIndex": 0, "result": -1},
            "select": {
                "type": 1,
                "context": 21,
                "option": [{"type": 14, "id": 7}],
            },
        }
        result, action_requests = self._run(observation)
        self.assertEqual(result["status"], "repeated_selection_forfeit")
        self.assertEqual(result["reward"], -1.0)
        self.assertEqual(action_requests, 20)

    def test_fifty_full_rounds_are_a_draw(self) -> None:
        observation = {
            "current": {"turn": 99, "yourIndex": 0, "result": -1},
            "select": {
                "type": 1,
                "context": 21,
                "option": [{"type": 14, "id": 7}],
            },
        }
        result, _action_requests = self._run(observation)
        self.assertEqual(result["status"], "turn_limit_draw")
        self.assertEqual(result["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
