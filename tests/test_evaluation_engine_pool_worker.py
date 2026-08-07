from __future__ import annotations

import ctypes
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.packages.loader import SubmissionPackage
from evaluation.runner.engine_pool_worker import _PointerBattle, run_pool_games
from evaluation.runner.models import GameRequest


class _FakeEngineLibrary:
    def __init__(self) -> None:
        self.next_pointer = 1
        self.steps: dict[int, int] = {}
        self.finished: list[int] = []
        self._buffers: list[ctypes.Array] = []

    @staticmethod
    def _pointer_value(pointer) -> int:
        return int(pointer.value if hasattr(pointer, "value") else pointer)

    def BattleStart(self, _cards):
        pointer = self.next_pointer
        self.next_pointer += 1
        self.steps[pointer] = 0
        return SimpleNamespace(battlePtr=pointer, errorPlayer=-1, errorType=0)

    def GetBattleData(self, pointer):
        value = self._pointer_value(pointer)
        step = self.steps[value]
        result = 0 if step >= 3 else -1
        payload = {
            "current": {"turn": step, "yourIndex": step % 2, "result": result},
            "select": None
            if result >= 0
            else {"type": 0, "option": [{"type": 14}], "minCount": 1, "maxCount": 1},
        }
        buffer = ctypes.create_string_buffer(b"")
        self._buffers.append(buffer)
        return SimpleNamespace(
            json=json.dumps(payload).encode(),
            data=ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
            count=0,
        )

    def Select(self, pointer, _action, _count):
        value = self._pointer_value(pointer)
        self.steps[value] += 1
        return 0

    def BattleFinish(self, pointer):
        self.finished.append(self._pointer_value(pointer))


def _package(root: Path, name: str, card: int) -> SubmissionPackage:
    return SubmissionPackage(
        name=name,
        root=root,
        deck=[card] * 60,
        entrypoint=root / "main.py",
        package_hash=f"{name}-package",
        deck_hash=f"{name}-deck",
        cg_manifest={"runtime": "fake"},
    )


class EvaluationEnginePoolWorkerTests(unittest.TestCase):
    def test_pointer_battles_keep_independent_state_and_finish_once(self) -> None:
        library = _FakeEngineLibrary()
        lock = threading.Lock()
        first = _PointerBattle(library, lock)
        second = _PointerBattle(library, lock)

        first_observation = first.start([1] * 60, [2] * 60)
        second_observation = second.start([3] * 60, [4] * 60)
        self.assertEqual(first_observation["current"]["turn"], 0)
        self.assertEqual(second_observation["current"]["turn"], 0)

        first.select([0])
        first_observation = first.select([0])
        second_observation = second.select([0])
        self.assertEqual(first_observation["current"]["turn"], 2)
        self.assertEqual(second_observation["current"]["turn"], 1)

        first.finish()
        first.finish()
        second.finish()
        self.assertEqual(library.finished, [1, 2])

    def test_pool_runs_two_games_concurrently_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _package(root, "candidate", 7)
            opponent = _package(root, "opponent", 8)
            jobs = []
            for index in range(2):
                request = GameRequest(
                    run_id="run-pool",
                    game_id=f"game-{index}",
                    candidate=candidate,
                    opponent=opponent,
                    candidate_first=index == 0,
                    max_steps=10,
                    visualize=False,
                )
                jobs.append((request, root / f"trace-{index}.json"))

            library = _FakeEngineLibrary()

            def agent_factory(request, _role):
                def agent(observation):
                    if observation["select"] is None:
                        return list(request.candidate.deck)
                    return [0]

                return agent, lambda: None

            results = run_pool_games(
                jobs,
                library=library,
                pool_size=2,
                agent_factory=agent_factory,
                validate_compatibility=False,
            )

            self.assertEqual([result.game_id for result in results], ["game-0", "game-1"])
            self.assertTrue(all(result.finished for result in results))
            self.assertTrue(all(result.steps == 3 for result in results))
            self.assertEqual(sorted(library.finished), [1, 2])
            self.assertTrue(all(trace.exists() for _, trace in jobs))


if __name__ == "__main__":
    unittest.main()
