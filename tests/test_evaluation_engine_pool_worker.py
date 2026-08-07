from __future__ import annotations

import ctypes
import json
from collections import Counter
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.packages.loader import SubmissionPackage
from evaluation.runner.engine_pool_worker import (
    _PointerBattle,
    _RemotePolicyChannelPool,
    run_pool_games,
)
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
    def test_profiled_channel_records_ipc_stages_and_client_timestamp(self) -> None:
        class Connection:
            def __init__(self):
                self.sent = []

            def send(self, payload):
                self.sent.append(payload)

            def recv(self):
                return {"ok": True, "action": [3]}

            def close(self):
                pass

        connection = Connection()
        pool = _RemotePolicyChannelPool(
            "/unused",
            "candidate",
            channel_count=1,
            profile=True,
            connection_factory=lambda _path: connection,
        )
        timing = Counter()

        action = pool.call("game-a:candidate", {"select": {}}, [1] * 60, timing)

        self.assertEqual(action, [3])
        self.assertIsInstance(connection.sent[0]["_profile_client_send_ns"], int)
        self.assertEqual(timing["ipc_calls"], 1)
        self.assertGreaterEqual(timing["ipc_roundtrip_seconds"], 0.0)
        self.assertGreaterEqual(timing["ipc_send_seconds"], 0.0)

    def test_worker_local_compiler_does_not_import_torch(self) -> None:
        policy_root = (
            Path(__file__).resolve().parents[1]
            / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1"
            / "policies/policy_0806"
        )
        source = """
import sys
from pathlib import Path
from evaluation.runner.engine_pool_worker import _WorkerLocalCompilerFactory
factory = _WorkerLocalCompilerFactory(
    Path(sys.argv[1]),
    {"max_options": 128, "max_action_steps": 64},
)
assert "torch" not in sys.modules
assert factory._prototypes is not None
"""
        result = subprocess.run(
            [sys.executable, "-c", source, str(policy_root)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_0035_incremental_worker_compiler_is_torch_free_and_record_exact(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        policy_root = (
            repository
            / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1"
            / "policies/policy_0806"
        )
        fixture = (
            repository
            / "train/0035_lifetime_aware_feature_compiler/tests/fixtures"
            / "incremental_feature_trajectory.json.gz"
        )
        source = r"""
import gzip
import json
import sys
from pathlib import Path
from evaluation.runner.engine_pool_worker import _WorkerLocalCompilerFactory

policy_root = Path(sys.argv[1])
with gzip.open(sys.argv[2], "rt", encoding="utf-8") as handle:
    trajectory = json.load(handle)["trajectories"][0]
deck = [
    card_id
    for card_id, count in trajectory["deck_manifest"]["counts"]
    for _ in range(count)
]
config = {"max_options": 1024, "max_action_steps": 64}
stateless = _WorkerLocalCompilerFactory(policy_root, config).create(
    trajectory["actor"], deck
)
incremental = _WorkerLocalCompilerFactory(
    policy_root, config, backend="0035_incremental"
).create(trajectory["actor"], deck)
for decision in trajectory["decisions"]:
    observation = decision["actor_observation"]
    assert incremental.encode_record(observation) == stateless.encode_record(observation)
assert "torch" not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", source, str(policy_root), str(fixture)],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_channel_pool_reuses_connections_with_explicit_sessions(self) -> None:
        class Connection:
            def __init__(self):
                self.sent = []

            def send(self, payload):
                self.sent.append(payload)

            def recv(self):
                if self.sent[-1].get("command") == "close_session":
                    return {"ok": True}
                return {"ok": True, "action": [len(self.sent)]}

            def close(self):
                self.sent.append({"command": "connection_closed"})

        connections = [Connection(), Connection()]
        pool = _RemotePolicyChannelPool(
            "/unused",
            "candidate",
            channel_count=2,
            connection_factory=lambda _path: connections.pop(0),
        )

        first = pool.call("game-a:candidate", {"select": {}}, [1] * 60)
        second = pool.call("game-b:candidate", {"select": {}}, [2] * 60)
        pool.close_session("game-a:candidate")
        pool.close()

        self.assertEqual(first, [1])
        self.assertEqual(second, [1])
        sent = [row for connection in pool._connections for row in connection.sent]
        request_sessions = [row["session_id"] for row in sent if "observation" in row]
        self.assertEqual(request_sessions, ["game-a:candidate", "game-b:candidate"])
        self.assertEqual(
            [row for row in sent if row.get("command") == "close_session"],
            [{"command": "close_session", "session_id": "game-a:candidate"}],
        )

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
