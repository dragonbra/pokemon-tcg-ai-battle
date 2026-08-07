from __future__ import annotations

import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from evaluation.runner.compiler_pool import CompilerPool, shard_for_session


class CompilerPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for relative in ("strategy", "strategy/model"):
            path = self.root / relative
            path.mkdir(parents=True, exist_ok=True)
            (path / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "strategy/model/config.py").write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass
                @dataclass
                class ModelConfig:
                    delay: float = 0.0
                """
            ),
            encoding="utf-8",
        )
        (self.root / "strategy/online_runtime.py").write_text(
            textwrap.dedent(
                """
                import time
                def collate_canonical_records(records):
                    return records
                class OnlineCausalEncoder:
                    def __init__(self, actor, deck, config):
                        self.actor = actor
                        self.delay = config.delay
                        self.step = 0
                    def encode(self, observation):
                        time.sleep(self.delay)
                        self.step += 1
                        return {"actor": self.actor, "step": self.step, "value": observation["value"]}
                """
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def request(session_id: str, value: int) -> dict:
        return {
            "session_id": session_id,
            "actor": 0,
            "deck": [1] * 60,
            "observation": {"value": value},
        }

    def distinct_sessions(self) -> tuple[str, str]:
        first = "session-0"
        for index in range(1, 100):
            second = f"session-{index}"
            if shard_for_session(first, 2) != shard_for_session(second, 2):
                return first, second
        raise AssertionError("could not find sessions on distinct shards")

    def test_session_hash_is_stable_and_validates_inputs(self) -> None:
        self.assertEqual(shard_for_session("battle-a", 4), shard_for_session("battle-a", 4))
        self.assertIn(shard_for_session("battle-a", 4), range(4))
        with self.assertRaises(ValueError):
            shard_for_session("", 4)
        with self.assertRaises(ValueError):
            shard_for_session("battle-a", 0)

    def test_pool_restores_order_and_preserves_session_chronology(self) -> None:
        first, second = self.distinct_sessions()
        with CompilerPool(self.root, {"delay": 0.0}, workers=2) as pool:
            initial = pool.encode([self.request(second, 2), self.request(first, 1)])
            following = pool.encode([self.request(first, 3), self.request(second, 4)])
            pool.close_session(first)
            reset = pool.encode([self.request(first, 5)])

        self.assertEqual([row["value"] for row in initial], [2, 1])
        self.assertEqual([row["step"] for row in initial], [1, 1])
        self.assertEqual([row["step"] for row in following], [2, 2])
        self.assertEqual(reset[0]["step"], 1)

    def test_distinct_shards_compile_concurrently(self) -> None:
        first, second = self.distinct_sessions()
        with CompilerPool(self.root, {"delay": 0.15}, workers=2) as pool:
            started = time.perf_counter()
            rows = pool.encode([self.request(first, 1), self.request(second, 2)])
            elapsed = time.perf_counter() - started

        self.assertEqual(len(rows), 2)
        self.assertLess(elapsed, 0.27)


if __name__ == "__main__":
    unittest.main()
