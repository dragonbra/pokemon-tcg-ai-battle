from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import threading
import unittest


PROJECT = "train.0042_full_model_design"
POOL = importlib.import_module(f"{PROJECT}.rollout.pool_worker")
COLLECTOR = importlib.import_module(f"{PROJECT}.rollout.collector")


class _FakeConnection:
    def __init__(self) -> None:
        self.payload = None
        self.closed = False

    def send(self, payload):
        self.payload = payload

    def recv(self):
        return {"kind": "action", "action": [int(self.payload["ordinal"])]}

    def close(self):
        self.closed = True


class _FakeLibrary:
    def __init__(self) -> None:
        self.finished = []

    def BattleFinish(self, pointer):
        self.finished.append(pointer)

    def ConfigureSeeds(self, engine_seed, search_seed):
        self.configured = (engine_seed, search_seed)
        return 0

    def BattleStartSeeded(self, argument, seed):
        return type("Started", (), {"battle_ptr": 123})()


class RolloutPoolTest(unittest.TestCase):
    def test_channel_pool_routes_interleaved_requests_and_closes_once(self) -> None:
        connections = [_FakeConnection(), _FakeConnection()]
        pool = POOL._ChannelPool(connections)
        with ThreadPoolExecutor(max_workers=8) as executor:
            actions = list(executor.map(
                lambda ordinal: pool.exchange({
                    "kind": "decision",
                    "session_id": f"game-{ordinal}",
                    "ordinal": ordinal,
                })["action"],
                range(32),
            ))
        self.assertEqual(actions, [[ordinal] for ordinal in range(32)])
        pool.close()
        self.assertTrue(all(connection.closed for connection in connections))

    def test_pointer_battle_finishes_exactly_once(self) -> None:
        library = _FakeLibrary()
        battle = POOL._PointerBattle(library, threading.Lock())
        battle._pointer = 123
        battle.finish()
        battle.finish()
        self.assertEqual(library.finished, [123])

    def test_pointer_battle_configures_engine_and_search_seed_atomically(self) -> None:
        class Battle(POOL._PointerBattle):
            def _observation(self):
                return {"current": {}}
        library = _FakeLibrary()
        battle = Battle(library, threading.Lock())
        battle.start((1,) * 60, (2,) * 60, 1234, 5678)
        self.assertEqual(library.configured, (1234, 5678))

    def test_collector_rejects_more_channels_than_engines(self) -> None:
        class _Model:
            def eval(self):
                return self

        with self.assertRaisesRegex(ValueError, "invalid collector configuration"):
            COLLECTOR.FullSemanticRolloutCollector(
                _Model(),
                _Model(),
                device=None,
                worker_processes=2,
                engines_per_worker=2,
                inference_channels_per_role=3,
            )

    def test_confusion_is_a_chance_boundary_before_phantom_allocation(self) -> None:
        current = {"players": [{"confused": True}, {"confused": False}]}
        self.assertTrue(POOL._chance_before_phantom_allocation(current, 0))
        self.assertFalse(POOL._chance_before_phantom_allocation(current, 1))

    def test_phantom_macro_is_not_precommitted_across_chance_boundary(self) -> None:
        self.assertFalse(COLLECTOR.phantom_macro_eligible(
            phantom_root=3, target_count=5, action_boundary_mode="enabled",
            chance_before_allocation=True,
        ))
        self.assertTrue(COLLECTOR.phantom_macro_eligible(
            phantom_root=3, target_count=5, action_boundary_mode="enabled",
            chance_before_allocation=False,
        ))


if __name__ == "__main__":
    unittest.main()
