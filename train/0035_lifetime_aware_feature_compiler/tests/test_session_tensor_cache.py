from __future__ import annotations

from dataclasses import replace
import importlib
import unittest

import torch


BASE = "train.0035_lifetime_aware_feature_compiler"
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")
COLLATE = importlib.import_module(f"{BASE}.features.collate")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
STATE = importlib.import_module(f"{BASE}.knowledge.state")
TENSOR_BANK = importlib.import_module(f"{BASE}.features.tensor_bank")
CACHE = importlib.import_module(f"{BASE}.deployment.session_tensor_cache")
EXTENDED = importlib.import_module(f"{BASE}.tests.extended_fixture")


class SessionTensorCacheTests(unittest.TestCase):
    def assert_batch_equal(self, actual, expected) -> None:
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.assertEqual(actual[name].dtype, expected[name].dtype, name)
            self.assertEqual(actual[name].shape, expected[name].shape, name)
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_interleaved_sessions_reconstruct_reference_batches(self) -> None:
        trajectories = EXTENDED.load_extended_trajectories()[:4]
        store = CACHE.GpuSessionTensorStore("cpu", torch.float32)
        sessions = []
        current_records = {}
        for index, trajectory in enumerate(trajectories):
            key = CACHE.SessionTensorKey(
                f"session-{index}", trajectory.actor, f"deck-{index}"
            )
            sessions.append(
                (
                    key,
                    trajectory,
                    STATE.CausalKnowledge(trajectory.actor, trajectory.deck),
                    TENSOR_BANK.PersistentTensorBank(),
                )
            )

        for decision_index in range(3):
            for key, trajectory, knowledge, bank in sessions:
                decision = trajectory.decisions[decision_index]
                snapshot = knowledge.consume(
                    decision.observation, decision.event_cursor
                )
                record = COMPILER.compile_canonical_row(
                    decision.row(), snapshot, BENCHMARK.PROTOTYPES
                )
                store.apply(key, bank.collate_delta(record))
                current_records[key] = record
            order = [item[0] for item in reversed(sessions)]
            actual = store.batch(order)
            expected = COLLATE.collate_canonical_records(
                [current_records[key] for key in order]
            )
            self.assert_batch_equal(actual, expected)

        stats = store.stats()
        self.assertEqual(stats["sessions"], 4)
        self.assertGreater(stats["delta_h2d_bytes"], 0)
        self.assertGreater(stats["d2d_batch_bytes"], 0)

    def test_lifecycle_and_malformed_updates_fail_closed(self) -> None:
        trajectory = BENCHMARK.load_parity_trajectories()[0]
        knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
        decision = trajectory.decisions[0]
        record = COMPILER.compile_canonical_row(
            decision.row(),
            knowledge.consume(decision.observation, decision.event_cursor),
            BENCHMARK.PROTOTYPES,
        )
        bank = TENSOR_BANK.PersistentTensorBank()
        delta = bank.collate_delta(record)
        key = CACHE.SessionTensorKey("one", trajectory.actor, "deck")
        store = CACHE.GpuSessionTensorStore("cpu", torch.float32)

        non_seed = replace(delta, full_reseed=False)
        with self.assertRaisesRegex(ValueError, "reseed"):
            store.apply(key, non_seed)
        store.apply(key, delta)
        with self.assertRaisesRegex(ValueError, "identity"):
            store.apply(
                CACHE.SessionTensorKey("one", 1 - trajectory.actor, "deck"),
                delta,
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            store.batch([key, key])

        patch = delta.patches[0]
        malformed = replace(
            patch,
            replace=False,
            rows=torch.tensor([999], dtype=torch.long),
            values=patch.values.new_zeros((1, *patch.values.shape[2:])),
        )
        with self.assertRaises((ValueError, RuntimeError)):
            store.apply(
                key,
                replace(delta, patches=(malformed,), full_reseed=False),
            )

        store.close_session(key)
        self.assertEqual(store.stats()["sessions"], 0)
        with self.assertRaisesRegex(KeyError, "session"):
            store.batch([key])
        store.apply(key, delta)
        store.clear("test")
        self.assertEqual(store.stats()["sessions"], 0)

    def test_float_inputs_are_cast_once_to_runtime_dtype(self) -> None:
        trajectory = BENCHMARK.load_parity_trajectories()[0]
        knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
        decision = trajectory.decisions[0]
        record = COMPILER.compile_canonical_row(
            decision.row(),
            knowledge.consume(decision.observation, decision.event_cursor),
            BENCHMARK.PROTOTYPES,
        )
        key = CACHE.SessionTensorKey("half", trajectory.actor, "deck")
        bank = TENSOR_BANK.PersistentTensorBank()
        store = CACHE.GpuSessionTensorStore("cpu", torch.float16)
        store.apply(key, bank.collate_delta(record))
        batch = store.batch([key])
        for name, value in batch.items():
            if name.endswith("_num"):
                self.assertEqual(value.dtype, torch.float16, name)


if __name__ == "__main__":
    unittest.main()
