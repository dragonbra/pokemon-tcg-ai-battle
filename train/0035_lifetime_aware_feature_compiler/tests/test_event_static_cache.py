from __future__ import annotations

import copy
import importlib
import unittest

import torch


BASE = "train.0035_lifetime_aware_feature_compiler"
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")
CACHE = importlib.import_module(f"{BASE}.deployment.event_static_cache")
COLLATE = importlib.import_module(f"{BASE}.features.collate")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
EXTENDED = importlib.import_module(f"{BASE}.tests.extended_fixture")
SESSION = importlib.import_module(f"{BASE}.deployment.session_tensor_cache")
STATE = importlib.import_module(f"{BASE}.knowledge.state")


class EventStaticCacheTests(unittest.TestCase):
    def assert_event_equal(self, actual, expected) -> None:
        self.assertEqual(set(actual), CACHE.EventStaticStore.EVENT_KEYS)
        for name in CACHE.EventStaticStore.EVENT_KEYS:
            self.assertEqual(actual[name].dtype, expected[name].dtype, name)
            self.assertEqual(actual[name].shape, expected[name].shape, name)
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_all_extended_decisions_match_stateless_event_tensors(self) -> None:
        checked = 0
        store = CACHE.EventStaticStore("cpu", torch.float32, max_sessions=16)
        for trajectory_index, trajectory in enumerate(
            EXTENDED.load_extended_trajectories()
        ):
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            key = SESSION.SessionTensorKey(
                f"trajectory-{trajectory_index}", trajectory.actor, f"deck-{trajectory_index}"
            )
            for decision in trajectory.decisions:
                snapshot = knowledge.consume(decision.observation, decision.event_cursor)
                record = COMPILER.compile_canonical_row(
                    decision.row(), snapshot, BENCHMARK.PROTOTYPES
                )
                identities = tuple(event.source_event for event in snapshot.recent_events)
                actual = store.update_batch([key], [record], [identities])
                expected = COLLATE.collate_canonical_records([record])
                self.assert_event_equal(actual, expected)
                checked += 1
            store.close_session(key)
        self.assertEqual(checked, 525)
        stats = store.stats()
        self.assertGreater(stats["static_hits"], stats["static_misses"])
        self.assertGreater(stats["static_upload_bytes"], 0)
        self.assertGreater(stats["dynamic_upload_bytes"], 0)

    def test_interleaved_sessions_and_immutable_commitment_guard(self) -> None:
        trajectories = EXTENDED.load_extended_trajectories()[:4]
        records = []
        identities = []
        keys = []
        for index, trajectory in enumerate(trajectories):
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            decision = trajectory.decisions[0]
            snapshot = None
            for candidate in trajectory.decisions:
                candidate_snapshot = knowledge.consume(
                    candidate.observation, candidate.event_cursor
                )
                decision = candidate
                snapshot = candidate_snapshot
                if snapshot.recent_events:
                    break
            self.assertIsNotNone(snapshot)
            record = COMPILER.compile_canonical_row(
                decision.row(), snapshot, BENCHMARK.PROTOTYPES
            )
            records.append(record)
            identities.append(tuple(event.source_event for event in snapshot.recent_events))
            keys.append(SESSION.SessionTensorKey(f"s-{index}", trajectory.actor, f"d-{index}"))

        store = CACHE.EventStaticStore("cpu", torch.float32, max_sessions=4)
        actual = store.update_batch(keys, records, identities)
        expected = COLLATE.collate_canonical_records(records)
        self.assert_event_equal(actual, expected)

        index = next(i for i, values in enumerate(identities) if values)
        changed = copy.deepcopy(records[index])
        changed["actor"]["event_cat"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "immutable event payload changed"):
            store.update_batch([keys[index]], [changed], [identities[index]])


if __name__ == "__main__":
    unittest.main()
