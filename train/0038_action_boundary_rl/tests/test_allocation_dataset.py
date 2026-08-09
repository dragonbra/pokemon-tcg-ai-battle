from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace

import torch

dataset = importlib.import_module("train.0038_action_boundary_rl.data.allocation_dataset")
collector = importlib.import_module("train.0038_action_boundary_rl.rollout.collector")
protocol = importlib.import_module("train.0038_action_boundary_rl.rollout.protocol")


class AllocationDatasetTest(unittest.TestCase):
    def test_split_is_battle_level(self):
        self.assertEqual(dataset.split_for_battle("battle-1"), dataset.split_for_battle("battle-1"))

    def test_rejects_incomplete_label(self):
        with self.assertRaises(ValueError):
            dataset.AllocationBCSample("b", 1, "o", True, 1, {}, 0, ({"serial": 1},),
                                       (5,), (1, 1, 1, 1, 1), 0, 0)

    def test_shadow_reconstructs_order_into_one_canonical_label(self):
        job = protocol.RolloutJob("b", "opp", True, 7, 0, (1,) * 60, (2,) * 60,
                                  __import__("pathlib").Path("."), action_boundary_mode="shadow")
        session = collector._Session(job, torch.Generator())
        bench = [{"serial": 10, "id": 100, "hp": 50},
                 {"serial": 20, "id": 200, "hp": 100}]
        root = {"actor": 0, "turn": 3, "select": {"context": 0, "option": []}}
        collector.FullSemanticRolloutCollector._observe_shadow_allocation(
            session, root, SimpleNamespace(indices=(0,)), {"x": torch.zeros(1, 2)},
            0, 0, bench,
        )
        order = [20, 10, 20, 20, 10, 20]
        for serial in order:
            slot = 0 if serial == 10 else 1
            message = {"actor": 0, "turn": 3, "select": {
                "context": 14, "option": [{"playerIndex": 1, "index": slot}]}}
            collector.FullSemanticRolloutCollector._observe_shadow_allocation(
                session, message, SimpleNamespace(indices=(0,)), {"x": torch.zeros(1, 2)},
                0, None, bench,
            )
        self.assertEqual(len(session.allocation_bc_samples), 1)
        self.assertEqual(session.allocation_bc_samples[0].counters, (2, 4))
        self.assertEqual(session.allocation_bc_samples[0].primitive_order, tuple(order))

    def test_stale_pending_chain_fails_closed(self):
        job = protocol.RolloutJob("b", "opp", True, 7, 0, (1,) * 60, (2,) * 60,
                                  __import__("pathlib").Path("."), action_boundary_mode="shadow")
        session = collector._Session(job, torch.Generator())
        bench = [{"serial": 10, "id": 100, "hp": 50}]
        collector.FullSemanticRolloutCollector._observe_shadow_allocation(
            session, {"actor": 0, "turn": 3, "select": {"context": 0}},
            SimpleNamespace(indices=(0,)), {"x": torch.zeros(1, 2)}, 0, 0, bench,
        )
        collector.FullSemanticRolloutCollector._observe_shadow_allocation(
            session, {"actor": 0, "turn": 3, "select": {"context": 9}},
            SimpleNamespace(indices=(0,)), {"x": torch.zeros(1, 2)}, 0, None, bench,
        )
        self.assertIsNone(session.pending_shadow_allocation)
        self.assertEqual(session.allocation_bc_invalid, 1)

    def test_expanded_bench_remains_one_macro_label(self):
        job = protocol.RolloutJob("b", "opp", True, 7, 0, (1,) * 60, (2,) * 60,
                                  __import__("pathlib").Path("."), action_boundary_mode="shadow")
        session = collector._Session(job, torch.Generator())
        bench = [{"serial": i, "id": 100 + i, "hp": 100} for i in range(6)]
        collector.FullSemanticRolloutCollector._observe_shadow_allocation(
            session, {"actor": 0, "turn": 3, "select": {"context": 0}},
            SimpleNamespace(indices=(0,)), {"x": torch.zeros(1, 2)}, 0, 0, bench,
        )
        self.assertIsNotNone(session.pending_shadow_allocation)
        self.assertEqual(len(session.pending_shadow_allocation["identities"]), 6)
        self.assertEqual(session.allocation_bc_invalid_reasons, {})


if __name__ == "__main__":
    unittest.main()
