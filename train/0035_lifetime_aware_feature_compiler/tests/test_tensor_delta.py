from __future__ import annotations

import copy
import importlib
import unittest

import torch


BASE = "train.0035_lifetime_aware_feature_compiler"
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")
COLLATE = importlib.import_module(f"{BASE}.features.collate")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
STATE = importlib.import_module(f"{BASE}.knowledge.state")
TENSOR_BANK = importlib.import_module(f"{BASE}.features.tensor_bank")
EXTENDED = importlib.import_module(f"{BASE}.tests.extended_fixture")


def _records(count: int = 2) -> list[dict]:
    trajectory = BENCHMARK.load_parity_trajectories()[0]
    knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
    output = []
    for decision in trajectory.decisions[:count]:
        snapshot = knowledge.consume(decision.observation, decision.event_cursor)
        output.append(
            COMPILER.compile_canonical_row(
                decision.row(), snapshot, BENCHMARK.PROTOTYPES
            )
        )
    return output


def _apply(
    resident: dict[str, torch.Tensor], delta: object
) -> dict[str, torch.Tensor]:
    for patch in delta.patches:
        if patch.replace or patch.name not in resident:
            resident[patch.name] = patch.values.clone()
            continue
        previous = resident[patch.name]
        target = previous.new_zeros(patch.logical_shape)
        common = tuple(min(a, b) for a, b in zip(previous.shape, target.shape))
        slices = tuple(slice(0, length) for length in common)
        target[slices] = previous[slices]
        if patch.rows.numel():
            target[0].index_copy_(0, patch.rows, patch.values)
        resident[patch.name] = target
    return resident


class TensorDeltaTests(unittest.TestCase):
    def assert_batch_equal(self, actual, expected) -> None:
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.assertEqual(actual[name].dtype, expected[name].dtype, name)
            self.assertEqual(actual[name].shape, expected[name].shape, name)
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_first_reseeds_every_key_and_unchanged_record_has_no_patches(self) -> None:
        record = _records(1)[0]
        bank = TENSOR_BANK.PersistentTensorBank()
        first = bank.collate_delta(record)
        expected = COLLATE.collate_canonical_records([record])
        self.assertTrue(first.full_reseed)
        self.assertEqual({patch.name for patch in first.patches}, set(expected))
        self.assertTrue(all(patch.replace for patch in first.patches))
        self.assert_batch_equal(first.tensors, expected)

        second = bank.collate_delta(copy.deepcopy(record))
        self.assertFalse(second.full_reseed)
        self.assertEqual(second.patches, ())
        self.assert_batch_equal(second.tensors, expected)

    def test_sparse_growth_shrink_and_empty_transitions_reconstruct_exactly(self) -> None:
        first_record, second_record = _records(2)
        bank = TENSOR_BANK.PersistentTensorBank()
        resident: dict[str, torch.Tensor] = {}
        for record in (first_record, second_record, first_record):
            delta = bank.collate_delta(record)
            _apply(resident, delta)
            self.assert_batch_equal(
                resident, COLLATE.collate_canonical_records([record])
            )

        changed = copy.deepcopy(first_record)
        changed["actor"]["global_num"][1] += 1.0
        delta = bank.collate_delta(changed)
        self.assertEqual(
            {patch.name for patch in delta.patches}, {"global_num"}
        )
        _apply(resident, delta)
        self.assert_batch_equal(
            resident, COLLATE.collate_canonical_records([changed])
        )

    def test_extended_chronology_reconstructs_all_39_tensors(self) -> None:
        checked = 0
        avoided_bytes = 0
        full_bytes = 0
        for trajectory in EXTENDED.load_extended_trajectories():
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            bank = TENSOR_BANK.PersistentTensorBank()
            resident: dict[str, torch.Tensor] = {}
            for decision in trajectory.decisions:
                snapshot = knowledge.consume(
                    decision.observation, decision.event_cursor
                )
                record = COMPILER.compile_canonical_row(
                    decision.row(), snapshot, BENCHMARK.PROTOTYPES
                )
                expected = COLLATE.collate_canonical_records([record])
                delta = bank.collate_delta(record)
                _apply(resident, delta)
                self.assert_batch_equal(resident, expected)
                total = sum(
                    value.numel() * value.element_size()
                    for value in expected.values()
                )
                transferred = sum(
                    patch.values.numel() * patch.values.element_size()
                    for patch in delta.patches
                )
                full_bytes += total
                avoided_bytes += total - transferred
                checked += 1
        self.assertEqual(checked, 525)
        self.assertGreater(avoided_bytes, 0)
        self.assertGreater(avoided_bytes / full_bytes, 0.25)


if __name__ == "__main__":
    unittest.main()
