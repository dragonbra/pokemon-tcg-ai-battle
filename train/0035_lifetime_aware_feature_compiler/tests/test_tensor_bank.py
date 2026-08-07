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
INCREMENTAL = importlib.import_module(f"{BASE}.features.incremental")
EXTENDED = importlib.import_module(f"{BASE}.tests.extended_fixture")


def _records(count: int = 2):
    trajectory = BENCHMARK.load_parity_trajectories()[0]
    knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
    output = []
    for decision in trajectory.decisions[:count]:
        snapshot = knowledge.consume(decision.observation, decision.event_cursor)
        output.append(COMPILER.compile_canonical_row(
            decision.row(), snapshot, BENCHMARK.PROTOTYPES
        ))
    return output


class PersistentTensorBankTests(unittest.TestCase):
    def assert_batch_equal(self, actual, expected) -> None:
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            self.assertEqual(actual[key].dtype, expected[key].dtype, key)
            self.assertEqual(actual[key].shape, expected[key].shape, key)
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_first_materialization_matches_reference_collator(self) -> None:
        record = _records(1)[0]
        bank = TENSOR_BANK.PersistentTensorBank()
        self.assert_batch_equal(
            bank.collate(record), COLLATE.collate_canonical_records([record])
        )

    def test_unchanged_record_reuses_every_storage(self) -> None:
        record = _records(1)[0]
        bank = TENSOR_BANK.PersistentTensorBank()
        first = bank.collate(record)
        pointers = {key: value.untyped_storage().data_ptr() for key, value in first.items()}
        second = bank.collate(copy.deepcopy(record))
        self.assert_batch_equal(second, COLLATE.collate_canonical_records([record]))
        self.assertEqual(
            pointers,
            {key: value.untyped_storage().data_ptr() for key, value in second.items()},
        )
        stats = bank.stats.snapshot()
        self.assertGreater(stats["slot_hits"], 0)
        self.assertEqual(stats["collates"], 2)

    def test_global_only_change_does_not_rewrite_card_storage(self) -> None:
        record = _records(1)[0]
        changed = copy.deepcopy(record)
        changed["actor"]["global_num"][1] += 1.0
        bank = TENSOR_BANK.PersistentTensorBank()
        first = bank.collate(record)
        card_pointer = first["card_cat"].untyped_storage().data_ptr()
        before_writes = bank.stats.snapshot()["elements_written"]
        second = bank.collate(changed)
        self.assert_batch_equal(
            second, COLLATE.collate_canonical_records([changed])
        )
        self.assertEqual(
            card_pointer, second["card_cat"].untyped_storage().data_ptr()
        )
        self.assertGreater(bank.stats.snapshot()["elements_written"], before_writes)

    def test_length_growth_changes_only_affected_logical_views(self) -> None:
        first_record, second_record = _records(2)
        bank = TENSOR_BANK.PersistentTensorBank()
        first = bank.collate(first_record)
        global_pointer = first["global_cat"].untyped_storage().data_ptr()
        second = bank.collate(second_record)
        self.assert_batch_equal(
            second, COLLATE.collate_canonical_records([second_record])
        )
        self.assertEqual(
            global_pointer, second["global_cat"].untyped_storage().data_ptr()
        )

    def test_nonempty_to_empty_ragged_field_zeroes_placeholder(self) -> None:
        first_record, second_record = _records(2)
        self.assertTrue(first_record["actor"]["option_skill_id"])
        self.assertFalse(second_record["actor"]["option_skill_id"])
        bank = TENSOR_BANK.PersistentTensorBank()
        bank.collate(first_record)
        actual = bank.collate(second_record)
        expected = COLLATE.collate_canonical_records([second_record])
        for key in (
            "option_skill_id", "option_skill_role", "option_skill_parent",
            "option_effect_id", "option_effect_role", "option_effect_parent",
        ):
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_reset_discards_session_storage(self) -> None:
        record = _records(1)[0]
        bank = TENSOR_BANK.PersistentTensorBank()
        first = bank.collate(record)
        pointer = first["card_cat"].untyped_storage().data_ptr()
        bank.reset()
        second = bank.collate(record)
        self.assertNotEqual(pointer, second["card_cat"].untyped_storage().data_ptr())

    def test_extended_525_decisions_match_stateless_tensor_authority(self) -> None:
        checked = 0
        for trajectory in EXTENDED.load_extended_trajectories():
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            compiler = INCREMENTAL.IncrementalCanonicalCompiler(
                BENCHMARK.PROTOTYPES
            )
            bank = TENSOR_BANK.PersistentTensorBank()
            for decision in trajectory.decisions:
                snapshot = knowledge.consume(
                    decision.observation, decision.event_cursor
                )
                reference = COMPILER.compile_canonical_row(
                    decision.row(), snapshot, BENCHMARK.PROTOTYPES
                )
                candidate = compiler.compile(decision.row(), snapshot)
                self.assert_batch_equal(
                    bank.collate(candidate),
                    COLLATE.collate_canonical_records([reference]),
                )
                checked += 1
        self.assertEqual(checked, 525)


if __name__ == "__main__":
    unittest.main()
