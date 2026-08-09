from __future__ import annotations

import importlib
import unittest


module = importlib.import_module("train.0038_action_boundary_rl.semantic_parity.random_outcomes")


def payload() -> dict:
    return {"schema_version": module.RANDOM_OUTCOME_SCHEMA_VERSION,
            "shuffle_permutation": [2, 0, 3, 1], "prize_indices": [0, 3],
            "coin_results": [1, 0], "random_target_indices": [2],
            "effect_results": {"random_discard": 1}}


class RandomOutcomeSchemaTest(unittest.TestCase):
    def test_fixture_is_stable_and_consumption_fails_closed(self):
        fixture = module.RandomOutcomeFixture.from_mapping(payload())
        self.assertEqual(fixture, module.RandomOutcomeFixture.from_mapping(fixture.to_mapping()))
        self.assertEqual(len(fixture.sha256), 64)
        cursor = module.OutcomeCursor(fixture)
        self.assertEqual(cursor.coin(), 1)
        self.assertEqual(cursor.coin(), 0)
        self.assertEqual(cursor.target(3), 2)
        self.assertEqual(cursor.effect("random_discard"), 1)
        cursor.assert_fully_consumed()
        with self.assertRaisesRegex(RuntimeError, "underflow"):
            cursor.coin()

    def test_invalid_permutation_and_target_fail_closed(self):
        invalid = payload()
        invalid["shuffle_permutation"] = [0, 0, 1, 2]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            module.RandomOutcomeFixture.from_mapping(invalid)
        fixture = module.RandomOutcomeFixture.from_mapping(payload())
        with self.assertRaisesRegex(RuntimeError, "outside"):
            module.OutcomeCursor(fixture).target(2)

    def test_unused_values_fail(self):
        with self.assertRaisesRegex(RuntimeError, "unused"):
            module.OutcomeCursor(module.RandomOutcomeFixture.from_mapping(payload())).assert_fully_consumed()


if __name__ == "__main__":
    unittest.main()
