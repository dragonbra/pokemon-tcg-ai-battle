from __future__ import annotations

import importlib
import unittest

contract = importlib.import_module("train.0013_semantic_goal_policy.features.action_contract")


class OrderedActionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.options = [
            {"type": 3, "playerIndex": 0, "area": 2, "index": 0, "cardId": 101},
            {"type": 3, "playerIndex": 0, "area": 2, "index": 1, "cardId": 102},
            {"type": 7, "playerIndex": 0, "area": 2, "index": 2, "cardId": 103},
        ]

    def test_preserves_selected_order_and_optional_stop(self) -> None:
        action = contract.validate_ordered_action(
            [2, 0], option_count=3, min_count=1, max_count=3, decoder_capacity=4
        )
        self.assertEqual(action.indices, (2, 0))
        self.assertEqual(action.termination, contract.ActionTermination.OPTIONAL_STOP)
        self.assertTrue(action.has_explicit_stop)

    def test_maximum_count_forces_termination_not_stop(self) -> None:
        action = contract.validate_ordered_action(
            [0, 2], option_count=3, min_count=1, max_count=2, decoder_capacity=4
        )
        self.assertEqual(action.termination, contract.ActionTermination.FORCED_MAX)
        self.assertFalse(action.has_explicit_stop)

    def test_rejects_duplicate_bounds_counts_and_capacity(self) -> None:
        cases = (
            ([0, 0], 3, 0, 3, 4, "distinct"),
            ([3], 3, 0, 3, 4, "out of range"),
            ([], 3, 1, 3, 4, "minimum"),
            ([0, 1, 2], 3, 0, 2, 4, "maximum"),
            ([0, 1, 2], 3, 0, 3, 2, "decoder capacity"),
        )
        for indices, count, minimum, maximum, capacity, message in cases:
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, message):
                    contract.validate_ordered_action(
                        indices, option_count=count, min_count=minimum, max_count=maximum,
                        decoder_capacity=capacity,
                    )

    def test_rejects_boolean_float_and_string_indices_or_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "integers"):
            contract.validate_ordered_action([True], option_count=1, min_count=0, max_count=1,
                                             decoder_capacity=1)
        for keyword in ("option_count", "min_count", "max_count", "decoder_capacity"):
            kwargs = dict(option_count=1, min_count=0, max_count=1, decoder_capacity=1)
            kwargs[keyword] = 1.0 if keyword != "min_count" else "0"
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(ValueError, "integer"):
                    contract.validate_ordered_action([], **kwargs)

    def test_semantic_identity_accepts_only_mapping_and_exact_integer_or_null_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "mapping"):
            contract.OptionSemanticIdentity.from_option([])
        for invalid in (True, 1.0, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "type"):
                    contract.OptionSemanticIdentity.from_option({"type": invalid})
        identity = contract.OptionSemanticIdentity.from_option({"type": 1, "count": None})
        self.assertEqual(identity.type, 1)
        self.assertIsNone(identity.count)

    def test_metadata_and_unknown_keys_do_not_change_semantic_identity(self) -> None:
        base = {"type": 1, "playerIndex": 0}
        metadata = {**base, "serial": 99}
        unknown = {**base, "futureEngineField": "unreviewed"}
        self.assertEqual(
            contract.OptionSemanticIdentity.from_option(base),
            contract.OptionSemanticIdentity.from_option(metadata),
        )
        self.assertEqual(
            contract.OptionSemanticIdentity.from_option(base),
            contract.OptionSemanticIdentity.from_option(unknown),
        )
        self.assertEqual(
            contract.unrecognized_option_fields([metadata, unknown]),
            {"futureEngineField": 1},
        )
        self.assertNotIn("serial", contract.OptionSemanticIdentity.__dataclass_fields__)
        self.assertNotIn("futureEngineField", contract.OptionSemanticIdentity.__dataclass_fields__)

    def test_unknown_field_audit_rejects_non_string_keys_deliberately(self) -> None:
        for options, option_index, key in (([{1: "x"}], 0, 1), ([{"future": 1, 2: "x"}], 0, 2)):
            with self.subTest(options=options):
                with self.assertRaisesRegex(
                    ValueError, rf"option {option_index} has non-string key {key!r}"
                ):
                    contract.unrecognized_option_fields(options)

    def test_unknown_field_audit_leaves_arbitrary_unknown_values_inert(self) -> None:
        self.assertEqual(
            contract.unrecognized_option_fields(
                [{"future": {"nested": [object()]}, "another": None}]
            ),
            {"another": 1, "future": 1},
        )

    def test_rejects_contradictory_target_player_aliases(self) -> None:
        with self.assertRaisesRegex(ValueError, "contradictory"):
            contract.OptionSemanticIdentity.from_option(
                {"inPlayPlayerIndex": 0, "targetPlayerIndex": 1}
            )

    def test_occurrence_identity_is_internal_and_not_feature_export(self) -> None:
        package = importlib.import_module("train.0013_semantic_goal_policy.features")
        self.assertNotIn("OptionOccurrenceIdentity", package.__all__)
        self.assertNotIn("occurrence", contract.OptionSemanticIdentity.__dataclass_fields__)
        self.assertNotIn("serial", contract.OptionSemanticIdentity.__dataclass_fields__)

    def test_duplicate_occurrences_follow_original_option_through_known_swap(self) -> None:
        duplicate = {"type": 3, "playerIndex": 0, "area": 2, "index": 0, "cardId": 101}
        original = [duplicate, duplicate, self.options[1]]
        original_occurrences = contract.build_occurrence_identities(original)
        action = contract.validate_ordered_action(
            [0, 1], option_count=3, min_count=1, max_count=3, decoder_capacity=4
        )

        remapped, carried = contract.remap_action(
            action, old_to_new=(1, 0, 2), original_occurrences=original_occurrences
        )

        self.assertEqual(remapped.indices, (1, 0))
        self.assertEqual(carried[0], original_occurrences[1])
        self.assertEqual(carried[1], original_occurrences[0])
        self.assertEqual([identity.occurrence for identity in carried], [1, 0, 0])

    def test_remap_round_trip_and_optional_target_validation(self) -> None:
        original_occurrences = contract.build_occurrence_identities(self.options)
        action = contract.validate_ordered_action(
            [2, 0], option_count=3, min_count=1, max_count=3, decoder_capacity=4
        )
        remapped, carried = contract.remap_action(
            action, old_to_new=(1, 2, 0), original_occurrences=original_occurrences,
            target_options=[self.options[2], self.options[0], self.options[1]],
        )
        restored, _ = contract.remap_action(
            remapped, old_to_new=(2, 0, 1), original_occurrences=carried
        )
        self.assertEqual(remapped.indices, (0, 1))
        self.assertEqual(restored, action)

    def test_remap_rejects_non_bijective_wrong_length_and_bad_target_semantics(self) -> None:
        occurrences = contract.build_occurrence_identities(self.options)
        action = contract.validate_ordered_action(
            [0], option_count=3, min_count=1, max_count=3, decoder_capacity=4
        )
        for permutation in ((0, 0, 2), (0, 1)):
            with self.subTest(permutation=permutation):
                with self.assertRaisesRegex(ValueError, "bijection"):
                    contract.remap_action(action, old_to_new=permutation,
                                          original_occurrences=occurrences)
        with self.assertRaisesRegex(ValueError, "target options"):
            contract.remap_action(action, old_to_new=(1, 2, 0), original_occurrences=occurrences,
                                  target_options=self.options)


if __name__ == "__main__":
    unittest.main()
