from __future__ import annotations

import importlib
import unittest


diagnostic = importlib.import_module(
    "train.0042_full_model_design.diagnostics.fp32_fp16_u40_cpu2048_pairwise"
)
action_difference = diagnostic.action_difference
compare_outcomes = diagnostic.compare_outcomes
merge_action_summaries = diagnostic.merge_action_summaries


class OutcomeComparisonTest(unittest.TestCase):
    def test_counts_directional_outcome_transitions(self) -> None:
        reference = [
            {"game_id": "g1", "outcome": 1},
            {"game_id": "g2", "outcome": -1},
            {"game_id": "g3", "outcome": 0},
            {"game_id": "g4", "outcome": 1},
        ]
        candidate = [
            {"game_id": "g1", "outcome": -1},
            {"game_id": "g2", "outcome": 1},
            {"game_id": "g3", "outcome": 1},
            {"game_id": "g4", "outcome": 1},
        ]
        result = compare_outcomes(reference, candidate)
        self.assertEqual(result["games"], 4)
        self.assertEqual(result["unchanged_games"], 1)
        self.assertEqual(result["changed_games"], 3)
        self.assertEqual(result["win_to_loss"], 1)
        self.assertEqual(result["loss_to_win"], 1)
        self.assertEqual(result["draw_to_win"], 1)
        self.assertEqual(result["transition_matrix"]["W"]["L"], 1)
        self.assertEqual(result["changed_game_ids"], ["g1", "g2", "g3"])

    def test_rejects_duplicate_or_mismatched_game_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_outcomes(
                [{"game_id": "g1", "outcome": 1}, {"game_id": "g1", "outcome": 1}],
                [{"game_id": "g1", "outcome": 1}],
            )
        with self.assertRaisesRegex(ValueError, "game-ID sets differ"):
            compare_outcomes(
                [{"game_id": "g1", "outcome": 1}],
                [{"game_id": "g2", "outcome": 1}],
            )


class ActionComparisonTest(unittest.TestCase):
    def test_separates_engine_root_decoder_stop_and_allocation_flips(self) -> None:
        same_root_different_allocation = action_difference(
            fp32_indices=(2,), fp32_stopped=True,
            fp16_indices=(2,), fp16_stopped=True,
            fp32_macro={"targets": [10, 20], "counters": [6, 0]},
            fp16_macro={"targets": [10, 20], "counters": [0, 6]},
        )
        self.assertFalse(same_root_different_allocation["engine_root_action_flip"])
        self.assertFalse(same_root_different_allocation["decoder_action_flip"])
        self.assertTrue(same_root_different_allocation["allocation_comparable"])
        self.assertTrue(same_root_different_allocation["allocation_flip"])
        self.assertEqual(same_root_different_allocation["primitive_slots_examined"], 6)
        self.assertEqual(same_root_different_allocation["primitive_slots_flipped"], 6)

        stop_only = action_difference(
            fp32_indices=(3,), fp32_stopped=True,
            fp16_indices=(3,), fp16_stopped=False,
            fp32_macro=None, fp16_macro=None,
        )
        self.assertFalse(stop_only["engine_root_action_flip"])
        self.assertTrue(stop_only["decoder_action_flip"])
        self.assertTrue(stop_only["stop_flip"])

    def test_merges_action_summaries_without_losing_denominators(self) -> None:
        first = {
            "strategic_decisions_examined": 2,
            "engine_root_action_flips": 1,
            "decoder_action_flips": 1,
            "first_token_flips": 1,
            "stop_flips": 0,
            "root_token_slots_examined": 3,
            "root_token_slots_flipped": 1,
            "allocation_decisions_examined": 0,
            "allocation_flips": 0,
            "primitive_slots_examined": 0,
            "primitive_slots_flipped": 0,
            "games_examined": ["g1"],
            "affected_game_ids": ["g1"],
            "flip_events": [{"game_id": "g1"}],
        }
        second = {
            **first,
            "strategic_decisions_examined": 3,
            "engine_root_action_flips": 0,
            "decoder_action_flips": 0,
            "first_token_flips": 0,
            "allocation_decisions_examined": 1,
            "allocation_flips": 1,
            "primitive_slots_examined": 6,
            "primitive_slots_flipped": 2,
            "games_examined": ["g2"],
            "affected_game_ids": ["g2"],
            "flip_events": [{"game_id": "g2"}],
        }
        merged = merge_action_summaries([first, second])
        self.assertEqual(merged["strategic_decisions_examined"], 5)
        self.assertEqual(merged["engine_root_action_flips"], 1)
        self.assertEqual(merged["allocation_flips"], 1)
        self.assertEqual(merged["games_examined"], 2)
        self.assertEqual(merged["affected_games"], 2)
        self.assertEqual(len(merged["flip_events"]), 2)

    def test_merges_persisted_chunk_cardinalities_without_collisions(self) -> None:
        merged = merge_action_summaries(
            [
                {"games_examined": 32, "strategic_decisions_examined": 100},
                {"games_examined": 32, "strategic_decisions_examined": 120},
            ]
        )
        self.assertEqual(merged["games_examined"], 64)
        self.assertEqual(merged["strategic_decisions_examined"], 220)


if __name__ == "__main__":
    unittest.main()
