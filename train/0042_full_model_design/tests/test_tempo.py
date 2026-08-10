from __future__ import annotations

import importlib
import unittest

tempo = importlib.import_module("train.0042_full_model_design.integrated.tempo")
pool = importlib.import_module("train.0042_full_model_design.rollout.pool_worker")


class TempoTest(unittest.TestCase):
    def test_detector_uses_current_legal_options(self):
        self.assertTrue(tempo.TempoTracker.attack_opportunity([{"attackId": 154}]))
        self.assertFalse(tempo.TempoTracker.attack_opportunity([{"action": "evolve"}]))

    def test_second_turn_and_streak(self):
        tracker = tempo.TempoTracker()
        for turn, attacked in [(1, True), (2, True), (3, False)]:
            tracker.add(tempo.TempoEvent(turn, True, "x", True, attacked))
        summary = tracker.summary()
        self.assertEqual(summary["second_turn_attacked"], 1)
        self.assertEqual(summary["longest_attack_streak"], 2)

    def test_no_future_state_argument_exists(self):
        with self.assertRaises(TypeError):
            tempo.TempoTracker.attack_opportunity([], future_deck=[1])

    def test_trace_aggregation_uses_existing_selects_only(self):
        trace = [
            {"observation": {"current": {"yourIndex": 0, "turn": 2},
                             "select": {"option": [{"attackId": 154}, {"id": 1}]}}},
            {"primitive_select": [0], "observation": {
                "current": {"yourIndex": 1, "turn": 3}, "select": {"option": []}}},
        ]
        self.assertEqual(pool._tempo_from_trace(trace), [
            {"own_turn_index": 1, "attack_legal": True, "attacked": True}
        ])


if __name__ == "__main__":
    unittest.main()
