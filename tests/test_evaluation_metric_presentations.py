from __future__ import annotations

import unittest

from evaluation.metrics import AggregateMetric
from evaluation.metrics.attack_quality import AttackQualityPlugin
from evaluation.metrics.setup_relay import SetupRelayPlugin


class EvaluationMetricPresentationTests(unittest.TestCase):
    def test_setup_relay_presentation_exposes_bridge_and_draw_sections(self) -> None:
        aggregate = AggregateMetric(
            "setup_relay",
            1,
            2,
            0.5,
            {},
            {
                "dunsparce_bridge": {"numerator": 1, "denominator": 2},
                "second_turn_draws": {
                    "all_games": {"average": 2.0},
                    "normal_draw_cards": {"all_games": {"average": 1.0}},
                },
            },
        )

        presentation = SetupRelayPlugin().render(aggregate, ())

        self.assertEqual(presentation.metric_id, "setup_relay")
        self.assertIn("Setup and relay", presentation.markdown)
        self.assertIn("bridge", presentation.markdown)
        self.assertIn("second-turn draws", presentation.html)

    def test_attack_quality_presentation_exposes_audit_counts(self) -> None:
        aggregate = AggregateMetric(
            "attack_quality",
            2,
            4,
            0.5,
            {},
            {
                "attack_submissions": 5,
                "resolved_attacks": 4,
                "unresolved_attacks": 1,
                "unknown_prize_attacks": 1,
                "non_prize_attacks": {"numerator": 2, "denominator": 3},
            },
        )

        presentation = AttackQualityPlugin().render(aggregate, ())

        self.assertEqual(presentation.metric_id, "attack_quality")
        self.assertIn("Attack quality", presentation.markdown)
        self.assertIn("unknown prize", presentation.markdown)
        self.assertIn("unresolved", presentation.html)


if __name__ == "__main__":
    unittest.main()
