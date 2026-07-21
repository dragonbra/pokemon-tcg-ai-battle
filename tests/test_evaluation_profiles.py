from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from evaluation.metrics.profiles import (
    AUTO_ITERATION_PROFILE_ID,
    get_metric_profile,
)
from evaluation.metrics.registry import create_metric_registry


class EvaluationProfileTests(unittest.TestCase):
    def test_auto_iteration_profile_declares_revision_priorities_and_metrics(self) -> None:
        profile = get_metric_profile(AUTO_ITERATION_PROFILE_ID)

        self.assertEqual(profile.revision, 2)
        self.assertEqual(
            profile.metric_ids,
            (
                "outcome",
                "length",
                "correctness",
                "powerful_hand",
                "rare_candy",
                "post_ko_relay",
                "run_away_draw",
                "library_pressure",
                "setup_relay",
                "attack_quality",
            ),
        )
        self.assertEqual(profile.priorities[0].metric_id, "outcome")
        self.assertEqual(profile.priorities[0].priority, "result_guardrail")
        self.assertEqual(profile.priorities[1].metric_id, "powerful_hand")

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown metric profile"):
            get_metric_profile("missing")

    def test_readme_documents_auto_iteration_profile_contract(self) -> None:
        readme = (Path(__file__).parents[1] / "evaluation" / "README.md").read_text(
            encoding="utf-8"
        )

        for value in (
            "auto_iteration_v8_setup_relay",
            "revision 2",
            "metrics.json",
            "report.html",
            "不负责晋级决策",
        ):
            self.assertIn(value, readme)

    def test_dynamic_plugin_cannot_override_metric_from_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "override.py"
            module_path.write_text(
                """
from evaluation.metrics.base import AggregateMetric, GameMetric

class OverridePlugin:
    metric_id = 'setup_relay'
    def analyze_game(self, trace, context):
        return GameMetric(self.metric_id, 'success', 0, 1, 0, (), ())
    def aggregate(self, results):
        return AggregateMetric(self.metric_id, 0, 1, 0, {})
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "selected metric profile"):
                create_metric_registry(
                    [str(module_path)],
                    profile_id=AUTO_ITERATION_PROFILE_ID,
                )


if __name__ == "__main__":
    unittest.main()
