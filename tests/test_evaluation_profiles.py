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

        self.assertEqual(profile.revision, 3)
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

    def test_auto_iteration_manifest_declares_semantic_stages_and_metric_targets(self) -> None:
        profile = get_metric_profile(AUTO_ITERATION_PROFILE_ID)
        manifest = profile.manifest()

        self.assertEqual(
            [group["id"] for group in manifest["semantic_groups"]],
            [
                "result_correctness_guardrail",
                "stage_1_setup",
                "stage_2_post_ko_relay",
                "stage_3_attack_quality",
                "auxiliary_health_audit",
            ],
        )
        semantics = {
            item["semantic_id"]: item for item in manifest["metric_semantics"]
        }
        self.assertNotIn("opening_components", semantics)
        self.assertEqual(semantics["powerful_hand"]["title"], "二回合 Alakazam 实际攻击")
        self.assertEqual(semantics["powerful_hand"]["direction"], "higher")
        self.assertEqual(semantics["post_ko_success"]["value_source"], "payload.success_rate")
        self.assertEqual(semantics["post_ko_success"]["role"], "target")
        self.assertEqual(
            semantics["attack_quality_powerful"]["title"],
            "Powerful Hand 子集未拿奖赏率",
        )

    def test_core_manifest_keeps_semantic_metadata_additive(self) -> None:
        manifest = get_metric_profile("core").manifest()

        self.assertNotIn("semantic_groups", manifest)
        self.assertNotIn("metric_semantics", manifest)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown metric profile"):
            get_metric_profile("missing")

    def test_readme_documents_setup_relay_profile_contract(self) -> None:
        readme = (Path(__file__).parents[1] / "evaluation" / "README.md").read_text(
            encoding="utf-8"
        )

        for value in (
            "auto_iteration_v8_setup_relay",
            "revision 3",
            "report_only",
            "report.html",
            "不执行自动迭代",
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
