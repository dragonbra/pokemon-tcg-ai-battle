from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "docs/environment-daily_kaggle_top100"


class EnvironmentTransitionAnalysisTests(unittest.TestCase):
    def test_transition_data_keeps_three_rosters_and_strict_two_day_delta(self) -> None:
        from data.processed.environment_daily.generate_transition_analysis import (
            build_transition_data,
        )

        data = build_transition_data(ROOT)
        self.assertEqual([snapshot["date"] for snapshot in data["snapshots"]], [
            "2026-07-26",
            "2026-07-27",
            "2026-07-28",
        ])
        self.assertEqual(data["strict_comparison_dates"], ["2026-07-27", "2026-07-28"])
        self.assertTrue(data["card_pool_delta"])
        self.assertGreater(data["continuity"]["shared_27_28"], 0)

    def test_rendered_transition_page_has_evidence_and_daily_ui_sections(self) -> None:
        from data.processed.environment_daily.generate_transition_analysis import (
            build_transition_data,
            render_transition_report,
        )

        data = build_transition_data(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "environment-transition.html"
            render_transition_report(data, output)
            text = output.read_text(encoding="utf-8")

        for section_id in (
            "transition-overview",
            "archetype-migration",
            "rank-movement",
            "deck-transition",
            "card-pool-delta",
            "evidence-boundary",
        ):
            self.assertIn(f'id="{section_id}"', text)
        payload = re.search(
            r'<script type="application/json" id="transition-data">(.*?)</script>',
            text,
            re.S,
        )
        self.assertIsNotNone(payload)
        embedded = json.loads(str(payload.group(1)))
        self.assertEqual(embedded["strict_comparison_dates"], ["2026-07-27", "2026-07-28"])
        self.assertEqual(embedded["audit"]["2026-07-28"]["consecutive_stable_sweeps"], 2)
        self.assertGreater(text.count("class=\"card-thumb"), 10)

    def test_checked_in_transition_page_is_linked_from_environment_index(self) -> None:
        report = REPORT_ROOT / "environment-transition.html"
        self.assertTrue(report.is_file())
        self.assertIn("environment-transition.html", (REPORT_ROOT / "index.html").read_text(
            encoding="utf-8"
        ))


if __name__ == "__main__":
    unittest.main()
