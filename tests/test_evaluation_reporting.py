from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics import MetricPresentation
from evaluation.reporting import ReportData, render_html, render_markdown, write_report


OPPONENTS = (
    "romanrozen_v9",
    "pilkwang_v2",
    "kokinn_search",
    "penguin_915",
    "crustle_wall",
    "crustle_v1",
    "kiyotah_lucario",
    "kiyotah_dragapult",
    "kiyotah_iono",
    "kiyotah_abomasnow",
    "kacchan_anti_wall",
    "nursrijan_lucario",
    "yakitori_raging_bolt",
    'zoli_dragapult <&"',
    "sue_alakazam",
    "maktha_1084",
    "yanxiaohan",
)


def report_data() -> ReportData:
    by_opponent = {
        opponent: {
            "games": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "errors": 0,
            "unfinished": 0,
            "win_rate": 0.5,
        }
        for opponent in OPPONENTS
    }
    by_opponent[OPPONENTS[0]]["win_rate"] = 0.125
    return ReportData(
        manifest={"run_id": "fixture-run", "candidate": {"name": "alakazam_v8"}},
        summary={
            "total_games": 34,
            "wins": 17,
            "losses": 12,
            "draws": 2,
            "errors": 3,
            "unfinished": 1,
            "completed_games": 30,
            "win_rate": 0.314159,
            "completion_rate": 0.88235,
            "by_opponent": by_opponent,
            "control": {
                "differences": {
                    "candidate": "alakazam_v8",
                    "control": "alakazam_v7",
                    "summary": "unavailable: control 未在本次 batch 中运行",
                    "win_rate": -0.12,
                    "powerful_hand": "7/17 vs 9/17",
                },
                "promotion": "must not be rendered",
            },
        },
        games=({"game_id": "fixture-001"},),
        metrics={
            "correctness": {
                "numerator": 3,
                "denominator": 34,
                "value": 0.088235,
                "by_opponent": {},
                "diagnostics": {
                    "failure_classes": {"candidate_error": 1, "worker_crash": 2}
                },
            },
            "powerful_hand": {
                "numerator": 7,
                "denominator": 17,
                "value": 0.4123,
                "by_opponent": {OPPONENTS[0]: {"numerator": 1, "denominator": 2}},
                "diagnostics": {"reached_target_turn": 17},
            },
        },
        cases=(
            {
                "game_id": "fixture-001",
                "opponent": OPPONENTS[0],
                "failure_class": "rare_candy_not_played",
                "metric_ids": ["rare_candy"],
                "evidence": [{"step": 12, "expected_reason": "complete route"}],
                "trace_path": "traces/fixture-001.json",
            },
        ),
    )


class EvaluationReportingTests(unittest.TestCase):
    def test_plugin_presentations_and_profile_metadata_are_rendered(self) -> None:
        base = report_data()
        data = ReportData(
            manifest={
                **base.manifest,
                "metric_profile": {
                    "id": "auto_iteration_v8_setup_relay",
                    "revision": 2,
                },
            },
            summary=base.summary,
            games=base.games,
            metrics=base.metrics,
            cases=base.cases,
            metric_profile={
                "id": "auto_iteration_v8_setup_relay",
                "revision": 2,
                "metric_ids": ["setup_relay", "attack_quality"],
            },
            presentations={
                "setup_relay": MetricPresentation(
                    "setup_relay",
                    "Setup and relay",
                    "## Setup and relay\n\nbridge rate: 0.5",
                    "<section><h2>Setup and relay</h2><p>bridge rate: 0.5</p></section>",
                ),
                "attack_quality": MetricPresentation(
                    "attack_quality",
                    "Attack quality",
                    "## Attack quality\n\nnon-prize attacks: 2",
                    "<section><h2>Attack quality</h2><p>non-prize attacks: 2</p></section>",
                ),
            },
        )

        markdown = render_markdown(data)
        html = render_html(data)

        for value in ("auto_iteration_v8_setup_relay", "revision", "Setup and relay", "Attack quality"):
            self.assertIn(value, markdown)
            self.assertIn(value, html)
        self.assertIn('"presentations"', html)

    def test_markdown_and_html_render_the_same_aggregated_fixture(self) -> None:
        data = report_data()

        markdown = render_markdown(data)
        html = render_html(data)

        for value in ("34", "powerful_hand", "rare_candy_not_played", "fixture-001"):
            self.assertIn(value, markdown)
            self.assertIn(value, html)
        for opponent in OPPONENTS:
            self.assertIn(opponent, markdown)
        for opponent in OPPONENTS:
            if opponent == 'zoli_dragapult <&"':
                continue
            self.assertIn(opponent, html)
        self.assertIn("zoli_dragapult &lt;&amp;&quot;", html)
        self.assertNotIn('zoli_dragapult <&"', html)
        self.assertIn("12.50%", markdown)
        self.assertIn("12.50%", html)
        self.assertIn("31.42%", markdown)
        self.assertIn("31.42%", html)
        self.assertIn("candidate_error", markdown)
        self.assertIn("worker_crash", html)
        self.assertIn("步骤 12", markdown)
        self.assertIn("traces/fixture-001.json", html)
        self.assertNotIn("promotion", markdown)
        self.assertNotIn("promotion", html)
        self.assertIn("alakazam_v7", markdown)
        self.assertIn("alakazam_v7", html)
        self.assertIn("unavailable", markdown)
        self.assertIn("unavailable", html)
        for forbidden in ("reject", "revert"):
            self.assertNotIn(forbidden, markdown.lower())
            self.assertNotIn(forbidden, html.lower())

    def test_html_embeds_script_safe_json_data(self) -> None:
        html = render_html(report_data())

        payload = html.split('<script id="report-data" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
        decoded = json.loads(payload)

        self.assertEqual(decoded["summary"]["total_games"], 34)
        self.assertEqual(decoded["metrics"]["powerful_hand"]["numerator"], 7)
        self.assertEqual(decoded["cases"][0]["game_id"], "fixture-001")

    def test_html_converts_non_finite_numbers_to_json_null(self) -> None:
        data = report_data()
        data = ReportData(
            manifest=data.manifest,
            summary={**data.summary, "win_rate": float("nan")},
            games=data.games,
            metrics=data.metrics,
            cases=data.cases,
        )

        html = render_html(data)
        payload = html.split('<script id="report-data" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
        decoded = json.loads(payload)

        self.assertIsNone(decoded["summary"]["win_rate"])

    def test_write_report_creates_both_standalone_outputs(self) -> None:
        data = report_data()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "nested" / "report"
            write_report(data, output_dir)

            markdown = (output_dir / "report.md").read_text(encoding="utf-8")
            html = (output_dir / "report.html").read_text(encoding="utf-8")

        self.assertIn("总体结果", markdown)
        self.assertIn("matchup-chart", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)


if __name__ == "__main__":
    unittest.main()
