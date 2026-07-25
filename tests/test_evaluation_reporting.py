from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics import MetricPresentation
from evaluation.metrics.profiles import AUTO_ITERATION_PROFILE_ID, get_metric_profile
from evaluation.reporting import (
    ReportData,
    render_html,
    render_markdown,
    write_evaluation_index,
    write_report,
)


OPPONENTS = (
    "mega_lucario_ex_solrock_01",
    "mega_lucario_ex_solrock_02",
    "mega_lucario_ex_solrock_03",
    "mega_lucario_ex_solrock_04",
    "crustle_01",
    "crustle_02",
    "mega_lucario_ex_solrock_05",
    "dragapult_ex_01",
    "ionos_bellibolt_ex_kilowattrel_01",
    "mega_abomasnow_ex_kyogre_01",
    "mega_lucario_ex_solrock_06",
    "mega_lucario_ex_solrock_07",
    "mega_lucario_ex_solrock_10",
    'dragapult_ex_02 <&"',
    "alakazam_dudunsparce_01",
    "mega_lucario_ex_solrock_08",
    "mega_lucario_ex_solrock_09",
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


def visible_html(html: str) -> str:
    return html.split('<script id="report-data" type="application/json">', 1)[0]


class EvaluationReportingTests(unittest.TestCase):
    def test_auto_iteration_html_maps_metric_ids_to_semantic_stages(self) -> None:
        metrics = {
            "outcome": {
                "numerator": 70,
                "denominator": 170,
                "value": 70 / 170,
                "payload": {
                    "by_turn_order": {
                        "first": {"numerator": 35, "denominator": 85, "value": 35 / 85},
                        "second": {"numerator": 35, "denominator": 85, "value": 35 / 85},
                    }
                },
            },
            "correctness": {
                "numerator": 1,
                "denominator": 170,
                "value": 1 / 170,
                "payload": {},
            },
            "powerful_hand": {
                "numerator": 4,
                "denominator": 170,
                "value": 4 / 170,
                "payload": {
                    "reached_value": 4 / 154,
                    "reached_numerator": 4,
                    "reached_denominator": 154,
                    "by_turn_order": {
                        "first": {"numerator": 2, "denominator": 85, "value": 2 / 85},
                        "second": {"numerator": 2, "denominator": 85, "value": 2 / 85},
                    },
                },
            },
            "setup_relay": {
                "numerator": 4,
                "denominator": 170,
                "value": 4 / 170,
                "payload": {
                    "dunsparce_bridge": {"numerator": 0, "denominator": 59},
                    "second_turn_draws": {
                        "all_games": {"total": 471, "games": 170, "average": 471 / 170},
                        "reached_second_turn": {"total": 471, "games": 154, "average": 471 / 154},
                        "first": {"total": 258, "games": 85, "average": 258 / 85},
                        "second": {"total": 213, "games": 85, "average": 213 / 85},
                        "normal_draw_cards": {
                            "all_games": {"total": 337, "games": 170, "average": 337 / 170}
                        },
                    },
                    "opening_four_components": {
                        "component_counts": {
                            "active_abra": 76,
                            "rare_candy": 66,
                            "alakazam_or_search": 156,
                            "psychic_energy_or_hilda": 133,
                        },
                        "sample_games": 170,
                    },
                },
            },
            "post_ko_relay": {
                "numerator": 339,
                "denominator": 455,
                "value": 339 / 455,
                "payload": {
                    "successes": 116,
                    "opportunities": 455,
                    "success_rate": 0.3,
                    "by_turn_order": {
                        "first": {"successes": 70, "opportunities": 234, "success_rate": 70 / 234},
                        "second": {"successes": 46, "opportunities": 221, "success_rate": 46 / 221},
                    },
                },
            },
            "attack_quality": {
                "numerator": 139,
                "denominator": 458,
                "value": 139 / 458,
                "payload": {
                    "non_prize_attacks": {"numerator": 139, "denominator": 458, "rate": 139 / 458},
                    "powerful_hand": {
                        "non_prize_attacks": 54,
                        "resolved_attacks": 367,
                        "unknown_prize_attacks": 0,
                    },
                },
            },
            "library_pressure": {
                "numerator": 564,
                "denominator": 170,
                "value": 564 / 170,
                "payload": {},
            },
            "length": {
                "numerator": 15,
                "denominator": 4,
                "value": 3.75,
                "payload": {
                    "by_outcome": {
                        "win": {
                            "numerator": 6,
                            "denominator": 2,
                            "average": 3.0,
                            "distribution": {
                                "3": {
                                    "count": 2,
                                    "candidate_first": 1,
                                    "candidate_second": 1,
                                }
                            },
                        },
                        "loss": {
                            "numerator": 9,
                            "denominator": 2,
                            "average": 4.5,
                            "distribution": {
                                "4": {
                                    "count": 1,
                                    "candidate_first": 1,
                                    "candidate_second": 0,
                                },
                                "5": {
                                    "count": 1,
                                    "candidate_first": 0,
                                    "candidate_second": 1,
                                },
                            },
                        },
                    }
                },
            },
        }
        data = ReportData(
            manifest={"run_id": "semantic-fixture"},
            summary={"total_games": 170, "wins": 70, "losses": 99, "draws": 0, "errors": 1},
            games=(),
            metrics=metrics,
            cases=(),
            metric_profile=get_metric_profile(AUTO_ITERATION_PROFILE_ID).manifest(),
        )

        html = render_html(data)

        expected = (
            "结果与正确性护栏",
            "阶段一：二回合基础能力",
            "阶段二：Post-KO 接力能力",
            "阶段三：攻击质量惩罚项",
            "辅助健康与审计指标",
            "二回合 Alakazam 实际攻击",
            "Post-KO 立即接力成功率",
            "攻击但未拿奖赏率",
            "Powerful Hand 子集未拿奖赏率",
            "metric_id",
            "post_ko_relay",
            "116/455",
            "30.00%",
            "139/458",
            "30.35%",
            "3.31765",
            "先手：70/234 = 29.91%",
            "后手：46/221 = 20.81%",
            "实际到达二回合：3.05844 张/局",
            "先手额外过牌：3.03529 张/局",
            "后手额外过牌：2.50588 张/局",
            "正常回合抽牌审计：1.98235 张/局",
            "对局结束回合数",
            "胜利平均 3 回合；失败平均 4.5 回合",
            "胜利对局分布",
            "失败对局分布",
            "第3回合",
            "第5回合",
            "我们先攻",
            "我们后攻",
            "追踪目标",
        )
        for value in expected:
            self.assertIn(value, html)

        self.assertLess(html.index("结果与正确性护栏"), html.index("阶段一：二回合基础能力"))
        self.assertLess(html.index("阶段一：二回合基础能力"), html.index("阶段二：Post-KO 接力能力"))
        self.assertLess(html.index("阶段二：Post-KO 接力能力"), html.index("阶段三：攻击质量惩罚项"))
        self.assertLess(html.index("阶段三：攻击质量惩罚项"), html.index("辅助健康与审计指标"))
        visible = visible_html(html)
        self.assertEqual(visible.rsplit("<h2>", 1)[1].split("</h2>", 1)[0], "辅助健康与审计指标")
        for hidden_section in (
            "<h2>指标</h2>",
            "failure_class 分布",
            "重点案例",
            "对照差异",
            "插件审计明细",
            "Presentation diagnostics",
        ):
            self.assertNotIn(hidden_section, visible)
        stage_two = html[html.index("阶段二：Post-KO 接力能力") : html.index("阶段三：攻击质量惩罚项")]
        self.assertIn("116/455 = 30.00%", stage_two)
        self.assertNotIn("339/455 = 74.51%", stage_two)
        self.assertNotIn("331.76%", html)
        visible = visible_html(html)
        self.assertNotIn("第一回合起始四组件状态", visible)
        self.assertNotIn("<th>分子 / 分母</th>", visible)
        self.assertIn("139/458 = 30.35%", visible)
        self.assertIn("54/367 = 14.71%", visible)

    def test_auto_iteration_html_marks_empty_event_rates_as_undefined(self) -> None:
        data = ReportData(
            manifest={"run_id": "empty-semantic-fixture"},
            summary={},
            games=(),
            metrics={
                "post_ko_relay": {
                    "numerator": 0,
                    "denominator": 0,
                    "value": None,
                    "payload": {"successes": 0, "opportunities": 0, "success_rate": None},
                },
                "attack_quality": {
                    "numerator": 0,
                    "denominator": 0,
                    "value": None,
                    "payload": {
                        "non_prize_attacks": {"numerator": 0, "denominator": 0, "rate": None},
                        "powerful_hand": {
                            "non_prize_attacks": 0,
                            "resolved_attacks": 0,
                            "unknown_prize_attacks": 0,
                        },
                    },
                },
            },
            cases=(),
            metric_profile=get_metric_profile(AUTO_ITERATION_PROFILE_ID).manifest(),
        )

        html = render_html(data)

        self.assertIn("未定义（无有效样本）", html)
        self.assertIn("无机会", html)
        self.assertIn("无有效攻击", html)
        stage_three = html[html.index("阶段三：攻击质量惩罚项") :]
        self.assertGreaterEqual(stage_three.count("未定义（无有效样本）"), 2)
        self.assertNotIn("0/0", html)
        self.assertNotIn("0/0 =", html)

    def test_plugin_presentations_remain_embedded_but_are_not_visible_after_semantic_metrics(self) -> None:
        base = report_data()
        data = ReportData(
            manifest={
                **base.manifest,
                "metric_profile": {
                    "id": "auto_iteration_v8_setup_relay",
                    "revision": 3,
                },
            },
            summary=base.summary,
            games=base.games,
            metrics=base.metrics,
            cases=base.cases,
            metric_profile={
                "id": "auto_iteration_v8_setup_relay",
                "revision": 3,
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
        visible = visible_html(html)
        self.assertNotIn("Setup and relay", visible)
        self.assertNotIn("Attack quality", visible)
        self.assertIn('"presentations"', html)

    def test_generic_html_keeps_legacy_metric_table_title(self) -> None:
        html = render_html(report_data())

        self.assertIn("<h2>指标</h2>", html)
        self.assertIn("<th>达成值</th>", html)

    def test_html_keeps_only_win_rate_chart_and_uses_plain_metric_ids(self) -> None:
        data = report_data()
        first_opponent = OPPONENTS[0]
        data = ReportData(
            manifest={
                **data.manifest,
                "opponents": [
                    {
                        "name": first_opponent,
                        "display_name": "Mega Lucario ex / Solrock 01",
                        "representative_cards": [
                            {
                                "card_id": 678,
                                "name": "Mega Lucario ex",
                                "image_url": "https://images.pokemontcg.io/me1/77.png",
                            }
                        ],
                    }
                ],
            },
            summary=data.summary,
            games=data.games,
            metrics=data.metrics,
            cases=data.cases,
            metric_profile=get_metric_profile(AUTO_ITERATION_PROFILE_ID).manifest(),
        )

        html = render_html(data)

        self.assertNotIn("<h2>对局矩阵</h2>", html)
        self.assertIn("<h2>对局胜率图</h2>", html)
        self.assertIn('<span class="metric-id">powerful_hand</span>', html)
        self.assertNotIn("<code>powerful_hand</code>", html)
        self.assertIn('class="hero"', html)
        self.assertIn("--brand:#217a58", html)
        self.assertIn("@media (max-width:760px)", html)
        self.assertIn("grid-template-columns:repeat(2,minmax(0,1fr))", html)
        self.assertIn("height:7px", html)
        self.assertIn("Mega Lucario ex / Solrock 01", html)
        self.assertIn('class="opponent-thumb"', html)
        self.assertIn("https://images.pokemontcg.io/me1/77.png", html)

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
            if opponent == 'dragapult_ex_02 <&"':
                continue
            self.assertIn(opponent, html)
        self.assertIn("dragapult_ex_02 &lt;&amp;&quot;", html)
        self.assertNotIn('dragapult_ex_02 <&"', html)
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

    def test_write_report_creates_only_standalone_html(self) -> None:
        data = report_data()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "nested" / "report"
            write_report(data, output_dir)

            html = (output_dir / "report.html").read_text(encoding="utf-8")

            self.assertEqual({path.name for path in output_dir.iterdir()}, {"report.html"})

        self.assertIn("matchup-chart", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)

    def test_project_index_summarizes_and_links_version_reports(self) -> None:
        data = report_data()
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary) / "0012-feature-test"
            write_report(data, project_root / "V2_second")
            write_report(data, project_root / "V1_first")
            (project_root / "V3_broken.html").write_text("broken", encoding="utf-8")

            for directory in (project_root / "V1_first", project_root / "V2_second"):
                (directory / "report.html").replace(project_root / f"{directory.name}.html")
                directory.rmdir()
            index_path = write_evaluation_index(project_root)
            index = index_path.read_text(encoding="utf-8")

        self.assertIn("0012-feature-test", index)
        self.assertLess(index.index("V1_first"), index.index("V2_second"))
        self.assertIn('href="V1_first.html"', index)
        self.assertIn("fixture-run", index)
        self.assertIn("34", index)
        self.assertIn("31.42%", index)
        self.assertIn("无法解析", index)


if __name__ == "__main__":
    unittest.main()
