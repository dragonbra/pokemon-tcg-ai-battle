from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/environment-daily_kaggle_top100/daily/2026-07-27.html"
CURRENT_REPORT = ROOT / "docs/environment-daily_kaggle_top100/daily/2026-07-30.html"
LATEST_REPORT = ROOT / "docs/environment-daily_kaggle_top100/daily/2026-08-16.html"


class _DailyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.player_rows = 0
        self.player_details = 0
        self.pool_cards = 0
        self.card_thumbs = 0
        self.pool_tables = 0
        self.investment_grids = 0
        self.rate_cards = 0
        self.archetype_visuals = 0
        self.heatmaps = 0
        self.card_images = 0
        self.invalid_card_images = 0
        self.high_score_decks = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if "data-player-row" in values:
            self.player_rows += 1
        if "data-player-detail" in values:
            self.player_details += 1
        if "data-pool-card" in values:
            self.pool_cards += 1
        if "data-high-score-deck" in values:
            self.high_score_decks += 1
        if "card-thumb" in str(values.get("class", "")).split():
            self.card_thumbs += 1
        if tag == "img" and str(values.get("src", "")).startswith("https://"):
            self.card_images += 1
            if not values.get("width") or not values.get("height"):
                self.invalid_card_images += 1
        classes = str(values.get("class", "")).split()
        if tag == "table" and "pool-table" in classes:
            self.pool_tables += 1
        if "investment-grid" in classes:
            self.investment_grids += 1
        if "rate" in classes:
            self.rate_cards += 1
        if "archetype-visual" in classes:
            self.archetype_visuals += 1
        if tag == "table" and "heatmap" in classes:
            self.heatmaps += 1


class EnvironmentDailyContractTests(unittest.TestCase):
    def test_rate_call_retries_transient_kaggle_transport_failure(self) -> None:
        from requests.exceptions import ConnectionError

        from data.processed.environment_daily.generate_live_snapshot import _rate_call

        operation = mock.Mock(side_effect=[ConnectionError("connection reset"), "ok"])
        with mock.patch(
            "data.processed.environment_daily.generate_live_snapshot.time.sleep"
        ) as sleep:
            self.assertEqual(_rate_call(operation), "ok")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(5)

    def test_episode_selection_is_bounded_by_leaderboard_capture(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _episodes_at_or_before,
        )

        episodes = [
            SimpleNamespace(id=3, create_time="2026-07-26T20:07:02+00:00"),
            SimpleNamespace(id=2, create_time="2026-07-26T20:07:01+00:00"),
            SimpleNamespace(id=1, create_time="2026-07-26T20:06:59+00:00"),
        ]
        bounded = _episodes_at_or_before(
            episodes,
            "2026-07-26T20:07:01.973692+00:00",
        )
        self.assertEqual([episode.id for episode in bounded], [2, 1])

    def test_leaderboard_binding_prefers_matching_score_over_latest_submission_date(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _select_leaderboard_submission,
        )

        leaderboard = SimpleNamespace(
            score="1148.2",
            submission_date="2026-07-29T02:34:52.576000+00:00",
        )
        lower_score_latest = SimpleNamespace(
            id=55070000,
            public_score="929.5",
            date_submitted="2026-07-29T02:34:52.577000+00:00",
        )
        leaderboard_best = SimpleNamespace(
            id=55069846,
            public_score="1148.2",
            date_submitted="2026-07-29T02:25:43.117000+00:00",
        )

        selected, binding = _select_leaderboard_submission(
            leaderboard,
            [lower_score_latest, leaderboard_best],
        )

        self.assertEqual(selected.id, 55069846)
        self.assertEqual(binding, "leaderboard_score")

    def test_leaderboard_binding_uses_date_only_to_disambiguate_equal_scores(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _select_leaderboard_submission,
        )

        leaderboard = SimpleNamespace(
            score="1100.0",
            submission_date="2026-07-29T02:34:52.576000+00:00",
        )
        selected, binding = _select_leaderboard_submission(
            leaderboard,
            [
                SimpleNamespace(
                    id=1,
                    public_score="1100.0",
                    date_submitted="2026-07-29T02:00:00+00:00",
                ),
                SimpleNamespace(
                    id=2,
                    public_score="1100.0",
                    date_submitted="2026-07-29T02:34:52.577000+00:00",
                ),
            ],
        )

        self.assertEqual(selected.id, 2)
        self.assertEqual(binding, "leaderboard_score_submission_date")

    def test_leaderboard_binding_rejects_an_ambiguous_score(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _select_leaderboard_submission,
        )

        leaderboard = SimpleNamespace(
            score="1100.0",
            submission_date="2026-07-29T03:00:00+00:00",
        )
        tied = [
            SimpleNamespace(
                id=submission_id,
                public_score="1100.0",
                date_submitted=submitted_at,
            )
            for submission_id, submitted_at in (
                (1, "2026-07-29T01:00:00+00:00"),
                (2, "2026-07-29T02:00:00+00:00"),
            )
        ]

        with self.assertRaisesRegex(ValueError, "leaderboard score"):
            _select_leaderboard_submission(leaderboard, tied)

    def test_scored_submissions_are_cutoff_bounded_and_score_ordered(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _scored_submissions_at_or_before,
        )

        submissions = [
            SimpleNamespace(id=1, public_score="1080.0", date_submitted="2026-08-15T01:00:00Z"),
            SimpleNamespace(id=2, public_score="1120.0", date_submitted="2026-08-14T01:00:00Z"),
            SimpleNamespace(id=3, public_score="1200.0", date_submitted="2026-08-16T02:00:01Z"),
            SimpleNamespace(id=4, public_score=None, date_submitted="2026-08-13T01:00:00Z"),
        ]

        ordered = _scored_submissions_at_or_before(
            submissions, "2026-08-16T02:00:00+00:00"
        )

        self.assertEqual([row.id for row in ordered], [2, 1])

    def test_latest_submission_is_cutoff_bounded(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _latest_submission_at_or_before,
        )

        submissions = [
            SimpleNamespace(id=1, date_submitted="2026-08-15T01:00:00Z"),
            SimpleNamespace(id=2, date_submitted="2026-08-16T01:59:59Z"),
            SimpleNamespace(id=3, date_submitted="2026-08-16T02:00:01Z"),
        ]

        selected = _latest_submission_at_or_before(
            submissions, "2026-08-16T02:00:00+00:00"
        )

        self.assertEqual(selected.id, 2)

    def test_distinct_high_score_decks_keep_only_best_two_hashes(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _select_distinct_deck_evidence,
        )

        evidence = [
            {"submission_id": 1, "public_score": "1200.0", "deck_sha256": "aaa"},
            {"submission_id": 2, "public_score": "1180.0", "deck_sha256": "aaa"},
            {"submission_id": 3, "public_score": "1170.0", "deck_sha256": "bbb"},
            {"submission_id": 4, "public_score": "1160.0", "deck_sha256": "ccc"},
        ]

        selected = _select_distinct_deck_evidence(evidence, limit=2)

        self.assertEqual([row["submission_id"] for row in selected], [1, 3])

    def test_relative_submission_age_uses_snapshot_time(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _relative_submission_age,
        )

        captured = "2026-08-16T12:00:00+00:00"
        self.assertEqual(_relative_submission_age("2026-08-16T11:59:40Z", captured), "刚刚")
        self.assertEqual(_relative_submission_age("2026-08-16T11:21:00Z", captured), "39 分钟前")
        self.assertEqual(_relative_submission_age("2026-08-16T08:00:00Z", captured), "4 小时前")
        self.assertEqual(_relative_submission_age("2026-08-14T11:59:59Z", captured), "2 天前")
        with self.assertRaisesRegex(ValueError, "after snapshot cutoff"):
            _relative_submission_age("2026-08-16T12:00:01Z", captured)

    def test_final_leaderboard_freeze_refreshes_every_team_view_after_cutoff(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _capture_score_bound_leaderboard,
        )

        first = [
            SimpleNamespace(
                team_id=team_id,
                team_name=f"team-{team_id}",
                score="1000.0",
                submission_date="2026-08-01T00:00:00+00:00",
            )
            for team_id in range(100)
        ]
        final = [SimpleNamespace(**vars(row)) for row in first]
        final[7].score = "1001.0"

        class FakeApi:
            def __init__(self) -> None:
                self.leaderboard_calls = 0
                self.submission_calls = Counter()

            def competition_leaderboard_view(
                self, competition: str, page_size: int
            ) -> list[SimpleNamespace]:
                self.leaderboard_calls += 1
                return first if self.leaderboard_calls == 1 else final

            def competition_team_submissions(self, team_id: int) -> list[SimpleNamespace]:
                self.submission_calls[team_id] += 1
                score = (
                    "1001.0"
                    if team_id == 7 and self.submission_calls[team_id] > 1
                    else "1000.0"
                )
                return [
                    SimpleNamespace(
                        id=10_000 + team_id,
                        public_score=score,
                        date_submitted="2026-08-01T00:00:00+00:00",
                    )
                ]

        api = FakeApi()
        leaderboard, submissions, captured_at_utc = _capture_score_bound_leaderboard(api)

        self.assertEqual(leaderboard[7].score, "1001.0")
        self.assertTrue(captured_at_utc.endswith("+00:00"))
        self.assertEqual(api.leaderboard_calls, 2)
        self.assertTrue(all(api.submission_calls[index] == 2 for index in range(100)))
        self.assertEqual(submissions[7][0].public_score, "1001.0")

    def test_episode_identity_uses_agent_index_not_list_position(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _episode_views,
            _result,
        )

        episode = SimpleNamespace(
            id=77,
            create_time="2026-07-26T20:00:00+00:00",
            end_time="2026-07-26T20:01:00+00:00",
            agents=[
                SimpleNamespace(
                    index=1, submission_id=42, team_id=4, team_name="self", reward=1
                ),
                SimpleNamespace(
                    index=0, submission_id=99, team_id=9, team_name="other", reward=-1
                ),
            ],
        )
        self.assertEqual(_result([episode], 42), (1, 0, 0, 1))
        view = _episode_views([episode], 42)[0]
        self.assertEqual(view["player_index"], 1)
        self.assertEqual(view["other"]["submission_id"], 99)

    def test_meta_sweeps_keep_a_monotonic_episode_union(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import (
            _merge_episode_views,
        )

        def view(episode_id: int) -> dict[str, object]:
            return {"episode_id": episode_id, "create_time": "2026-07-26T20:00:00+00:00"}

        merged = _merge_episode_views([view(1), view(2)], [view(2), view(3)])
        self.assertEqual({row["episode_id"] for row in merged}, {1, 2, 3})

    def test_completed_capture_cannot_be_silently_reused(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import collect

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            snapshot = {
                "status": "complete",
                "report_date": "2026-07-27",
            }
            (work / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completed leaderboard capture is immutable"):
                collect(work, work / "report.html", "2026-07-27")

    def test_temp_preview_resolves_the_canonical_ui_baseline(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import _baseline_css

        with tempfile.TemporaryDirectory() as directory:
            css = _baseline_css(Path(directory) / "report.html")
        self.assertIn("body", css)

    def test_0727_keeps_the_reviewed_daily_structure(self) -> None:
        parser = _DailyParser()
        parser.feed(REPORT.read_text(encoding="utf-8"))
        expected_ids = {
            "new-summary",
            "personal-winrates",
            "construction-distribution",
            "snapshot-comparison",
            "matchup-boundary",
            "card-pool",
            "archetype-builds",
            "rank-index",
            "player-details",
            "boundaries",
        }
        self.assertTrue(expected_ids.issubset(parser.ids))
        self.assertEqual(parser.player_rows, 100)
        self.assertEqual(parser.player_details, 100)
        self.assertGreater(parser.pool_cards, 40)
        self.assertGreater(parser.card_thumbs, 1_000)

    def test_0727_uses_the_real_0725_pool_and_0726_visual_components(self) -> None:
        parser = _DailyParser()
        parser.feed(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(parser.pool_tables, 1)
        self.assertEqual(parser.investment_grids, 1)
        self.assertEqual(parser.rate_cards, 300)
        self.assertGreaterEqual(parser.archetype_visuals, 100)
        self.assertEqual(parser.heatmaps, 2)
        self.assertGreater(parser.card_images, 1_000)
        self.assertEqual(parser.invalid_card_images, 0)

    def test_0727_embeds_the_100_player_identity_audit(self) -> None:
        text = REPORT.read_text(encoding="utf-8")
        match = re.search(
            r'<script type="application/json" id="snapshot-audit">(.*?)</script>',
            text,
            re.S,
        )
        self.assertIsNotNone(match)
        audit = json.loads(match.group(1))
        self.assertEqual(audit["audited_players"], 100)
        self.assertEqual(len(audit["selected"]), 100)
        self.assertTrue(audit["all_episode_times_at_or_before_capture"])
        self.assertTrue(audit["all_submission_matches_unique"])
        self.assertTrue(audit["all_decks_exactly_60"])
        self.assertGreater(audit["bounded_player_views"], 10_000)
        self.assertEqual(audit["stabilization"]["consecutive_stable_sweeps"], 2)

    def test_project_contract_names_both_ui_baselines(self) -> None:
        contract = (ROOT / "docs/environment-daily_kaggle_top100/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("daily/2026-07-26.html", contract)
        self.assertIn("daily/2026-07-25.html", contract)
        self.assertIn("tests.test_environment_daily_contract", contract)
        self.assertIn("submissionDate", contract)
        self.assertIn("PUBLIC + COMPLETED", contract)
        self.assertIn("leaderboard `score`", contract)
        self.assertIn("同分", contract)

    def test_current_report_highlights_the_user_team_everywhere(self) -> None:
        text = CURRENT_REPORT.read_text(encoding="utf-8")

        self.assertIn("我的位置", text)
        self.assertIn('data-current-user="true"', text)
        self.assertGreaterEqual(text.count('data-current-user="true"'), 4)
        self.assertIn("宝糕手", text)

    def test_current_report_embeds_100_matching_leaderboard_scores(self) -> None:
        text = CURRENT_REPORT.read_text(encoding="utf-8")
        match = re.search(
            r'<script type="application/json" id="snapshot-audit">(.*?)</script>',
            text,
            re.S,
        )
        self.assertIsNotNone(match)
        audit = json.loads(match.group(1))
        selected = audit["selected"]

        self.assertEqual(len(selected), 100)
        self.assertTrue(all("leaderboard_score" in row for row in selected))
        self.assertTrue(
            all(row["leaderboard_score"] == row["submission_public_score"] for row in selected)
        )
        self.assertIn("leaderboard score = submission publicScore", text)

    def test_latest_report_shows_relative_last_submission_and_audited_dual_decks(self) -> None:
        text = LATEST_REPORT.read_text(encoding="utf-8")
        parser = _DailyParser()
        parser.feed(text)
        audit_match = re.search(
            r'<script type="application/json" id="snapshot-audit">(.*?)</script>',
            text,
            re.S,
        )
        self.assertIsNotNone(audit_match)
        audit = json.loads(audit_match.group(1))

        self.assertEqual(parser.player_rows, 100)
        self.assertEqual(parser.player_details, 100)
        self.assertEqual(parser.high_score_decks, audit["audited_high_score_decks"])
        self.assertEqual(audit["dual_deck_players"], 55)
        self.assertIn("最后提交", text)
        self.assertIn("精确提交时间：", text)
        self.assertIn("第二高分不同构筑", text)

    def test_generator_is_date_parameterized(self) -> None:
        source = (
            ROOT / "data/processed/environment_daily/generate_live_snapshot.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('REPORT_DATE = "2026-07-27"', source)
        self.assertIn('parser.add_argument("--date"', source)
        self.assertIn("_validate_snapshot", source)
        self.assertIn("_update_index", source)
        self.assertNotIn("archive.train_legacy", source)

    def test_previous_report_parser_reads_the_latest_published_daily(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import _previous_players

        players = _previous_players("2026-07-30")
        self.assertEqual(len(players), 100)
        self.assertEqual(players["Dries @ Tufa Labs"]["rank"], 1)
        self.assertEqual(
            players["Dries @ Tufa Labs"]["archetype"],
            "Marnie's Grimmsnarl ex / Froslass",
        )
        self.assertEqual(players["Dries @ Tufa Labs"]["deck_hash"], "c20a8a46f5c6")

    def test_deck_hash_comparison_accepts_a_shared_display_prefix(self) -> None:
        from data.processed.environment_daily import generate_live_snapshot

        comparison = getattr(generate_live_snapshot, "_same_deck_hash", None)
        self.assertIsNotNone(comparison)
        self.assertTrue(comparison("c20a8a46f5c6", "c20a8a46f5c6"))
        self.assertTrue(comparison("c20a8a46f5c6", "c20a8a46f5"))
        self.assertFalse(comparison("c20a8a46f5c6", "f50fa3a23cdf"))
        self.assertFalse(comparison("", "c20a8a46f5c6"))

    def test_report_index_is_reverse_chronological(self) -> None:
        text = (
            ROOT / "docs/environment-daily_kaggle_top100/index.html"
        ).read_text(encoding="utf-8")
        dates = re.findall(r'<time datetime="(\d{4}-\d{2}-\d{2})">', text)
        self.assertEqual(dates, sorted(dates, reverse=True))


if __name__ == "__main__":
    unittest.main()
