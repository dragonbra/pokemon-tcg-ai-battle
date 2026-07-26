from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/environment-daily_kaggle_top100/daily/2026-07-27.html"
SNAPSHOT = ROOT / ".tmp/environment_daily_0727_live_20260727_0406_bounded/snapshot.json"


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
        if "card-thumb" in str(values.get("class", "")).split():
            self.card_thumbs += 1
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

    def test_completed_capture_cannot_be_silently_reused(self) -> None:
        from data.processed.environment_daily.generate_live_snapshot import collect

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            snapshot = {
                "rows": [{"rank": rank} for rank in range(1, 101)],
                "players": {str(rank): {} for rank in range(1, 101)},
            }
            (work / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completed leaderboard capture is immutable"):
                collect(work, work / "report.html")

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

    def test_0727_snapshot_has_100_exact_60_card_decks(self) -> None:
        snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        cutoff = snapshot["captured_at_utc"]
        self.assertEqual(len(snapshot["rows"]), 100)
        self.assertEqual(len(snapshot["players"]), 100)
        for rank in range(1, 101):
            player = snapshot["players"][str(rank)]
            self.assertEqual(len(player["deck"]), 60)
            self.assertIn(player["episode_player_index"], (0, 1))
            self.assertLessEqual(player["episode_create_time"], cutoff)
            self.assertEqual(len(player["deck_sha256"]), 64)

    def test_project_contract_names_both_ui_baselines(self) -> None:
        contract = (ROOT / "docs/environment-daily_kaggle_top100/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("daily/2026-07-26.html", contract)
        self.assertIn("daily/2026-07-25.html", contract)
        self.assertIn("tests.test_environment_daily_contract", contract)


if __name__ == "__main__":
    unittest.main()
