from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    ROOT
    / "docs/environment-daily_kaggle_top100/ranked/data/2026-08-06-top500-exact.json"
)


class _ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.player_rows = 0
        self.player_details = 0
        self.variant_rows = 0
        self.pool_tables = 0
        self.card_images = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if "data-player-row" in values:
            self.player_rows += 1
        if "data-player-detail" in values:
            self.player_details += 1
        if "data-variant-row" in values:
            self.variant_rows += 1
        if tag == "table" and "pool-table" in str(values.get("class", "")).split():
            self.pool_tables += 1
        if tag == "img" and str(values.get("src", "")).startswith("https://"):
            self.card_images += 1


class RankedEnvironmentReportTests(unittest.TestCase):
    def test_snapshot_is_a_complete_top500_rank_and_deck_view(self) -> None:
        payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        players = payload["players"]

        self.assertEqual(payload["schema"], "pokemon_tcg_top500_exact_v1")
        self.assertEqual(len(players), 500)
        self.assertEqual(sorted(player["rank"] for player in players), list(range(1, 501)))
        self.assertTrue(all(len(player["deck"]) == 60 for player in players))

    def test_renderer_emits_ranked_deck_analysis_contract(self) -> None:
        from data.processed.environment_daily.generate_ranked_deck_report import (
            render_report,
        )

        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.html"
            summary = render_report(SNAPSHOT, report)
            source = report.read_text(encoding="utf-8")

        self.assertEqual(summary["players"], 500)
        self.assertEqual(summary["exact_score_bindings"], 240)
        self.assertEqual(summary["top100"], 100)
        self.assertEqual(summary["top500"], 500)
        self.assertIn("Top 1000", source)
        self.assertIn("Top 5000", source)
        self.assertIn("当前快照不可重建", source)
        self.assertIn("不使用 15 局胜率", source)
        self.assertIn("完全相同构筑", source)

        parser = _ReportParser()
        parser.feed(source)
        required_ids = {
            "summary",
            "rank-bands",
            "archetype-distribution",
            "exact-variants",
            "notable-decks",
            "card-pool",
            "rank-index",
            "player-details",
            "evidence-boundary",
        }
        self.assertTrue(required_ids <= parser.ids)
        self.assertEqual(parser.player_rows, 500)
        self.assertEqual(parser.player_details, 500)
        self.assertGreater(parser.variant_rows, 50)
        self.assertEqual(parser.pool_tables, 2)
        self.assertGreater(parser.card_images, 500)


if __name__ == "__main__":
    unittest.main()
