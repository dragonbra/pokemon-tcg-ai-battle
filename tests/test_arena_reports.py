from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arena.models import DeckRecord, RunRecord
from arena.reports import build_report_snapshot, render_html, render_markdown, write_reports
from arena.storage import ArenaStore


def fixture_snapshot() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "generated_at": "2026-07-22T00:00:00Z",
        "quality_filter": "all",
        "ratings": [
            {
                "deck_id": "good-deck",
                "display_name": "Lucario - Good",
                "archetype": "Lucario",
                "mu": 650,
                "sigma": 120,
                "elo": 640,
                "games": 20,
                "status": "active",
                "role": "public",
                "primary_pokemon": [
                    {"card_id": 123, "name": "Lucario", "image_url": "https://img.example/lucario.png"}
                ],
            },
            {
                "deck_id": "bad-deck",
                "display_name": "Bad",
                "archetype": "Unknown Archetype",
                "mu": 250,
                "sigma": 100,
                "elo": 260,
                "games": 40,
                "status": "demoted",
                "role": "public",
                "primary_pokemon": [],
            },
        ],
        "pair_matrix": [
            {"a": "good-deck", "b": "bad-deck", "wins_a": 8, "wins_b": 2, "draws": 0, "games": 10}
        ],
        "sources": [
            {
                "deck_id": "good-deck",
                "title": "<unsafe>",
                "url": "https://www.kaggle.com/code/u/k",
                "author": "u",
                "votes": 8,
                "public_score": 900,
                "status": "exact_submission",
            }
        ],
        "summary": {"total_games": 10, "wins": 8, "losses": 2, "draws": 0, "win_rate": 0.8},
        "demoted": [{"deck_id": "bad-deck", "display_name": "Bad", "mu": 250}],
        "config": {"rating_method": "official_gaussian_approx"},
    }


class ReportTests(unittest.TestCase):
    def test_eligible_report_excludes_demoted_but_keeps_all_report_rows(self) -> None:
        snapshot = fixture_snapshot()
        all_html = render_html(snapshot)
        eligible_html = render_html(snapshot | {"quality_filter": "eligible"})

        self.assertIn("低质量附录", all_html)
        self.assertIn('data-deck-id="bad-deck"', all_html)
        self.assertNotIn('data-deck-id="bad-deck"', eligible_html)
        self.assertIn("bad-deck", eligible_html)

    def test_html_escapes_source_title_and_contains_heatmap_and_image_url(self) -> None:
        html = render_html(fixture_snapshot())

        self.assertIn("&lt;unsafe&gt;", html)
        self.assertIn("克制关系热力图", html)
        self.assertIn('loading="lazy"', html)
        self.assertIn("https://img.example/lucario.png", html)
        self.assertNotIn("<unsafe>", html)

    def test_markdown_contains_two_rating_methods_and_source_link(self) -> None:
        markdown = render_markdown(fixture_snapshot())

        self.assertIn("official_gaussian_approx", markdown)
        self.assertIn("elo_compat", markdown)
        self.assertIn("https://www.kaggle.com/code/u/k", markdown)

    def test_write_reports_creates_immutable_and_latest_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            written = write_reports(fixture_snapshot(), root)
            self.assertEqual({path.name for path in written}, {"report.html", "report.md"})
            self.assertTrue((root / "run-1" / "report.html").is_file())
            self.assertTrue((root / "latest.html").is_file())

    def test_snapshot_uses_checkpoint_status_for_demotion_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ArenaStore(Path(temporary) / "arena")
            store.initialize()
            store.upsert_deck(DeckRecord("bad", "source", "Bad", "Unknown", (), "bad", None, "public", "valid", {}))
            store.create_run(RunRecord("run-1", "continuous", {}, "now", None, "running"))
            store.record_checkpoint(
                "run-1",
                20,
                {"bad": {"mu": 250, "sigma": 100, "elo": 250, "games": 20, "status": "demoted", "below_threshold_checkpoints": 2}},
                "later",
            )

            snapshot = build_report_snapshot(store, "run-1", include_demoted=True)

            self.assertEqual(snapshot["ratings"][0]["status"], "demoted")

    def test_snapshot_excludes_invalid_and_metadata_only_from_ranking_but_keeps_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ArenaStore(Path(temporary) / "arena")
            store.initialize()
            for deck_id, status in (("valid", "valid"), ("invalid", "invalid"), ("metadata", "metadata_only")):
                store.upsert_deck(
                    DeckRecord(deck_id, deck_id, deck_id, "Unknown", (), deck_id, None, "public", status, {})
                )
            store.create_run(RunRecord("run-1", "smoke", {}, "now", None, "running"))

            snapshot = build_report_snapshot(store, "run-1", include_demoted=True)

            self.assertEqual({row["deck_id"] for row in snapshot["ratings"]}, {"valid"})
            self.assertEqual(
                {row["deck_id"] for row in snapshot["sources"]},
                {"valid", "invalid", "metadata"},
            )


if __name__ == "__main__":
    unittest.main()
