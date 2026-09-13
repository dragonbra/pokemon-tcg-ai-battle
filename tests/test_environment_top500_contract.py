from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest


class EnvironmentTop500ContractTests(unittest.TestCase):
    def _catalog(self):
        from data.processed.environment_daily.generate_current_top500_report import (
            load_reference_catalog,
        )

        return load_reference_catalog()

    def test_score_band_boundaries(self) -> None:
        from data.processed.environment_daily.generate_current_top500_report import score_band

        self.assertEqual(score_band(1000), "1000+")
        self.assertEqual(score_band(999.9), "900-1000")
        self.assertEqual(score_band(900), "900-1000")
        self.assertEqual(score_band(899.9), "800-900")
        self.assertEqual(score_band(800), "800-900")
        self.assertEqual(score_band(799.9), "700-800")
        self.assertEqual(score_band(700), "700-800")
        self.assertEqual(score_band(699.9), "<700")

    def test_exact_001_070_match_has_precedence(self) -> None:
        from data.processed.environment_daily.generate_current_top500_report import (
            classify_decks,
        )

        reference = self._catalog()
        known = reference["decks"][0]
        payload = {"players": [{"decks": [{"deck": list(known["deck"])}]}]}
        classify_decks(payload, reference=reference)
        result = payload["players"][0]["decks"][0]["classification"]
        self.assertEqual(result["kind"], "catalog_exact")
        self.assertEqual(result["deck_id"], "001")
        self.assertEqual(result["report_deck_id"], "001")

    def test_unknown_decks_receive_stable_meta_scoped_ids(self) -> None:
        from data.processed.environment_daily.generate_current_top500_report import (
            classify_decks,
        )

        reference = self._catalog()
        base = list(reference["decks"][0]["deck"])
        first = base.copy()
        first[-1] = 1 if first[-1] != 1 else 2
        second = base.copy()
        second[-2] = 3 if second[-2] != 3 else 4
        payload = {
            "players": [
                {"decks": [{"deck": second}]},
                {"decks": [{"deck": first}]},
                {"decks": [{"deck": second}]},
            ]
        }
        classify_decks(payload, reference=reference)
        rows = [player["decks"][0]["classification"] for player in payload["players"]]
        self.assertTrue(all(row["kind"] == "meta_nearest" for row in rows))
        self.assertRegex(rows[0]["report_deck_id"], r"^00_[a-z0-9_]+_P\d{2}$")
        self.assertEqual(rows[0]["report_deck_id"], rows[2]["report_deck_id"])
        self.assertNotEqual(rows[0]["report_deck_id"], rows[1]["report_deck_id"])
        self.assertIn("nearest_deck_id", rows[0]["audit"])
        self.assertIn("similarity", rows[0]["audit"])

    def test_validation_counts_people_once_and_accepts_two_distinct_decks(self) -> None:
        from data.processed.environment_daily.generate_current_top500_report import (
            canonical_deck_sha256,
            validate_snapshot,
        )

        reference = self._catalog()
        deck_a = list(reference["decks"][0]["deck"])
        deck_b = list(reference["decks"][1]["deck"])
        payload = {
            "schema": "pokemon_tcg_current_top500_decks_v1",
            "captured_at_utc": "2026-08-16T12:00:00+00:00",
            "leaderboard_count": 1,
            "players": [
                {
                    "rank": 1,
                    "team_id": 9,
                    "team_name": "team",
                    "score": 1001.0,
                    "submission_date": "2026-08-16T11:00:00+00:00",
                    "submission_id": 10,
                    "submission_public_score": "1001.0",
                    "score_binding_delta": 0.0,
                    "binding": "leaderboard_score",
                    "decks": [
                        {
                            "role": "leaderboard_primary",
                            "submission_id": 10,
                            "public_score": "1001.0",
                            "submitted_at": "2026-08-16T11:00:00+00:00",
                            "episode_id": 100,
                            "episode_create_time": "2026-08-16T11:30:00+00:00",
                            "episode_player_index": 0,
                            "deck_sha256": canonical_deck_sha256(deck_a),
                            "deck": deck_a,
                        },
                        {
                            "role": "alternate_high_score",
                            "submission_id": 11,
                            "public_score": "990.0",
                            "submitted_at": "2026-08-15T11:00:00+00:00",
                            "episode_id": 101,
                            "episode_create_time": "2026-08-15T11:30:00+00:00",
                            "episode_player_index": 1,
                            "deck_sha256": canonical_deck_sha256(deck_b),
                            "deck": deck_b,
                        },
                    ],
                }
            ],
        }
        audit = validate_snapshot(payload, expected_count=1)
        self.assertEqual(audit["players"], 1)
        self.assertEqual(audit["deck_evidence"], 2)
        self.assertEqual(audit["dual_deck_players"], 1)
        self.assertEqual(audit["score_bands"]["1000+"], 1)

        broken = deepcopy(payload)
        broken["players"][0]["decks"][1]["deck"].pop()
        with self.assertRaisesRegex(ValueError, "60 cards"):
            validate_snapshot(broken, expected_count=1)

        drifted = deepcopy(payload)
        drifted["players"][0]["binding"] = "active_submission_score_at_team_capture"
        drifted["players"][0]["submission_public_score"] = "1002.5"
        drifted["players"][0]["score_binding_delta"] = 1.5
        drift_audit = validate_snapshot(drifted, expected_count=1)
        self.assertEqual(drift_audit["strict_score_bindings"], 0)
        self.assertEqual(drift_audit["dynamic_score_bindings"], 1)
        self.assertFalse(drift_audit["all_scores_bound"])

    def test_renderer_has_required_distribution_and_player_contract(self) -> None:
        from data.processed.environment_daily.generate_current_top500_report import (
            canonical_deck_sha256,
            classify_decks,
            render_report,
        )

        reference = self._catalog()
        deck = list(reference["decks"][0]["deck"])
        payload = {
            "schema": "pokemon_tcg_current_top500_decks_v1",
            "captured_at_utc": "2026-08-16T12:00:00+00:00",
            "leaderboard_count": 1,
            "coverage": {"lowest_rank": 1, "lowest_score": 1001.0, "reached_700": False},
            "players": [
                {
                    "rank": 1,
                    "team_id": 9,
                    "team_name": "team",
                    "score": 1001.0,
                    "submission_date": "2026-08-16T11:00:00+00:00",
                    "submission_id": 10,
                    "submission_public_score": "1001.0",
                    "score_binding_delta": 0.0,
                    "binding": "leaderboard_score",
                    "decks": [
                        {
                            "role": "leaderboard_primary",
                            "submission_id": 10,
                            "public_score": "1001.0",
                            "submitted_at": "2026-08-16T11:00:00+00:00",
                            "episode_id": 100,
                            "episode_create_time": "2026-08-16T11:30:00+00:00",
                            "episode_player_index": 0,
                            "deck_sha256": canonical_deck_sha256(deck),
                            "deck": deck,
                        }
                    ],
                }
            ],
        }
        classify_decks(payload, reference=reference)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            render_report(payload, output, reference=reference)
            text = output.read_text(encoding="utf-8")
        for section_id in (
            "summary",
            "score-bands",
            "catalog-distribution",
            "meta-distribution",
            "rank-index",
            "player-decks",
            "evidence-boundary",
        ):
            self.assertIn(f'id="{section_id}"', text)
        self.assertEqual(text.count('data-catalog-bar="'), 70)
        self.assertIn('id="catalog-score-table"', text)
        self.assertIn("提交时间", text)
        self.assertIn("data-player-row", text)
        self.assertIn("card-thumb", text)


if __name__ == "__main__":
    unittest.main()
