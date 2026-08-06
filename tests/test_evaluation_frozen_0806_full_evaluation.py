from __future__ import annotations

import unittest
from dataclasses import replace

from evaluation.frozen_0806_full_evaluation import (
    EXPECTED_FIRST,
    EXPECTED_GAMES,
    EXPECTED_SECOND,
    _schedule_counts,
    _turn_order,
    validate_report_payload,
)
from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.reporting.html import _matchup_html


class Frozen0806FullEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_frozen_0806_runtime_catalog()

    def test_schedule_is_exactly_256_and_seat_balanced(self) -> None:
        counts = _schedule_counts(self.catalog)
        self.assertEqual(len(counts), 55)
        self.assertEqual(sum(counts), EXPECTED_GAMES)
        games = [
            {"candidate_first": index % 2 == 0, "winner": 0}
            for index in range(EXPECTED_GAMES)
        ]
        order = _turn_order(games)
        self.assertEqual(order["first"]["games"], EXPECTED_FIRST)
        self.assertEqual(order["second"]["games"], EXPECTED_SECOND)

    def test_catalog_assigns_stable_three_digit_report_numbers(self) -> None:
        by_number = sorted(
            self.catalog.opponents,
            key=lambda package: package.package_manifest["frozen_deck_number"],
        )
        self.assertEqual(
            [package.package_manifest["frozen_deck_number"] for package in by_number],
            [f"{number:03d}" for number in range(1, 56)],
        )
        schedule_by_id = {entry.deck_id: entry for entry in self.catalog.pool.schedule}
        frequency_keys = [
            (
                -schedule_by_id[package.name].games,
                schedule_by_id[package.name].best_rank,
                package.name,
            )
            for package in by_number
        ]
        self.assertEqual(frequency_keys, sorted(frequency_keys))
        self.assertEqual(
            self.catalog.opponents[0].package_manifest["frozen_report_href"],
            "007_dragapult_ex.html",
        )

    def test_matchup_row_links_numbered_report_and_shows_sample_size(self) -> None:
        opponent = self.catalog.opponents[0]
        rendered = _matchup_html(
            {
                "by_opponent": {
                    opponent.name: {
                        "games": 7,
                        "wins": 4,
                        "losses": 3,
                        "draws": 0,
                        "win_rate": 4 / 7,
                    }
                }
            },
            {
                "opponents": [
                    {
                        "name": opponent.name,
                        "display_name": opponent.display_name,
                        "representative_cards": opponent.representative_cards,
                        "package_manifest": opponent.package_manifest,
                    }
                ]
            },
        )
        number = opponent.package_manifest["frozen_deck_number"]
        self.assertIn(f'class="deck-number">{number}</span>', rendered)
        self.assertIn(
            f'href="{opponent.package_manifest["frozen_report_href"]}"', rendered
        )
        self.assertIn("4-3-0 / 7局", rendered)

    def test_acceptance_rejects_partial_report(self) -> None:
        candidate = self.catalog.candidates[0]
        with self.assertRaisesRegex(ValueError, "partial"):
            validate_report_payload(
                {
                    "manifest": {
                        "candidate": {
                            "deck": candidate.deck,
                            "package_manifest": candidate.package_manifest,
                        },
                        "opponent_pool": {
                            "pool_id": self.catalog.pool.pool_id,
                            "catalog_sha256": self.catalog.pool.manifest_sha256,
                            "policy_hash": self.catalog.pool.policies["opponent"]["weights_sha256"],
                        },
                        "opponent_schedule_id": self.catalog.pool.manifest["schedule_sha256"],
                        "games_per_opponent": list(_schedule_counts(self.catalog)),
                        "opponents": [{}] * 55,
                        "games": 256,
                    },
                    "summary": {
                        "total_games": 255,
                        "completed_games": 255,
                        "errors": 0,
                        "unfinished": 0,
                    },
                    "games": [],
                },
                candidate,
                self.catalog,
            )

    def test_acceptance_rejects_wrong_candidate_checkpoint(self) -> None:
        candidate = self.catalog.candidates[0]
        manifest = dict(candidate.package_manifest)
        manifest["checkpoint_sha256"] = "0" * 64
        wrong = replace(candidate, package_manifest=manifest)
        with self.assertRaisesRegex(ValueError, "candidate policy"):
            validate_report_payload(
                {
                    "manifest": {"candidate": {"package_manifest": wrong.package_manifest}},
                    "summary": {},
                    "games": [],
                },
                wrong,
                self.catalog,
            )


if __name__ == "__main__":
    unittest.main()
