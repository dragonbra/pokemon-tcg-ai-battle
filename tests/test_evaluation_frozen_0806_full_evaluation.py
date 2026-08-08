from __future__ import annotations

import unittest
from dataclasses import replace

from evaluation.frozen_0806_full_evaluation import (
    EXPECTED_FIRST,
    EXPECTED_GAMES,
    EXPECTED_SECOND,
    LEGACY_POLICY_0806_OUTPUT_ROOT,
    POLICY_0806_TARGET,
    _schedule_counts,
    _turn_order,
    validate_report_payload,
)
from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.frozen_0806_contract import evaluation_schedule_id
from evaluation.reporting.html import _matchup_html


class Frozen0806FullEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_frozen_0806_runtime_catalog()

    def test_schedule_is_exactly_seeded_2048_and_seat_balanced(self) -> None:
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
        self.assertEqual(EXPECTED_GAMES, 2048)
        self.assertEqual(EXPECTED_FIRST, 1024)
        self.assertEqual(EXPECTED_SECOND, 1024)

    def test_policy_0806_seeded2048_output_is_independent_from_legacy(self) -> None:
        self.assertNotEqual(POLICY_0806_TARGET.output_root, LEGACY_POLICY_0806_OUTPUT_ROOT)
        self.assertEqual(
            POLICY_0806_TARGET.output_root.name,
            "0806_kaggle_top100_plus_v1_seeded_2048_v2",
        )
        self.assertEqual(
            LEGACY_POLICY_0806_OUTPUT_ROOT.name,
            "0806_kaggle_top100_plus_v1",
        )

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

    def test_policy_0806_opponents_reuse_the_candidate_runtime_and_exact_decks(self) -> None:
        catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
        self.assertEqual(catalog.opponent_policy.root, catalog.candidate_policy.root)
        self.assertEqual(len(catalog.opponents), 55)
        self.assertEqual(POLICY_0806_TARGET.label, "Policy-0806")
        for candidate, opponent in zip(
            catalog.candidates, catalog.opponents, strict=True
        ):
            self.assertEqual(opponent.deck, candidate.deck)
            self.assertEqual(
                opponent.package_manifest["frozen_policy_label"], "Policy-0806"
            )
            self.assertEqual(
                opponent.package_manifest["frozen_deck_number"],
                candidate.package_manifest["frozen_deck_number"],
            )

    def test_pending_candidates_can_be_ordered_by_display_number(self) -> None:
        published_ids = {self.catalog.candidates[0].name}
        pending = sorted(
            (
                candidate
                for candidate in self.catalog.candidates
                if candidate.name not in published_ids
            ),
            key=lambda candidate: int(
                candidate.package_manifest["frozen_deck_number"]
            ),
        )
        numbers = [
            int(candidate.package_manifest["frozen_deck_number"])
            for candidate in pending
        ]
        self.assertEqual(numbers, sorted(numbers))

    def test_special_deck_presentations_match_exact_deck_identity(self) -> None:
        by_number = {
            package.package_manifest["frozen_deck_number"]: package
            for package in self.catalog.candidates
        }
        ogerpon = by_number["015"]
        slowking = by_number["016"]
        self.assertEqual(
            ogerpon.display_name,
            "Wellspring Mask Ogerpon ex / Teal Mask Ogerpon ex",
        )
        self.assertEqual(
            [card["card_id"] for card in ogerpon.representative_cards],
            [108, 96],
        )
        self.assertEqual(ogerpon.package_manifest["frozen_report_href"], "015_wellspring_mask_ogerpon_ex_teal_mask_ogerpon_ex.html")
        self.assertEqual(slowking.display_name, "Slowking Toolbox")
        self.assertEqual(
            [card["card_id"] for card in slowking.representative_cards],
            [163],
        )
        self.assertEqual(
            slowking.package_manifest["frozen_report_href"],
            "016_slowking_toolbox.html",
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
                        "opponent_schedule_id": evaluation_schedule_id(
                            self.catalog.pool.manifest["schedule_sha256"]
                        ),
                        "seed": 341_512_806,
                        "games_per_opponent": list(_schedule_counts(self.catalog)),
                        "opponents": [{}] * 55,
                        "games": 512,
                    },
                    "summary": {
                        "total_games": 511,
                        "completed_games": 511,
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
