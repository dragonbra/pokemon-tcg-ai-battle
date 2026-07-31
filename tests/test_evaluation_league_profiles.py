from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.frozen import load_frozen_catalog
from evaluation.metrics.league_profiles import (
    COMMON_METRICS,
    PROFILES,
    all_profiled_decks,
    profile_for_deck,
)


ROOT = Path(__file__).resolve().parents[1]


class LeagueQualityProfileTest(unittest.TestCase):
    def test_every_frozen_deck_has_exactly_one_nonempty_profile(self) -> None:
        catalog = load_frozen_catalog(
            ROOT / "evaluation" / "configs" / "frozen.json",
            ROOT / "evaluation",
        )
        frozen_names = tuple(sorted(package.name for package in catalog.opponents))

        self.assertEqual(all_profiled_decks(), frozen_names)
        self.assertEqual(len(PROFILES), 21)
        for name in frozen_names:
            profile = profile_for_deck(name)
            self.assertIn(name, profile.deck_ids)
            self.assertTrue(profile.key_card_ids)
            self.assertTrue(profile.focus_metrics)
            self.assertTrue(profile.interpretation)
            self.assertTrue(profile.reward_warning)

    def test_profiles_select_auditable_generic_metric_vocabulary(self) -> None:
        allowed = set(COMMON_METRICS) | {
            "ability_actions",
            "damage_events",
            "key_setup_round",
            "self_field_discards",
        }
        for profile in PROFILES:
            with self.subTest(profile=profile.profile_id):
                self.assertTrue(set(profile.focus_metrics) <= allowed)

    def test_reward_hazard_profiles_cover_non_greedy_archetypes(self) -> None:
        dusknoir = profile_for_deck("dragapult_ex_dusknoir_001")
        control = profile_for_deck("mega_kangaskhan_ex_crustle_004")
        spread = profile_for_deck("dragapult_ex_001")

        self.assertIn("自我 KO", dusknoir.reward_warning)
        self.assertIn("拖长比赛", control.reward_warning)
        self.assertIn("铺伤", spread.reward_warning)


if __name__ == "__main__":
    unittest.main()
