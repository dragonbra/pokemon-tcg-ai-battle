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
EXTRA_PROFILED_CANDIDATES = {
    "0024_lucario_hariyama_zero_shot",
    "raging_bolt_ogerpon_user_zero_shot",
}


class LeagueQualityProfileTest(unittest.TestCase):
    def test_every_frozen_deck_has_exactly_one_nonempty_profile(self) -> None:
        catalog = load_frozen_catalog(
            ROOT / "evaluation" / "configs" / "frozen.json",
            ROOT / "evaluation",
        )
        frozen_names = tuple(sorted(package.name for package in catalog.opponents))

        expected_names = tuple(sorted(set(frozen_names) | EXTRA_PROFILED_CANDIDATES))

        self.assertEqual(all_profiled_decks(), expected_names)
        self.assertEqual(len(PROFILES), 25)
        for name in frozen_names:
            profile = profile_for_deck(name)
            self.assertIn(name, profile.deck_ids)
            self.assertTrue(profile.key_card_ids)
            self.assertTrue(profile.focus_metrics)
            self.assertTrue(profile.interpretation)
            self.assertTrue(profile.reward_warning)

    def test_lucario_hariyama_candidate_has_auditable_profile(self) -> None:
        profile = profile_for_deck("0024_lucario_hariyama_zero_shot")

        self.assertEqual(profile.profile_id, "lucario_hariyama_energy")
        self.assertEqual(profile.key_card_ids[:2], (678, 674))
        self.assertIn("max_evolved_pokemon", profile.focus_metrics)
        self.assertIn("Hariyama", profile.interpretation)

    def test_raging_bolt_ogerpon_candidate_has_auditable_profile(self) -> None:
        profile = profile_for_deck("raging_bolt_ogerpon_user_zero_shot")

        self.assertEqual(profile.profile_id, "raging_bolt_ogerpon_energy")
        self.assertEqual(profile.key_card_ids, (63, 96, 226))
        self.assertIn("attack_continuity", profile.focus_metrics)
        self.assertIn("Prize", profile.interpretation)

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
