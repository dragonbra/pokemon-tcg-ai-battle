from __future__ import annotations

import unittest

from train.kaggle_bc_top20.training.render_top20_evaluation import (
    ARCHETYPE_ORDER,
    _card_group,
    aggregate_matchups,
    card_image_url,
    classify_archetype,
)


class Top20EvaluationReportTests(unittest.TestCase):
    def test_classifies_all_audited_archetypes_by_key_pokemon(self) -> None:
        key_names = {
            "Dragapult ex": ("Dragapult ex",),
            "Cynthia's Garchomp ex": ("Cynthia's Garchomp ex",),
            "Alakazam": ("Alakazam",),
            "Marnie's Grimmsnarl ex": ("Marnie's Grimmsnarl ex",),
            "Mega Lopunny ex / Mega Froslass ex": (
                "Mega Lopunny ex",
                "Mega Froslass ex",
            ),
            "Team Rocket's Mewtwo ex / Spidops": (
                "Team Rocket's Mewtwo ex",
                "Team Rocket's Spidops",
            ),
            "Archaludon ex / Cinderace": ("Archaludon ex", "Cinderace"),
            "Mega Kangaskhan ex / Crustle": ("Mega Kangaskhan ex", "Crustle"),
        }
        self.assertEqual(tuple(key_names), ARCHETYPE_ORDER)
        for expected, pokemon in key_names.items():
            with self.subTest(expected=expected):
                profile = {"pokemon": [{"name": name} for name in pokemon]}
                self.assertEqual(classify_archetype(profile), expected)

    def test_rejects_an_unclassified_deck(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected one archetype"):
            classify_archetype({"pokemon": [{"name": "Pikachu"}]})

    def test_builds_legacy_and_recent_set_image_urls(self) -> None:
        self.assertEqual(
            card_image_url("TWM", "130"),
            "https://images.pokemontcg.io/sv6/130.png",
        )
        self.assertEqual(
            card_image_url("ASC", "47"),
            "https://images.scrydex.com/pokemon/me2pt5-47/small",
        )
        self.assertEqual(
            card_image_url("POR", "62"),
            "https://images.scrydex.com/pokemon/me3-62/small",
        )
        self.assertIsNone(card_image_url("UNKNOWN", "1"))

    def test_pokemon_tools_stay_in_the_trainer_group(self) -> None:
        metadata = {"stage_or_type": "Pokémon Tool", "hp": "n/a"}
        self.assertEqual(_card_group(metadata), "Trainer")

    def test_aggregates_exact_decks_into_archetypes(self) -> None:
        player = {
            "known_top_deck_matchups": {
                "by_opponent_deck": [
                    {
                        "opponent_deck_sha256": "dragapult-a",
                        "episodes": 3,
                        "wins": 2,
                        "losses": 1,
                        "draws": 0,
                        "unknown": 0,
                    },
                    {
                        "opponent_deck_sha256": "dragapult-b",
                        "episodes": 2,
                        "wins": 1,
                        "losses": 0,
                        "draws": 1,
                        "unknown": 0,
                    },
                ]
            }
        }
        result = aggregate_matchups(
            player,
            {
                "dragapult-a": "Dragapult ex",
                "dragapult-b": "Dragapult ex",
            },
        )
        self.assertEqual(result["Dragapult ex"]["episodes"], 5)
        self.assertEqual(result["Dragapult ex"]["wins"], 3)
        self.assertEqual(result["Dragapult ex"]["draws"], 1)
        self.assertEqual(result["Dragapult ex"]["win_rate"], 0.6)
        self.assertEqual(result["Alakazam"]["episodes"], 0)
        self.assertIsNone(result["Alakazam"]["win_rate"])


if __name__ == "__main__":
    unittest.main()
