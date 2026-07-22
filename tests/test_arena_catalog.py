from __future__ import annotations

import unittest

from arena.catalog import classify_deck


class CatalogTests(unittest.TestCase):
    def test_classification_uses_pokemon_only_and_prefers_core_evolution(self) -> None:
        metadata = {
            1: {"Card Name": "Dunsparce", "Stage (Pokémon)/Type (Energy and Trainer)": "Basic Pokémon"},
            2: {"Card Name": "Alakazam", "Stage (Pokémon)/Type (Energy and Trainer)": "Stage 2 Pokémon", "Expansion": "MEG", "Collection No.": "1"},
            3: {"Card Name": "Hero's Cape", "Stage (Pokémon)/Type (Energy and Trainer)": "Pokémon Tool"},
        }

        archetype, primary_ids, display_name = classify_deck([1] * 4 + [2] * 4 + [3], metadata)

        self.assertEqual(archetype, "Alakazam")
        self.assertEqual(primary_ids[0], 2)
        self.assertEqual(display_name, "Alakazam")


if __name__ == "__main__":
    unittest.main()
