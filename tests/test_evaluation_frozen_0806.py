from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.frozen_0806 import (
    POLICY_0019_SHA256,
    POLICY_0806_SHA256,
    allocate_largest_remainder,
    load_frozen_0806_pool,
)
from evaluation.frozen_0806_assets import materialize


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "evaluation"
CONFIG = EVALUATION_ROOT / "configs" / "frozen_0806.json"
SOURCE = (
    ROOT
    / "docs"
    / "environment-daily_kaggle_top100"
    / "ranked"
    / "data"
    / "2026-08-06-top500-exact.json"
)


class Frozen0806AllocationTest(unittest.TestCase):
    def test_largest_remainder_is_deterministic(self) -> None:
        groups = [
            ("later", 1, 20),
            ("larger", 2, 30),
            ("earlier", 1, 10),
        ]

        allocation = allocate_largest_remainder(groups, population=4, total=10)

        self.assertEqual(allocation, {"later": 2, "larger": 5, "earlier": 3})


class Frozen0806PoolTest(unittest.TestCase):
    def test_committed_pool_has_fixed_256_game_distribution(self) -> None:
        pool = load_frozen_0806_pool(CONFIG, EVALUATION_ROOT)

        self.assertEqual(pool.pool_id, "0806_kaggle_top100_plus_v1")
        self.assertEqual(pool.total_games, 256)
        self.assertEqual(len(pool.decks), 55)
        self.assertEqual(sum(item.games for item in pool.schedule), 256)
        self.assertEqual(
            sum(item.games for item in pool.schedule if item.segment == "top100"), 240
        )
        self.assertEqual(
            sum(item.games for item in pool.schedule if item.segment == "potential_101_500"),
            16,
        )
        self.assertEqual(pool.policies["opponent"]["weights_sha256"], POLICY_0019_SHA256)
        self.assertEqual(pool.policies["main"]["weights_sha256"], POLICY_0806_SHA256)
        self.assertEqual(len({deck.exact_deck_sha256 for deck in pool.decks}), 55)
        self.assertTrue(all(len(deck.cards) == 60 for deck in pool.decks))

    def test_top100_schedule_reproduces_observed_population(self) -> None:
        pool = load_frozen_0806_pool(CONFIG, EVALUATION_ROOT)
        top = [item for item in pool.schedule if item.segment == "top100"]

        self.assertEqual(len(top), 41)
        self.assertEqual(sum(item.observed_players for item in top), 100)
        expected = allocate_largest_remainder(
            [
                (item.exact_deck_sha256, item.observed_players, item.best_rank)
                for item in top
            ],
            population=100,
            total=240,
        )
        self.assertEqual(
            {item.exact_deck_sha256: item.games for item in top}, expected
        )

    def test_tail_selection_and_corrected_starmie_labels_are_explicit(self) -> None:
        pool = load_frozen_0806_pool(CONFIG, EVALUATION_ROOT)
        tail = [item for item in pool.schedule if item.segment == "potential_101_500"]
        by_archetype: dict[str, int] = {}
        for item in tail:
            by_archetype[item.archetype] = by_archetype.get(item.archetype, 0) + item.games

        self.assertEqual(
            by_archetype,
            {
                "Arboliva ex / Meganium / Teal Mask Ogerpon ex": 2,
                "Mega Starmie ex / Mega Froslass ex": 3,
                "Mega Starmie ex / Dusknoir": 2,
                "Archaludon ex / Cinderace": 4,
                "Dragapult ex / Dusknoir": 2,
                "Erika's Vileplume ex / Cinderace": 1,
                "N's Zoroark ex / Munkidori": 2,
            },
        )
        froslass = [item for item in tail if "Mega Froslass" in item.archetype]
        dusknoir = [item for item in tail if item.archetype == "Mega Starmie ex / Dusknoir"]
        self.assertEqual({item.selection_rank for item in froslass}, {204, 210, 425})
        self.assertEqual({item.selection_rank for item in dusknoir}, {228, 350})

    def test_materializer_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as left_tmp, tempfile.TemporaryDirectory() as right_tmp:
            left = Path(left_tmp) / "pool"
            right = Path(right_tmp) / "pool"
            materialize(source_snapshot=SOURCE, target=left)
            materialize(source_snapshot=SOURCE, target=right)

            def tree_hash(root: Path) -> str:
                digest = hashlib.sha256()
                for path in sorted(item for item in root.rglob("*") if item.is_file()):
                    digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                    digest.update(path.read_bytes())
                return digest.hexdigest()

            self.assertEqual(tree_hash(left), tree_hash(right))
            manifest = json.loads((left / "manifest.json").read_text())
            self.assertEqual(manifest["deck_count"], 55)


if __name__ == "__main__":
    unittest.main()
