from __future__ import annotations

import unittest
from collections import Counter

from engine_cuda_2_0.tools import evaluate_sp_series_cuda as subject


class SpSeriesCudaEvaluationTests(unittest.TestCase):
    def test_archive_contract_contains_sp01_through_sp09(self) -> None:
        self.assertEqual(
            subject.EXPECTED_IDS,
            tuple(f"SP{index:02d}_MAGA" for index in range(1, 10)),
        )

    def test_sp09_is_exact_user_provided_sixty_card_deck(self) -> None:
        deck = subject._read_deck(
            subject.SERIES_ROOT / "decks/SP09_MAGA/deck.csv"
        )
        counts = Counter(deck)
        self.assertEqual(len(deck), 60)
        self.assertEqual(
            counts,
            Counter(
                {
                    5: 2,
                    13: 1,
                    19: 4,
                    66: 2,
                    140: 1,
                    305: 3,
                    343: 1,
                    741: 4,
                    742: 4,
                    743: 3,
                    1079: 3,
                    1081: 4,
                    1086: 4,
                    1097: 1,
                    1129: 1,
                    1152: 4,
                    1182: 3,
                    1184: 1,
                    1197: 3,
                    1225: 4,
                    1231: 4,
                    1264: 3,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
