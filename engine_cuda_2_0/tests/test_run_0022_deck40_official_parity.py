from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from engine_cuda_2_0.tools.run_0022_deck40_official_parity import make_cases


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]


class Deck40PairGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plugins = [
            SimpleNamespace(
                deck_id=deck_id,
                root=CUDA_ENGINE_ROOT / "tests" / "fixtures" / deck_id,
            )
            for deck_id in ("alpha_one", "beta_two", "gamma_three")
        ]

    def test_mirrors_generates_one_self_pair_per_deck(self) -> None:
        cases = make_cases(
            self.plugins,
            focal_deck_id="alpha_one",
            pair_mode="mirrors",
            include_mirrors=False,
            case_limit=0,
        )

        self.assertEqual(len(cases), len(self.plugins))
        self.assertEqual(
            [(case["deck0_name"], case["deck1_name"]) for case in cases],
            [(plugin.deck_id, plugin.deck_id) for plugin in self.plugins],
        )

    def test_all_mode_still_controls_mirrors_with_flag(self) -> None:
        without_mirrors = make_cases(
            self.plugins,
            focal_deck_id="alpha_one",
            pair_mode="all",
            include_mirrors=False,
            case_limit=0,
        )
        with_mirrors = make_cases(
            self.plugins,
            focal_deck_id="alpha_one",
            pair_mode="all",
            include_mirrors=True,
            case_limit=0,
        )

        self.assertEqual(len(without_mirrors), 6)
        self.assertEqual(len(with_mirrors), 9)


if __name__ == "__main__":
    unittest.main()
