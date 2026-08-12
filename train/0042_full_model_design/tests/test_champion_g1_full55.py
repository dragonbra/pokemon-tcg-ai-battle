from __future__ import annotations

import importlib
import unittest


controller = importlib.import_module(
    "train.0042_full_model_design.evaluation.run_champion_g1_full55"
)


class ChampionG1Full55Test(unittest.TestCase):
    def test_archetype_first_order_is_exact(self) -> None:
        order = controller.evaluation_order()
        self.assertEqual(
            order[:14],
            ("001", "002", "003", "006", "007", "009", "010", "011",
             "013", "019", "020", "021", "044", "048"),
        )
        self.assertEqual(len(order), 55)
        self.assertEqual(set(order), {f"{number:03d}" for number in range(1, 56)})
        self.assertEqual(order[14:], tuple(sorted(order[14:])))

    def test_other_is_deferred_to_second_phase(self) -> None:
        self.assertNotIn("047", controller.evaluation_order()[:14])


if __name__ == "__main__":
    unittest.main()
