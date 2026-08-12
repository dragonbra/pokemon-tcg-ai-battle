from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "engine_cuda_2_0/src/official_engine_kernels.cu"


class Semantic0031SignedHpContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = KERNEL.read_text(encoding="utf-8")
        start = source.index("__device__ bool semantic0031_add_card_entity(")
        end = source.index("\n__device__ ", start + 1)
        cls.function = source[start:end]

    def test_card_hp_preserves_signed_canonical_difference(self) -> None:
        self.assertIn(
            "const std::int32_t hp = max_hp - damage;",
            self.function,
        )
        self.assertNotIn(
            "const std::int32_t hp = max_hp > damage ? max_hp - damage : 0;",
            self.function,
        )

    def test_card_hp_arithmetic_and_output_are_signed(self) -> None:
        for name in ("max_hp", "damage", "hp"):
            self.assertRegex(
                self.function,
                rf"const\s+std::int32_t\s+{name}\s*=",
            )
        self.assertIn("float* card_num", self.function)
        self.assertRegex(
            self.function,
            re.compile(r"num\[0\]\s*=\s*static_cast<float>\(hp\);"),
        )


if __name__ == "__main__":
    unittest.main()
