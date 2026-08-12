from __future__ import annotations

import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from engine_cuda_2_0.tests.test_official_ir import minimal_payload  # noqa: E402
from ptcg_cuda_engine.official_rule_pack import (  # noqa: E402
    ATTACK,
    CARD,
    CONDITION,
    CONTINUATION,
    EFFECT,
    HEADER,
    NAME_SET,
    SKILL,
    TARGET,
    TRIGGER,
    compile_official_rule_pack,
    validate_official_rule_pack,
)


class OfficialRulePackTest(unittest.TestCase):
    def test_python_layout_matches_cuda_abi(self) -> None:
        self.assertEqual(HEADER.size, 256)
        self.assertEqual(CARD.size, 96)
        self.assertEqual(SKILL.size, 80)
        self.assertEqual(ATTACK.size, 96)
        self.assertEqual(EFFECT.size, 80)
        self.assertEqual(TARGET.size, 48)
        self.assertEqual(CONDITION.size, 32)
        self.assertEqual(TRIGGER.size, 16)
        self.assertEqual(NAME_SET.size, 48)
        self.assertEqual(CONTINUATION.size, 16)

    def test_compile_is_byte_deterministic(self) -> None:
        left, left_summary, _ = compile_official_rule_pack(minimal_payload())
        right, right_summary, _ = compile_official_rule_pack(minimal_payload())
        self.assertEqual(left, right)
        self.assertEqual(left_summary, right_summary)
        loaded = validate_official_rule_pack(left)
        self.assertEqual(loaded.counts["cards"], 1)
        self.assertEqual(loaded.counts["effects"], 2)
        self.assertEqual(loaded.counts["targets"], 3)

    def test_payload_corruption_fails_closed(self) -> None:
        data, _, _ = compile_official_rule_pack(minimal_payload())
        corrupted = bytearray(data)
        corrupted[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            validate_official_rule_pack(bytes(corrupted))


if __name__ == "__main__":
    unittest.main()
