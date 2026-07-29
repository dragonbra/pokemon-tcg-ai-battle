from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.schema import (  # noqa: E402
    ACTION_STRUCT,
    HEADER_STRUCT,
    INSTRUCTION_STRUCT,
    RULE_MAGIC,
    RulePack,
)
from ptcg_cuda_engine.native import RESET_STRUCT  # noqa: E402


class RulePackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pack = RulePack.load(CUDA_ENGINE_ROOT / "rules" / "smoke_rules.json")

    def test_smoke_pack_is_stable_and_compiles(self) -> None:
        report = self.pack.report()
        self.assertEqual(report["actions"], 5)
        self.assertEqual(report["instructions"], 14)
        self.assertEqual(len(report["canonical_sha256"]), 64)
        blob = self.pack.to_bytes()
        magic, rule_version, abi_version, instruction_count, action_count, _ = HEADER_STRUCT.unpack_from(blob)
        self.assertEqual(magic, RULE_MAGIC)
        self.assertEqual(rule_version, 1)
        self.assertEqual(abi_version, 1)
        self.assertEqual(instruction_count, 14)
        self.assertEqual(action_count, 5)
        self.assertEqual(
            len(blob),
            HEADER_STRUCT.size + 14 * INSTRUCTION_STRUCT.size + 5 * ACTION_STRUCT.size,
        )

    def test_binary_structs_match_cuda_abi(self) -> None:
        self.assertEqual(INSTRUCTION_STRUCT.size, 16)
        self.assertEqual(ACTION_STRUCT.size, 16)
        self.assertEqual(struct.calcsize("<BBHiii"), 16)
        self.assertEqual(RESET_STRUCT.size, 32)

    def test_duplicate_action_id_is_rejected(self) -> None:
        raw = {
            "name": "bad",
            "actions": [
                {"action_id": 0, "program": [{"opcode": "HALT"}]},
                {"action_id": 0, "program": [{"opcode": "HALT"}]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            RulePack.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
