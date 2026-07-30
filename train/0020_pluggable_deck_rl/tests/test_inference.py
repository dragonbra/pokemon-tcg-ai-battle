from __future__ import annotations

from importlib import import_module
from pathlib import Path
import unittest

import torch


INFERENCE = import_module("train.0020_pluggable_deck_rl.inference")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _deck(name: str) -> list[int]:
    path = REPOSITORY_ROOT / f"evaluation/arena/opponents/{name}/deck.csv"
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class InferenceTests(unittest.TestCase):
    def test_exact_decks_produce_distinct_registered_tensors(self) -> None:
        alakazam = INFERENCE.registered_deck_tensors(
            _deck("alakazam_dudunsparce_04_sota")
        )
        dragapult = INFERENCE.registered_deck_tensors(
            _deck("dragapult_ex_03_v20260729_rl")
        )
        self.assertFalse(
            torch.equal(
                alakazam["registered_card_ids"],
                dragapult["registered_card_ids"],
            )
        )
        self.assertEqual(INFERENCE.DEPLOYMENT_SOURCE_ID, 0)

    def test_deck_validation_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 60"):
            INFERENCE.validate_deck([1] * 59)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            INFERENCE.validate_deck([True] + [1] * 59)

    def test_frozen_checkpoint_loads_strictly(self) -> None:
        root = (
            REPOSITORY_ROOT
            / "rl_runs/0020_pluggable_deck_rl/versions/V1_frozen_0019_epoch13"
        )
        policy = INFERENCE.FrozenNeutralPolicy.from_checkpoint(
            root / "checkpoint/epoch-0013-da9b13d6f82d19d4.pt",
            root / "artifact/card_ontology.json",
            _deck("dragapult_ex_03_v20260729_rl"),
        )
        self.assertEqual(policy.source_id, 0)
        self.assertEqual(
            policy.deck,
            tuple(_deck("dragapult_ex_03_v20260729_rl")),
        )
        self.assertIsNone(policy.encoder)

    def test_audit_persona_checkpoint_loads_explicit_source(self) -> None:
        root = (
            REPOSITORY_ROOT
            / "rl_runs/0020_pluggable_deck_rl/versions/V1_frozen_0019_epoch13"
        )
        policy = INFERENCE.FrozenNeutralPolicy.from_checkpoint(
            root / "checkpoint/epoch-0013-da9b13d6f82d19d4.pt",
            root / "artifact/card_ontology.json",
            _deck("dragapult_ex_03_v20260729_rl"),
            source_id=98,
        )
        self.assertEqual(policy.source_id, 98)

    def test_audit_persona_rejects_unknown_source(self) -> None:
        root = (
            REPOSITORY_ROOT
            / "rl_runs/0020_pluggable_deck_rl/versions/V1_frozen_0019_epoch13"
        )
        with self.assertRaisesRegex(ValueError, "outside the frozen vocabulary"):
            INFERENCE.FrozenNeutralPolicy.from_checkpoint(
                root / "checkpoint/epoch-0013-da9b13d6f82d19d4.pt",
                root / "artifact/card_ontology.json",
                _deck("dragapult_ex_03_v20260729_rl"),
                source_id=510,
            )


if __name__ == "__main__":
    unittest.main()
