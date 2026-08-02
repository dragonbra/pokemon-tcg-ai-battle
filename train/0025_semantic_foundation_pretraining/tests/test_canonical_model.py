from __future__ import annotations

import importlib
from pathlib import Path
import unittest

import torch

BASE = "train.0025_semantic_foundation_pretraining"
_row = importlib.import_module(f"{BASE}.tests.test_canonical_features")._row
PROTOTYPES = importlib.import_module(f"{BASE}.features.prototypes")
COMPILER = importlib.import_module(f"{BASE}.features.canonical.compiler")
BATCHING = importlib.import_module(f"{BASE}.features.canonical.batching")
MODEL = importlib.import_module(f"{BASE}.model.canonical")


class CanonicalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            Path("train/0025_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json")
        )

    def _model(self):
        torch.manual_seed(17)
        return MODEL.CanonicalSemanticPolicy(
            MODEL.CanonicalModelConfig(
                d_model=64,
                heads=4,
                state_layers=1,
                option_layers=1,
                dropout=0.0,
            ),
            self.prototypes,
        ).eval()

    def _batch(self, energy_id: int):
        row, snapshot = _row(energy_id)
        record = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
        return BATCHING.collate_canonical_records([record])

    def test_forward_is_finite_and_greedy_is_legal(self) -> None:
        model = self._model()
        batch = self._batch(4)
        with torch.inference_mode():
            logits = model.teacher_logits(batch)
            action = model.greedy_action(batch)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(len(action), 1)
        self.assertIn(action[0], (0, 1))

    def test_typed_energy_change_reaches_policy_logits(self) -> None:
        model = self._model()
        lightning = self._batch(4)
        grass = self._batch(1)
        with torch.inference_mode():
            left = model(lightning)
            right = model(grass)
        self.assertFalse(torch.allclose(left, right))

    def test_batch_contract_rejects_unused_or_missing_fields(self) -> None:
        model = self._model()
        batch = self._batch(4)
        batch["unused_field"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "extra"):
            model(batch)


if __name__ == "__main__":
    unittest.main()
