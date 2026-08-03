from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path

import torch


DATASET = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.training.dataset"
)
OBJECTIVE = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.training.objective"
)
LEGACY = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.legacy.base_model"
)
SEMANTIC = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.model.multi_memory"
)
PROTOTYPES = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.features.prototypes"
)

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "rl_runs/0025_semantic_foundation_pretraining/dataset/V1_james_cox_raging_bolt_semantic"
PROTOTYPE_PATH = ROOT / "train/0025_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json"


class BCObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        dataset = DATASET.SemanticDecisionDataset(DATA_ROOT)
        cls.batch = next(dataset.iter_batches("train", 4, seed=17))
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(PROTOTYPE_PATH)

    def test_semantic_teacher_logits_and_loss_are_finite(self) -> None:
        torch.manual_seed(3)
        model = SEMANTIC.SemanticFoundationPolicy(
            SEMANTIC.SemanticModelConfig(d_model=64, heads=4), self.prototypes
        )
        logits = model.teacher_logits(self.batch)
        self.assertEqual(tuple(logits.shape[:2]), tuple(self.batch["targets"].shape))
        metrics = OBJECTIVE.teacher_batch(model, self.batch)
        self.assertTrue(torch.isfinite(metrics.loss))
        metrics.loss.backward()
        self.assertIsNotNone(model.board_query.gate.grad)
        self.assertIsNotNone(model.prototype_query.gate.grad)

    def test_batched_decode_is_legal_for_both_arms(self) -> None:
        models = [
            LEGACY.IDOnlyPointerPolicy(
                LEGACY.IDOnlyConfig(d_model=64, heads=4, encoder_layers=2)
            ),
            SEMANTIC.SemanticFoundationPolicy(
                SEMANTIC.SemanticModelConfig(d_model=64, heads=4), self.prototypes
            ),
        ]
        for model in models:
            decoded = OBJECTIVE.deterministic_decode(model.eval(), self.batch)
            self.assertTrue(decoded.legal.all())
            self.assertTrue(decoded.lengths.ge(self.batch["min_count"]).all())
            self.assertTrue(decoded.lengths.le(self.batch["max_count"]).all())

    def test_evaluation_uses_shared_metric_contract(self) -> None:
        model = LEGACY.IDOnlyPointerPolicy(
            LEGACY.IDOnlyConfig(d_model=64, heads=4, encoder_layers=2)
        )
        result = OBJECTIVE.evaluate_batches(model, [self.batch], device=torch.device("cpu"))
        self.assertEqual(result["decisions"], 4)
        self.assertEqual(
            set(result),
            {
                "loss",
                "token_accuracy",
                "teacher_exact_action",
                "exact_action",
                "legal_action",
                "action_length_accuracy",
                "decisions",
                "tokens",
            },
        )
        self.assertTrue(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
