from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import importlib

import torch

BASE = "train.0031_rule_faithful_semantic_foundation_pretraining"
DOMAIN = importlib.import_module(f"{BASE}.domain")
MODEL = importlib.import_module(f"{BASE}.model")
DATASET = importlib.import_module(f"{BASE}.training.dataset")
OBJECTIVE = importlib.import_module(f"{BASE}.training.objective")
FIXTURE = importlib.import_module(f"{BASE}.tests.test_dataset")


class TrainingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        fixture = FIXTURE.CanonicalDatasetTests()
        fixture.prototypes = DOMAIN.PrototypeIndex.load(
            Path("train/0031_rule_faithful_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json")
        )
        root = fixture._dataset(Path(self.temporary.name))
        self.dataset = DATASET.CanonicalDecisionDataset(root)
        self.batch = next(self.dataset.iter_batches("train", 1, seed=17))
        self.model = MODEL.SemanticPolicy(
            MODEL.ModelConfig(
                d_model=64,
                heads=4,
                state_layers=2,
                option_layers=2,
                ffn_multiplier=2,
                dropout=0.0,
            ),
            fixture.prototypes,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_teacher_loss_is_finite_and_backward_reaches_decoder(self) -> None:
        result = OBJECTIVE.teacher_batch(self.model, self.batch)
        self.assertTrue(torch.isfinite(result.loss))
        pending = [result.loss.grad_fn]
        gradient_nodes = set()
        while pending:
            node = pending.pop()
            if node is None or node in gradient_nodes:
                continue
            gradient_nodes.add(node)
            pending.extend(child for child, _ in node.next_functions)
        index_backward_count = sum(
            type(node).__name__ == "IndexBackward0" for node in gradient_nodes
        )
        self.assertLessEqual(index_backward_count, 4)
        result.loss.backward()
        self.assertIsNotNone(self.model.action_decoder.query.weight.grad)

    def test_greedy_decode_is_legal(self) -> None:
        decoded = OBJECTIVE.deterministic_decode(self.model.eval(), self.batch)
        self.assertTrue(decoded.legal.all())
        self.assertTrue(decoded.lengths.ge(self.batch["min_count"]).all())
        self.assertTrue(decoded.lengths.le(self.batch["max_count"]).all())

    def test_validation_metric_contract(self) -> None:
        result = OBJECTIVE.evaluate_batches(
            self.model,
            [self.batch],
            device=torch.device("cpu"),
        )
        self.assertEqual(result["decisions"], 1)
        self.assertTrue(0 <= result["legal_action"] <= 1)

if __name__ == "__main__":
    unittest.main()
