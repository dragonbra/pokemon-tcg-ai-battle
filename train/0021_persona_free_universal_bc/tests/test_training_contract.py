from __future__ import annotations

from importlib import import_module
import unittest
from types import SimpleNamespace

import torch


MATERIALIZED = import_module(
    "train.0021_persona_free_universal_bc.training.materialized"
)
TRAINER = import_module("train.0021_persona_free_universal_bc.training.a0_trainer")


def _cache_tensors() -> dict[str, torch.Tensor]:
    return {
        "legacy_global_cat": torch.zeros((2, 4), dtype=torch.int16),
        "legacy_global_num": torch.zeros((2, 12), dtype=torch.float32),
        "legacy_entity_cat": torch.zeros((2, 1, 7), dtype=torch.int16),
        "legacy_entity_num": torch.zeros((2, 1, 5), dtype=torch.float32),
        "legacy_entity_mask": torch.ones((2, 1), dtype=torch.bool),
        "legacy_option_cat": torch.zeros((2, 3, 11), dtype=torch.int16),
        "legacy_option_mask": torch.ones((2, 3), dtype=torch.bool),
        "ordered_action": torch.tensor([[2, 0], [0, 1]], dtype=torch.int16),
        "action_mask": torch.tensor([[True, False], [True, True]]),
        "min_count": torch.tensor([0, 2], dtype=torch.int16),
        "max_count": torch.tensor([3, 2], dtype=torch.int16),
        "action_termination": torch.tensor(
            [
                MATERIALIZED.TERMINATION_CODES["optional_stop"],
                MATERIALIZED.TERMINATION_CODES["forced_max"],
            ],
            dtype=torch.uint8,
        ),
    }


class TrainingContractTests(unittest.TestCase):
    def test_stop_target_exists_only_for_optional_termination(self) -> None:
        batch = MATERIALIZED.select_a0_batch(_cache_tensors(), torch.tensor([0, 1]))
        stop_index = batch["option_mask"].size(1)
        self.assertEqual(batch["targets"][0].tolist(), [2, stop_index, -100])
        self.assertEqual(batch["targets"][1].tolist(), [0, 1, -100])
        self.assertEqual(batch["target_mask"][0].tolist(), [True, True, False])
        self.assertEqual(batch["target_mask"][1].tolist(), [True, True, False])

    def test_each_decision_has_equal_outer_loss_weight(self) -> None:
        token_nll = torch.tensor([[2.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        mask = torch.tensor([[True, False, False], [True, True, True]])
        losses = TRAINER.per_decision_sequence_nll(token_nll, mask)
        torch.testing.assert_close(losses, torch.tensor([2.0, 2.0]))

    def test_empty_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no supervised target"):
            TRAINER.per_decision_sequence_nll(
                torch.zeros((1, 2)), torch.zeros((1, 2), dtype=torch.bool)
            )

    def test_validation_metrics_use_explicit_namespace(self) -> None:
        class DummyModel:
            def eval(self) -> None:
                return None

            def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
                return (torch.zeros(1),)

            def teacher_logits_from_encoding(
                self,
                batch: dict[str, torch.Tensor],
                encoded: tuple[torch.Tensor, ...],
            ) -> torch.Tensor:
                return torch.tensor(
                    [
                        [[8.0, 0.0, 0.0], [0.0, 0.0, 8.0]],
                        [[0.0, 8.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                )

            def deterministic_action_tensors(
                self,
                batch: dict[str, torch.Tensor],
                *,
                encoded: tuple[torch.Tensor, ...],
            ) -> SimpleNamespace:
                return SimpleNamespace(
                    sequences=torch.tensor([[0], [1]]),
                    lengths=torch.tensor([1, 1]),
                    legal=torch.tensor([True, True]),
                )

        batch = {
            "targets": torch.tensor([[0, 2], [1, -100]]),
            "target_mask": torch.tensor([[True, True], [True, False]]),
            "option_mask": torch.ones((2, 2), dtype=torch.bool),
        }
        metrics = TRAINER.evaluate(
            DummyModel(),
            [batch],
            device=torch.device("cpu"),
            amp=False,
            namespace="bc/validation_deck_ood",
        )
        self.assertEqual(metrics["bc/validation_deck_ood/exact_action"], 1.0)
        self.assertNotIn("bc/validation/loss", metrics)


if __name__ == "__main__":
    unittest.main()
