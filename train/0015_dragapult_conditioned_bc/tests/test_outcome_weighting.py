from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from ..training.outcome_weighting import (
    outcome_weights,
    weighted_token_cross_entropy,
)
from ..training.outcome_weighted_trainer import make_optimization_step


class OutcomeWeightingTest(unittest.TestCase):
    def test_outcome_ids_map_to_declared_weights(self) -> None:
        outcome_ids = torch.tensor([1, -1, 0, 1], dtype=torch.int8)
        result = outcome_weights(
            outcome_ids,
            win_weight=1.0,
            loss_weight=0.75,
            draw_weight=0.5,
        )
        torch.testing.assert_close(result, torch.tensor([1.0, 0.75, 0.5, 1.0]))

    def test_unknown_outcome_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outcome_id"):
            outcome_weights(
                torch.tensor([2], dtype=torch.int8),
                win_weight=1.0,
                loss_weight=0.75,
                draw_weight=1.0,
            )

    def test_nonpositive_or_nonfinite_weights_are_rejected(self) -> None:
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "positive finite"
            ):
                outcome_weights(
                    torch.tensor([1], dtype=torch.int8),
                    win_weight=invalid,
                    loss_weight=1.0,
                    draw_weight=1.0,
                )

    def test_uniform_weighted_loss_matches_token_cross_entropy(self) -> None:
        logits = torch.tensor(
            [
                [[2.0, -1.0], [-0.5, 1.5], [0.0, 0.0]],
                [[-0.2, 0.8], [1.2, -0.3], [0.0, 0.0]],
            ],
            requires_grad=True,
        )
        targets = torch.tensor([[0, 1, -100], [1, 0, -100]])
        expected = F.cross_entropy(logits[targets != -100], targets[targets != -100])
        loss, numerator, denominator = weighted_token_cross_entropy(
            logits, targets, torch.ones(2)
        )
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(loss, numerator / denominator)
        self.assertEqual(float(denominator), 4.0)

    def test_label_smoothing_matches_pytorch_token_mean(self) -> None:
        logits = torch.tensor(
            [[[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0], [9.0, -9.0, 0.0]]],
            requires_grad=True,
        )
        targets = torch.tensor([[0, 1, -100]])
        expected = F.cross_entropy(
            logits[targets != -100],
            targets[targets != -100],
            label_smoothing=0.05,
        )
        loss, numerator, denominator = weighted_token_cross_entropy(
            logits,
            targets,
            torch.ones(1),
            label_smoothing=0.05,
        )
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(loss, numerator / denominator)
        self.assertEqual(float(denominator), 2.0)

    def test_invalid_label_smoothing_is_rejected(self) -> None:
        for invalid in (-0.01, 1.0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "label smoothing"
            ):
                weighted_token_cross_entropy(
                    torch.zeros(1, 1, 2),
                    torch.zeros(1, 1, dtype=torch.long),
                    torch.ones(1),
                    label_smoothing=invalid,
                )

    def test_label_smoothing_excludes_illegal_negative_infinity_classes(self) -> None:
        floor = torch.finfo(torch.float32).min
        logits = torch.tensor(
            [[[2.0, 0.0, float("-inf")], [floor, 1.0, -1.0]]],
            requires_grad=True,
        )
        targets = torch.tensor([[0, 2]])
        loss, _, denominator = weighted_token_cross_entropy(
            logits,
            targets,
            torch.ones(1),
            label_smoothing=0.05,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(denominator), 2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad[torch.isfinite(logits)]).all())

    def test_padding_is_excluded_and_backward_is_finite(self) -> None:
        logits = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0], [100.0, -100.0]],
                [[1.0, 0.0], [0.0, 1.0], [-100.0, 100.0]],
            ],
            requires_grad=True,
        )
        targets = torch.tensor([[1, 0, -100], [0, 1, -100]])
        weights = torch.tensor([1.0, 0.5])
        loss, numerator, denominator = weighted_token_cross_entropy(logits, targets, weights)
        self.assertEqual(float(denominator), 3.0)
        torch.testing.assert_close(loss, numerator / denominator)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[:, 2], torch.zeros_like(logits.grad[:, 2])))

    def test_optimization_step_reports_unweighted_accuracy_and_weight_counts(self) -> None:
        class FakeModel:
            def teacher_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
                return batch["logits"]

        batch = {
            "logits": torch.tensor(
                [
                    [[2.0, 0.0], [0.0, 2.0]],
                    [[0.0, 2.0], [2.0, 0.0]],
                ],
                requires_grad=True,
            ),
            "targets": torch.tensor([[0, 1], [0, 1]]),
            "outcome_id": torch.tensor([1, -1], dtype=torch.int8),
        }
        step = make_optimization_step(
            win_weight=1.0, loss_weight=0.5, draw_weight=1.0
        )
        loss, tokens, correct, exact, _, diagnostics = step(
            FakeModel(), batch  # type: ignore[arg-type]
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(tokens), 4)
        self.assertEqual(int(correct), 2)
        self.assertEqual(int(exact), 1)
        self.assertEqual(float(diagnostics["decision_weight_sum"]), 1.5)
        self.assertEqual(int(diagnostics["win_decisions"]), 1)
        self.assertEqual(int(diagnostics["loss_decisions"]), 1)


if __name__ == "__main__":
    unittest.main()
