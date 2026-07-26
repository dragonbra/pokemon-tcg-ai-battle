from __future__ import annotations

import importlib
import unittest

import torch

probability = importlib.import_module("train.0013_semantic_goal_policy.model.decoder")


class FixedScorer:
    def __init__(self, option_steps, stop_steps, value=0.25):
        self.option_steps = [torch.tensor(step, dtype=torch.float32) for step in option_steps]
        self.stop_steps = [torch.tensor(value, dtype=torch.float32) for value in stop_steps]
        self.value = torch.tensor([value], dtype=torch.float32)

    def initial(self):
        return 0

    def logits(self, hidden, chosen):
        return self.option_steps[hidden].clone(), self.stop_steps[hidden].clone()

    def advance(self, hidden, option_index):
        return hidden + 1


class ProbabilityContractTest(unittest.TestCase):
    def test_optional_stop_and_forced_max_probability(self):
        scorer = FixedScorer([[3, 2, 1], [1, 4, 2]], [-5, -4])
        evaluated = probability.evaluate_action(scorer, [0, 1], option_mask=torch.ones(3, dtype=torch.bool), min_count=1, max_count=2)
        self.assertTrue(evaluated.forced_terminal)
        self.assertEqual(evaluated.terminal_log_prob, 0.0)
        self.assertEqual(evaluated.terminal_entropy, 0.0)
        self.assertEqual(evaluated.decision_count, 2)
        optional = probability.evaluate_action(FixedScorer([[3, 2], [1, 0]], [-5, 4]), [0], option_mask=torch.ones(2, dtype=torch.bool), min_count=1, max_count=2)
        self.assertFalse(optional.forced_terminal)
        self.assertLess(optional.terminal_log_prob, 0.0)
        self.assertEqual(optional.decision_count, 2)

    def test_sample_and_reevaluate_parity(self):
        scorer = FixedScorer([[4, 1, 0], [0, 5, 1], [0, 0, 6]], [-9, -9, 9])
        sample = probability.sample_action(scorer, option_mask=torch.ones(3, dtype=torch.bool), min_count=1, max_count=3, deterministic=True)
        reevaluated = probability.evaluate_action(scorer, sample.sequence, option_mask=torch.ones(3, dtype=torch.bool), min_count=1, max_count=3)
        self.assertEqual(sample.sequence, reevaluated.sequence)
        self.assertAlmostEqual(sample.log_prob, reevaluated.log_prob, places=6)
        self.assertAlmostEqual(sample.entropy_sum, reevaluated.entropy_sum, places=6)

    def test_min_count_legality_uniqueness_and_permutation(self):
        scorer = FixedScorer([[1, 3, 2], [4, 0, 5]], [100, 100])
        sample = probability.sample_action(scorer, option_mask=torch.ones(3, dtype=torch.bool), min_count=2, max_count=2, deterministic=True)
        self.assertEqual(len(sample.sequence), 2)
        self.assertEqual(len(set(sample.sequence)), 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probability.evaluate_action(scorer, [1, 1], option_mask=torch.ones(3, dtype=torch.bool), min_count=2, max_count=2)
        old_to_new = (2, 0, 1)
        remapped = tuple(old_to_new[index] for index in sample.sequence)
        self.assertEqual(tuple(sorted(remapped)), tuple(sorted(old_to_new[index] for index in sample.sequence)))

    def test_no_legal_option_before_minimum_fails(self):
        with self.assertRaisesRegex(RuntimeError, "minCount"):
            probability.sample_action(FixedScorer([[1]], [0]), option_mask=torch.zeros(1, dtype=torch.bool), min_count=1, max_count=1, deterministic=True)


if __name__ == "__main__":
    unittest.main()
