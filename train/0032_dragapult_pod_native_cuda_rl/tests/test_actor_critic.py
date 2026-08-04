from __future__ import annotations

import importlib
import unittest

import torch


contract_tests = importlib.import_module(
    "train.0032_dragapult_pod_native_cuda_rl.tests.test_contract"
)
model_module = importlib.import_module(
    "train.0032_dragapult_pod_native_cuda_rl.model"
)
ModelConfig = model_module.ModelConfig
PodNativeActorCritic = model_module.PodNativeActorCritic


class PodNativeActorCriticTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = PodNativeActorCritic(
            ModelConfig(
                d_model=32,
                heads=4,
                state_layers=1,
                option_layers=1,
                ffn_multiplier=2,
                dropout=0.0,
                max_action_steps=6,
            )
        ).eval()

    def test_forward_shapes_and_gradients(self) -> None:
        batch = contract_tests.valid_batch()
        logits, value = self.model(batch)
        self.assertEqual(tuple(logits.shape), (2, 129))
        self.assertEqual(tuple(value.shape), (2,))
        (logits[:, :4].mean() + value.mean()).backward()
        self.assertIsNotNone(self.model.value_head[-1].weight.grad)
        self.assertIsNotNone(self.model.action_decoder.query.weight.grad)

    def test_greedy_respects_selection_bounds_and_uniqueness(self) -> None:
        batch = contract_tests.valid_batch()
        batch["min_count"][:] = 2
        batch["max_count"][:] = 3
        actions = self.model.greedy(batch)
        self.assertTrue(bool(actions.legal.all()))
        for row, length in zip(actions.sequences, actions.lengths):
            chosen = row[: int(length)].tolist()
            self.assertGreaterEqual(len(chosen), 2)
            self.assertLessEqual(len(chosen), 3)
            self.assertEqual(len(chosen), len(set(chosen)))
            self.assertTrue(all(0 <= index < 4 for index in chosen))

    def test_padding_content_does_not_change_outputs(self) -> None:
        original = contract_tests.valid_batch(batch_size=1)
        changed = {name: value.clone() for name, value in original.items()}
        changed["entity_cat"][:, 3:] = 1
        changed["entity_num"][:, 3:] = 17.0
        changed["option_cat"][:, 4:] = 1
        changed["option_num"][:, 4:] = 19.0
        with torch.no_grad():
            logits_a, value_a = self.model(original)
            logits_b, value_b = self.model(changed)
        torch.testing.assert_close(logits_a, logits_b)
        torch.testing.assert_close(value_a, value_b)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_action_stays_resident(self) -> None:
        model = self.model.cuda()
        batch = contract_tests.valid_batch(device="cuda")
        with torch.inference_mode():
            sequences, lengths = model.act_device(batch)
        self.assertEqual(sequences.device.type, "cuda")
        self.assertEqual(lengths.device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_greedy_is_graph_capturable(self) -> None:
        model = self.model.cuda()
        batch = contract_tests.valid_batch(device="cuda")
        with torch.inference_mode():
            expected = model.greedy(batch)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = model.greedy(batch)
            graph.replay()
            torch.cuda.synchronize()
        torch.testing.assert_close(captured.sequences, expected.sequences)
        torch.testing.assert_close(captured.lengths, expected.lengths)
        torch.testing.assert_close(captured.legal, expected.legal)


if __name__ == "__main__":
    unittest.main()
