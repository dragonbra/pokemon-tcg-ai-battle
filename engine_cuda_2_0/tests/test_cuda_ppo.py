from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.cuda_ppo import (  # noqa: E402
    CudaRolloutBuffer,
    active_deck_lane_mask,
    ppo_update_device,
)
from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    foundation_r15_static_fields_for_decks,
)
from run_foundation_ppo_cuda_train import (  # noqa: E402
    _balanced_lane_game_quotas,
    _set_actor_value_learning_rates,
)


DECK_A = [1] * 4 + [2] * 4 + list(range(3, 55))
DECK_B = [10] * 2 + [11] * 3 + list(range(12, 67))


class _DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def eval(self):  # type: ignore[override]
        super().eval()
        return self


class _DummyActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = _DummyPolicy()
        self.value_head = nn.Linear(1, 1)
        self.encoded_batch_sizes: list[int] = []

    def encode(self, batch):
        self.encoded_batch_sizes.append(int(batch["obs"].shape[0]))
        state = batch["obs"] * self.policy.weight
        return state, state.unsqueeze(1)

    def evaluate_targets_from_encoding(self, batch, encoded):
        targets = batch["targets"].clamp_min(0).to(encoded[0].dtype)
        logprob = encoded[0].view(-1) * targets.sum(-1)
        value = self.value_head(encoded[0]).view(-1)
        return logprob, value

    def target_entropy_from_encoding(self, batch, encoded):
        del batch, encoded
        return torch.zeros((), dtype=self.policy.weight.dtype, device=self.policy.weight.device)


class ActiveDeckLaneMaskTest(unittest.TestCase):
    def test_target_masks_decks_that_already_reached_the_cap(self) -> None:
        completed = torch.tensor([199, 200, 215], dtype=torch.int64)
        lane_deck_ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
        mask = active_deck_lane_mask(
            completed_by_deck=completed,
            lane_deck_ids=lane_deck_ids,
            target_games_per_deck=200,
        )
        self.assertEqual(mask.tolist(), [True, False, False, True, False])

    def test_zero_target_keeps_every_lane_active(self) -> None:
        completed = torch.tensor([10, 20], dtype=torch.int64)
        lane_deck_ids = torch.tensor([0, 1, 0], dtype=torch.int64)
        mask = active_deck_lane_mask(
            completed_by_deck=completed,
            lane_deck_ids=lane_deck_ids,
            target_games_per_deck=0,
        )
        self.assertEqual(mask.tolist(), [True, True, True])


class ResumeLearningRateTest(unittest.TestCase):
    def test_cli_learning_rates_replace_restored_group_values(self) -> None:
        actor = nn.Parameter(torch.tensor(1.0))
        value = nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.AdamW(
            [
                {"params": [actor], "lr": 9.0e-3},
                {"params": [value], "lr": 8.0e-3},
            ]
        )

        _set_actor_value_learning_rates(
            optimizer, actor_lr=3.0e-6, value_lr=1.0e-4
        )

        self.assertEqual(optimizer.param_groups[0]["lr"], 3.0e-6)
        self.assertEqual(optimizer.param_groups[1]["lr"], 1.0e-4)


class BalancedLaneQuotaTest(unittest.TestCase):
    def test_each_deck_gets_exactly_the_requested_games(self) -> None:
        lane_deck_ids = [0, 1, 2, 0, 1, 2, 0]
        quotas = _balanced_lane_game_quotas(
            lane_deck_ids, deck_count=3, target_games_per_deck=5
        )
        self.assertEqual(sum(quotas), 15)
        self.assertEqual(
            [
                sum(quota for quota, deck_id in zip(quotas, lane_deck_ids) if deck_id == target)
                for target in range(3)
            ],
            [5, 5, 5],
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaPPOTest(unittest.TestCase):
    def test_shared_deck_fields_are_padded_on_device(self) -> None:
        fields = foundation_r15_static_fields_for_decks([DECK_A, DECK_B], "cuda")
        self.assertTrue(all(value.is_cuda for value in fields.values()))
        self.assertEqual(tuple(fields["registered_card_ids"].shape), (2, 57))
        self.assertEqual(tuple(fields["zone_inventory_num"].shape), (2, 2, 16))
        self.assertEqual(tuple(fields["ledger_cat"].shape), (2, 57, 4))
        self.assertEqual(fields["registered_mask"].tolist()[0][-1], False)
        self.assertEqual(fields["registered_mask"].tolist()[1][-1], True)

    def test_rollout_and_ppo_update_stay_on_cuda(self) -> None:
        device = torch.device("cuda")
        model = _DummyActorCritic().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        buffer = CudaRolloutBuffer(max_steps=2, batch_size=3)
        for step in range(2):
            obs = torch.full((3, 1), float(step + 1), device=device)
            targets = torch.tensor([[0, 2], [1, 2], [0, 2]], device=device)
            batch = {"obs": obs, "targets": targets}
            old_value = torch.zeros(3, device=device)
            buffer.append(
                batch,
                old_logprob=torch.zeros(3, device=device),
                old_value=old_value,
                reward=torch.ones(3, device=device),
                done=torch.zeros(3, dtype=torch.bool, device=device),
            )
        stats = ppo_update_device(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            last_value=torch.zeros(3, device=device),
            epochs=1,
            minibatch_size=3,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=1.0,
        )
        self.assertTrue(stats.loss.is_cuda)
        self.assertTrue(model.policy.weight.grad is not None)
        self.assertTrue(torch.isfinite(stats.loss))
        self.assertEqual(len(buffer), 0)

    def test_train_mask_filters_rows_before_encoder_forward(self) -> None:
        device = torch.device("cuda")
        model = _DummyActorCritic().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        buffer = CudaRolloutBuffer(max_steps=2, batch_size=3)
        masks = (
            torch.tensor([True, False, False], device=device),
            torch.tensor([False, True, False], device=device),
        )
        for step, train_mask in enumerate(masks):
            batch = {
                "obs": torch.full((3, 1), float(step + 1), device=device),
                "targets": torch.tensor([[0, 2], [1, 2], [0, 2]], device=device),
                "train_mask": train_mask,
            }
            buffer.append(
                batch,
                old_logprob=torch.zeros(3, device=device),
                old_value=torch.zeros(3, device=device),
                reward=torch.ones(3, device=device),
                done=torch.ones(3, dtype=torch.bool, device=device),
            )
        stats = ppo_update_device(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            last_value=torch.zeros(3, device=device),
            epochs=1,
            minibatch_size=8,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=1.0,
            target_kl=1.0,
        )
        self.assertEqual(model.encoded_batch_sizes, [2])
        self.assertEqual(len(buffer), 0)
        self.assertEqual(int(stats.train_samples.item()), 2)
        self.assertEqual(int(stats.optimizer_steps.item()), 1)
        for name in (
            "explained_variance",
            "value_mean",
            "return_mean",
            "advantage_std",
            "value_sign_accuracy",
            "value_vs_outcome_corr",
        ):
            self.assertTrue(torch.isfinite(getattr(stats, name)), name)


if __name__ == "__main__":
    unittest.main()
