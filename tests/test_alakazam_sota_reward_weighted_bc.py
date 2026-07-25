from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from train.project_0010_alakazam_sota_model.model import IDOnlyConfig, IDOnlyPointerPolicy
from train.project_0011_alakazam_sota_reward_weighted_bc.config import (
    ExperimentConfig,
    RewardComponent,
    RewardConfig,
    WeightingConfig,
)
from train.project_0011_alakazam_sota_reward_weighted_bc.dataset import build_reward_dataset
from train.project_0011_alakazam_sota_reward_weighted_bc.export_candidate import export_candidate
from train.project_0011_alakazam_sota_reward_weighted_bc.rewards import (
    compose_reward,
    imitation_weights,
    weighted_token_cross_entropy,
)
from train.project_0011_alakazam_sota_reward_weighted_bc.training import train


def _encoded_row(episode_id: int = 1) -> dict:
    return {
        "dataset_schema_version": "alakazam_sota_id_only_dataset_v1",
        "episode_id": episode_id,
        "episode_step": 3,
        "player_index": 0,
        "global_cat": [1, 1, 1, 0],
        "global_num": [0.0] * 12,
        "entity_cat": [[1, 1, 1, 1, 1, 0, 0]],
        "entity_num": [[0.0] * 5],
        "option_cat": [[1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]],
        "action": [0],
        "min_count": 1,
        "max_count": 1,
    }


def _reward_row(episode_id: int = 1) -> dict:
    return {
        "episode_id": episode_id,
        "episode_step": 3,
        "player_index": 0,
        "reward_schema_version": "alakazam_offline_reward_metrics_v2",
        "reward_metrics": {
            "episode": {"game_turn_count": 8},
            "decision": {"second_turn_setup_credit": 1.0},
            "audit": {},
        },
    }


class AlakazamSotaRewardWeightedBCTests(unittest.TestCase):
    def test_export_candidate_accepts_reward_weighted_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "deck.csv").write_text("\n".join(["1"] * 60) + "\n")
            (source / "cg").mkdir()
            (source / "cg" / "__init__.py").write_text("")
            config = IDOnlyConfig(
                d_model=32,
                heads=4,
                encoder_layers=1,
                ffn_multiplier=2,
                dropout=0.0,
                max_entities=8,
                max_options=8,
                max_action_steps=2,
            )
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "model": IDOnlyPointerPolicy(config).state_dict(),
                    "metadata": {
                        "model_version": "alakazam_sota_reward_weighted_pointer_bc_v1",
                        "model_config": config.__dict__,
                    },
                },
                checkpoint,
            )
            output = root / "candidate"
            manifest = export_candidate(checkpoint, source, output)
            self.assertEqual(
                manifest["schema_version"],
                "alakazam_sota_reward_weighted_candidate_v1",
            )
            self.assertTrue((output / "strategy" / "model.bin").is_file())
            self.assertTrue((output / "main.py").is_file())

    def test_all_rewards_config_loads_and_blocks_two_epoch_runs(self) -> None:
        config = ExperimentConfig.load(
            "train/project_0011_alakazam_sota_reward_weighted_bc/configs/0011_v1_all_rewards.json"
        )
        self.assertEqual(config.model.d_model, 320)
        self.assertEqual(config.minimum_epochs_before_stop, 8)
        self.assertTrue(all(component.enabled for component in config.reward.components))
        payload = config.to_dict()
        payload["epochs"] = 2
        payload["minimum_epochs_before_stop"] = 2
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict(payload)

    def test_reward_composition_and_weighted_token_loss(self) -> None:
        reward = RewardConfig(
            components=(
                RewardComponent(
                    name="setup",
                    field="reward_metrics.decision.second_turn_setup_credit",
                    weight=0.5,
                ),
            )
        )
        total, components = compose_reward(_reward_row(), reward)
        self.assertEqual(total, 0.5)
        self.assertEqual(components, {"setup": 0.5})

        rewards = torch.tensor([0.5, -0.5])
        _, weights = imitation_weights(
            rewards,
            WeightingConfig(),
            dataset_mean=0.0,
        )
        self.assertGreater(float(weights[0]), float(weights[1]))
        logits = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])
        targets = torch.tensor([[1], [1]])
        uniform = weighted_token_cross_entropy(logits, targets, torch.ones(2))
        weighted = weighted_token_cross_entropy(logits, targets, weights)
        self.assertLess(float(weighted), float(uniform))

    def test_dataset_join_and_cpu_training_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            reward_root = root / "rewards"
            merged = root / "merged"
            base.mkdir()
            reward_root.mkdir()
            (base / "dataset_audit.json").write_text("{}\n", encoding="utf-8")
            (reward_root / "reward_sidecar.jsonl.reward_audit.json").write_text(
                "{}\n", encoding="utf-8"
            )
            for split in ("train", "validation", "test"):
                rows = [_encoded_row()] if split != "test" else []
                rewards = [_reward_row()] if split != "test" else []
                with gzip.open(base / f"{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                with (reward_root / f"reward_sidecar.{split}.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:
                    for row in rewards:
                        handle.write(json.dumps(row) + "\n")
            audit = build_reward_dataset(base, reward_root, merged)
            self.assertEqual(audit["records_by_split"]["train"], 1)
            with gzip.open(merged / "train.jsonl.gz", "rt", encoding="utf-8") as handle:
                row = json.loads(next(handle))
            self.assertEqual(row["terminal_outcome"], 1.0)
            self.assertIn("reward_metrics", row)

            config = ExperimentConfig(
                model=IDOnlyConfig(
                    d_model=32,
                    heads=4,
                    encoder_layers=1,
                    ffn_multiplier=2,
                    dropout=0.0,
                    max_entities=8,
                    max_options=8,
                    max_action_steps=2,
                ),
                reward=RewardConfig(
                    components=(
                        RewardComponent(
                            name="terminal",
                            field="terminal_outcome",
                            weight=1.0,
                        ),
                    )
                ),
                weighting=WeightingConfig(mode="uniform"),
                batch_size=1,
                epochs=3,
                learning_rate=1e-3,
                weight_decay=0.0,
                seed=3,
                shuffle_buffer_size=2,
                train_eval_interval=1,
                minimum_epochs_before_stop=3,
                early_stopping_patience=2,
                log_every_batches=1,
                amp=False,
                device="cpu",
                max_model_mib=5.0,
                storage_path=str(root),
                min_free_gib=0.0,
            )
            summary = train(merged, root / "smoke", config)
            self.assertEqual(summary["status"], "completed")
            self.assertTrue((root / "smoke" / "checkpoints" / "best_validation.pt").is_file())
            self.assertTrue((root / "smoke" / "checkpoints" / "best_exact.pt").is_file())


if __name__ == "__main__":
    unittest.main()
