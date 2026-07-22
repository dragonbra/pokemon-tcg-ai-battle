from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from rl.core.model import ModelConfig
from rl.model.features import PTCGFeatureConfig, feature_config_for_schema
from rl.model.features import feature_schema_for_config
from rl.model.full_action_inference import FullActionPolicy
from rl.model.full_action_model import FullActionPolicyValueNet
from rl.train.build_full_action_submission import build as build_submission
from rl.train.build_kaggle_bc_dataset import iter_kaggle_records


class FullActionBCMiniTests(unittest.TestCase):
    def _model(self) -> FullActionPolicyValueNet:
        return FullActionPolicyValueNet(
            ModelConfig(
                state_numeric_dim=46,
                state_token_count=40,
                candidate_numeric_dim=10,
                max_candidates=64,
                card_vocab_size=4096,
                action_type_vocab_size=4,
                d_model=16,
                hidden_dim=32,
                num_heads=2,
            ),
            max_selection_count=64,
        )

    def test_count_head_and_candidate_logits_keep_legal_contract(self) -> None:
        model = self._model()
        value, logits, counts = model.forward_with_count(
            state_numeric=torch.zeros((1, 46)),
            state_card_ids=torch.zeros((1, 40), dtype=torch.long),
            action_type_ids=torch.tensor([[1, 2, 3] + [0] * 61]),
            action_card_ids=torch.zeros((1, 64), dtype=torch.long),
            action_target_ids=torch.zeros((1, 64), dtype=torch.long),
            action_numeric=torch.zeros((1, 64, 10)),
            action_mask=torch.tensor([[True, True, True] + [False] * 61]),
        )
        self.assertEqual(tuple(value.shape), (1,))
        self.assertEqual(tuple(logits.shape), (1, 64))
        self.assertEqual(tuple(counts.shape), (1, 65))
        self.assertLess(logits[0, 3].item(), -1e10)

    def test_policy_returns_distinct_sorted_top_k_without_teacher(self) -> None:
        policy = FullActionPolicy(
            self._model(),
            feature_config=PTCGFeatureConfig(),
        )
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [{"active": [], "bench": []}, {"active": [], "bench": []}],
            },
            "select": {
                "type": 1,
                "context": 7,
                "minCount": 2,
                "maxCount": 2,
                "option": [{"type": 1}, {"type": 2}, {"type": 3}],
            },
        }
        action = policy.select(observation)
        self.assertEqual(len(action), 2)
        self.assertEqual(action, sorted(set(action)))
        self.assertTrue(all(index < 3 for index in action))

    def test_kaggle_adapter_pairs_observation_with_next_step_action(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {"active": [], "bench": [], "hand": [], "discard": []},
                    {"active": [], "bench": [], "hand": [], "discard": []},
                ],
            },
            "select": {
                "type": 1,
                "context": 7,
                "minCount": 2,
                "maxCount": 2,
                "option": [{"type": 1}, {"type": 2}, {"type": 3}],
            },
        }
        payload = {
            "info": {
                "EpisodeId": 123,
                "Agents": [{"Name": "Yushin Ito"}, {"Name": "opponent"}],
            },
            "rewards": [1, -1],
            "steps": [
                [
                    {
                        "action": [],
                        "status": "ACTIVE",
                        "observation": observation,
                        "visualize": [{"action": [[5] * 60, [3] * 60]}],
                    },
                    {"action": [], "status": "INACTIVE", "observation": {}},
                ],
                [
                    {"action": [0, 2], "status": "DONE", "observation": {}},
                    {"action": [], "status": "DONE", "observation": {}},
                ],
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode-123-replay.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = list(
                iter_kaggle_records(
                    path,
                    agent_name="Yushin Ito",
                    split_map={"123": "train"},
                    submission_id=54773249,
                    feature_config=PTCGFeatureConfig(),
                )
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["targets"], [0, 2])
        self.assertEqual(records[0]["target_count"], 2)

    def test_universal_schema_encodes_deck_entities_history_and_expert(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [
                            {
                                "id": 343,
                                "hp": 70,
                                "maxHp": 80,
                                "energies": [5],
                                "tools": [],
                                "serial": 3,
                            }
                        ],
                        "bench": [],
                        "hand": [{"id": 5}],
                        "discard": [],
                    },
                    {"active": [], "bench": [], "hand": None, "discard": []},
                ],
            },
            "select": {
                "type": 1,
                "context": 7,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 7, "cardId": 5}],
            },
        }
        payload = {
            "info": {
                "EpisodeId": 123,
                "Agents": [{"Name": "Yushin Ito"}, {"Name": "opponent"}],
            },
            "rewards": [1, -1],
            "steps": [
                [
                    {
                        "action": [],
                        "status": "ACTIVE",
                        "observation": observation,
                        "visualize": [{"action": [[5] * 60, [3] * 60]}],
                    },
                    {"action": [], "status": "INACTIVE", "observation": {}},
                ],
                [
                    {"action": [0], "status": "DONE", "observation": {}},
                    {"action": [], "status": "DONE", "observation": {}},
                ],
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode-123-replay.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = next(
                iter_kaggle_records(
                    path,
                    agent_name="Yushin Ito",
                    split_map={"123": "train"},
                    submission_id=54773249,
                    feature_config=feature_config_for_schema("ptcg_features_universal"),
                )
            )
        encoded = record["encoded"]
        self.assertEqual(record["dataset_version"], "ptcg_kaggle_bc_universal")
        self.assertEqual(len(encoded["state_numeric"]), 64)
        self.assertEqual((len(encoded["deck_card_ids"]), len(encoded["deck_card_numeric"][0])), (60, 18))
        self.assertEqual((len(encoded["entity_card_ids"]), len(encoded["entity_numeric"][0])), (12, 28))
        self.assertEqual((len(encoded["history_card_ids"]), len(encoded["history_numeric"][0])), (32, 8))
        self.assertEqual(len(encoded["action_numeric"][0]), 22)
        self.assertEqual(encoded["state_card_ids"][0], 344)
        self.assertEqual(encoded["entity_card_ids"][0], 344)
        self.assertEqual(encoded["action_card_ids"][0], 6)
        self.assertEqual(encoded["deck_card_numeric"][0][1], 0.25)
        self.assertEqual(encoded["expert_ids"], 1)

    def test_universal_schema_rejects_incomplete_auxiliary_dimensions(self) -> None:
        config = feature_config_for_schema("ptcg_features_universal")
        invalid = PTCGFeatureConfig(**{**config.__dict__, "entity_numeric_dim": 16})
        with self.assertRaisesRegex(ValueError, "universal feature configuration"):
            feature_schema_for_config(invalid)

    def test_submission_builder_strips_training_state(self) -> None:
        model = self._model()
        feature_config = PTCGFeatureConfig()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "deck.csv").write_text("5\n" * 60, encoding="utf-8")
            (source / "cg").mkdir()
            (source / "cg" / "libcg.so").write_bytes(b"test")
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "step": 3,
                    "model": model.state_dict(),
                    "optimizer": {"training_only": True},
                    "metadata": {
                        "model_config": model.config.to_dict(),
                        "feature_config": {
                            **feature_config.__dict__,
                            "schema_version": "ptcg_features_v6",
                        },
                        "max_selection_count": 64,
                    },
                },
                checkpoint,
            )
            output = build_submission(
                root / "package",
                checkpoint,
                source,
                experiment_id="0001-test",
            )
            exported = torch.load(
                output / "strategy" / "model.bin",
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(set(exported), {"step", "model", "metadata"})
        self.assertEqual(exported["step"], 3)


if __name__ == "__main__":
    unittest.main()
