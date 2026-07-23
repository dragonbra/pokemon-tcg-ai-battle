from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from rl_environment.model import ModelConfig
from train.alakazam_bc_rl.features import PTCGFeatureConfig
from train.alakazam_reward_weighted_bc.card_categories import (
    CATEGORY_NAMES,
    load_card_category_lookup,
)
from train.alakazam_reward_weighted_bc.inference import RewardWeightedFullActionPolicy
from train.alakazam_reward_weighted_bc.model import (
    CategoryAugmentedFullActionPolicyValueNet,
)
from train.alakazam_reward_weighted_bc.config import (
    ExperimentConfig,
    LossConfig,
    RewardComponentConfig,
    RewardConfig,
    WeightingConfig,
)
from train.alakazam_reward_weighted_bc.reward_annotations import annotate_episode_records
from train.alakazam_reward_weighted_bc.rewards import compose_reward, imitation_weights
from train.alakazam_reward_weighted_bc.train import train


class RewardWeightedBCTests(unittest.TestCase):
    def test_reward_components_are_independently_switchable(self) -> None:
        config = RewardConfig(
            components=(
                RewardComponentConfig("terminal", "terminal_outcome", weight=1.0),
                RewardComponentConfig(
                    "relay", "reward_metrics.decision.post_ko_relay_credit", weight=0.5
                ),
                RewardComponentConfig(
                    "disabled", "missing", enabled=False
                ),
            ),
            clip_min=-1.0,
            clip_max=1.0,
        )
        total, components = compose_reward(
            {
                "terminal_outcome": 1.0,
                "reward_metrics": {"decision": {"post_ko_relay_credit": -1.0}},
            },
            config,
        )
        self.assertEqual(total, 0.5)
        self.assertEqual(components, {"terminal": 1.0, "relay": -0.5})

    def test_advantage_weights_are_non_negative_and_clipped(self) -> None:
        advantage, weights = imitation_weights(
            torch.tensor([-10.0, 0.0, 10.0]),
            WeightingConfig(
                mode="exponential",
                baseline="zero",
                temperature=1.0,
                min_weight=0.1,
                max_weight=5.0,
            ),
        )
        self.assertTrue(torch.equal(advantage, torch.tensor([-10.0, 0.0, 10.0])))
        self.assertTrue(torch.allclose(weights, torch.tensor([0.1, 1.0, 5.0])))

    def test_official_card_category_becomes_a_separate_embedding(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cards.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "Card ID",
                        "Stage (Pokémon)/Type (Energy and Trainer)",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Card ID": 5,
                        "Stage (Pokémon)/Type (Energy and Trainer)": "Basic Energy",
                    }
                )
                writer.writerow(
                    {
                        "Card ID": 741,
                        "Stage (Pokémon)/Type (Energy and Trainer)": "Basic Pokémon",
                    }
                )
            lookup, counts = load_card_category_lookup(path, card_vocab_size=1024)
        self.assertEqual(lookup[6], CATEGORY_NAMES.index("Basic Energy") + 1)
        self.assertEqual(lookup[742], CATEGORY_NAMES.index("Basic Pokémon") + 1)
        self.assertEqual(counts, {"Basic Energy": 1, "Basic Pokémon": 1})

        model = CategoryAugmentedFullActionPolicyValueNet(
            ModelConfig(
                state_numeric_dim=2,
                state_token_count=2,
                candidate_numeric_dim=2,
                max_candidates=2,
                card_vocab_size=1024,
                action_type_vocab_size=4,
                d_model=8,
                hidden_dim=16,
                num_heads=2,
            ),
            card_category_lookup=lookup,
            category_vocab_size=len(CATEGORY_NAMES),
            max_selection_count=2,
        )
        encoded = model._state_card_embedding(torch.tensor([[6, 742]]))
        id_only = model.state_card_embedding(torch.tensor([[6, 742]]))
        self.assertFalse(torch.equal(encoded, id_only))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "category.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "metadata": {
                        "model_config": model.config.to_dict(),
                        "feature_config": PTCGFeatureConfig().__dict__,
                        "max_selection_count": 2,
                        "card_category_embedding": {
                            "enabled": True,
                            "category_vocab_size": len(CATEGORY_NAMES),
                            "scale": 1.0,
                        },
                    },
                },
                checkpoint,
            )
            restored = RewardWeightedFullActionPolicy.from_checkpoint(checkpoint)
        self.assertIsInstance(restored.model, CategoryAugmentedFullActionPolicyValueNet)
        self.assertTrue(
            torch.equal(restored.model.card_category_lookup, model.card_category_lookup)
        )

    @staticmethod
    def _current(
        turn: int,
        active_id: int,
        *,
        hand: list[int] | None = None,
        bench: list[int] | None = None,
        retreated: bool = False,
        energies: list[int] | None = None,
    ) -> dict:
        return {
            "turn": turn,
            "yourIndex": 0,
            "firstPlayer": 0,
            "retreated": retreated,
            "players": [
                {
                    "active": [
                        {
                            "id": active_id,
                            "serial": 1,
                            "hp": 100,
                            "energies": list(energies or []),
                        }
                    ],
                    "bench": [
                        {"id": card_id, "serial": index + 2, "hp": 70}
                        for index, card_id in enumerate(bench or [])
                    ],
                    "hand": [{"id": card_id} for card_id in hand or []],
                    "discard": [],
                    "deckCount": 40,
                    "prizeCount": 6,
                },
                {"active": [{"id": 999, "serial": 99, "hp": 200}], "prizeCount": 6},
            ],
        }

    @staticmethod
    def _record(step: int, attack_id: int | None = None) -> dict:
        return {
            "episode_id": 1,
            "episode_step": step,
            "player_index": 0,
            "targets": [0],
            "terminal_outcome": 1.0,
            "_attack_id": attack_id,
        }

    def test_strict_dunsparce_bridge_records_every_milestone(self) -> None:
        currents = [
            self._current(1, 305, hand=[1079]),
            self._current(1, 305, hand=[1079], bench=[741]),
            self._current(3, 305, hand=[66, 1079, 743, 5], bench=[741]),
            self._current(3, 66, hand=[1079, 743, 5], bench=[741]),
            self._current(3, 741, hand=[1079, 743, 5], bench=[66], retreated=True),
            self._current(3, 743, hand=[5], bench=[66]),
            self._current(3, 743, hand=[], bench=[66], energies=[5]),
        ]
        records = [self._record(index, 1072 if index == 6 else None) for index in range(7)]
        payload = {
            "steps": [
                [
                    {
                        "observation": {
                            "current": current,
                            "select": {
                                "option": [
                                    {
                                        "type": 1,
                                        **(
                                            {"attackId": record["_attack_id"]}
                                            if record["_attack_id"] is not None
                                            else {}
                                        ),
                                    }
                                ]
                            },
                            "logs": [],
                        }
                    },
                    {"observation": {}},
                ]
                for current, record in zip(currents, records)
            ]
        }
        for record in records:
            record.pop("_attack_id")
        annotated = annotate_episode_records(records, payload)
        episode = annotated[-1]["reward_metrics"]["episode"]
        components = episode["dunsparce_bridge_components"]
        self.assertTrue(episode["dunsparce_bridge_opportunity"])
        self.assertTrue(episode["dunsparce_bridge_success"])
        self.assertTrue(all(components.values()))
        self.assertEqual(
            annotated[1]["reward_metrics"]["decision"]["dunsparce_first_turn_setup_credit"],
            1.0,
        )
        self.assertEqual(
            annotated[-1]["reward_metrics"]["decision"][
                "dunsparce_second_turn_execution_credit"
            ],
            1.0,
        )

    def test_control_training_smoke_loads_a_full_action_checkpoint(self) -> None:
        model = CategoryAugmentedFullActionPolicyValueNet(
            ModelConfig(
                state_numeric_dim=2,
                state_token_count=2,
                candidate_numeric_dim=2,
                max_candidates=2,
                card_vocab_size=8,
                action_type_vocab_size=4,
                d_model=8,
                hidden_dim=16,
                num_heads=2,
            ),
            card_category_lookup=[0] * 9,
            category_vocab_size=1,
            max_selection_count=2,
        )
        # A source BC checkpoint has no category module. Strip the experimental
        # parameters to exercise the same initialization path used in practice.
        source_state = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("card_category_")
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source.pt"
            torch.save(
                {
                    "model": source_state,
                    "metadata": {
                        "model_config": model.config.to_dict(),
                        "max_selection_count": 2,
                    },
                },
                checkpoint,
            )
            encoded = {
                "state_numeric": [0.0, 0.0],
                "state_card_ids": [1, 0],
                "action_type_ids": [1, 2],
                "action_card_ids": [1, 2],
                "action_target_ids": [0, 0],
                "action_numeric": [[0.0, 0.0], [0.0, 0.0]],
                "action_mask": [True, True],
            }
            dataset = root / "dataset.jsonl"
            rows = []
            for index, split in enumerate(("train", "train", "validation", "validation")):
                rows.append(
                    {
                        "dataset_version": "ptcg_kaggle_bc_v1",
                        "expert_team_name": "test expert",
                        "episode_id": index,
                        "episode_step": 0,
                        "player_index": 0,
                        "split": split,
                        "targets": [index % 2],
                        "target_count": 1,
                        "selection_min_count": 1,
                        "selection_max_count": 1,
                        "encoded": encoded,
                    }
                )
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = train(
                dataset,
                checkpoint,
                root / "output",
                ExperimentConfig(
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    storage_path=str(root),
                    min_free_gib=0.0,
                    weighting=WeightingConfig(
                        mode="uniform", min_weight=1.0, max_weight=1.0
                    ),
                    loss=LossConfig(policy=1.0, count=1.0, value=0.0),
                ),
            )
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
