from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from rl_environment.model import ModelConfig
from train.alakazam_bc_rl.features import PTCGFeatureConfig
from train.alakazam_reward_weighted_bc.card_categories import (
    CATEGORY_COUNTS,
    CATEGORY_NAMES,
    OFFICIAL_CARD_CATEGORY_BY_ID,
    SOURCE_CSV_SHA256,
    STATIC_MAPPING_SHA256,
    build_card_category_lookup,
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
        lookup = build_card_category_lookup(card_vocab_size=1024)
        self.assertEqual(lookup[6], CATEGORY_NAMES.index("Basic Energy") + 1)
        self.assertEqual(lookup[742], CATEGORY_NAMES.index("Basic Pokémon") + 1)
        self.assertEqual(len(OFFICIAL_CARD_CATEGORY_BY_ID), 1267)
        self.assertEqual(
            CATEGORY_COUNTS,
            {
                "Basic Pokémon": 595,
                "Stage 1 Pokémon": 345,
                "Stage 2 Pokémon": 116,
                "Item": 77,
                "Supporter": 61,
                "Pokémon Tool": 27,
                "Stadium": 26,
                "Special Energy": 12,
                "Basic Energy": 8,
            },
        )
        self.assertEqual(
            SOURCE_CSV_SHA256,
            "a0ea63cf7adcb65d35436ce0eb390de6e2e35654a7c67c065a45f4abaa00f373",
        )
        self.assertEqual(
            STATIC_MAPPING_SHA256,
            "46cd549b46bfed98dc4afff0466912ab67edf65d1e8689f50a98678c7a4b3de9",
        )
        mapping_payload = json.dumps(
            OFFICIAL_CARD_CATEGORY_BY_ID,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.assertEqual(hashlib.sha256(mapping_payload).hexdigest(), STATIC_MAPPING_SHA256)

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
        bench_energies: dict[int, list[int]] | None = None,
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
                        {
                            "id": card_id,
                            "serial": index + 2,
                            "hp": 70,
                            "energies": list((bench_energies or {}).get(card_id, [])),
                        }
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
    def _record(
        step: int,
        attack_id: int | None = None,
        *,
        dudunsparce_ability: bool = False,
    ) -> dict:
        return {
            "episode_id": 1,
            "episode_step": step,
            "player_index": 0,
            "targets": [0],
            "terminal_outcome": 1.0,
            "_attack_id": attack_id,
            "_dudunsparce_ability": dudunsparce_ability,
        }

    def test_opening_abra_is_preferred_over_dunsparce(self) -> None:
        cases = (
            (self._current(1, 741, hand=[305]), 1.0, False),
            (self._current(1, 305, hand=[741]), -1.0, False),
        )
        for current, expected_credit, expected_bridge in cases:
            record = self._record(0)
            payload = {
                "steps": [
                    [
                        {
                            "observation": {
                                "current": current,
                                "select": {"option": [{"type": 1}]},
                                "logs": [],
                            }
                        },
                        {"observation": {}},
                    ]
                ]
            }
            record.pop("_attack_id")
            record.pop("_dudunsparce_ability")
            annotated = annotate_episode_records([record], payload)[0]
            self.assertEqual(
                annotated["reward_metrics"]["decision"]["opening_active_abra_credit"],
                expected_credit,
            )
            self.assertEqual(
                annotated["reward_metrics"]["episode"]["dunsparce_bridge_opportunity"],
                expected_bridge,
            )

    def test_strict_dunsparce_bridge_records_every_milestone(self) -> None:
        currents = [
            self._current(1, 305, hand=[1079]),
            self._current(1, 305, hand=[1079], bench=[741], bench_energies={741: [5]}),
            self._current(
                3,
                305,
                hand=[66, 1079, 743],
                bench=[741],
                bench_energies={741: [5]},
            ),
            self._current(
                3,
                66,
                hand=[1079, 743],
                bench=[741],
                bench_energies={741: [5]},
            ),
            self._current(
                3,
                741,
                hand=[1079, 743],
                bench=[66],
                energies=[5],
            ),
            self._current(3, 743, hand=[], bench=[66], energies=[5]),
            self._current(3, 743, hand=[], bench=[66], energies=[5]),
        ]
        records = [
            self._record(
                index,
                1072 if index == 6 else None,
                dudunsparce_ability=index == 3,
            )
            for index in range(7)
        ]
        payload = {
            "steps": [
                [
                    {
                        "observation": {
                            "current": current,
                            "select": {
                                "option": [
                                    {
                                        "type": (
                                            10 if record["_dudunsparce_ability"] else 1
                                        ),
                                        **(
                                            {"area": 4, "index": 0}
                                            if record["_dudunsparce_ability"]
                                            else {}
                                        ),
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
            record.pop("_dudunsparce_ability")
        annotated = annotate_episode_records(records, payload)
        episode = annotated[-1]["reward_metrics"]["episode"]
        components = episode["dunsparce_bridge_components"]
        self.assertTrue(episode["dunsparce_bridge_opportunity"])
        self.assertTrue(episode["dunsparce_bridge_success"])
        self.assertFalse(components["target_energy_in_hand"])
        self.assertTrue(components["target_energy_already_attached"])
        self.assertTrue(components["target_resources_ready"])
        self.assertTrue(components["used_dudunsparce_ability"])
        self.assertTrue(components["switched_to_abra_after_ability"])
        self.assertTrue(components["dudunsparce_ability_handoff"])
        self.assertTrue(components["active_psychic_at_attack"])
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

        retreat_payload = json.loads(json.dumps(payload))
        retreat_payload["steps"][3][0]["observation"]["select"]["option"][0] = {
            "type": 13
        }
        retreat_payload["steps"][4][0]["observation"]["current"]["retreated"] = True
        retreat_records = [
            self._record(index, 1072 if index == 6 else None) for index in range(7)
        ]
        for record in retreat_records:
            record.pop("_attack_id")
            record.pop("_dudunsparce_ability")
        retreat_episode = annotate_episode_records(
            retreat_records, retreat_payload
        )[-1]["reward_metrics"]["episode"]
        self.assertFalse(retreat_episode["dunsparce_bridge_success"])
        self.assertFalse(
            retreat_episode["dunsparce_bridge_components"]["used_dudunsparce_ability"]
        )

    def test_official_visual_frame_contract_aligns_selected_actions(self) -> None:
        currents = [
            self._current(1, 305, hand=[1079]),
            self._current(1, 305, hand=[1079], bench=[741], bench_energies={741: [5]}),
            self._current(
                3,
                66,
                hand=[1079, 743],
                bench=[741],
                bench_energies={741: [5]},
            ),
            self._current(3, 741, hand=[1079, 743], bench=[66], energies=[5]),
            self._current(3, 743, hand=[], bench=[66], energies=[5]),
        ]
        records = [
            {**self._record(index), "targets": [5 if index == 2 else 2 if index == 4 else 0]}
            for index in range(len(currents))
        ]
        for record in records:
            record.pop("_attack_id")
            record.pop("_dudunsparce_ability")
        frames = []
        for index, current in enumerate(currents):
            options = [{"type": 1} for _ in range(6)]
            selected = [0]
            if index == 2:
                options[5] = {"type": 10, "area": 4, "index": 0}
                selected = [5]
            elif index == 4:
                options[2] = {"type": 14, "attackId": 1072}
                selected = [2]
            frames.append(
                {
                    "obs": {
                        "current": current,
                        "select": {"option": options},
                        "logs": [],
                    },
                    "selected": selected,
                }
            )
        payload = {
            "steps": [
                [
                    {"visualize": frames},
                    {"visualize": frames[:-1]},
                ]
            ]
        }

        annotated = annotate_episode_records(records, payload)
        episode = annotated[-1]["reward_metrics"]["episode"]
        self.assertTrue(episode["dunsparce_bridge_success"])
        self.assertTrue(
            episode["dunsparce_bridge_components"]["used_dudunsparce_ability"]
        )
        self.assertEqual(
            annotated[-1]["reward_metrics"]["audit"]["selected_attack_id"],
            1072,
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
                    train_eval_interval=1,
                    weighting=WeightingConfig(
                        mode="uniform", min_weight=1.0, max_weight=1.0
                    ),
                    loss=LossConfig(policy=1.0, count=1.0, value=0.0),
                ),
            )
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["best_epoch"], 1)
        self.assertIn("train_eval/exact_action_rate", summary["last_epoch"])


if __name__ == "__main__":
    unittest.main()
