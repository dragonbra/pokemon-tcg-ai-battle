from __future__ import annotations

import unittest
import gzip
import json
import tempfile
from pathlib import Path

import torch

from train.project_0012_alakazam_sota_feature_engineering.batching import (
    collate_id_only,
    permute_candidates,
)
from train.project_0012_alakazam_sota_feature_engineering.config import ExperimentConfig
from train.project_0012_alakazam_sota_feature_engineering.codec import FeatureEngineeringCodec
from train.project_0012_alakazam_sota_feature_engineering.inference import FeatureEngineeringInference
from train.project_0012_alakazam_sota_feature_engineering.model import (
    FeatureEngineeringPolicy,
    FeatureModelConfig,
)
from train.project_0012_alakazam_sota_feature_engineering.training import train
from train.project_0010_alakazam_sota_model.model import IDOnlyCodec, IDOnlyConfig, IDOnlyPointerPolicy


def _observation() -> dict[str, object]:
    return {
        "current": {
            "yourIndex": 0,
            "firstPlayer": 0,
            "turn": 3,
            "turnActionCount": 2,
            "players": [
                {
                    "active": [{"id": 741, "hp": 60, "maxHp": 60}],
                    "bench": [{"id": 305, "hp": 50, "maxHp": 50}],
                    "hand": [{"id": 743}, {"id": 1079}, {"id": 5}],
                    "discard": [],
                    "deckCount": 40,
                    "handCount": 3,
                    "prize": [1, 2, 3, 4, 5, 6],
                },
                {
                    "active": [{"id": 306, "hp": 120, "maxHp": 150}],
                    "bench": [],
                    "discard": [],
                    "deckCount": 42,
                    "handCount": 5,
                    "prize": [1, 2, 3, 4, 5, 6],
                },
            ],
        },
        "select": {
            "type": 1,
            "context": 3,
            "minCount": 2,
            "maxCount": 2,
            "option": [
                {"type": 2, "playerIndex": 0, "area": 2, "index": 0, "cardId": 743},
                {"type": 3, "playerIndex": 0, "area": 2, "index": 1, "cardId": 1079},
                {"type": 4, "playerIndex": 0, "area": 2, "index": 2, "cardId": 5},
            ],
        },
    }


class AlakazamSotaFeatureEngineeringTests(unittest.TestCase):
    def test_baseline_is_state_dict_and_logit_identical_to_0010(self) -> None:
        base_config = IDOnlyConfig(d_model=32, heads=4, encoder_layers=1, dropout=0.0)
        feature_config = FeatureModelConfig(
            d_model=32,
            heads=4,
            encoder_layers=1,
            dropout=0.0,
        )
        torch.manual_seed(19)
        reference = IDOnlyPointerPolicy(base_config).eval()
        torch.manual_seed(19)
        experiment = FeatureEngineeringPolicy(feature_config).eval()
        self.assertEqual(reference.state_dict().keys(), experiment.state_dict().keys())
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, experiment.state_dict()[name]), name)
        row = IDOnlyCodec(base_config).encode(_observation(), [2, 0])
        batch = collate_id_only([row])  # type: ignore[list-item]
        with torch.inference_mode():
            expected = reference.teacher_logits(batch)
            actual = experiment.teacher_logits(batch)
        self.assertTrue(torch.equal(expected, actual))

    def test_candidate_permutation_remaps_each_selected_index_without_reordering(self) -> None:
        config = IDOnlyConfig(d_model=32, heads=4, encoder_layers=1)
        row = IDOnlyCodec(config).encode(_observation(), [2, 0])
        batch = collate_id_only([row])  # type: ignore[list-item]
        original_cards = batch["option_cat"][0, :, 4].tolist()
        permuted = permute_candidates(batch, seed=7)
        remapped = permuted["targets"][0, :2].tolist()
        selected_cards = [permuted["option_cat"][0, index, 4].item() for index in remapped]
        self.assertEqual(selected_cards, [original_cards[2], original_cards[0]])
        self.assertEqual(permuted["targets"][0, 2].item(), 3)
        self.assertEqual(sorted(permuted["option_cat"][0, :, 4].tolist()), sorted(original_cards))

    def test_no_position_model_is_permutation_equivariant(self) -> None:
        config = FeatureModelConfig(
            d_model=32,
            heads=4,
            encoder_layers=1,
            dropout=0.0,
            remove_option_position=True,
        )
        row = IDOnlyCodec(config).encode(_observation(), [2, 0])
        original = collate_id_only([row])  # type: ignore[list-item]
        permuted = permute_candidates(original, seed=31)
        model = FeatureEngineeringPolicy(config).eval()
        with torch.inference_mode():
            original_loss = torch.nn.functional.cross_entropy(
                model.teacher_logits(original).flatten(0, 1),
                original["targets"].flatten(),
            )
            permuted_loss = torch.nn.functional.cross_entropy(
                model.teacher_logits(permuted).flatten(0, 1),
                permuted["targets"].flatten(),
            )
        self.assertTrue(torch.allclose(original_loss, permuted_loss, atol=1e-6))

    def test_config_rejects_permutation_training_with_position_feature(self) -> None:
        with self.assertRaisesRegex(ValueError, "remove_option_position"):
            ExperimentConfig(permutation_augmentation=True).validate()

    def test_action_primitive_codec_and_model_cover_audited_fields(self) -> None:
        observation = _observation()
        select = observation["select"]  # type: ignore[index]
        select["effect"] = {"id": 1086, "playerIndex": 0, "serial": 38}  # type: ignore[index]
        select["contextCard"] = {"id": 742, "playerIndex": 0, "serial": 23}  # type: ignore[index]
        select["option"][1].update(  # type: ignore[index]
            {"attackId": 17, "count": 2, "energyIndex": 1, "toolIndex": 0}
        )
        config = FeatureModelConfig(
            d_model=32,
            heads=4,
            encoder_layers=1,
            dropout=0.0,
            remove_option_position=True,
            role_separated_action=True,
            action_primitive_context=True,
            action_primitive_dim=8,
        )
        row = FeatureEngineeringCodec(config).encode(observation, [2, 0])
        self.assertEqual(row["option_primitive_cat"][1], [18, 3, 2, 1])  # type: ignore[index]
        self.assertEqual(row["action_context_cat"], [1086, 742, 3, 3])  # type: ignore[index]
        batch = collate_id_only([row])  # type: ignore[list-item]
        model = FeatureEngineeringPolicy(config).eval()
        with torch.inference_mode():
            logits = model.teacher_logits(batch)
        self.assertEqual(tuple(logits.shape), (1, 3, 4))
        self.assertTrue(torch.isfinite(logits).all())

    def test_action_primitive_permutation_stays_aligned_and_equivariant(self) -> None:
        observation = _observation()
        options = observation["select"]["option"]  # type: ignore[index]
        for index, option in enumerate(options):  # type: ignore[union-attr]
            option["attackId"] = 20 + index
            option["count"] = index
        config = FeatureModelConfig(
            d_model=32,
            heads=4,
            encoder_layers=1,
            dropout=0.0,
            remove_option_position=True,
            role_separated_action=True,
            action_primitive_context=True,
            action_primitive_dim=8,
        )
        row = FeatureEngineeringCodec(config).encode(observation, [2, 0])
        original = collate_id_only([row])  # type: ignore[list-item]
        permuted = permute_candidates(original, seed=31)
        new_to_old = permuted["permutation_new_to_old"][0]
        expected = original["option_primitive_cat"][0, new_to_old]
        self.assertTrue(torch.equal(permuted["option_primitive_cat"][0], expected))

        model = FeatureEngineeringPolicy(config).eval()
        with torch.inference_mode():
            original_logits = model.teacher_logits(original)
            permuted_logits = model.teacher_logits(permuted)
        old_to_new = permuted["permutation_old_to_new"]
        aligned = torch.empty_like(permuted_logits)
        aligned[..., :3] = permuted_logits[..., :3].gather(
            2, old_to_new.unsqueeze(1).expand(-1, permuted_logits.size(1), -1)
        )
        aligned[..., 3] = permuted_logits[..., 3]
        self.assertTrue(torch.allclose(original_logits, aligned, atol=1e-6))

    def test_transition_auxiliary_uses_selected_action_and_masked_labels(self) -> None:
        config = FeatureModelConfig(
            d_model=32,
            heads=4,
            encoder_layers=1,
            dropout=0.0,
            remove_option_position=True,
            role_separated_action=True,
            action_primitive_context=True,
            action_primitive_dim=8,
            action_transition_auxiliary=True,
        )
        row = FeatureEngineeringCodec(config).encode(_observation(), [2, 0])
        row.update(  # type: ignore[union-attr]
            {
                "transition_delta": [0.1] * 11,
                "transition_delta_mask": True,
                "transition_turn_changed": 0.0,
                "transition_next_mask": True,
            }
        )
        batch = collate_id_only([row])  # type: ignore[list-item]
        model = FeatureEngineeringPolicy(config)
        logits, delta, turn = model.teacher_logits_and_transition(batch)
        self.assertEqual(tuple(logits.shape), (1, 3, 4))
        self.assertEqual(tuple(delta.shape), (1, 11))
        self.assertEqual(tuple(turn.shape), (1,))
        loss = logits.mean() + delta.mean() + turn.mean()
        loss.backward()
        self.assertIsNotNone(model.transition_delta_head.weight.grad)

    def test_transition_auxiliary_uses_stop_dtype_for_autocast_mask(self) -> None:
        config = FeatureModelConfig(
            d_model=32,
            heads=4,
            encoder_layers=1,
            dropout=0.0,
            remove_option_position=True,
            role_separated_action=True,
            action_primitive_context=True,
            action_primitive_dim=8,
            action_transition_auxiliary=True,
        )
        row = FeatureEngineeringCodec(config).encode(_observation(), [2, 0])
        row.update(  # type: ignore[union-attr]
            {
                "transition_delta": [0.0] * 11,
                "transition_delta_mask": True,
                "transition_turn_changed": 0.0,
                "transition_next_mask": True,
            }
        )
        batch = collate_id_only([row])  # type: ignore[list-item]
        model = FeatureEngineeringPolicy(config).eval()
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            logits, delta, turn = model.teacher_logits_and_transition(batch)
        self.assertTrue(torch.isfinite(logits[logits > torch.finfo(logits.dtype).min]).all())
        self.assertTrue(torch.isfinite(delta).all())
        self.assertTrue(torch.isfinite(turn).all())

    def test_inference_preserves_live_min_count_beyond_training_step_cap(self) -> None:
        observation = _observation()
        select = observation["select"]  # type: ignore[index]
        select["minCount"] = 17  # type: ignore[index]
        select["maxCount"] = 17  # type: ignore[index]
        select["option"] = [  # type: ignore[index]
            {"type": 2, "playerIndex": 0, "area": 2, "index": index, "cardId": 743}
            for index in range(20)
        ]
        config = FeatureModelConfig(
            d_model=32,
            heads=4,
            encoder_layers=1,
            dropout=0.0,
            max_action_steps=16,
        )
        policy = FeatureEngineeringInference(
            FeatureEngineeringPolicy(config), FeatureEngineeringCodec(config)
        )
        action = policy.select(observation)
        self.assertEqual(len(action), 17)
        self.assertEqual(len(set(action)), 17)
        self.assertTrue(all(0 <= index < 20 for index in action))

    def test_action_history_enters_state_sequence_without_future_label(self) -> None:
        config = FeatureModelConfig(
            d_model=32,
            heads=4,
            encoder_layers=1,
            dropout=0.0,
            remove_option_position=True,
            role_separated_action=True,
            action_history=True,
            max_action_history=4,
        )
        row = FeatureEngineeringCodec(config).encode(_observation(), [2, 0])
        row["action_history_cat"] = [  # type: ignore[index]
            [2, 4, 3, 743, 0, 1],
            [2, 4, 4, 1079, 741, 2],
        ]
        batch = collate_id_only([row])  # type: ignore[list-item]
        self.assertEqual(tuple(batch["action_history_cat"].shape), (1, 2, 6))
        model = FeatureEngineeringPolicy(config).eval()
        with torch.inference_mode():
            logits = model.teacher_logits(batch)
        self.assertEqual(tuple(logits.shape), (1, 3, 4))
        self.assertTrue(torch.isfinite(logits).all())

    def test_cpu_training_smoke_writes_reproducible_artifacts_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            model_config = FeatureModelConfig(
                d_model=32,
                heads=4,
                encoder_layers=1,
                dropout=0.0,
                remove_option_position=True,
            )
            row = IDOnlyCodec(model_config).encode(_observation(), [2, 0])
            for split, count in (("train", 3), ("validation", 2), ("test", 0)):
                with gzip.open(dataset_root / f"{split}.jsonl.gz", "wt") as handle:
                    for _ in range(count):
                        handle.write(json.dumps(row) + "\n")
            (dataset_root / "dataset_audit.json").write_text(
                json.dumps(
                    {
                        "schema_version": "alakazam_sota_id_only_dataset_v1",
                        "records_by_split": {"train": 3, "validation": 2, "test": 0},
                        "outputs": {},
                    }
                ),
                encoding="utf-8",
            )
            config = ExperimentConfig(
                experiment_name="cpu_smoke",
                model=model_config,
                batch_size=2,
                epochs=1,
                learning_rate=1e-3,
                permutation_augmentation=True,
                device="cpu",
                amp=False,
                storage_path=str(root),
                min_free_gib=0.0,
            )
            output = root / "run"
            summary = train(dataset_root, output, config)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["last_completed_epoch"], 1)
            self.assertTrue((output / "checkpoints" / "best_exact.pt").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "ANALYSIS.md").is_file())


if __name__ == "__main__":
    unittest.main()
