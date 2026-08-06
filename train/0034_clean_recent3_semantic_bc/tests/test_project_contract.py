from __future__ import annotations

import importlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ID = "0034_clean_recent3_semantic_bc"
MANIFEST = REPOSITORY_ROOT / "experiments" / PROJECT_ID / "manifest.json"
FEATURE_AUDIT = (
    REPOSITORY_ROOT / "train" / PROJECT_ID / "contracts" / "feature_audit.json"
)
FIELDS = importlib.import_module(f"train.{PROJECT_ID}.contracts.fields")
EVALUATION_PROTOCOL = (
    REPOSITORY_ROOT / "experiments" / PROJECT_ID / "evaluation_protocol.json"
)
EVALUATION_RUNNER = (
    REPOSITORY_ROOT / "train" / PROJECT_ID / "tools" / "evaluate_marnie_six_bc.py"
)


class ProjectContractTests(unittest.TestCase):
    def test_manifest_declares_persona_free_self_contained_lineage(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["project_id"], PROJECT_ID)
        self.assertEqual(manifest["status"], "cleaned_recent3_trained_and_evaluated")
        self.assertFalse(manifest["actor_source_identity_visible"])
        self.assertEqual(
            manifest["implementation_lineage"],
            {
                "source_project": "0032_audited_semantic_foundation_pretraining",
                "semantic_relationship": "type_aware_effect_summary_extension_with_minimal_action_cleaning",
                "executable_dependency": False,
            },
        )
        design = REPOSITORY_ROOT / manifest["design"]
        self.assertTrue(design.is_file())

    def test_feature_audit_exactly_matches_executable_field_contract(self) -> None:
        audit = json.loads(FEATURE_AUDIT.read_text(encoding="utf-8"))
        fields = audit["observation_fields"]
        self.assertEqual(fields["global_categorical"], list(FIELDS.GLOBAL_CAT_FIELDS))
        self.assertEqual(fields["global_numeric"], list(FIELDS.GLOBAL_NUM_FIELDS))
        self.assertEqual(fields["card_categorical"], list(FIELDS.CARD_CAT_FIELDS))
        self.assertEqual(fields["card_numeric"], list(FIELDS.CARD_NUM_FIELDS))
        self.assertEqual(fields["resource_categorical"], list(FIELDS.RESOURCE_CAT_FIELDS))
        self.assertEqual(fields["resource_numeric"], list(FIELDS.RESOURCE_NUM_FIELDS))
        self.assertEqual(fields["event_categorical"], list(FIELDS.EVENT_CAT_FIELDS))
        self.assertEqual(fields["event_numeric"], list(FIELDS.EVENT_NUM_FIELDS))
        self.assertEqual(fields["option_categorical"], list(FIELDS.OPTION_CAT_FIELDS))
        self.assertEqual(fields["option_numeric"], list(FIELDS.OPTION_NUM_FIELDS))
        self.assertEqual(
            (
                FIELDS.WIDTHS.global_cat,
                FIELDS.WIDTHS.global_num,
                FIELDS.WIDTHS.card_cat,
                FIELDS.WIDTHS.card_num,
                FIELDS.WIDTHS.resource_cat,
                FIELDS.WIDTHS.resource_num,
                FIELDS.WIDTHS.event_cat,
                FIELDS.WIDTHS.event_num,
                FIELDS.WIDTHS.option_cat,
                FIELDS.WIDTHS.option_num,
            ),
            (7, 11, 6, 14, 3, 10, 17, 4, 11, 2),
        )

    def test_formal_dataset_scope_is_the_approved_recent_three_days(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        scope = manifest["dataset_scope"]
        self.assertEqual(scope["status"], "accepted")
        self.assertEqual(scope["perspective"], "unique_positive_terminal_winner")
        self.assertEqual(
            (scope["start_date"], scope["end_date"]),
            ("2026-08-01", "2026-08-03"),
        )
        self.assertEqual(scope["records"], 1_204_592)
        self.assertEqual(
            scope["split_counts"],
            {"train": 1_081_527, "validation": 123_065},
        )
        self.assertEqual(scope["sample_weights"], [0.95, 1.0, 1.15])
        self.assertEqual(scope["event_source_relations"], 0)
        self.assertEqual(scope["event_target_relations"], 0)

    def test_formal_training_contract_is_recorded_after_completion(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        training = manifest["formal_training"]
        self.assertEqual(training["status"], "complete")
        self.assertEqual(training["epochs_completed"], 3)
        self.assertEqual(training["full_train_passes_completed"], 3)
        self.assertEqual(training["full_validation_passes_completed"], 3)
        self.assertEqual(training["model_config"]["d_model"], 320)
        self.assertEqual(training["model_config"]["state_layers"], 4)
        self.assertEqual(training["model_config"]["option_layers"], 3)
        self.assertEqual(training["best_model_sha256"], "6bb23f0b23c04c478c7c0926310967c73325685b0308959f147240219e3538ff")
        self.assertTrue(training["random_initialization"])
        self.assertIsNone(training["initialized_from_checkpoint"])
        self.assertIsNone(training["wandb"])

    def test_marnie_evaluation_protocol_fixes_six_bc_opponents(self) -> None:
        protocol = json.loads(EVALUATION_PROTOCOL.read_text(encoding="utf-8"))
        pool = protocol["opponent_pool"]
        self.assertEqual(pool["games_per_opponent"], 50)
        self.assertEqual(pool["expected_total_games"], 300)
        self.assertEqual(
            [item["name"] for item in pool["opponents"]],
            [
                "pure_lucario_bc512_e4",
                "cynthia_core_meanpool_epochmix_v1",
                "kangaskhan_crustle_meanpool_epochmix_v2",
                "marnie_prize_control_v4",
                "yushin_alakazam_idonly_bc_v1",
                "dragapult_large_data_decoder_2layer",
            ],
        )
        self.assertEqual(protocol["candidate"]["inference_device"], "cuda:0")
        self.assertEqual(protocol["candidate"]["required_fallback_rate"], 0.0)

    def test_marnie_evaluation_runner_loads_only_the_0034_package(self) -> None:
        source = EVALUATION_RUNNER.read_text(encoding="utf-8")
        self.assertIn('package_dir / "load_model.py"', source)
        self.assertIn('package_dir / "best_model.pt"', source)
        self.assertIn('default=50', source)
        self.assertIn('report["totals"]["errors"]', source)
        self.assertIn('item["fallback_decisions"]', source)
        self.assertNotIn("train.0033_effect_summary_semantic", source)


if __name__ == "__main__":
    unittest.main()
