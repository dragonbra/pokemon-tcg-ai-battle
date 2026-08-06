from __future__ import annotations

import importlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ID = "0033_effect_summary_semantic_foundation_pretraining"
MANIFEST = REPOSITORY_ROOT / "experiments" / PROJECT_ID / "manifest.json"
FEATURE_AUDIT = (
    REPOSITORY_ROOT / "train" / PROJECT_ID / "contracts" / "feature_audit.json"
)
FIELDS = importlib.import_module(f"train.{PROJECT_ID}.contracts.fields")


class ProjectContractTests(unittest.TestCase):
    def test_manifest_declares_persona_free_self_contained_lineage(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["project_id"], PROJECT_ID)
        self.assertEqual(manifest["status"], "prototype_model_structure_verified")
        self.assertFalse(manifest["actor_source_identity_visible"])
        self.assertEqual(
            manifest["implementation_lineage"],
            {
                "source_project": "0032_audited_semantic_foundation_pretraining",
                "semantic_relationship": "type_aware_effect_summary_extension",
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

    def test_formal_dataset_scope_is_the_approved_recent_five_days(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        scope = manifest["dataset_scope"]
        self.assertEqual(scope["status"], "not_rematerialized_for_0033")
        self.assertEqual(
            scope["compatible_source_project"],
            "0032_audited_semantic_foundation_pretraining",
        )
        self.assertEqual(scope["perspective"], "unique_positive_terminal_winner")
        self.assertEqual(
            (scope["start_date"], scope["end_date"]),
            ("2026-07-28", "2026-08-01"),
        )
        self.assertEqual(scope["records"], 2_048_069)
        self.assertEqual(
            scope["split_counts"],
            {"train": 1_851_008, "validation": 197_061},
        )

    def test_formal_training_contract_is_fixed_independently_of_run_status(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertFalse(manifest["formal_training_started"])
        self.assertIsNone(manifest["training_notebook"])
        self.assertEqual(manifest["training_schema"], "not_started")
        self.assertFalse(manifest["verification"]["formal_training_claimed"])


if __name__ == "__main__":
    unittest.main()
