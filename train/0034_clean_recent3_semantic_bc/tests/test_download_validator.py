from __future__ import annotations

import importlib
import inspect
import tempfile
import unittest
from pathlib import Path

import torch


VALIDATOR = importlib.import_module(
    "train.0034_clean_recent3_semantic_bc.tools.validate_downloaded_partition"
)
PACKAGE_VALIDATOR = importlib.import_module(
    "train.0034_clean_recent3_semantic_bc.tools.validate_model_package"
)


def _cleaning_report() -> dict:
    return {
        "schema_version": "0034_cleaning_report_v1",
        "status": "passed",
        "date": "2026-08-01",
        "input_records": 11,
        "kept_records": 9,
        "rejected_records": 2,
        "rejected_actor_schema": 1,
        "rejected_illegal_target": 1,
        "event_history_compacted_records": 9,
        "sample_weight_distribution": {
            "0.95": 2,
            "1.0": 5,
            "1.15": 2,
        },
    }


class DownloadValidatorTests(unittest.TestCase):
    def test_cleaning_report_reconciles_rejections_and_weights(self) -> None:
        distribution = VALIDATOR._validate_cleaning_report(
            _cleaning_report(),
            expected_day="2026-08-01",
            records=9,
        )
        self.assertEqual(dict(distribution), {0.95: 2, 1.0: 5, 1.15: 2})

    def test_cleaning_report_rejects_black_box_deletions(self) -> None:
        report = _cleaning_report()
        report["rejected_illegal_target"] = 0
        with self.assertRaisesRegex(ValueError, "rejected-sample reasons"):
            VALIDATOR._validate_cleaning_report(
                report,
                expected_day="2026-08-01",
                records=9,
            )

    def test_cleaning_report_requires_all_weight_classes(self) -> None:
        report = _cleaning_report()
        report["sample_weight_distribution"].pop("0.95")
        report["sample_weight_distribution"]["1.0"] += 2
        with self.assertRaisesRegex(ValueError, "sample-weight classes"):
            VALIDATOR._validate_cleaning_report(
                report,
                expected_day="2026-08-01",
                records=9,
            )

    def test_output_root_validator_commits_all_three_dates_and_hashes(self) -> None:
        source = inspect.getsource(VALIDATOR.validate_output_root)
        self.assertIn("EXPECTED_DAYS", source)
        self.assertIn('entry.get("manifest_sha256")', source)
        self.assertIn("validate_partition(", source)
        self.assertIn('partition.get("total_records"', source)
        self.assertIn('output_root / "cleaning_report.json"', source)

    def test_model_package_validator_requires_smoke_forward_and_full_epochs(self) -> None:
        source = inspect.getsource(PACKAGE_VALIDATOR.validate_model_package)
        self.assertIn("smoke_input.pt", PACKAGE_VALIDATOR.REQUIRED_FILES)
        self.assertIn("policy.teacher_logits(smoke)", source)
        self.assertIn('report.get("full_train_passes_completed") == 3', source)
        self.assertIn('report.get("full_validation_passes_completed") == 3', source)
        self.assertIn('all("T4" in str(name)', source)

    def test_model_package_checkpoint_contract_is_model_only(self) -> None:
        metadata = {
            "architecture": "SemanticPolicy",
            "actor_schema": "0034_clean_recent3_semantic_decision_v1",
            "parameter_count": PACKAGE_VALIDATOR.EXPECTED_PARAMETER_COUNT,
            "model_config": PACKAGE_VALIDATOR.EXPECTED_MODEL_CONFIG,
            "random_initialization": True,
            "initialized_from_checkpoint": None,
            "winner_only": True,
        }
        payload = {
            "schema_version": "0034_model_only_checkpoint_v1",
            "state_dict": {"weight": torch.ones(1)},
            "metadata": metadata,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_model.pt"
            torch.save(payload, path)
            loaded = PACKAGE_VALIDATOR._load_checkpoint(path)
        self.assertEqual(set(loaded), {"schema_version", "state_dict", "metadata"})


if __name__ == "__main__":
    unittest.main()
