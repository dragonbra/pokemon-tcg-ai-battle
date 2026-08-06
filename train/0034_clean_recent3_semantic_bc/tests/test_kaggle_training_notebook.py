from __future__ import annotations

import hashlib
import importlib
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG = importlib.import_module(
    "train.0034_clean_recent3_semantic_bc.model.config"
).ModelConfig
NOTEBOOK = (
    ROOT
    / "experiments"
    / "0034_clean_recent3_semantic_bc"
    / "kaggle"
    / "02_train_recent3_cleaned_bc"
    / "02_train_recent3_cleaned_bc.ipynb"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class KaggleTrainingNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.notebook = notebook
        cls.code_source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        cls.package_source = "".join(notebook["cells"][6]["source"])
        cls.function_source = "".join(notebook["cells"][8]["source"])
        cls.run_source = "".join(notebook["cells"][10]["source"])
        cls.metadata = json.loads(
            (NOTEBOOK.parent / "kernel-metadata.json").read_text(encoding="utf-8")
        )

    def _function_namespace(self) -> dict:
        policy = nn.Linear(2, 1)
        namespace = {
            "F": F,
            "Path": Path,
            "dataset": SimpleNamespace(manifest_sha256="combined-manifest"),
            "fields": SimpleNamespace(
                SCHEMA_VERSION="0034_clean_recent3_semantic_decision_v1"
            ),
            "math": math,
            "model_config": SimpleNamespace(to_dict=lambda: {"d_model": 320}),
            "nn": nn,
            "parallel": SimpleNamespace(
                module=SimpleNamespace(policy=policy)
            ),
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "sha256_file": _sha256,
            "time": time,
            "torch": torch,
        }
        exec(self.function_source, namespace)
        return namespace

    def test_stop_has_its_own_action_type(self) -> None:
        namespace = self._function_namespace()
        batch = {
            "targets": torch.tensor([[2]], dtype=torch.long),
            "option_cat": torch.zeros((1, 2, 11), dtype=torch.long),
        }
        batch["option_cat"][0, 0, 0] = 9
        batch["option_cat"][0, 1, 0] = 14

        correct = namespace["_new_action_metrics"]()
        namespace["_update_action_metrics"](
            correct,
            batch,
            torch.tensor([[[0.0, 0.0, 1.0]]]),
        )
        self.assertEqual(correct["action_type_tokens"], 1)
        self.assertEqual(correct["action_type_correct"], 1)

        wrong = namespace["_new_action_metrics"]()
        namespace["_update_action_metrics"](
            wrong,
            batch,
            torch.tensor([[[0.0, 1.0, 0.0]]]),
        )
        self.assertEqual(wrong["action_type_tokens"], 1)
        self.assertEqual(wrong["action_type_correct"], 0)

    def test_notebook_checkpoint_payload_is_model_only(self) -> None:
        namespace = self._function_namespace()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_model.pt"
            namespace["save_model"](path, 1, {"weighted_loss": 1.0})
            payload = torch.load(path, weights_only=True)
        self.assertEqual(
            set(payload),
            {"schema_version", "state_dict", "metadata"},
        )
        self.assertEqual(payload["metadata"]["epoch"], 1)
        self.assertFalse(
            any(str(name).startswith("module.") for name in payload["state_dict"])
        )

    def test_package_selects_aggregate_cleaning_report(self) -> None:
        self.assertIn('"training_report": "training_report.json"', self.package_source)
        self.assertIn("aggregate_cleaning_reports", self.package_source)
        self.assertNotIn(
            'for cleaning_report in INPUT.rglob("cleaning_report.json"):\n    shutil.copy2',
            self.package_source,
        )

    def test_model_capacity_and_training_mode_are_fixed(self) -> None:
        config = MODEL_CONFIG()
        self.assertEqual(config.d_model, 320)
        self.assertEqual(config.state_layers, 4)
        self.assertEqual(config.option_layers, 3)
        self.assertEqual(config.ffn_multiplier, 3)
        self.assertIn("EPOCHS = 3", self.code_source)
        self.assertIn('"epoch_definition": "one_epoch_is_full_merged_recent3_train"', self.run_source)
        self.assertIn('"random_initialization": True', self.run_source)
        self.assertIn('"initialized_from_checkpoint": None', self.run_source)
        self.assertIn('"wandb": None', self.run_source)
        self.assertIn('"gpu_device_names": device_names', self.run_source)
        self.assertIn('"gpu_device_names": device_names', self.package_source)

    def test_every_epoch_uses_the_full_combined_three_day_dataset(self) -> None:
        self.assertIn(
            'EXPECTED_DATES = {\n        \'2026-08-01\',\n        \'2026-08-02\',\n        \'2026-08-03\'',
            self.code_source,
        )
        self.assertIn("CombinedCanonicalDecisionDataset(roots)", self.code_source)
        self.assertIn('dataset.manifest.get("partition_count") != 3', self.code_source)
        self.assertIn(
            'train_partition_pass(epoch, "combined_recent3", dataset)',
            self.run_source,
        )
        self.assertIn(
            'validate_dataset(epoch, "merged_validation_recent3", dataset)',
            self.run_source,
        )
        self.assertIn(
            'decisions != partition_dataset.split_counts["train"]',
            self.function_source,
        )
        self.assertIn(
            'decisions != eval_dataset.split_counts["validation"]',
            self.function_source,
        )
        self.assertIn('report["full_train_passes_completed"] = len(history)', self.run_source)
        self.assertIn('report["full_validation_passes_completed"] = len(history)', self.run_source)

    def test_weighted_loss_and_requested_action_metrics_are_reported(self) -> None:
        self.assertIn('cleaning.get("sample_weight", 1.0)', self.function_source)
        self.assertIn('reduction="none"', self.function_source)
        self.assertIn("token_losses * token_weights", self.function_source)
        for metric in (
            "action_type_accuracy",
            "attach_target_token_accuracy",
            "attach_target_exact_action",
            "attack_action_token_accuracy",
            "attack_action_exact",
            "weighted_loss",
        ):
            self.assertIn(metric, self.function_source)

    def test_kernel_uses_only_cleaning_output_and_no_pretrained_model(self) -> None:
        self.assertEqual(
            self.metadata["kernel_sources"],
            ["horizen12/ptcg-0034-01-clean-recent3-data"],
        )
        self.assertEqual(self.metadata["model_sources"], [])
        self.assertTrue(self.metadata["enable_gpu"])
        self.assertEqual(self.metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertNotIn("import wandb", self.code_source)
        self.assertNotIn("wandb.init", self.code_source)

    def test_package_contract_includes_prototypes_reports_and_loader(self) -> None:
        for artifact in (
            '"weights": ["best_model.pt", "last_model.pt"]',
            '"contract": "model_contract.json"',
            '"prototype_assets": ["official_public_prototypes_v1.json", "official_full_engine_prototypes_v2.json"]',
            '"feature_audit": "feature_audit.json"',
            '"cleaning_report": "cleaning_report.json"',
            '"training_report": "training_report.json"',
            '"usage_loader": "load_model.py"',
            '"usage_readme": "MODEL_USAGE.md"',
            '"smoke_input": "smoke_input.pt"',
        ):
            self.assertIn(artifact, self.package_source)
        self.assertIn(
            'torch.save(smoke_input, OUTPUT / "smoke_input.pt")',
            self.package_source,
        )


if __name__ == "__main__":
    unittest.main()
