from __future__ import annotations

import ast
import hashlib
from importlib import import_module
import json
from pathlib import Path
import unittest

import torch


PROJECT = import_module("train.0020_pluggable_deck_rl")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_ROOT = (
    REPOSITORY_ROOT
    / "rl_runs/0020_pluggable_deck_rl/versions/V1_frozen_0019_epoch13"
)
EXPECTED_SHA256 = "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"


class FoundationContractTests(unittest.TestCase):
    def test_project_identity_and_frozen_checkpoint(self) -> None:
        self.assertEqual(PROJECT.PROJECT_ID, "0020_pluggable_deck_rl")
        manifest = json.loads(
            (VERSION_ROOT / "artifact/foundation_manifest.json").read_text(encoding="utf-8")
        )
        checkpoint = VERSION_ROOT / "checkpoint/epoch-0013-da9b13d6f82d19d4.pt"
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), EXPECTED_SHA256)
        self.assertEqual(manifest["checkpoint"]["sha256"], EXPECTED_SHA256)
        self.assertEqual(manifest["checkpoint"]["epoch"], 13)
        self.assertEqual(manifest["deployment_source_id"], 0)
        self.assertFalse(manifest["checkpoint"]["optimizer_state_saved"])
        self.assertFalse(manifest["checkpoint"]["resumable_training_state_saved"])

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(set(payload), {"model", "epoch", "global_step", "metadata"})
        self.assertEqual(payload["epoch"], 13)

    def test_no_other_numbered_project_runtime_imports(self) -> None:
        forbidden: list[tuple[str, str]] = []
        for path in PROJECT_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    if name.startswith("train.00") and not name.startswith(
                        "train.0020_pluggable_deck_rl"
                    ):
                        forbidden.append((str(path), name))
        self.assertEqual(forbidden, [])

    def test_design_documents_share_authoritative_contract(self) -> None:
        experiment = REPOSITORY_ROOT / "experiments/0020_pluggable_deck_rl"
        markdown = (experiment / "DESIGN.md").read_text(encoding="utf-8")
        html = (experiment / "DESIGN.html").read_text(encoding="utf-8")
        for fact in (
            "epoch-0013-da9b13d6f82d19d4.pt",
            EXPECTED_SHA256,
            "source_id=0",
            "registered_card_ids",
            "V1_frozen_0019_epoch13",
        ):
            self.assertIn(fact, markdown)
            self.assertIn(fact, html)


if __name__ == "__main__":
    unittest.main()
