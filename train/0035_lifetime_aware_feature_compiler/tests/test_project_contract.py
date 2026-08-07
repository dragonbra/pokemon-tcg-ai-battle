from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ID = "0035_lifetime_aware_feature_compiler"
MANIFEST = REPOSITORY_ROOT / "experiments" / PROJECT_ID / "manifest.json"


class ProjectContractTests(unittest.TestCase):
    def test_manifest_declares_persona_free_actor_and_self_contained_lineage(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["project_id"], PROJECT_ID)
        self.assertFalse(manifest["actor_source_identity_visible"])
        self.assertEqual(
            manifest["implementation_lineage"],
            {
                "source_project": "0031_rule_faithful_semantic_foundation_pretraining",
                "source_commit": "473f6e5bd83468c70fd977a841803aea19d854c5",
                "semantic_relationship": "self_contained_semantics_preserving_runtime_fork",
                "executable_dependency": False,
            },
        )

    def test_runtime_has_no_cross_numbered_project_import(self) -> None:
        project_root = REPOSITORY_ROOT / "train" / PROJECT_ID
        for path in project_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            self.assertFalse(
                any(name.startswith("train.00") for name in imported),
                (path.relative_to(REPOSITORY_ROOT).as_posix(), imported),
            )

    def test_runtime_project_does_not_claim_a_new_training_run(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertFalse(manifest["training_performed"])
        self.assertNotIn("runs", manifest["paths"])

    def test_actor_schema_remains_the_0031_contract(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["actor_schema"], "0031_rule_faithful_semantic_decision_v2"
        )
        self.assertEqual(manifest["expected_actor_tensor_keys"], 39)


if __name__ == "__main__":
    unittest.main()
