from __future__ import annotations

import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ID = "0028_universal_semantic_foundation_pretraining"
MANIFEST = REPOSITORY_ROOT / "experiments" / PROJECT_ID / "manifest.json"


class ProjectContractTests(unittest.TestCase):
    def test_manifest_declares_persona_free_actor_and_0025_provenance(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["project_id"], PROJECT_ID)
        self.assertFalse(manifest["actor_source_identity_visible"])
        self.assertEqual(
            manifest["implementation_lineage"],
            {
                "source_project": "0025_semantic_foundation_pretraining",
                "source_version": "V4_canonical_semantic_foundation",
                "semantic_relationship": "same_actor_semantics_readable_rewrite",
                "executable_dependency": False,
            },
        )

    def test_project_number_0027_is_not_created(self) -> None:
        forbidden = (
            REPOSITORY_ROOT / "train" / "0027_universal_semantic_foundation_pretraining",
            REPOSITORY_ROOT / "experiments" / "0027_universal_semantic_foundation_pretraining",
            REPOSITORY_ROOT / "rl_runs" / "0027_universal_semantic_foundation_pretraining",
        )
        self.assertFalse(any(path.exists() for path in forbidden))

    def test_initial_dataset_interval_matches_available_archives(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["dataset_scope"],
            {
                "perspective": "unique_positive_terminal_winner",
                "start_date": "2026-07-10",
                "end_date": "2026-08-01",
                "end_date_basis": "latest_locally_available_official_archive_at_project_creation",
            },
        )


if __name__ == "__main__":
    unittest.main()
