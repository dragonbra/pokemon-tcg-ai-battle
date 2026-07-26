from __future__ import annotations

import importlib
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ID = "0013_semantic_goal_policy"
PACKAGE_NAME = f"train.{PROJECT_ID}"
COMMANDS = ("validate-protocol", "audit-visibility", "build-dataset", "smoke-model", "train", "schedule", "select-project", "export", "evaluate")
SOURCE_MANIFEST = "data/raw/episodes/source_manifest.json"
PATCH_PATH = "data/raw/episodes/patches/2026-07-24/episode-87841523-replay.json"


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def _local_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        parser = _LinkParser()
        parser.feed(text)
        links = parser.links
    else:
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
    return [link for link in links if "://" not in link and not link.startswith("#")]


class SemanticGoalPolicyScaffoldTests(unittest.TestCase):
    def test_package_exports_exact_project_identity_and_structured_source_contract(self) -> None:
        package = importlib.import_module(PACKAGE_NAME)
        config = importlib.import_module(f"{PACKAGE_NAME}.config").PROJECT_CONFIG
        self.assertEqual(package.PROJECT_ID, PROJECT_ID)
        self.assertEqual(config.project_id, PROJECT_ID)
        self.assertEqual(config.deck, "alakazam_dudunsparce")
        self.assertEqual(config.expert_name, "Yushin Ito")
        self.assertEqual(config.expert_name_normalization, "casefold_whitespace_collapse_exact")
        self.assertEqual(config.source_dates, ("2026-07-18", "2026-07-25"))
        self.assertTrue(config.winner_only)
        self.assertTrue(config.submission_id_unavailable)
        self.assertEqual(config.source_manifest, SOURCE_MANIFEST)
        self.assertEqual(config.patch_path, PATCH_PATH)
        self.assertEqual(config.patch_precedence, "patch_replaces_same_episode_from_archive")
        self.assertEqual(config.deck_conditioning, "actual_registered_deck_per_episode_player")
        self.assertEqual(config.action_contract, "ordered_full_action")
        self.assertEqual(config.visibility_contract, "causal_actor_visible_only")

    def test_manifest_records_exact_structured_source_contract(self) -> None:
        archive = REPOSITORY_ROOT / "experiments" / PROJECT_ID
        manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["project_id"], PROJECT_ID)
        self.assertEqual(manifest["source_contract"], {
            "action_contract": "ordered_full_action",
            "date_range": {"first": "2026-07-18", "last": "2026-07-25"},
            "deck_conditioning": "actual_registered_deck_per_episode_player",
            "expert": {"name": "Yushin Ito", "normalization": "casefold_whitespace_collapse_exact"},
            "patch": {"path": PATCH_PATH, "precedence": "patch_replaces_same_episode_from_archive"},
            "source_manifest": SOURCE_MANIFEST,
            "submission_id_unavailable": True,
            "visibility_contract": "causal_actor_visible_only",
            "winner_only": True,
        })
        self.assertNotIn("audit", manifest)

    def test_module_help_lists_all_thin_cli_commands(self) -> None:
        result = subprocess.run([sys.executable, "-m", PACKAGE_NAME, "--help"], cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in COMMANDS:
            self.assertIn(command, result.stdout)

    def test_three_roots_and_all_readme_design_links_resolve(self) -> None:
        train = REPOSITORY_ROOT / "train" / PROJECT_ID
        archive = REPOSITORY_ROOT / "experiments" / PROJECT_ID
        runs = REPOSITORY_ROOT / "rl_runs" / PROJECT_ID
        manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(train.is_dir())
        self.assertTrue(archive.is_dir())
        self.assertTrue(runs.is_dir())
        self.assertEqual(manifest["paths"], {"archive": f"experiments/{PROJECT_ID}", "runs": f"rl_runs/{PROJECT_ID}", "training": f"train/{PROJECT_ID}"})
        for document in (archive / "README.md", archive / "DESIGN.md", archive / "DESIGN.html"):
            with self.subTest(document=document.name):
                for link in _local_links(document):
                    self.assertTrue((document.parent / link).resolve().exists(), link)

    def test_tracked_archive_landings_and_local_generated_runtime_state(self) -> None:
        archive = REPOSITORY_ROOT / "experiments" / PROJECT_ID
        for landing in ("data_audit/README.md", "decisions/README.md", "versions/README.md"):
            self.assertTrue((archive / landing).is_file(), landing)
        runtime_root = REPOSITORY_ROOT / "rl_runs" / PROJECT_ID
        for generated in (runtime_root / "dataset", runtime_root / "versions"):
            self.assertTrue(generated.is_dir())
            self.assertTrue(all(path.name.startswith(("V", ".V")) for path in generated.iterdir()))
        readme = (archive / "README.md").read_text(encoding="utf-8")
        self.assertIn("local generated runtime directories", readme)
        self.assertIn("not tracked", readme)

    def test_scaffold_contains_no_obsolete_identity_or_tracked_training_assets(self) -> None:
        source_roots = [
            REPOSITORY_ROOT / "train" / PROJECT_ID,
            REPOSITORY_ROOT / "experiments" / PROJECT_ID,
        ]
        text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for root in source_roots
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
        self.assertNotIn("0013_alakazam_rollout_" + "value_calibration", text)
        for suffix in (".pt", ".pth", ".ckpt", ".npz", ".parquet"):
            self.assertFalse(
                any(path.suffix == suffix for root in source_roots for path in root.rglob("*"))
            )


if __name__ == "__main__":
    unittest.main()
