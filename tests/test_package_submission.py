from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = REPO_ROOT / "scripts" / "package_submission.sh"


class PackageSubmissionTests(unittest.TestCase):
    def write_candidate(self, path: Path, marker: str) -> None:
        (path / "cg").mkdir(parents=True)
        (path / "main.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
        (path / "deck.csv").write_text("card,1\n", encoding="utf-8")

    def write_strategy(self, path: Path) -> None:
        (path / "strategy").mkdir(parents=True)
        (path / "strategy" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        (path / "strategy" / "policy.py").write_text("VALUE = 2\n", encoding="utf-8")

    def run_packager(self, root: Path, name: str) -> subprocess.CompletedProcess[str]:
        script = root / "scripts" / "package_submission.sh"
        return subprocess.run(
            ["bash", str(script), name],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_work_candidate_is_preferred_and_written_under_submission_dist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "scripts").mkdir()
            shutil.copy2(PACKAGE_SCRIPT, root / "scripts" / PACKAGE_SCRIPT.name)
            self.write_candidate(root / "work" / "candidate", "work")
            self.write_candidate(root / "submission" / "candidate", "historical")

            result = self.run_packager(root, "candidate")

            self.assertEqual(result.returncode, 0, result.stderr)
            archive = root / "submission" / "dist" / "candidate.tar.gz"
            self.assertEqual(Path(result.stdout.strip()), archive)
            with tarfile.open(archive, "r:gz") as handle:
                self.assertEqual(
                    sorted(handle.getnames()),
                    ["cg", "deck.csv", "main.py"],
                )
                self.assertIn(b"work", handle.extractfile("main.py").read())

    def test_historical_candidate_is_used_when_work_candidate_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "scripts").mkdir()
            shutil.copy2(PACKAGE_SCRIPT, root / "scripts" / PACKAGE_SCRIPT.name)
            self.write_candidate(root / "submission" / "legacy", "historical")

            result = self.run_packager(root, "legacy")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "submission" / "dist" / "legacy.tar.gz").is_file())

    def test_historical_candidate_is_used_when_work_candidate_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "scripts").mkdir()
            shutil.copy2(PACKAGE_SCRIPT, root / "scripts" / PACKAGE_SCRIPT.name)
            (root / "work" / "candidate").mkdir(parents=True)
            self.write_candidate(root / "submission" / "candidate", "historical")

            result = self.run_packager(root, "candidate")

            self.assertEqual(result.returncode, 0, result.stderr)
            archive = root / "submission" / "dist" / "candidate.tar.gz"
            with tarfile.open(archive, "r:gz") as handle:
                self.assertIn(b"historical", handle.extractfile("main.py").read())

    def test_multimodule_candidate_includes_strategy_at_archive_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "scripts").mkdir()
            shutil.copy2(PACKAGE_SCRIPT, root / "scripts" / PACKAGE_SCRIPT.name)
            candidate = root / "work" / "candidate"
            self.write_candidate(candidate, "multi")
            self.write_strategy(candidate)
            (candidate / "strategy" / "__pycache__").mkdir()
            (candidate / "strategy" / "__pycache__" / "bad.pyc").write_bytes(b"bad")

            result = self.run_packager(root, "candidate")

            self.assertEqual(result.returncode, 0, result.stderr)
            archive = root / "submission" / "dist" / "candidate.tar.gz"
            with tarfile.open(archive, "r:gz") as handle:
                self.assertEqual(
                    sorted(handle.getnames()),
                    [
                        "cg",
                        "deck.csv",
                        "main.py",
                        "strategy",
                        "strategy/__init__.py",
                        "strategy/policy.py",
                    ],
                )

    def test_real_v8_archive_imports_from_kaggle_agent_directory(self) -> None:
        result = subprocess.run(
            ["bash", str(PACKAGE_SCRIPT), "alakazam_v8_current"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = REPO_ROOT / "submission" / "dist" / "alakazam_v8_current.tar.gz"
        with tempfile.TemporaryDirectory() as temp_name:
            agent_dir = Path(temp_name) / "kaggle_simulations" / "agent"
            agent_dir.mkdir(parents=True)
            with tarfile.open(archive, "r:gz") as handle:
                handle.extractall(agent_dir)

            imported = subprocess.run(
                [
                    "python3",
                    "-c",
                    (
                        "import main; "
                        "assert len(main.DECK) == 60; "
                        "assert main.read_deck_csv() == main.DECK; "
                        "assert len(main.agent({'select': None})) == 60"
                    ),
                ],
                cwd=agent_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)

            main_path = agent_dir / "main.py"
            raw_exec = subprocess.run(
                [
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"path = Path({str(main_path)!r}); "
                        "scope = {'__file__': str(path)}; "
                        "exec(compile(path.read_bytes(), str(path), 'exec'), scope); "
                        "assert len(scope['DECK']) == 60"
                    ),
                ],
                cwd=temp_name,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(raw_exec.returncode, 0, raw_exec.stderr)


if __name__ == "__main__":
    unittest.main()
