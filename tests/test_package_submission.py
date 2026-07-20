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


if __name__ == "__main__":
    unittest.main()
