from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.submission_paths import (
    historical_submission_dirs,
    is_complete_submission,
    resolve_submission,
    work_submission_dirs,
)


class SubmissionPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_submission(self, path: Path, *, complete: bool = True) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "main.py").write_text("", encoding="utf-8")
        (path / "deck.csv").write_text("", encoding="utf-8")
        if complete:
            (path / "cg").mkdir()

    def test_work_candidates_only_include_complete_direct_children(self) -> None:
        self.add_submission(self.root / "work" / "candidate")
        self.add_submission(self.root / "work" / "incomplete", complete=False)
        self.add_submission(self.root / "work" / "docs")
        self.add_submission(self.root / "work" / "auto-iteration")

        self.assertEqual(
            [self.root / "work" / "candidate"],
            work_submission_dirs(self.root),
        )

    def test_historical_candidates_ignore_dist(self) -> None:
        self.add_submission(self.root / "submission" / "legacy")
        self.add_submission(self.root / "submission" / "dist")

        self.assertEqual(
            [self.root / "submission" / "legacy"],
            historical_submission_dirs(self.root),
        )

    def test_resolve_prefers_work_and_can_prefer_historical(self) -> None:
        self.add_submission(self.root / "work" / "candidate")
        self.add_submission(self.root / "submission" / "candidate")

        self.assertEqual(
            self.root / "work" / "candidate",
            resolve_submission("candidate", self.root),
        )
        self.assertEqual(
            self.root / "submission" / "candidate",
            resolve_submission("candidate", self.root, prefer_work=False),
        )

    def test_resolve_rejects_unknown_or_incomplete_submission(self) -> None:
        self.add_submission(self.root / "work" / "incomplete", complete=False)

        with self.assertRaises(ValueError):
            resolve_submission("missing", self.root)
        with self.assertRaises(ValueError):
            resolve_submission("incomplete", self.root)

    def test_is_complete_submission_requires_runtime_directory(self) -> None:
        path = self.root / "candidate"
        self.add_submission(path)
        self.assertTrue(is_complete_submission(path))
        (path / "cg").rmdir()
        self.assertFalse(is_complete_submission(path))


if __name__ == "__main__":
    unittest.main()
