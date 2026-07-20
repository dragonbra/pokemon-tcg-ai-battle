import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.v7_best_submission import stage_best_for_schedule, validate_best_strategy


class V7BestSubmissionTests(unittest.TestCase):
    def _write_archive(self, root: Path, *, main: str = "print('best')\n") -> Path:
        archive = root / "submission" / "dist" / "alakazam_v7_auto_iter_best_iter-15-test.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp_name:
            source = Path(temp_name)
            (source / "main.py").write_text(main, encoding="utf-8")
            (source / "deck.csv").write_text("a,b\n", encoding="utf-8")
            (source / "cg").mkdir()
            (source / "cg" / "game.py").write_text("# runtime\n", encoding="utf-8")
            with tarfile.open(archive, "w:gz") as handle:
                handle.add(source / "main.py", arcname="main.py")
                handle.add(source / "deck.csv", arcname="deck.csv")
                handle.add(source / "cg", arcname="cg")
        return archive

    def _write_marker(self, root: Path, archive: Path, *, digest: str | None = None) -> Path:
        marker = root / "submission" / "alakazam_v7_auto_iter" / "BEST_STRATEGY.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "source_submission": "alakazam_v7_auto_iter",
                    "best_iteration": 15,
                    "label": "iter-15-test",
                    "artifact": str(archive.relative_to(root)),
                    "artifact_sha256": digest
                    or hashlib.sha256(archive.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        return marker

    def test_best_marker_validates_immutable_archive(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = self._write_archive(root)
            marker = self._write_marker(root, archive)

            result = validate_best_strategy(marker, root)

            self.assertEqual(result.best_iteration, 15)
            self.assertEqual(result.label, "iter-15-test")
            self.assertEqual(result.artifact, archive.resolve())

    def test_best_marker_rejects_changed_archive(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = self._write_archive(root)
            marker = self._write_marker(root, archive, digest="0" * 64)

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_best_strategy(marker, root)

    def test_best_marker_rejects_archive_without_submission_runtime(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = root / "submission" / "dist" / "bad.tar.gz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "w:gz") as handle:
                payload = root / "payload.txt"
                payload.write_text("bad\n", encoding="utf-8")
                handle.add(payload, arcname="payload.txt")
            marker = self._write_marker(root, archive)

            with self.assertRaisesRegex(ValueError, "required archive members"):
                validate_best_strategy(marker, root)

    def test_best_marker_rejects_root_dist_archive(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = self._write_archive(root)
            root_archive = root / "dist" / archive.name
            root_archive.parent.mkdir(parents=True, exist_ok=True)
            archive.rename(root_archive)
            archive = root_archive
            marker = self._write_marker(root, archive)

            with self.assertRaisesRegex(ValueError, "inside submission/dist/"):
                validate_best_strategy(marker, root)

    def test_stage_best_for_schedule_copies_archive_and_rewrites_portable_marker(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = self._write_archive(root)
            marker = self._write_marker(root, archive)
            schedule_dir = root / "schedule-state"

            staged = stage_best_for_schedule(marker, root, schedule_dir)

            self.assertEqual(staged, (schedule_dir / archive.name).resolve())
            self.assertEqual(staged.read_bytes(), archive.read_bytes())
            staged_marker = json.loads(
                (schedule_dir / "BEST_STRATEGY.json").read_text(encoding="utf-8")
            )
            self.assertEqual(staged_marker["artifact"], archive.name)
            self.assertEqual(
                staged_marker["artifact_sha256"],
                hashlib.sha256(archive.read_bytes()).hexdigest(),
            )
            source_marker = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(source_marker["artifact"], str(archive.relative_to(root)))


if __name__ == "__main__":
    unittest.main()
