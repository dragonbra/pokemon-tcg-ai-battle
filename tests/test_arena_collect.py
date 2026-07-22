from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from subprocess import CompletedProcess

from arena.collect import (
    KernelMetadata,
    collect_kernel_files,
    collect_kernel_index,
    collect_leaderboard_snapshot,
    collect_submissions_snapshot,
    safe_extract_archive,
)


class CollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_kernel_metadata_keeps_votes_and_builds_stable_source_id(self) -> None:
        metadata = KernelMetadata.from_mapping(
            {
                "ref": "u/k",
                "title": "Title",
                "author": "U",
                "totalVotes": 8,
            },
            competition="pokemon-tcg-ai-battle",
        )
        self.assertEqual(metadata.source_id(), "pokemon-tcg-ai-battle__u__k")
        self.assertEqual(metadata.total_votes, 8)

    def test_kernel_index_paginates_until_empty_page(self) -> None:
        pages = [
            [{"ref": "u/one", "title": "One", "author": "U", "totalVotes": 3}],
            [{"ref": "u/two", "title": "Two", "author": "U", "totalVotes": 2}],
            [],
        ]

        def runner(command: list[str], **_: object) -> CompletedProcess[str]:
            page = int(command[command.index("--page") + 1])
            return CompletedProcess(command, 0, json.dumps(pages[page - 1]), "")

        records = collect_kernel_index(
            "pokemon-tcg-ai-battle",
            self.root,
            runner=runner,
        )

        self.assertEqual([record.ref for record in records], ["u/one", "u/two"])
        self.assertTrue((self.root / "pokemon-tcg-ai-battle__u__one" / "metadata.json").is_file())

    def test_archive_extraction_rejects_parent_traversal(self) -> None:
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../../escape.txt", "bad")
        with self.assertRaises(ValueError):
            safe_extract_archive(archive, self.root / "out")

    def test_kernel_index_accepts_kaggle_not_found_terminal_page(self) -> None:
        def runner(command: list[str], **_: object) -> CompletedProcess[str]:
            if command[command.index("--page") + 1] == "1":
                return CompletedProcess(command, 0, "[]", "")
            return CompletedProcess(command, 0, "Not found\n", "")

        records = collect_kernel_index(
            "pokemon-tcg-ai-battle",
            self.root,
            runner=runner,
        )

        self.assertEqual(records, ())

    def test_kernel_file_timeout_is_recorded_without_aborting_collection(self) -> None:
        metadata = KernelMetadata.from_mapping(
            {"ref": "u/k", "title": "Title", "author": "U"},
            competition="pokemon-tcg-ai-battle",
        )

        def runner(command: list[str], **_: object) -> CompletedProcess[str]:
            raise TimeoutError("subprocess timed out")

        result = collect_kernel_files(metadata, self.root, runner=runner)

        self.assertEqual(result.status, "collection_error")
        self.assertIn("timed out", result.error or "")

    def test_leaderboard_snapshot_parses_page_token_prefix_and_preserves_score(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], **_: object) -> CompletedProcess[str]:
            calls.append(command)
            if len(calls) == 1:
                return CompletedProcess(
                    command,
                    0,
                    'Next Page Token = next-token\n[{"teamName":"U","score":"1012.3"}]',
                    "",
                )
            return CompletedProcess(command, 0, "[]", "")

        result = collect_leaderboard_snapshot(
            "pokemon-tcg-ai-battle",
            self.root,
            runner=runner,
        )

        self.assertEqual(result["entries"][0]["score"], "1012.3")
        self.assertEqual(len(calls), 2)
        self.assertIn("--page-token", calls[1])
        self.assertTrue((self.root / "leaderboard-latest.json").is_file())

    def test_submissions_snapshot_preserves_public_score_separately(self) -> None:
        def runner(command: list[str], **_: object) -> CompletedProcess[str]:
            return CompletedProcess(
                command,
                0,
                '[{"ref":"u/submission.tar.gz","publicScore":"770.3","status":"COMPLETE"}]',
                "",
            )

        result = collect_submissions_snapshot("pokemon-tcg-ai-battle", self.root, runner=runner)

        self.assertEqual(result["entries"][0]["publicScore"], "770.3")
        self.assertTrue((self.root / "submissions-latest.json").is_file())


if __name__ == "__main__":
    unittest.main()
