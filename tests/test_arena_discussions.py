from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess

from arena.discussions import collect_discussion_index


class DiscussionTests(unittest.TestCase):
    def test_topics_show_enriches_author_and_reply_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calls: list[list[str]] = []

            def runner(command: list[str], **_: object) -> CompletedProcess[str]:
                calls.append(command)
                if "show" in command:
                    return CompletedProcess(command, 0, json.dumps({"authorName": "detail-author", "replies": [{"id": 2}]}), "")
                return CompletedProcess(command, 0, json.dumps([{"id": 7, "title": "Topic", "authorName": "list-author", "votes": 3}]), "")

            records = collect_discussion_index(
                "pokemon-tcg-ai-battle",
                Path(temporary),
                runner=runner,
            )

            self.assertEqual(records[0].author, "detail-author")
            self.assertEqual(records[0].raw["detail"], {"authorName": "detail-author", "replies": [{"id": 2}]})
            self.assertTrue(any("show" in command for command in calls))


if __name__ == "__main__":
    unittest.main()
