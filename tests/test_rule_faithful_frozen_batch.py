from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation import rule_faithful_frozen_batch as batch


def _deck(deck_id: str, card_id: int) -> dict[str, object]:
    cards = [card_id] * 60
    return {
        "deck_id": deck_id,
        "display_name": deck_id.replace("_", " ").title(),
        "deck": cards,
        "deck_sha256": batch._deck_hash(cards),
    }


def _report(path: Path, deck: dict[str, object], checkpoint: str, run_id: str) -> None:
    games = [
        {
            "status": "finished",
            "candidate_first": index % 2 == 0,
            "winner": 0 if index < 300 else 1,
        }
        for index in range(batch.EXPECTED_GAMES)
    ]
    payload = {
        "manifest": {
            "run_id": run_id,
            "wall_time_seconds": 600.0,
            "games": batch.EXPECTED_GAMES,
            "opponents": [{}] * 51,
            "opponent_pool": {"pool_id": batch.POOL_ID},
            "candidate": {
                "deck": deck["deck"],
                "package_manifest": {
                    "project_id": batch.PROJECT_ID,
                    "version": batch.VERSION,
                    "arm": batch.ARM,
                    "checkpoint_selection": "latest",
                    "checkpoint_sha256": checkpoint,
                    "deck_id": deck["deck_id"],
                    "deck_sha256": deck["deck_sha256"],
                },
            },
        },
        "summary": {
            "total_games": batch.EXPECTED_GAMES,
            "completed_games": batch.EXPECTED_GAMES,
            "wins": 300,
            "losses": 210,
            "draws": 0,
            "errors": 0,
            "unfinished": 0,
            "completion_rate": 1.0,
            "win_rate": 300 / 510,
        },
        "games": games,
    }
    path.write_text(
        '<script id="report-data" type="application/json">'
        + json.dumps(payload)
        + "</script>",
        encoding="utf-8",
    )


class RuleFaithfulFrozenBatchTests(unittest.TestCase):
    def test_validate_and_publish_refuses_conflicting_report(self) -> None:
        deck = _deck("dragapult_ex_001", 1)
        checkpoint = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.html", root / "second.html"
            output = root / "published"
            _report(first, deck, checkpoint, "run-first")
            record = batch.publish_report(first, deck, checkpoint, output)
            self.assertEqual(record["games"], 510)
            self.assertEqual(record["turn_order"]["first"]["wins"], 150)
            batch.publish_report(first, deck, checkpoint, output)
            _report(second, deck, checkpoint, "run-second")
            with self.assertRaises(FileExistsError):
                batch.publish_report(second, deck, checkpoint, output)

    def test_refresh_index_preserves_requested_order(self) -> None:
        checkpoint = "b" * 64
        first = _deck("dragapult_ex_001", 1)
        second = _deck("marnies_grimmsnarl_ex_froslass_001", 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source_first, source_second = output / "first.html", output / "second.html"
            _report(source_first, first, checkpoint, "run-first")
            _report(source_second, second, checkpoint, "run-second")
            batch.publish_report(source_second, second, checkpoint, output)
            batch.publish_report(source_first, first, checkpoint, output)
            records = batch.refresh_index([first, second], checkpoint, output)
            self.assertEqual(
                [record["deck_id"] for record in records],
                ["dragapult_ex_001", "marnies_grimmsnarl_ex_froslass_001"],
            )
            document = (output / "index.html").read_text(encoding="utf-8")
            self.assertLess(document.index(str(first["deck_id"])), document.index(str(second["deck_id"])))


if __name__ == "__main__":
    unittest.main()
