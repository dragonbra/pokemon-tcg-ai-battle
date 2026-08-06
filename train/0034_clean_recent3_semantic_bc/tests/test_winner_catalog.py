from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


CONTRACT = importlib.import_module(
    "train.0034_clean_recent3_semantic_bc.data.replay_contract"
)
CATALOG = importlib.import_module(
    "train.0034_clean_recent3_semantic_bc.data.winner_catalog"
)


def _payload(
    episode_id: int,
    *,
    winner: int = 0,
    deck_size: int = 60,
    team_names: tuple[str, str] = ("Alpha", "Beta"),
) -> dict:
    decks = [[1] * deck_size, [2] * 60]
    return {
        "info": {"EpisodeId": episode_id, "TeamNames": list(team_names)},
        "rewards": [1 if index == winner else -1 for index in range(2)],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [
                {
                    "visualize": [
                        {
                            "action": decks,
                            "current": {"firstPlayer": 0},
                        }
                    ]
                },
                {},
            ]
        ],
    }


def _archive(root: Path, date: str, payloads: list[dict]) -> Path:
    path = root / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for payload in payloads:
            episode_id = payload["info"]["EpisodeId"]
            bundle.writestr(
                f"{episode_id}.json",
                json.dumps(payload, separators=(",", ":")),
            )
    return path


class WinnerCatalogTests(unittest.TestCase):
    def test_catalog_accepts_episode_json_files_without_an_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "2026-07-10"
            root.mkdir()
            episode = root / "101.json"
            episode.write_text(json.dumps(_payload(101, winner=1)), encoding="utf-8")

            catalog = CATALOG.scan_archives([], [episode], workers=1)

        self.assertEqual(catalog["totals"]["archive_count"], 0)
        self.assertEqual(catalog["totals"]["accepted_gap_patches"], 1)
        self.assertEqual(catalog["totals"]["unique_winning_episodes"], 1)
        self.assertEqual(catalog["episodes"][0]["episode_date"], "2026-07-10")

    def test_catalog_keeps_unique_positive_terminal_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = _archive(Path(directory), "2026-07-10", [_payload(101, winner=1)])
            catalog = CATALOG.scan_archives([archive], workers=1)
        self.assertEqual(catalog["totals"]["unique_winning_episodes"], 1)
        record = catalog["episodes"][0]
        self.assertEqual(record["player_index"], 1)
        self.assertEqual(record["terminal_outcome"], "win")
        self.assertEqual(sum(count for _, count in record["deck_counts"]), 60)
        self.assertEqual(record["team_name"], "Beta")
        self.assertEqual(catalog["contracts"]["source_identity"], "provenance_only")

    def test_registration_rejects_non_60_card_deck(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 60"):
            CONTRACT.registration_decks(_payload(102, deck_size=59))

    def test_duplicate_episode_payload_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = _archive(root, "2026-07-10", [_payload(103, team_names=("A", "B"))])
            right = _archive(root, "2026-07-11", [_payload(103, team_names=("Changed", "B"))])
            with self.assertRaisesRegex(ValueError, "payload drift"):
                CATALOG.scan_archives([left, right], workers=1)

    def test_split_digest_does_not_use_source_identity(self) -> None:
        left = CATALOG.assign_split(
            episode_id=104,
            deck_sha256="abc",
            seat="first",
        )
        right = CATALOG.assign_split(
            episode_id=104,
            deck_sha256="abc",
            seat="first",
        )
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
