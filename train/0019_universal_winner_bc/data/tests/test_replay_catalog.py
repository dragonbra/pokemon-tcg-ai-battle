from __future__ import annotations

import json
from importlib import import_module
import tempfile
import unittest
import zipfile
from pathlib import Path


CATALOG = import_module("train.0019_universal_winner_bc.data.replay_catalog")
scan_archives = CATALOG.scan_archives


def _episode(
    episode_id: int,
    rewards: list[int | None],
    *,
    team: str = "expert",
    statuses: list[str] | None = None,
) -> dict:
    decks = [list(range(1, 61)), list(range(61, 121))]
    return {
        "info": {"EpisodeId": episode_id, "TeamNames": ["other", team]},
        "rewards": rewards,
        "statuses": statuses or ["DONE", "DONE"],
        "steps": [
            [
                {
                    "visualize": [
                        {"action": decks, "current": {"firstPlayer": 0}}
                    ]
                },
                {},
            ],
            [{"action": decks[0]}, {"action": decks[1]}],
        ],
    }


class ReplayCatalogTests(unittest.TestCase):
    def _archive(self, root: Path, day: str, episodes: list[dict]) -> Path:
        path = root / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            for payload in episodes:
                episode_id = payload["info"]["EpisodeId"]
                bundle.writestr(
                    f"{episode_id}.json",
                    json.dumps(payload, separators=(",", ":")),
                )
        return path

    def test_winner_only_deduplication_and_source_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            winner = _episode(100, [-1, 1], team="Cafe\u0301")
            first = self._archive(root, "2026-07-10", [winner, _episode(101, [0, 0])])
            second = self._archive(root, "2026-07-11", [winner])
            catalog = scan_archives([first, second])
        self.assertEqual(catalog["totals"]["unique_winning_episodes"], 1)
        self.assertEqual(catalog["totals"]["duplicate_episode_members"], 1)
        self.assertEqual(catalog["totals"]["excluded_episode_members"], 1)
        self.assertEqual(catalog["sources"], [{"source_id": 1, "team_name": "Caf\u00e9"}])
        row = catalog["episodes"][0]
        self.assertEqual(row["player_index"], 1)
        self.assertEqual(sum(count for _, count in row["deck_counts"]), 60)
        self.assertIn(row["split"], {"train", "validation"})

    def test_parallel_scan_is_identical_to_serial_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives = []
            for day, episode_id in (("2026-07-10", 101), ("2026-07-11", 102)):
                archive = self._archive(root, day, [_episode(episode_id, [-1, 1])])
                archives.append(archive)
            serial = scan_archives(archives, workers=1)
            parallel = scan_archives(archives, workers=2)
        self.assertEqual(serial, parallel)

    def test_duplicate_payload_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._archive(root, "2026-07-10", [_episode(100, [-1, 1])])
            second = self._archive(root, "2026-07-11", [_episode(100, [1, -1])])
            with self.assertRaisesRegex(ValueError, "payload drift"):
                scan_archives([first, second])

    def test_accepts_done_winner_against_terminal_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(
                root,
                "2026-07-10",
                [_episode(102, [None, 1], statuses=["TIMEOUT", "DONE"])],
            )
            catalog = scan_archives([archive])
        self.assertEqual(catalog["totals"]["unique_winning_episodes"], 1)
        self.assertEqual(catalog["episodes"][0]["opponent_terminal_status"], "TIMEOUT")

    def test_includes_audited_file_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root, "2026-07-10", [_episode(100, [-1, 1])])
            patch_dir = root / "patches" / "2026-07-10"
            patch_dir.mkdir(parents=True)
            patch = patch_dir / "episode-102-replay.json"
            patch.write_text(json.dumps(_episode(102, [-1, 1])), encoding="utf-8")
            catalog = scan_archives([archive], [patch])
        self.assertEqual(catalog["totals"]["accepted_gap_patches"], 1)
        self.assertEqual(catalog["totals"]["unique_winning_episodes"], 2)
        patched = next(row for row in catalog["episodes"] if row["episode_id"] == 102)
        self.assertEqual(patched["locator"]["kind"], "file")


if __name__ == "__main__":
    unittest.main()
