from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.combat_matrix import (
    build_combat_matrix_data,
    render_combat_matrix,
    write_combat_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "evaluation" / "configs" / "opponents.json"


class CombatMatrixTests(unittest.TestCase):
    def test_builds_package_and_archetype_heatmaps_from_full_reports(self) -> None:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["opponents"]
        names = [entry["name"] for entry in catalog]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports_root = root / "reports"
            for candidate_index, candidate in enumerate(names):
                by_opponent = {}
                turns_by_opponent = {}
                games = []
                total_wins = 0
                for opponent_index, opponent in enumerate(names):
                    wins = (candidate_index + opponent_index) % 11
                    turns = 10 + candidate_index + opponent_index
                    total_wins += wins
                    by_opponent[opponent] = {
                        "games": 10,
                        "wins": wins,
                        "losses": 10 - wins,
                        "draws": 0,
                        "errors": 0,
                        "unfinished": 0,
                        "win_rate": wins / 10,
                    }
                    turns_by_opponent[opponent] = {
                        "numerator": turns * 10,
                        "denominator": 10,
                        "value": turns,
                    }
                    for game_index in range(10):
                        games.append(
                            {
                                "candidate_first": game_index % 2 == 0,
                                "opponent": opponent,
                                "metric_refs": {
                                    "outcome": {
                                        "value": "win" if game_index < wins else "loss"
                                    }
                                },
                            }
                        )
                payload = {
                    "manifest": {
                        "run_id": f"run-{candidate_index}",
                        "candidate": {"name": candidate},
                        "opponents": [{"name": name} for name in names],
                        "wall_time_seconds": candidate_index + 1,
                    },
                    "summary": {
                        "total_games": len(names) * 10,
                        "wins": total_wins,
                        "losses": len(names) * 10 - total_wins,
                        "draws": 0,
                        "errors": 0,
                        "unfinished": 0,
                        "win_rate": total_wins / (len(names) * 10),
                        "by_opponent": by_opponent,
                    },
                    "metrics": {
                        "length": {"payload": {"rounds": {"by_opponent": turns_by_opponent}}}
                    },
                    "games": games,
                }
                report = reports_root / candidate / f"run-{candidate_index}" / "report.html"
                report.parent.mkdir(parents=True)
                report.write_text(
                    '<script id="report-data" type="application/json">'
                    + json.dumps(payload)
                    + "</script>",
                    encoding="utf-8",
                )

            data = build_combat_matrix_data(CATALOG, reports_root)

            self.assertEqual(data["protocol"]["packages"], 20)
            self.assertEqual(data["protocol"]["archetypes"], 11)
            self.assertEqual(data["summary"]["total_games"], 20 * 20 * 10)
            first = names[0]
            second = names[1]
            self.assertEqual(data["package_matrix"][first][second]["win_rate"], 0.1)
            self.assertEqual(data["package_matrix"][first][second]["first_games"], 5)
            self.assertEqual(data["package_matrix"][first][second]["second_games"], 5)
            self.assertEqual(data["package_matrix"][first][second]["average_turns"], 11.0)
            self.assertEqual(
                data["archetype_matrix"]["alakazam_dudunsparce"]
                ["alakazam_dudunsparce"]["games"],
                2 * 2 * 10,
            )

            html = render_combat_matrix(data, ROOT / "evaluation" / "arena")
            for title in (
                "Package 对局胜率",
                "Package 先攻胜率",
                "Package 后攻胜率",
                "Archetype 对局胜率",
                "Package 平均完整回合数",
                "Archetype 平均完整回合数",
            ):
                self.assertIn(title, html)
            self.assertIn('data-sort="win_rate"', html)
            self.assertIn('data-sort="wall_time"', html)
            self.assertIn("先攻胜率", html)
            self.assertIn("后攻胜率", html)
            self.assertIn("ceil(engine_turn / 2)", html)
            self.assertNotIn("<th>对局</th>", html)
            self.assertIn('class="win-rate-pill"', html)
            self.assertIn('class="matrix"', html)
            self.assertIn('class="card-stack"', html)
            self.assertIn('id="combat-mat-data"', html)

            output = root / "combat_mat" / "index.html"
            source = root / "combat_mat" / "matrix.json"
            write_combat_matrix(data, output, source)
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(source.read_text())["summary"]["total_games"], 4000)


if __name__ == "__main__":
    unittest.main()
