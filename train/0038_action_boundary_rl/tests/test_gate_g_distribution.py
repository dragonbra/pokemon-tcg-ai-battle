from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.frozen_0806_contract import evaluation_game_seed


gate = importlib.import_module("train.0038_action_boundary_rl.semantic_parity.gate_g_distribution")


def games(wins: int, total: int, backend: str) -> list[dict]:
    return [{"outcome": "win" if index < wins else "loss",
             "seat": "first" if index % 2 == 0 else "second", "opponent": "deck-a",
             "game_length": 7 + (index % 3), "prize_differential": (index % 5) - 2,
             "strategic_decisions": 40 + index % 4, "macro_count": index % 2,
             "forced_count": 10 + index % 3, "action_type_counts": {"attack": 2},
             "error": False, "continuation_error": False, "fallback": False,
             "backend": backend} for index in range(total)]


class GateGDistributionTest(unittest.TestCase):
    def test_equivalent_large_panels_pass(self):
        cpu = games(2_500, 5_000, "cpu")
        cuda = games(2_505, 5_000, "cuda")
        for index, row in enumerate(cpu):
            row["engine_seed"] = index
        for index, row in enumerate(cuda):
            row["engine_seed"] = 1_000_000 + index
        result = gate.compare_runtime_distributions(cpu, cuda)
        self.assertEqual(result["status"], "PASS")

    def test_small_panel_is_incomplete_not_false_pass(self):
        result = gate.compare_runtime_distributions(games(152, 256, "cpu"), games(1003, 1792, "cuda"))
        self.assertEqual(result["status"], "INCOMPLETE")

    def test_semantic_errors_fail_even_with_equal_win_rate(self):
        cuda = games(2_500, 5_000, "cuda")
        cuda[0]["fallback"] = True
        result = gate.compare_runtime_distributions(games(2_500, 5_000, "cpu"), cuda)
        self.assertEqual(result["status"], "FAIL")

    def test_missing_runtime_diagnostic_is_incomplete_not_silent_zero(self):
        cpu = games(2_500, 5_000, "cpu")
        cuda = games(2_500, 5_000, "cuda")
        for row in cuda:
            row.pop("fallback")
        result = gate.compare_runtime_distributions(cpu, cuda)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["error_rates"]["fallback"]["status"], "INCOMPLETE")
        self.assertIn("missing", result["error_rates"]["fallback"]["reason"])

    def test_unbalanced_opponent_seat_mix_is_incomplete(self):
        cpu = games(2_500, 5_000, "cpu")
        cuda = games(2_500, 5_000, "cuda")
        for row in cuda:
            row["seat"] = "first"
        result = gate.compare_runtime_distributions(cpu, cuda)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["input_contract"]["strata"]["status"], "INCOMPLETE")

    def test_absent_seed_provenance_is_incomplete(self):
        result = gate.compare_runtime_distributions(
            games(2_500, 5_000, "cpu"), games(2_500, 5_000, "cuda")
        )
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["input_contract"]["engine_seeds"]["status"], "INCOMPLETE")

    def test_disjoint_seeded_equivalent_panels_pass(self):
        cpu = games(2_500, 5_000, "cpu")
        cuda = games(2_505, 5_000, "cuda")
        for index, row in enumerate(cpu):
            row["engine_seed"] = index
        for index, row in enumerate(cuda):
            row["engine_seed"] = 1_000_000 + index
        result = gate.compare_runtime_distributions(cpu, cuda)
        self.assertEqual(result["status"], "PASS")

    def test_normalizers_copy_only_observed_diagnostics(self):
        official = {
            "games": [{
                "game_id": "cpu-1", "opponent": "deck-a", "candidate_first": True,
                "status": "finished", "error_kind": None, "winner": 0,
                "metric_refs": {
                    "outcome": {"value": "error"},
                    "length": {"value": 7},
                },
            }]
        }
        cuda = {"games": [{
            "game_id": "cuda-1", "opponent_id": "deck-a", "focal_first": False,
            "focal_outcome": "loss", "engine_seed": 17, "raw_engine_result": 2,
        }]}
        official_row = gate.normalize_official_cpu_report(official)[0]
        cuda_row = gate.normalize_cuda_game_results(cuda)[0]
        self.assertEqual(official_row["game_length"], 7.0)
        self.assertEqual(official_row["outcome"], "win")
        self.assertFalse(official_row["error"])
        self.assertNotIn("fallback", official_row)
        self.assertNotIn("continuation_error", official_row)
        self.assertEqual(cuda_row["engine_seed"], 17)
        self.assertNotIn("error", cuda_row)
        self.assertNotIn("fallback", cuda_row)

    def test_official_html_loader_reads_report_data(self):
        payload = {"games": []}
        document = (
            '<html><script id="report-data" type="application/json">'
            + json.dumps(payload)
            + "</script></html>"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            path.write_text(document, encoding="utf-8")
            self.assertEqual(gate.load_official_cpu_report(path), payload)

    def test_official_frozen_2048_seed_provenance_is_reconstructed_fail_closed(self):
        opponent = "deck-a"
        candidate = "candidate-a"
        raw_games = []
        base_slots = 2_048 // 8
        for index in range(2_048):
            replica = index // base_slots
            raw_games.append({
                "game_id": f"{opponent}-{index + 1:03d}",
                "opponent": opponent,
                "candidate_first": replica % 2 == 0,
                "status": "finished",
                "winner": 0,
                "metric_refs": {
                    "outcome": {"value": "win"},
                    "length": {"value": 7},
                },
            })
        payload = {
            "manifest": {
                "games": 2_048,
                "games_per_opponent": [2_048],
                "seed": 12345,
                "opponent_schedule_id": "formal-schedule",
                "candidate": {"name": candidate},
                "opponents": [{"name": opponent}],
            },
            "games": raw_games,
        }
        rows = gate.normalize_official_cpu_report(payload)
        self.assertEqual(
            rows[0]["engine_seed"],
            evaluation_game_seed(
                evaluation_seed=12345,
                focal_identity=candidate,
                opponent_identity=opponent,
                slot=0,
                replica=0,
            ),
        )
        payload["games"][base_slots]["candidate_first"] = True
        rows = gate.normalize_official_cpu_report(payload)
        self.assertTrue(all("engine_seed" not in row for row in rows))


if __name__ == "__main__":
    unittest.main()
