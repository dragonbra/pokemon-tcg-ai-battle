from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.frozen_0806_contract import evaluation_game_seed, evaluation_schedule_id


reference = importlib.import_module(
    "train.0042_full_model_design.semantic_parity.standard_panel_reference"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(outcomes: list[str]) -> dict[str, int]:
    return {
        "wins": outcomes.count("win"),
        "losses": outcomes.count("loss"),
        "draws": outcomes.count("draw"),
    }


def _refresh_cuda_commit(cuda_path: Path, manifest_path: Path) -> None:
    cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
    schedule_payload = {
        "schema": "policy_0806_cuda_seeded2048_v2",
        "evaluation_seed": cuda["evaluation_seed"],
        "focal_deck_id": cuda["deck_id"],
        "jobs": [{
            key: game[key]
            for key in (
                "game_id", "opponent_id", "replica", "slot", "engine_seed",
                "search_seed", "focal_first",
            )
        } for game in cuda["games"]],
    }
    cuda["schedule_sha256"] = hashlib.sha256(
        json.dumps(schedule_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cuda_path.write_text(json.dumps(cuda), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reports"][0]["game_records_sha256"] = _sha256(cuda_path)
    manifest["reports"][0]["schedule_sha256"] = cuda["schedule_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    opponent = "opponent-a"
    focal = "candidate-a"
    exact_deck = "c" * 64
    policy = "d" * 64
    cpu_games = []
    cuda_games = []
    cpu_outcomes: list[str] = []
    cuda_outcomes: list[str] = []
    cuda_guard_indices = {7, 777}
    cpu_guard_indices = {5, 100}
    for index in range(2_048):
        replica = index // 256
        slot = index % 256
        focal_first = replica % 2 == 0
        cpu_outcome = "win" if index % 3 else "loss"
        cuda_outcome = (
            "win" if cpu_outcome == "loss" and index % 10 == 0
            else "loss" if cpu_outcome == "win" and index % 17 == 0
            else cpu_outcome
        )
        cpu_outcomes.append(cpu_outcome)
        cuda_outcomes.append(cuda_outcome)
        guarded = index in cpu_guard_indices
        cpu_games.append({
            "candidate_first": focal_first,
            "error_kind": None,
            "game_id": f"{opponent}-{index + 1:03d}",
            "metric_refs": {
                "correctness": {
                    "value": "engine_error" if guarded else "ok",
                },
                "outcome": {"value": "error" if guarded else cpu_outcome},
                "length": {
                    "payload": {
                        "round": 5 + index % 4,
                        "action_selections": 100 + index % 7,
                    }
                },
            },
            "opponent": opponent,
            "status": "finished",
            "winner": 0 if cpu_outcome == "win" else 1,
        })
        raw_engine_result = (
            0
            if cuda_outcome == "draw"
            else 1
            if (cuda_outcome == "win") is focal_first
            else 2
        )
        cuda_games.append({
            "engine_seed": evaluation_game_seed(
                focal_identity=focal,
                opponent_identity=opponent,
                slot=slot,
                replica=replica,
            ),
            "focal_first": focal_first,
            "focal_outcome": cuda_outcome,
            "game_id": f"r{replica + 1:02d}-{opponent}-{slot + 1:03d}",
            "opponent_id": opponent,
            "raw_engine_result": raw_engine_result,
            "repeat_forfeit": index in cuda_guard_indices,
            "replica": replica,
            "schedule_index": index,
            "search_seed": evaluation_game_seed(
                focal_identity=focal,
                opponent_identity=opponent,
                slot=slot,
                replica=replica,
                namespace="search",
            ),
            "slot": slot,
            "turn_limit_draw": False,
        })

    cpu_counts = _counts(cpu_outcomes)
    cpu_payload = {
        "games": cpu_games,
        "manifest": {
            "candidate": {"package_manifest": {
                "checkpoint_sha256": policy,
                "deck_id": focal,
                "exact_deck_sha256": exact_deck,
                "frozen_deck_number": "007",
            }},
            "games": 2_048,
            "games_per_opponent": [2_048],
            "opponent_schedule_id": reference._FROZEN_0806_SCHEDULE_ID,
            "seed": 341_512_806,
        },
        "metrics": {"correctness": {"numerator": len(cpu_guard_indices)}},
        "summary": {
            **cpu_counts,
            "performance": {"wall_time_seconds": 1_024.0},
        },
    }
    cpu_path = root / "cpu.html"
    cpu_path.write_text(
        '<html><script id="report-data" type="application/json">'
        + json.dumps(cpu_payload)
        + "</script></html>",
        encoding="utf-8",
    )

    schedule_payload = {
        "schema": "policy_0806_cuda_seeded2048_v2",
        "evaluation_seed": 341_512_806,
        "focal_deck_id": focal,
        "jobs": [{
            key: game[key]
            for key in (
                "game_id", "opponent_id", "replica", "slot", "engine_seed",
                "search_seed", "focal_first",
            )
        } for game in cuda_games],
    }
    schedule = hashlib.sha256(
        json.dumps(schedule_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    game_results_sha256 = hashlib.sha256(
        json.dumps(
            [game["raw_engine_result"] for game in cuda_games], separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()
    cuda_payload = {
        "contract_id": "frozen_0806_seeded_2048_v2",
        "deck_id": focal,
        "deck_number": "007",
        "evaluation_seed": 341_512_806,
        "exact_deck_sha256": exact_deck,
        "game_results_sha256": game_results_sha256,
        "games": cuda_games,
        "schedule_sha256": schedule,
    }
    games_dir = root / "games"
    games_dir.mkdir()
    cuda_path = games_dir / "007.json"
    cuda_path.write_text(json.dumps(cuda_payload), encoding="utf-8")
    cuda_counts = _counts(cuda_outcomes)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({
        "policy_sha256": policy,
        "reports": [{
            "deck_id": focal,
            "draws": cuda_counts["draws"],
            "game_records_sha256": _sha256(cuda_path),
            "games": 2_048,
            "losses": cuda_counts["losses"],
            "progress_guard_forfeits": sorted(cuda_guard_indices),
            "schedule_sha256": schedule,
            "wall_seconds": 64.0,
            "wins": cuda_counts["wins"],
        }],
    }), encoding="utf-8")
    return cpu_path, cuda_path, manifest_path


class StandardPanelReferenceTest(unittest.TestCase):
    def test_full_panel_mapping_statistics_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cpu_path, cuda_path, manifest_path = _fixture(Path(directory))
            expected_hashes = {
                "cpu": _sha256(cpu_path),
                "cuda": _sha256(cuda_path),
                "manifest": _sha256(manifest_path),
            }
            result = reference.analyze_standard_panel_reference(cpu_path, cuda_path)

        self.assertEqual(
            result["evidence_type"],
            "shared_standard_panel_reference_not_independent_gate_g",
        )
        self.assertFalse(result["gate_g_eligible"])
        self.assertEqual(result["overall"]["cpu"]["games"], 2_048)
        self.assertEqual(result["by_seat"]["first"]["cpu"]["games"], 1_024)
        self.assertEqual(result["by_seat"]["second"]["cuda"]["games"], 1_024)
        self.assertEqual([row["cpu"]["games"] for row in result["by_shard"]], [256] * 8)
        self.assertGreater(result["overall"]["paired"]["cpu_loss_to_cuda_win"], 0)
        self.assertGreater(result["overall"]["paired"]["cpu_win_to_cuda_loss"], 0)
        self.assertEqual(result["diagnostics"]["cpu"]["progress_guard_adjudications"], 2)
        self.assertEqual(result["diagnostics"]["cpu"]["lifecycle_errors"], 0)
        self.assertEqual(result["diagnostics"]["cuda"]["repeat_forfeits"], 2)
        self.assertEqual(result["throughput"]["cuda_over_cpu_games_per_second"], 16.0)
        self.assertEqual(result["inputs"]["cpu_report_html"]["sha256"], expected_hashes["cpu"])
        self.assertEqual(result["inputs"]["cuda_games_json"]["sha256"], expected_hashes["cuda"])
        self.assertEqual(result["inputs"]["cuda_manifest_json"]["sha256"], expected_hashes["manifest"])

    def test_seat_mapping_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cpu_path, cuda_path, manifest_path = _fixture(Path(directory))
            cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
            cuda["games"][256]["focal_first"] = True
            cuda_path.write_text(json.dumps(cuda), encoding="utf-8")
            _refresh_cuda_commit(cuda_path, manifest_path)
            with self.assertRaisesRegex(ValueError, "mapping or seat mismatch"):
                reference.analyze_standard_panel_reference(cpu_path, cuda_path)

    def test_seed_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cpu_path, cuda_path, manifest_path = _fixture(Path(directory))
            cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
            cuda["games"][0]["engine_seed"] += 1
            cuda_path.write_text(json.dumps(cuda), encoding="utf-8")
            _refresh_cuda_commit(cuda_path, manifest_path)
            with self.assertRaisesRegex(ValueError, "seed contract mismatch"):
                reference.analyze_standard_panel_reference(cpu_path, cuda_path)


if __name__ == "__main__":
    unittest.main()
