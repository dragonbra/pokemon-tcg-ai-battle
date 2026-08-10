from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from engine_cuda.tools.evaluate_policy_0809_cuda import (
    CANDIDATE_CONTRACT,
    DIAGNOSTIC_SCHEMA,
    EXPECTED_GAMES,
    POLICY_ID,
    POLICY_SHA256,
    RESULT_SCHEMA,
    TOP_TEN,
    _sha256,
    _catalog,
    _classify_meta,
    _meta_aggregates,
    _meta_contract,
    _opponent_aggregates,
    _opponent_meta_aggregates,
    _render_deck,
    _valid_result,
    build_cuda_schedule,
    preflight_identity,
)
from evaluation.combat_mat_contract import (
    validate_combat_mat_detail,
    validate_combat_mat_index,
)


@dataclass(frozen=True)
class Entry:
    deck_id: str
    games: int


class Policy0809CudaEvaluationTest(unittest.TestCase):
    def test_top_ten_is_exactly_001_through_010(self) -> None:
        self.assertEqual(TOP_TEN, tuple(f"{number:03d}" for number in range(1, 11)))

    def test_schedule_is_eight_complete_unique_replicas_with_policy_identity(self) -> None:
        schedule = build_cuda_schedule(
            focal_deck_id="candidate-a",
            entries=(Entry("opponent-a", 128), Entry("opponent-b", 128)),
        )
        jobs = schedule["jobs"]
        self.assertEqual(len(jobs), EXPECTED_GAMES)
        self.assertEqual({row["replica"] for row in jobs}, set(range(8)))
        self.assertEqual(
            {row["replica"]: sum(item["replica"] == row["replica"] for item in jobs)
             for row in jobs},
            {replica: 256 for replica in range(8)},
        )
        self.assertEqual(len({row["engine_seed"] for row in jobs}), EXPECTED_GAMES)
        self.assertEqual(len({row["search_seed"] for row in jobs}), EXPECTED_GAMES)
        self.assertTrue(all(row["focal_policy_id"] == POLICY_ID for row in jobs))
        self.assertTrue(all(row["opponent_policy_id"] == POLICY_ID for row in jobs))
        self.assertTrue(all(type(row["focal_won_toss"]) is bool for row in jobs))

    def test_registered_policy_0809_preflight_passes_for_both_roles(self) -> None:
        audit = preflight_identity()
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["focal_source"]["requested_policy_id"], POLICY_ID)
        self.assertEqual(audit["opponent"]["requested_policy_id"], POLICY_ID)
        self.assertEqual(audit["focal_source"]["checkpoint_sha256"], POLICY_SHA256)
        self.assertEqual(audit["opponent"]["checkpoint_sha256"], POLICY_SHA256)

    def test_full_catalog_has_55_decks_and_14_explicit_meta_classes(self) -> None:
        _catalog_value, candidates = _catalog()
        contract = _meta_contract()
        classified = [_classify_meta(item.deck) for item in candidates]
        self.assertEqual(len(candidates), 55)
        self.assertEqual(len(contract["classes"]), 14)
        self.assertEqual(
            [item["class_id"] for item in contract["classes"]], list(range(14))
        )
        self.assertTrue(all(0 <= item["class_id"] <= 14 for item in classified))

    def test_meta_aggregation_conserves_completed_games(self) -> None:
        _catalog_value, candidates = _catalog()
        first = candidates[0]
        meta = _classify_meta(first.deck)
        rows = [{
            "deck_number": first.package_manifest["frozen_deck_number"],
            "games": 2048,
            "wins": 1000,
            "losses": 1040,
            "draws": 8,
        }]
        aggregates, other = _meta_aggregates(candidates, rows)
        self.assertEqual(len(aggregates), 14)
        self.assertEqual(
            sum(item["games"] for item in aggregates) + other["games"], 2048
        )
        target = other if meta["class_id"] == 14 else aggregates[meta["class_id"]]
        self.assertEqual(target["tested_decks"], 1)
        self.assertEqual((target["wins"], target["losses"], target["draws"]),
                         (1000, 1040, 8))

    def test_detail_contract_requires_60_cards_55_opponents_and_14_meta_rows(self) -> None:
        catalog, candidates = _catalog()
        candidate = candidates[6]
        games = []
        for opponent in candidates:
            for outcome in ("win", "loss"):
                games.append({
                    "opponent_id": opponent.name,
                    "focal_outcome": outcome,
                    "focal_first": outcome == "win",
                })
        by_opponent = _opponent_aggregates(games, candidates)
        by_meta, other = _opponent_meta_aggregates(by_opponent, candidates)
        detail = {
            "deck_total": len(candidate.deck),
            "deck_cards": [{"count": len(candidate.deck), "image_url": "card.png"}],
            "by_opponent": by_opponent,
            "by_meta_archetype": by_meta,
            "meta_archetype_other": other,
            "games": len(games),
            "wins": 55,
            "losses": 55,
            "draws": 0,
        }
        validate_combat_mat_detail(detail, expected_opponent_ids={item.name for item in candidates})
        self.assertEqual(len(by_opponent), 55)
        self.assertEqual(len(by_meta), 14)
        self.assertEqual(sum(item["games"] for item in by_meta) + other["games"], 110)

    def test_index_and_detail_renderers_expose_the_fixed_ui_contract(self) -> None:
        catalog, candidates = _catalog()
        index_payload = {
            "catalog_decks": 55,
            "meta_archetype_aggregates": [{"class_id": index} for index in range(14)],
            "meta_archetype_other": {"class_id": 14},
            "catalog": [
                {
                    "deck_number": f"{number:03d}",
                    "status": "tested" if number <= 10 else "pending",
                    "representative_cards": [{"image_url": "card.png"}],
                }
                for number in range(1, 56)
            ],
            "reports": [],
        }
        validate_combat_mat_index(index_payload)
        candidate = candidates[6]
        games = []
        for opponent in candidates:
            games.append({
                "opponent_id": opponent.name,
                "focal_outcome": "win",
                "focal_first": True,
            })
        by_opponent = _opponent_aggregates(games, candidates)
        by_meta, other = _opponent_meta_aggregates(by_opponent, candidates)
        detail = {
            "deck_total": 60,
            "deck_cards": [{"count": 60, "image_url": "card.png"}],
            "by_opponent": by_opponent,
            "by_meta_archetype": by_meta,
            "meta_archetype_other": other,
            "games": 55,
            "wins": 55,
            "losses": 0,
            "draws": 0,
        }
        page = _render_deck(
            {"deck_number": "007", "display_name": candidate.display_name},
            games,
            catalog=catalog,
            candidates=candidates,
            detail=detail,
            completed_numbers=set(TOP_TEN),
        )
        self.assertIn('id="exact-deck"', page)
        self.assertIn('id="opponent-matchups"', page)
        self.assertIn('id="meta-archetype-matchups"', page)
        self.assertEqual(page.count('class="chart-row opponent-row"'), 55)
        self.assertEqual(page.count('class="chart-row meta-row"'), 14)
        self.assertEqual(page.count('class="chart-row meta-other-row"'), 1)

        missing_opponent = dict(detail)
        missing_opponent["by_opponent"] = dict(by_opponent)
        missing_opponent["by_opponent"].pop(next(iter(by_opponent)))
        with self.assertRaisesRegex(ValueError, "every catalog opponent"):
            validate_combat_mat_detail(
                missing_opponent,
                expected_opponent_ids={item.name for item in candidates},
            )

        missing_meta = dict(detail)
        missing_meta["by_meta_archetype"] = by_meta[:-1]
        with self.assertRaisesRegex(ValueError, "meta classes 0-13"):
            validate_combat_mat_detail(
                missing_meta,
                expected_opponent_ids={item.name for item in candidates},
            )

    def test_cached_result_requires_both_complete_identity_audits(self) -> None:
        games = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            (package / "strategy").mkdir(parents=True)
            portable = package / "strategy/model.bin"
            portable.write_bytes(b"portable")
            schedule = root / "schedule.json"
            schedule.write_text(json.dumps({"jobs": [1, 2]}), encoding="utf-8")
            result = {
                "passed": True,
                "schema_version": RESULT_SCHEMA,
                "collector": {"completed_games": games, "errors": 0},
                "determinism": {"game_results": [1, 2]},
                "schedule": {"sha256": _sha256(schedule)},
                "models": {
                    "actor_checkpoint_sha256": _sha256(portable),
                    "opponent_checkpoint_sha256": POLICY_SHA256,
                },
                "candidate_deployment_identity_audit": {
                    "status": "PASS",
                    "contract_id": CANDIDATE_CONTRACT,
                    "source_checkpoint_sha256": POLICY_SHA256,
                    "portable_checkpoint_sha256": _sha256(portable),
                    "effective_candidate_sha256": "a" * 64,
                    "storage_dtype": "fp16",
                    "runtime_dtype": "fp32",
                },
                "opponent_policy_identity_audit": {
                    "status": "PASS",
                    "requested_policy_id": POLICY_ID,
                    "checkpoint_sha256": POLICY_SHA256,
                    "effective_policy_sha256": "b" * 64,
                },
                "device": {
                    "float32_matmul_precision": "highest",
                    "matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                },
                "per_game_diagnostics": {
                    "schema": DIAGNOSTIC_SCHEMA,
                    "terminal_turns": [1] * games,
                    "engine_selections": [1] * games,
                    "terminal_prize_counts": [[0, 0]] * games,
                    "first_player_choosers": [0] * games,
                    "first_player_actions": [0] * games,
                    "actual_first_players": [0] * games,
                },
            }
            self.assertTrue(_valid_result(
                result, schedule_path=schedule, package=package, games=games
            ))
            result["candidate_deployment_identity_audit"]["status"] = "FAIL"
            self.assertFalse(_valid_result(
                result, schedule_path=schedule, package=package, games=games
            ))
            result["candidate_deployment_identity_audit"]["status"] = "PASS"
            result["opponent_policy_identity_audit"].pop("effective_policy_sha256")
            self.assertFalse(_valid_result(
                result, schedule_path=schedule, package=package, games=games
            ))


if __name__ == "__main__":
    unittest.main()
