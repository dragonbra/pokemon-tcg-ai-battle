from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rl.train.top20_bc_campaign import _attempt_gate, _evaluation_summary, _select_attempt


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _gate(attempt_id: str, *, exact: float, loss: float, passed: bool) -> dict:
    return {
        "attempt_id": attempt_id,
        "validation_gate_passed": passed,
        "offline_gate_passed": False,
        "checkpoint": f"checkpoint/{attempt_id}.pt",
        "checkpoint_sha256": attempt_id,
        "metrics": {
            "validation_exact_action_rate": exact,
            "validation_policy_loss": loss,
        },
    }


class Top20BCCampaignTests(unittest.TestCase):
    def test_offline_gate_stays_pending_until_frozen_test_is_evaluated(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            summary = {
                "best_validation_exact_action_rate": 0.8,
                "best_epoch": 4,
                "best_epoch_metrics": {
                    "validation/exact_action_rate": 0.8,
                    "validation/multi_action_exact_rate": 0.7,
                    "validation/selection_count_accuracy": 0.99,
                    "validation/legal_action_rate": 1.0,
                    "validation/policy_loss": 0.2,
                    "train/exact_action_rate": 0.82,
                },
                "final_test": {},
                "checkpoint": str(checkpoint),
                "runtime": {
                    "train_function_seconds": 1.0,
                    "peak_gpu_memory_bytes": 2,
                    "parameter_count": 3,
                },
            }
            summary_path = root / "V1_shared_config/training_summary.json"
            _write_json(summary_path, summary)
            pending = _attempt_gate(root, "V1_shared_config")
            summary["final_test"] = {"legal_action_rate": 1.0}
            _write_json(summary_path, summary)
            final = _attempt_gate(root, "V1_shared_config")
        self.assertTrue(pending["validation_gate_passed"])
        self.assertFalse(pending["test_evaluated"])
        self.assertFalse(pending["offline_gate_passed"])
        self.assertTrue(final["test_evaluated"])
        self.assertTrue(final["offline_gate_passed"])

    def test_attempt_selection_never_uses_a_failing_tie_over_a_passing_gate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_json(
                root / "V1_shared_config/gate_decision.json",
                _gate("V1_shared_config", exact=0.751, loss=0.4, passed=True),
            )
            _write_json(
                root / "V2_lr_lower/gate_decision.json",
                _gate("V2_lr_lower", exact=0.753, loss=0.1, passed=False),
            )
            decision = _select_attempt(root)
        self.assertEqual(decision["selected_attempt_id"], "V1_shared_config")
        self.assertTrue(decision["validation_gate_passed"])

    def test_evaluation_summary_uses_revision_two_post_ko_semantics(self) -> None:
        with TemporaryDirectory() as directory:
            run_root = Path(directory)
            result_root = run_root / "evaluation/V1/run-test"
            _write_json(
                result_root / "summary.json",
                {"completed_games": 180, "errors": 0, "unfinished": 0},
            )
            _write_json(
                result_root / "metrics.json",
                {
                    "outcome": {
                        "payload": {
                            "by_turn_order": {
                                "first": {"value": 0.6},
                                "second": {"value": 0.4},
                            }
                        }
                    },
                    "correctness": {
                        "numerator": 0,
                        "diagnostics": {"failure_classes": {}},
                    },
                    "powerful_hand": {"value": 0.5},
                    "setup_relay": {
                        "value": 0.5,
                        "payload": {
                            "second_turn_draws": {"all_games": {"average": 3.25}}
                        },
                    },
                    "post_ko_relay": {
                        "value": 0.7,
                        "payload": {"success_rate": 0.3},
                    },
                    "attack_quality": {"value": 0.2},
                    "library_pressure": {"value": 1.5},
                },
            )
            (result_root / "report.html").write_text("", encoding="utf-8")
            (result_root / "report.md").write_text("", encoding="utf-8")
            summary = _evaluation_summary(run_root, "V1")
        metrics = summary["semantic_metrics"]
        self.assertEqual(metrics["post_ko_relay_success_rate"], 0.3)
        self.assertEqual(metrics["first_player_win_rate"], 0.6)
        self.assertEqual(metrics["second_turn_ability_draws_per_game"], 3.25)


if __name__ == "__main__":
    unittest.main()
