from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from train.kaggle_bc_top20.training.bc_capacity_search import (
    ARCHITECTURES,
    TrialSpec,
    _can_launch,
    _run_trial,
    choose_baseline_learning_rate,
    choose_phase_c_spec,
    choose_phase_d_spec,
    phase_a_specs,
    phase_b_specs,
)
from train.kaggle_bc_top20.training.render_bc_capacity_search import _update_index


def _result(
    version: str,
    architecture: str,
    phase: str,
    learning_rate: float,
    *,
    train_exact: float,
    validation_exact: float,
    policy_loss: float = 0.4,
    stability: float = 0.001,
    train_slope: float = 0.0,
    validation_slope: float = 0.0,
    parameter_count: int = 5_000_000,
) -> dict[str, object]:
    d_model, layers, hidden_dim = ARCHITECTURES[architecture]
    spec = TrialSpec(
        version=version,
        phase=phase,
        architecture=architecture,
        d_model=d_model,
        layers=layers,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
    )
    return {
        "status": "completed",
        "spec": asdict(spec),
        "offline": {
            "train_exact": train_exact,
            "validation_exact": validation_exact,
            "exact_gap": train_exact - validation_exact,
            "validation_policy_loss": policy_loss,
            "validation_last5_std": stability,
            "train_last5_slope": train_slope,
            "validation_last5_slope": validation_slope,
        },
        "model": {"parameter_count": parameter_count},
        "training": {"wall_seconds": 900.0, "epoch_median_seconds": 40.0},
    }


class BCCapacitySearchTests(unittest.TestCase):
    def test_required_phase_plan_is_stable(self) -> None:
        phase_a = phase_a_specs()
        self.assertEqual([spec.learning_rate for spec in phase_a], [1e-4, 3e-4, 5e-4])
        self.assertEqual([spec.version.split("_", 1)[0] for spec in phase_a], ["V1", "V2", "V3"])

        phase_b = phase_b_specs(3e-4)
        self.assertEqual([spec.architecture for spec in phase_b], ["S", "W", "D", "L"])
        self.assertEqual([spec.version.split("_", 1)[0] for spec in phase_b], ["V4", "V5", "V6", "V7"])
        self.assertTrue(all(spec.epochs == 20 and spec.batch_size == 256 for spec in phase_b))

    def test_baseline_lr_selection_uses_validation_tie_rules(self) -> None:
        results = [
            _result("V1", "B", "A", 1e-4, train_exact=0.82, validation_exact=0.800),
            _result(
                "V2",
                "B",
                "A",
                3e-4,
                train_exact=0.83,
                validation_exact=0.798,
                policy_loss=0.35,
            ),
            _result("V3", "B", "A", 5e-4, train_exact=0.84, validation_exact=0.790),
        ]
        for index, result in enumerate(results):
            result["evaluation"] = {"win_rate": index / 2}
        self.assertEqual(choose_baseline_learning_rate(results), 3e-4)

    def test_phase_c_compares_against_calibrated_baseline(self) -> None:
        results = [
            _result("V1", "B", "A", 1e-4, train_exact=0.95, validation_exact=0.70),
            _result("V2", "B", "A", 3e-4, train_exact=0.80, validation_exact=0.80),
            _result("V3", "B", "A", 5e-4, train_exact=0.99, validation_exact=0.60),
            _result("V7", "L", "B", 3e-4, train_exact=0.82, validation_exact=0.79),
        ]
        spec, reason = choose_phase_c_spec(results, 3e-4)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.phase, "C")
        self.assertEqual(spec.architecture, "L")
        self.assertEqual(spec.dropout, 0.05)
        self.assertIn("gap worsened", reason)

    def test_phase_d_ignores_nonselected_baseline_lr(self) -> None:
        results = [
            _result(
                "V1",
                "B",
                "A",
                1e-4,
                train_exact=0.99,
                validation_exact=0.99,
                parameter_count=5_000_000,
            ),
            _result(
                "V2",
                "B",
                "A",
                3e-4,
                train_exact=0.80,
                validation_exact=0.80,
                parameter_count=5_000_000,
            ),
            _result(
                "V4",
                "S",
                "B",
                3e-4,
                train_exact=0.80,
                validation_exact=0.799,
                parameter_count=3_000_000,
            ),
        ]
        spec, _ = choose_phase_d_spec(results, 8, 3e-4)
        self.assertEqual(spec.architecture, "S")
        self.assertEqual(spec.seed, 17)

    def test_budget_gate_includes_observed_epoch_cost(self) -> None:
        completed = _result(
            "V4",
            "S",
            "B",
            3e-4,
            train_exact=0.8,
            validation_exact=0.8,
        )
        completed["training"] = {"wall_seconds": 100.0, "epoch_median_seconds": 50.0}
        comparison = {
            "trials": [completed, {"training": {"wall_seconds": 10_600.0}}]
        }
        d_model, layers, hidden_dim = ARCHITECTURES["S"]
        spec = TrialSpec(
            version="V8_s_budget",
            phase="D",
            architecture="S",
            d_model=d_model,
            layers=layers,
            hidden_dim=hidden_dim,
            learning_rate=3e-4,
        )
        self.assertFalse(_can_launch(comparison, spec))

    def test_training_log_staging_keeps_new_version_unoccupied(self) -> None:
        with TemporaryDirectory() as directory:
            repository_root = Path(directory)
            experiment_root = repository_root / "rl_runs/0004-test"
            experiment_root.mkdir(parents=True)
            spec = phase_a_specs()[0]
            run_root = experiment_root / spec.version

            def fail_training(command, cwd, log_path, **kwargs):  # type: ignore[no-untyped-def]
                self.assertFalse(run_root.exists())
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("expected smoke failure\n", encoding="utf-8")
                return 2.0, 9

            with patch(
                "train.kaggle_bc_top20.training.bc_capacity_search._run_logged_process",
                side_effect=fail_training,
            ):
                result = _run_trial(
                    repository_root,
                    experiment_root,
                    repository_root / "dataset.jsonl",
                    repository_root / "source",
                    repository_root / "runtime",
                    spec,
                )

            self.assertEqual(result["status"], "failed")
            self.assertTrue((run_root / "train.log").is_file())
            self.assertTrue((run_root / "status.json").is_file())

    def test_index_marker_replacement_is_idempotent(self) -> None:
        with TemporaryDirectory() as directory:
            repository_root = Path(directory)
            runs_root = repository_root / "rl_runs"
            experiment_root = runs_root / "0004-test"
            experiment_root.mkdir(parents=True)
            index = runs_root / "INDEX.html"
            index.write_text("<html><body><h2>当前方向</h2></body></html>", encoding="utf-8")
            comparison = {"status": "running", "trials": []}

            _update_index(repository_root, experiment_root, comparison)
            _update_index(repository_root, experiment_root, comparison)

            content = index.read_text(encoding="utf-8")
            self.assertEqual(content.count("BC_CAPACITY_SEARCH_START:0004-test"), 1)
            self.assertEqual(content.count("bc_capacity_search_report.html"), 1)


if __name__ == "__main__":
    unittest.main()
