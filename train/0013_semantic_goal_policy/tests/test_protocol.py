from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

protocol_module = importlib.import_module("train.0013_semantic_goal_policy.protocol")
PreRunProtocol = protocol_module.PreRunProtocol
freeze_protocol = protocol_module.freeze_protocol


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "configs" / "pre_run_protocol.json"
MODULE = "train.0013_semantic_goal_policy"


class PreRunProtocolTests(unittest.TestCase):
    def test_frozen_defaults_match_plan(self) -> None:
        protocol = PreRunProtocol.from_json(PROTOCOL_PATH)
        data = json.loads(protocol.canonical_bytes())

        self.assertEqual(data["schema_version"], "pre_run_protocol_v1")
        self.assertEqual(
            data["split"],
            {
                "train_fraction": 0.9,
                "validation_fraction": 0.1,
                "seed": 20260726,
                "group_unit": "complete_episode_player",
                "stratify_by": ["date", "deck_manifest_hash"],
                "assignment": "deterministic_digest_rank",
                "digest_algorithm": "sha256",
                "digest_input": (
                    "semantic_goal_policy_split_v1|20260726|<date>|<deck_hash>|"
                    "<episode_id>|<player>"
                ),
            },
        )
        self.assertEqual(
            data["source_precedence"],
            {
                "identity": "canonical_episode_id",
                "rule": "patched_episode_replaces_archive_original",
                "ambiguity": "fail_closed",
                "first_date": "2026-07-18",
                "last_date": "2026-07-25",
                "date_count": 8,
            },
        )
        self.assertEqual(data["event_window"], {"max_events": 64, "visibility": "actor_visible", "selection": "newest"})
        self.assertEqual(data["ontology"]["version"], "card_effect_ontology_v1")
        self.assertEqual(data["ontology"]["coverage_population"], [
            "registered_deck_identities",
            "observed_legal_option_identities",
        ])
        self.assertEqual(data["ontology"]["card_identity_coverage_min"], 1.0)
        self.assertEqual(data["ontology"]["effect_instance_coverage_min"], 0.99)
        self.assertEqual(data["ontology"]["residual_effect"], "UNKNOWN_EFFECT")
        self.assertEqual(
            data["view_contract"],
            {
                "kinds": ["none", "eligible_subset", "full_membership", "ordered_view"],
                "exact_prize_inference_requires": "verified_full_membership",
            },
        )
        self.assertEqual(
            data["formal_evaluation"],
            {
                "metric_profile": "auto_iteration_v8_setup_relay",
                "metric_revision": 7,
                "games_per_enabled_opponent": 20,
                "seed_formula": "20260726 + game_index",
                "seat_schedule": "alternating",
                "games_per_seat": 10,
            },
        )
        self.assertEqual(data["thresholds"]["invariance_aligned_agreement_min"], 0.999)
        self.assertEqual(data["thresholds"]["invariance_mean_kl_max"], 1e-4)
        self.assertEqual(data["thresholds"]["free_running_legality"], 1.0)
        self.assertEqual(data["thresholds"]["free_running_completion"], 1.0)
        self.assertEqual(data["thresholds"]["counterfactual_direction_rate_min"], 0.8)
        self.assertEqual(data["thresholds"]["unknown_vs_zero_rate_min"], 0.8)
        self.assertEqual(data["thresholds"]["irrelevant_goal_mean_abs_logit_change_max"], 0.05)
        self.assertEqual(
            data["numerical_health"],
            {"nan_inf_count_max": 0, "corrupt_checkpoint_count_max": 0, "unacknowledged_amp_overflow_count_max": 0},
        )
        self.assertEqual(
            data["minimum_formal_allocation"],
            {"models": ["M0", "M1", "M2", "M3", "M4", "M5"], "seeds": 1, "complete_epochs": 3, "complete_train_passes_min": 1, "requires_smoke_pass": True},
        )
        self.assertEqual(
            data["runtime_floor"],
            {"formula": "min(10.0, 0.8 * measured_m0_smoke_decisions_per_second)", "units": "decisions_per_second"},
        )
        self.assertNotIn("measured_m0_smoke_decisions_per_second", data["runtime_floor"])
        self.assertEqual(
            data["gpu_time"],
            {"telemetry_interval_seconds": 10, "active_seconds_target": 36000, "merge_gap_seconds_max": 60, "intervals": "non_overlapping_healthy_gpu_work", "report_separately": ["idle", "failure", "overlap"]},
        )
        self.assertEqual(data["wandb_snapshot_limits"], {"bytes_per_file_max": 1048576, "bytes_per_generation_max": 4194304})

    def test_canonical_hash_is_stable_across_key_order_and_whitespace(self) -> None:
        protocol = PreRunProtocol.from_json(PROTOCOL_PATH)
        self.assertEqual(
            protocol.sha256(),
            "da6482cf2d4cdc8d9e56fd4c03431e60dbf63b8327720c83185e907a039d44d3",
        )
        reordered = json.dumps(json.loads(protocol.canonical_bytes()), sort_keys=False, indent=7)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "protocol.json"
            path.write_text(reordered, encoding="utf-8")
            loaded = PreRunProtocol.from_json(path)
        self.assertEqual(loaded.canonical_bytes(), protocol.canonical_bytes())
        self.assertEqual(loaded.sha256(), protocol.sha256())
        self.assertTrue(protocol.canonical_bytes().endswith(b"\n"))

    def test_missing_threshold_is_rejected(self) -> None:
        data = dict(PreRunProtocol.from_json(PROTOCOL_PATH).data)
        data["thresholds"] = dict(data["thresholds"])
        del data["thresholds"]["free_running_legality"]
        with self.assertRaisesRegex(ValueError, "free_running_legality"):
            PreRunProtocol(data).validate()

    def test_invalid_mutations_are_rejected_across_every_section(self) -> None:
        mutations = {
            "top-level unexpected": lambda d: d.update(extra=True),
            "schema": lambda d: d.__setitem__("schema_version", "v2"),
            "split missing": lambda d: d["split"].pop("seed"),
            "split unexpected": lambda d: d["split"].update(extra=True),
            "split bool number": lambda d: d["split"].__setitem__("seed", True),
            "split sum": lambda d: d["split"].__setitem__("validation_fraction", 0.2),
            "split algorithm": lambda d: d["split"].__setitem__("digest_algorithm", "md5"),
            "source precedence": lambda d: d["source_precedence"].__setitem__("ambiguity", "guess"),
            "event window": lambda d: d["event_window"].__setitem__("max_events", 0),
            "ontology population": lambda d: d["ontology"].__setitem__("coverage_population", ["registered_deck_identities"]),
            "ontology coverage": lambda d: d["ontology"].__setitem__("effect_instance_coverage_min", 1.1),
            "view kinds": lambda d: d["view_contract"].__setitem__("kinds", ["none"]),
            "formal profile": lambda d: d["formal_evaluation"].__setitem__("metric_revision", 8),
            "formal seed": lambda d: d["formal_evaluation"].__setitem__("seed_formula", "random"),
            "threshold nonfinite": lambda d: d["thresholds"].__setitem__("invariance_mean_kl_max", float("inf")),
            "threshold bool": lambda d: d["thresholds"].__setitem__("free_running_legality", True),
            "threshold range": lambda d: d["thresholds"].__setitem__("unknown_vs_zero_rate_min", -0.1),
            "numerical health": lambda d: d["numerical_health"].__setitem__("nan_inf_count_max", 1),
            "allocation models": lambda d: d["minimum_formal_allocation"].__setitem__("models", ["M0", "M1"]),
            "allocation count": lambda d: d["minimum_formal_allocation"].__setitem__("seeds", 0),
            "runtime result field": lambda d: d["runtime_floor"].update(measured_m0_smoke_decisions_per_second=10),
            "runtime units": lambda d: d["runtime_floor"].__setitem__("units", "games_per_second"),
            "gpu duration": lambda d: d["gpu_time"].__setitem__("active_seconds_target", 0),
            "gpu reports": lambda d: d["gpu_time"].__setitem__("report_separately", ["idle"]),
            "snapshot limit": lambda d: d["wandb_snapshot_limits"].__setitem__("bytes_per_file_max", False),
        }
        baseline = json.loads(PreRunProtocol.from_json(PROTOCOL_PATH).canonical_bytes())
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(baseline)
                mutate(candidate)
                with self.assertRaises(ValueError):
                    PreRunProtocol(candidate).validate()

    def test_strict_json_rejects_nonstandard_constants(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "protocol.json"
                payload = PROTOCOL_PATH.read_text(encoding="utf-8").replace("0.999", constant, 1)
                path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                    PreRunProtocol.from_json(path)

    def test_freeze_refuses_overwrite(self) -> None:
        protocol = PreRunProtocol.from_json(PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "frozen.json"
            freeze_protocol(protocol, destination)
            original = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                freeze_protocol(protocol, destination)
            self.assertEqual(destination.read_bytes(), original)

    def test_runtime_floor_resolution_is_external_and_validated(self) -> None:
        protocol = PreRunProtocol.from_json(PROTOCOL_PATH)
        self.assertEqual(protocol.resolve_runtime_floor(20.0), 10.0)
        self.assertEqual(protocol.resolve_runtime_floor(8.0), 6.4)
        with self.assertRaisesRegex(ValueError, "positive"):
            protocol.resolve_runtime_floor(0.0)

    def test_validate_protocol_cli_prints_canonical_hash(self) -> None:
        result = self.run_cli("validate-protocol")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), PreRunProtocol.from_json(PROTOCOL_PATH).sha256())

    def test_formal_command_rejects_missing_protocol_hash(self) -> None:
        result = self.run_cli("train", "--m0-smoke-throughput", "12")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--protocol-sha256", result.stderr)

    def test_formal_command_rejects_protocol_hash_mismatch(self) -> None:
        result = self.run_cli(
            "train", "--protocol-sha256", "0" * 64, "--m0-smoke-throughput", "12"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("protocol hash mismatch", result.stderr)

    def test_training_rejects_missing_runtime_measurement_after_valid_hash(self) -> None:
        digest = PreRunProtocol.from_json(PROTOCOL_PATH).sha256()
        result = self.run_cli("train", "--protocol-sha256", digest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--m0-smoke-throughput", result.stderr)

    def test_non_protocol_scaffold_commands_do_not_load_protocol(self) -> None:
        original = PROTOCOL_PATH.read_bytes()
        try:
            PROTOCOL_PATH.write_text("not json", encoding="utf-8")
            for command in ("audit-visibility", "smoke-model"):
                with self.subTest(command=command):
                    result = self.run_cli(command)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("not implemented in the scaffold", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
        finally:
            PROTOCOL_PATH.write_bytes(original)

    def test_training_invalid_runtime_measurement_is_clean_parser_error(self) -> None:
        digest = PreRunProtocol.from_json(PROTOCOL_PATH).sha256()
        for measurement in ("0", "nan", "inf"):
            with self.subTest(measurement=measurement):
                result = self.run_cli(
                    "train", "--protocol-sha256", digest,
                    "--m0-smoke-throughput", measurement,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("finite and positive", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    @staticmethod
    def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", MODULE, *args],
            cwd=PACKAGE_ROOT.parents[1],
            text=True,
            capture_output=True,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
