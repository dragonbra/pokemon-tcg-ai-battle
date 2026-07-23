from __future__ import annotations

import json
import hashlib
import tempfile
import tarfile
import unittest
from pathlib import Path

from train.kaggle_bc_top20.import_candidate import validate_archive_members
from train.kaggle_bc_top20.prepare_input import select_campaign_jobs
from train.kaggle_bc_top20.render_kernel import render_kernel
from train.kaggle_bc_top20.worker import (
    PREPARE_RESULT_SCHEMA,
    _discover_mounted_replay_datasets,
    _load_prebuilt_dataset,
    choose_rescue_side,
    gate_from_summary,
    select_attempt,
)
from train.kaggle_bc_top20.training.build_daily_winner_bc_dataset import (
    _is_winner,
    _normalize_team,
    _reference_entity_count,
    _scan_list,
    _valid_action,
    _visual_frames,
)


def _summary(
    *,
    validation_exact: float,
    validation_loss: float = 1.0,
    train_exact: float = 0.8,
    test_legal: float | None = None,
) -> dict[str, object]:
    test = {} if test_legal is None else {"legal_action_rate": test_legal}
    return {
        "best_validation_exact_action_rate": validation_exact,
        "best_epoch": 4,
        "best_epoch_metrics": {
            "validation/exact_action_rate": validation_exact,
            "validation/multi_action_exact_rate": 0.7,
            "validation/selection_count_accuracy": 0.99,
            "validation/legal_action_rate": 1.0,
            "validation/policy_loss": validation_loss,
            "train/exact_action_rate": train_exact,
        },
        "final_test": test,
        "checkpoint": "/tmp/checkpoint.pt",
        "runtime": {
            "train_function_seconds": 12.0,
            "peak_gpu_memory_bytes": 123,
            "parameter_count": 456,
        },
    }


class KaggleBCGateTests(unittest.TestCase):
    def test_gate_is_validation_first_until_test_is_evaluated(self) -> None:
        validation_only = gate_from_summary(
            _summary(validation_exact=0.8),
            attempt_id="V1_shared_config",
            checkpoint_sha256="a" * 64,
        )
        self.assertTrue(validation_only["validation_gate_passed"])
        self.assertFalse(validation_only["test_evaluated"])
        self.assertFalse(validation_only["offline_gate_passed"])

        with_test = gate_from_summary(
            _summary(validation_exact=0.8, test_legal=1.0),
            attempt_id="V1_shared_config",
            checkpoint_sha256="a" * 64,
        )
        self.assertTrue(with_test["offline_gate_passed"])

    def test_selection_prefers_passing_attempt_then_loss_inside_tie(self) -> None:
        failing = gate_from_summary(
            _summary(validation_exact=0.9),
            attempt_id="V1",
            checkpoint_sha256="1" * 64,
        )
        failing["validation_gate_passed"] = False
        passing_a = gate_from_summary(
            _summary(validation_exact=0.801, validation_loss=0.8),
            attempt_id="V2",
            checkpoint_sha256="2" * 64,
        )
        passing_b = gate_from_summary(
            _summary(validation_exact=0.8, validation_loss=0.7),
            attempt_id="V3",
            checkpoint_sha256="3" * 64,
        )
        decision = select_attempt([failing, passing_a, passing_b])
        self.assertEqual(decision["selected_attempt_id"], "V3")
        self.assertTrue(decision["passing_attempt_preferred"])

    def test_rescue_uses_lower_lr_for_clear_overfit_gap(self) -> None:
        gate = gate_from_summary(
            _summary(validation_exact=0.6, train_exact=0.8),
            attempt_id="V1",
            checkpoint_sha256="a" * 64,
        )
        curve = [
            {
                "validation/exact_action_rate": 0.55 + index * 0.01,
                "train/exact_action_rate": 0.65 + index * 0.02,
            }
            for index in range(5)
        ]
        side, evidence = choose_rescue_side(gate, curve)
        self.assertEqual(side, "lower")
        self.assertGreater(evidence["train_validation_gap"], 0.12)


class KaggleBCInputTests(unittest.TestCase):
    def test_daily_winner_selector_matches_reference_contract(self) -> None:
        self.assertEqual(_normalize_team("  YUSHIN   Ito "), "yushin ito")
        self.assertTrue(_is_winner([-1, 1], 1))
        self.assertFalse(_is_winner([1, -1], 1))
        select = {"option": [{}, {}, {}], "minCount": 1, "maxCount": 2}
        self.assertEqual(_valid_action([2, 0], select), [0, 2])
        self.assertIsNone(_valid_action([0, 0], select))
        payload = {
            "steps": [
                [{"visualize": [{"obs": {"step": 1}}]}],
                [{"visualize": [{"obs": {"step": 1}}, {"obs": {"step": 2}}]}],
            ]
        }
        self.assertEqual(len(_visual_frames(payload)), 2)
        observation = {
            "current": {
                "players": [
                    {"active": [{"cardId": 1, "energyCards": [2]}]},
                    {"bench": [{"cardId": 3}]},
                ]
            },
            "select": {"deck": [4]},
        }
        self.assertEqual(_reference_entity_count(observation, 0), 4)

    def test_daily_header_scan_handles_whitespace_without_full_trace_parse(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ptcg-daily-header-test-") as temporary:
            path = Path(temporary) / "episode.json"
            path.write_text(
                '{"info": {"TeamNames": ["Yushin Ito", "Other"]}, '
                '"rewards": [1, -1], "steps": ["unneeded"]}',
                encoding="utf-8",
            )
            self.assertEqual(_scan_list(path, "info.TeamNames"), ["Yushin Ito", "Other"])
            self.assertEqual(_scan_list(path, "rewards"), [1, -1])

    def test_prebuilt_dataset_requires_hashes_and_exact_source_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ptcg-kaggle-prebuilt-test-") as temporary:
            root = Path(temporary)
            dataset = root / "dataset/dataset.jsonl"
            dataset.parent.mkdir()
            dataset.write_text('{"record": 1}\n', encoding="utf-8")
            dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
            job = {
                "schema_version": "ptcg_kaggle_single_expert_job_v1",
                "selected_roster_order": 3,
                "source_identity": {
                    "team_id": 1,
                    "submission_id": 2,
                    "deck_sha256": "d" * 64,
                },
            }
            data_manifest = dataset.with_suffix(".jsonl.data_manifest.json")
            data_manifest.write_text(
                json.dumps(
                    {
                        "dataset_sha256": dataset_hash,
                        "source_identity": job["source_identity"],
                    }
                ),
                encoding="utf-8",
            )
            file_hashes = {
                path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (dataset, data_manifest)
            }
            result = {
                "schema_version": PREPARE_RESULT_SCHEMA,
                "status": "dataset_ready",
                "job": job,
                "dataset": {
                    "relative_path": "dataset/dataset.jsonl",
                    "sha256": dataset_hash,
                    "bytes": dataset.stat().st_size,
                    "audit": {"status": "passed", "dataset_sha256": dataset_hash},
                },
                "files_sha256": file_hashes,
            }
            (root / "PREPARE_RESULT.json").write_text(
                json.dumps(result), encoding="utf-8"
            )

            loaded, audit, _ = _load_prebuilt_dataset(root, expected_job=job)
            self.assertEqual(loaded, dataset.resolve())
            self.assertEqual(audit["status"], "passed")
            mismatched = {**job, "selected_roster_order": 4}
            with self.assertRaisesRegex(ValueError, "job identity"):
                _load_prebuilt_dataset(root, expected_job=mismatched)

    def test_mounted_replay_discovery_supports_versioned_kaggle_layout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ptcg-kaggle-bc-mount-test-") as temporary:
            root = Path(temporary)
            daily = (
                root
                / "datasets/kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22"
            )
            daily.mkdir(parents=True)
            (daily / "87364041.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_discover_mounted_replay_datasets(root), [daily.resolve()])

    def test_candidate_archive_rejects_wrapping_directory(self) -> None:
        members = [
            tarfile.TarInfo("wrapped/main.py"),
            tarfile.TarInfo("wrapped/deck.csv"),
        ]
        with self.assertRaisesRegex(ValueError, "unexpected candidate archive root"):
            validate_archive_members(members)

    def test_candidate_archive_requires_direct_package_roots(self) -> None:
        members = [
            tarfile.TarInfo("main.py"),
            tarfile.TarInfo("deck.csv"),
            tarfile.TarInfo("cg/api.py"),
            tarfile.TarInfo("strategy/model.bin"),
        ]
        self.assertEqual(validate_archive_members(members), members)

    def test_completed_local_jobs_are_excluded(self) -> None:
        sources = [
            {"selected_roster_order": 3, "v1_complete": False},
            {"selected_roster_order": 1, "v1_complete": True},
            {"selected_roster_order": 2, "v1_complete": True},
            {"selected_roster_order": 4, "v1_complete": False},
        ]
        jobs = select_campaign_jobs(sources)
        self.assertEqual(
            [row["selected_roster_order"] for row in jobs],
            [3, 4],
        )

    def test_notebook_and_metadata_are_valid_json(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        for path in (
            repository_root / "notebooks/kaggle_bc_worker/ptcg_single_expert_bc.ipynb",
            repository_root / "notebooks/kaggle_bc_worker/kernel-metadata.json",
        ):
            with self.subTest(path=path):
                self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_order_filter_fails_closed_for_completed_job(self) -> None:
        sources = [
            {"selected_roster_order": 1, "v1_complete": True},
            {"selected_roster_order": 3, "v1_complete": False},
        ]
        with self.assertRaisesRegex(ValueError, "unavailable or already complete"):
            select_campaign_jobs(sources, include_orders={1})

    def test_rendered_kernel_has_an_independent_job_slug(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="ptcg-kaggle-bc-test-") as temporary:
            output = Path(temporary) / "kernel"
            result = render_kernel(
                template_root=repository_root / "notebooks/kaggle_bc_worker",
                output=output,
                owner="tommycyd",
                dataset_source="tommycyd/pokemon-tcg-bc-cloud-input",
                replay_dataset_sources=[
                    "kaggle/pokemon-tcg-ai-battle-episodes-2026-07-21",
                    "kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22",
                ],
                job_order=7,
                retention="package",
            )
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            notebook = json.loads((output / "ptcg_single_expert_bc.ipynb").read_text())
            source = "".join(notebook["cells"][1]["source"])
            self.assertEqual(result["kernel"], "tommycyd/pokemon-tcg-bc-job-07")
            self.assertEqual(
                metadata["dataset_sources"],
                [
                    "tommycyd/pokemon-tcg-bc-cloud-input",
                    "kaggle/pokemon-tcg-ai-battle-episodes-2026-07-21",
                    "kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22",
                ],
            )
            self.assertIn("JOB_ORDER = 7", source)
            self.assertIn('RETENTION = "package"', source)

    def test_daily_winner_kernel_is_offline_and_records_corpus_mode(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="ptcg-kaggle-winner-test-") as temporary:
            output = Path(temporary) / "kernel"
            result = render_kernel(
                template_root=repository_root / "notebooks/kaggle_bc_worker",
                output=output,
                owner="tommycyd",
                dataset_source="tommycyd/pokemon-tcg-bc-cloud-input",
                replay_dataset_sources=[
                    "kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22"
                ],
                job_order=4,
                corpus_mode="daily_team_winners",
            )
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            notebook = json.loads((output / "ptcg_single_expert_bc.ipynb").read_text())
            source = "".join(notebook["cells"][1]["source"])
            dependencies = next(
                cell for cell in notebook["cells"] if cell.get("id") == "dependencies"
            )
            self.assertEqual(result["corpus_mode"], "daily_team_winners")
            self.assertIs(metadata["enable_internet"], False)
            self.assertIn('CORPUS_MODE = "daily_team_winners"', source)
            self.assertNotIn("pip install", "".join(dependencies["source"]))

    def test_rendered_kernel_can_embed_private_input_without_dataset_source(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="ptcg-kaggle-bc-embed-test-") as temporary:
            root = Path(temporary)
            archive = root / "input.tar.gz"
            archive.write_bytes(b"frozen private input")
            output = root / "kernel"
            result = render_kernel(
                template_root=repository_root / "notebooks/kaggle_bc_worker",
                output=output,
                owner="tommycyd",
                dataset_source=None,
                input_archive=archive,
                replay_dataset_sources=[
                    "kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22"
                ],
                job_order=3,
            )
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            notebook = json.loads((output / "ptcg_single_expert_bc.ipynb").read_text())
            source = "".join(notebook["cells"][1]["source"])

            self.assertEqual(result["input_mode"], "embedded_archive")
            self.assertEqual(
                metadata["dataset_sources"],
                ["kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22"],
            )
            self.assertIn("ZnJvemVuIHByaXZhdGUgaW5wdXQ=", source)
            self.assertIn(result["input_archive_sha256"], source)

    def test_renderer_builds_cpu_prepare_and_gpu_train_kernel_pair(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="ptcg-kaggle-bc-pair-test-") as temporary:
            root = Path(temporary)
            archive = root / "input.tar.gz"
            archive.write_bytes(b"frozen private input")
            prepare_output = root / "prepare"
            prepare = render_kernel(
                template_root=repository_root / "notebooks/kaggle_bc_worker",
                output=prepare_output,
                owner="tommycyd",
                dataset_source=None,
                input_archive=archive,
                replay_dataset_sources=[
                    "kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22"
                ],
                job_order=3,
                mode="prepare",
            )
            prepare_metadata = json.loads(
                (prepare_output / "kernel-metadata.json").read_text()
            )
            self.assertEqual(prepare["accelerator"], "cpu")
            self.assertIs(prepare_metadata["enable_gpu"], False)
            self.assertNotIn("machine_shape", prepare_metadata)

            train_output = root / "train"
            train = render_kernel(
                template_root=repository_root / "notebooks/kaggle_bc_worker",
                output=train_output,
                owner="tommycyd",
                dataset_source=None,
                input_archive=archive,
                kernel_sources=[prepare["kernel"]],
                job_order=3,
                mode="train",
                retention="package",
            )
            train_metadata = json.loads(
                (train_output / "kernel-metadata.json").read_text()
            )
            self.assertEqual(train["accelerator"], "NvidiaTeslaT4")
            self.assertEqual(train_metadata["dataset_sources"], [])
            self.assertEqual(train_metadata["kernel_sources"], [prepare["kernel"]])


if __name__ == "__main__":
    unittest.main()
