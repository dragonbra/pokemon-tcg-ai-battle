from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rl_environment.wandb_snapshot import (
    SnapshotPolicy,
    upload_snapshot_generation,
    validate_snapshot_files,
)


class FakeWandb:
    def __init__(
        self,
        *,
        fail_on: str | None = None,
        mutate_after: str | None = None,
    ) -> None:
        self.fail_on = fail_on
        self.mutate_after = mutate_after
        self.saved: list[tuple[str, str, str]] = []

    def save(self, path: str, *, policy: str, base_path: str) -> None:
        name = Path(path).name
        if name == self.fail_on:
            raise RuntimeError(f"failed to upload {name}")
        self.saved.append((name, policy, base_path))
        if name == self.mutate_after:
            Path(path).write_text('{"changed": true}', encoding="utf-8")


class WandbSnapshotTests(unittest.TestCase):
    def _write_payloads(self, artifact: Path) -> list[Path]:
        values = {
            "training_summary.json": {"loss": 0.2},
            "status.json": {"state": "synced"},
            "checkpoint_selection.json": {"checkpoint": "best"},
            "model_contract.json": {"schema": "v1"},
            "dataset_reference.json": {"dataset": "sha256:abc"},
            "metrics_snapshot.json": {"epoch": 3},
        }
        paths: list[Path] = []
        for name, value in values.items():
            path = artifact / name
            path.write_text(json.dumps(value), encoding="utf-8")
            paths.append(path)
        return paths

    def test_policy_has_fixed_non_broadenable_exact_allowlist(self) -> None:
        self.assertEqual(
            SnapshotPolicy().allowed_names,
            frozenset(
                {
                    "training_summary.json",
                    "status.json",
                    "checkpoint_selection.json",
                    "model_contract.json",
                    "dataset_reference.json",
                    "metrics_snapshot.json",
                }
            ),
        )
        with self.assertRaises(TypeError):
            SnapshotPolicy(allowed_names=frozenset({"extra.json"}))  # type: ignore[call-arg]

    def test_validation_rejects_outside_root_symlink_wrong_name_directory_oversize_and_secrets(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.mkdir()
            outside = root / "status.json"
            outside.write_text("{}", encoding="utf-8")
            wrong = artifact / "extra.json"
            wrong.write_text("{}", encoding="utf-8")
            linked = artifact / "status.json"
            linked.symlink_to(outside)
            directory_payload = artifact / "training_summary.json"
            directory_payload.mkdir()
            policy = SnapshotPolicy(max_file_bytes=2)

            with self.assertRaisesRegex(ValueError, "outside"):
                validate_snapshot_files(artifact, [outside])
            with self.assertRaisesRegex(ValueError, "symlink"):
                validate_snapshot_files(artifact, [linked])
            with self.assertRaisesRegex(ValueError, "not allowed"):
                validate_snapshot_files(artifact, [wrong])
            with self.assertRaisesRegex(ValueError, "regular file"):
                validate_snapshot_files(artifact, [directory_payload])

            directory_payload.rmdir()
            directory_payload.write_text('{"loss": 0.25}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "too large"):
                validate_snapshot_files(artifact, [directory_payload], policy=policy)

            directory_payload.write_text('{"api_key": "secret"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "secret"):
                validate_snapshot_files(artifact, [directory_payload])

    def test_validation_rejects_aggregate_limit_and_nested_sensitive_raw_and_jsonl_content(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.mkdir()
            summary = artifact / "training_summary.json"
            status = artifact / "status.json"
            summary.write_text('{"nested": {"access_token": "x"}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "secret"):
                validate_snapshot_files(artifact, [summary])

            summary.write_text('{"nested": {"replay": "x"}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw-data"):
                validate_snapshot_files(artifact, [summary])

            summary.write_text('{"metrics": [{"step": 1}\n{"step": 2}]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSONL"):
                validate_snapshot_files(artifact, [summary])

            summary.write_text('{"a": 1}', encoding="utf-8")
            status.write_text('{"b": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "total"):
                validate_snapshot_files(
                    artifact,
                    [summary, status],
                    policy=SnapshotPolicy(max_total_bytes=10),
                )

    def test_uploads_payloads_before_hash_manifest_with_policy_now(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.mkdir()
            payloads = self._write_payloads(artifact)
            manifest = artifact / "wandb_snapshot_manifest.json"
            fake = FakeWandb()

            result = upload_snapshot_generation(fake, artifact, payloads, manifest)
            written = json.loads(manifest.read_text(encoding="utf-8"))
            training_summary_hash = hashlib.sha256(
                (artifact / "training_summary.json").read_bytes()
            ).hexdigest()

        self.assertEqual(result["state"], "synced")
        self.assertEqual([call[0] for call in fake.saved], [path.name for path in payloads] + [manifest.name])
        self.assertTrue(all(call[1] == "now" for call in fake.saved))
        self.assertTrue(all(call[2] == str(artifact) for call in fake.saved))
        self.assertEqual(set(written["payloads"]), {path.name for path in payloads})
        self.assertNotIn(manifest.name, written["payloads"])
        self.assertEqual(
            written["payloads"]["training_summary.json"]["sha256"],
            training_summary_hash,
        )

    def test_invalid_late_payload_or_manifest_causes_zero_uploads(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.mkdir()
            payloads = self._write_payloads(artifact)
            fake = FakeWandb()
            payloads[-1].write_text('{"secret": "x"}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "secret"):
                upload_snapshot_generation(fake, artifact, payloads, artifact / "wandb_snapshot_manifest.json")
            self.assertEqual(fake.saved, [])

            payloads[-1].write_text('{"epoch": 3}', encoding="utf-8")
            invalid_manifest = artifact / "manifest.json"
            with self.assertRaisesRegex(ValueError, "manifest"):
                upload_snapshot_generation(fake, artifact, payloads, invalid_manifest)
            self.assertEqual(fake.saved, [])

    def test_existing_or_directory_manifest_is_rejected_before_uploads(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.mkdir()
            payloads = self._write_payloads(artifact)
            manifest = artifact / "wandb_snapshot_manifest.json"
            fake = FakeWandb()
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                upload_snapshot_generation(fake, artifact, payloads, manifest)
            self.assertEqual(fake.saved, [])

            manifest.unlink()
            manifest.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                upload_snapshot_generation(fake, artifact, payloads, manifest)
            self.assertEqual(fake.saved, [])

    def test_mutated_payload_prevents_manifest_publication(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.mkdir()
            payloads = self._write_payloads(artifact)
            manifest = artifact / "wandb_snapshot_manifest.json"
            fake = FakeWandb(mutate_after="training_summary.json")

            result = upload_snapshot_generation(fake, artifact, payloads, manifest)

            self.assertEqual(result["state"], "failed")
            self.assertIn("changed", result["reason"])
            self.assertFalse(manifest.exists())
            self.assertNotIn(manifest.name, [call[0] for call in fake.saved])

    def test_upload_failures_preserve_local_files_including_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.mkdir()
            payloads = self._write_payloads(artifact)
            manifest = artifact / "wandb_snapshot_manifest.json"

            payload_failure = upload_snapshot_generation(
                FakeWandb(fail_on="model_contract.json"), artifact, payloads, manifest
            )
            self.assertEqual(payload_failure["state"], "failed")
            self.assertTrue(all(path.exists() for path in payloads))
            self.assertFalse(manifest.exists())

            manifest_failure = upload_snapshot_generation(
                FakeWandb(fail_on=manifest.name), artifact, payloads, manifest
            )
            self.assertEqual(manifest_failure["state"], "failed")
            self.assertTrue(manifest.exists())
            self.assertTrue(all(path.exists() for path in payloads))


if __name__ == "__main__":
    unittest.main()
