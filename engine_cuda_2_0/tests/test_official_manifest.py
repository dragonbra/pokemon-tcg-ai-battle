from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from engine_cuda_2_0.tests.test_official_ir import minimal_payload  # noqa: E402
from ptcg_cuda_engine.official_ir import validate_official_ir  # noqa: E402
from ptcg_cuda_engine.official_manifest import (  # noqa: E402
    verify_extraction_manifest,
    verify_rule_pack_manifest,
)
from ptcg_cuda_engine.official_provenance import (  # noqa: E402
    canonical_sha256,
    git_value,
    sha256_file,
    source_tree_manifest,
)
from ptcg_cuda_engine.official_rule_pack import compile_official_rule_pack  # noqa: E402


FIXTURE_SOURCE_COMMIT = git_value(WORKSPACE_ROOT, "rev-parse", "HEAD") or "a" * 40


class OfficialManifestTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        source = root / "fixture-source"
        source.mkdir()
        header = source / "header.h"
        header.write_text("struct Fixture {};\n", encoding="utf-8")
        oracle = root / "oracle.bin"
        oracle.write_bytes(b"frozen-oracle")
        card_data = root / "cards.csv"
        card_data.write_text("id,name\n1,fixture\n", encoding="utf-8")

        payload = minimal_payload()
        payload_path = root / "official_rules.json"
        payload_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        ir_summary = validate_official_ir(payload)
        source_hash, source_files = source_tree_manifest(source)
        field_contract = {
            "unsupported_runtime_fields": [],
            "audited_source_sha256": {"header.h": sha256_file(header)},
        }
        field_contract["audit_contract_sha256"] = canonical_sha256(field_contract)
        exporter = CUDA_ENGINE_ROOT / "extractor" / "official_rule_exporter.cpp"
        compiler = root / "compiler.txt"
        compiler.write_text("fixture compiler\n", encoding="utf-8")
        manifest = {
            "schema_version": 2,
            "competition_private": True,
            "redistribution_allowed": False,
            "source_relpath": source.relative_to(WORKSPACE_ROOT).as_posix(),
            "source_git_commit": FIXTURE_SOURCE_COMMIT,
            "source_tree_sha256": source_hash,
            "source_files": source_files,
            "official_oracle": {
                "relpath": oracle.relative_to(WORKSPACE_ROOT).as_posix(),
                "bytes": oracle.stat().st_size,
                "sha256": sha256_file(oracle),
            },
            "card_data": [{
                "relpath": card_data.relative_to(WORKSPACE_ROOT).as_posix(),
                "bytes": card_data.stat().st_size,
                "sha256": sha256_file(card_data),
            }],
            "exporter_sha256": sha256_file(exporter),
            "extractor_git_commit": "b" * 40,
            "docker_image": "fixture-image",
            "docker_image_id": "fixture-image-id",
            "compiler_sha256": sha256_file(compiler),
            "semantic_field_coverage": field_contract,
            "payload_file": payload_path.name,
            "payload_sha256": sha256_file(payload_path),
            "summary": ir_summary.to_dict(),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

        binary, pack_summary, _ = compile_official_rule_pack(payload)
        binary_path = root / "official_rules.bin"
        binary_path.write_bytes(binary)
        binary_manifest = {
            "schema_version": 2,
            "competition_private": True,
            "redistribution_allowed": False,
            "input_sha256": sha256_file(payload_path),
            "input_canonical_sha256": ir_summary.canonical_sha256,
            "output_sha256": sha256_file(binary_path),
            "extraction_manifest_sha256": sha256_file(manifest_path),
            "official_oracle": manifest["official_oracle"],
            "source_git_commit": manifest["source_git_commit"],
            "source_tree_sha256": manifest["source_tree_sha256"],
            "exporter_sha256": manifest["exporter_sha256"],
            "extractor_git_commit": manifest["extractor_git_commit"],
            "card_data": manifest["card_data"],
            "rule_pack": pack_summary.to_dict(),
        }
        binary_manifest_path = root / "official_rules.bin.manifest.json"
        binary_manifest_path.write_text(
            json.dumps(binary_manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest_path, binary_manifest_path

    def test_extraction_manifest_and_rule_pack_are_verified(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as temporary:
            manifest_path, binary_manifest_path = self._write_fixture(Path(temporary))
            extraction = verify_extraction_manifest(
                manifest_path.parent,
                workspace=WORKSPACE_ROOT,
                verify_docker=False,
            )
            self.assertEqual(extraction["summary"]["canonical_sha256"], validate_official_ir(minimal_payload()).canonical_sha256)
            verified = verify_rule_pack_manifest(
                binary_manifest_path,
                workspace=WORKSPACE_ROOT,
                verify_docker=False,
            )
            self.assertEqual(verified["rule_pack"]["total_bytes"], 1040)

    def test_tampered_field_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as temporary:
            manifest_path, _ = self._write_fixture(Path(temporary))
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["semantic_field_coverage"]["unsupported_runtime_fields"] = ["future_field"]
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported runtime fields"):
                verify_extraction_manifest(
                    manifest_path.parent,
                    workspace=WORKSPACE_ROOT,
                    verify_docker=False,
                )

    def test_tampered_binary_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as temporary:
            _, binary_manifest_path = self._write_fixture(Path(temporary))
            binary_path = binary_manifest_path.with_name("official_rules.bin")
            data = bytearray(binary_path.read_bytes())
            data[-1] ^= 1
            binary_path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "rule-pack output hash mismatch"):
                verify_rule_pack_manifest(
                    binary_manifest_path,
                    workspace=WORKSPACE_ROOT,
                    verify_docker=False,
                )


if __name__ == "__main__":
    unittest.main()
