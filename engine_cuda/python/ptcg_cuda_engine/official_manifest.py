from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .official_ir import load_and_validate_official_ir
from .official_provenance import (
    canonical_sha256,
    git_value,
    require_workspace_relative,
    sha256_file,
    source_tree_manifest,
)
from .official_rule_pack import (
    compile_official_rule_pack,
    validate_official_rule_pack,
)


EXTRACTION_MANIFEST_SCHEMA_VERSION = 2
RULE_PACK_MANIFEST_SCHEMA_VERSION = 2


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _private_marker(manifest: dict[str, Any], path: Path) -> None:
    if manifest.get("competition_private") is not True:
        raise ValueError(f"{path} is missing competition_private=true")
    if manifest.get("redistribution_allowed") is not False:
        raise ValueError(f"{path} does not prohibit redistribution")


def _workspace_path(
    value: Any,
    *,
    field: str,
    workspace: Path,
    override: Path | None = None,
) -> Path:
    recorded = require_workspace_relative(value, field)
    path = (override if override is not None else workspace / recorded).resolve()
    if not path.exists():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def _verify_docker_image(manifest: dict[str, Any]) -> None:
    image = manifest.get("docker_image")
    expected = manifest.get("docker_image_id")
    if not isinstance(image, str) or not image or not isinstance(expected, str) or not expected:
        raise ValueError("Docker image provenance is incomplete")
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Docker image cannot be inspected: {image}") from error
    actual = result.stdout.strip()
    if actual != expected:
        raise ValueError(f"Docker image ID mismatch: recorded={expected} actual={actual}")


def verify_extraction_manifest(
    directory: str | Path,
    *,
    workspace: str | Path,
    source: Path | None = None,
    oracle: Path | None = None,
    card_data: Sequence[Path] | None = None,
    verify_docker: bool = True,
) -> dict[str, Any]:
    directory_path = Path(directory).resolve()
    manifest_path = directory_path / "manifest.json"
    manifest = _read_json(manifest_path)
    _private_marker(manifest, manifest_path)
    if manifest.get("schema_version") != EXTRACTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported extraction manifest schema")

    workspace_path = Path(workspace).resolve()
    source_path = _workspace_path(
        manifest.get("source_relpath"),
        field="source_relpath",
        workspace=workspace_path,
        override=source,
    )
    source_hash, source_files = source_tree_manifest(source_path)
    if source_hash != manifest.get("source_tree_sha256"):
        raise ValueError("official source tree hash mismatch")
    if source_files != manifest.get("source_files"):
        raise ValueError("official source file manifest mismatch")
    source_commit = manifest.get("source_git_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("source_git_commit is missing or malformed")
    actual_source_commit = git_value(source_path, "rev-parse", "HEAD")
    if actual_source_commit and actual_source_commit != source_commit:
        raise ValueError(
            "official source commit mismatch: "
            f"recorded={source_commit} actual={actual_source_commit}"
        )

    coverage = manifest.get("semantic_field_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("semantic_field_coverage is missing")
    if coverage.get("unsupported_runtime_fields") != []:
        raise ValueError("unsupported runtime fields are present")
    recorded_contract_hash = coverage.get("audit_contract_sha256")
    contract = dict(coverage)
    contract.pop("audit_contract_sha256", None)
    if canonical_sha256(contract) != recorded_contract_hash:
        raise ValueError("semantic field audit contract hash mismatch")
    for relative, expected_hash in coverage.get("audited_source_sha256", {}).items():
        header = source_path / relative
        if not header.is_file() or sha256_file(header) != expected_hash:
            raise ValueError(f"audited source header hash mismatch: {relative}")

    oracle_record = manifest.get("official_oracle")
    if not isinstance(oracle_record, dict):
        raise ValueError("official_oracle provenance is missing")
    oracle_path = _workspace_path(
        oracle_record.get("relpath"),
        field="official_oracle.relpath",
        workspace=workspace_path,
        override=oracle,
    )
    if oracle_path.stat().st_size != oracle_record.get("bytes"):
        raise ValueError("official oracle byte count mismatch")
    if sha256_file(oracle_path) != oracle_record.get("sha256"):
        raise ValueError("official oracle hash mismatch")

    card_records = manifest.get("card_data")
    if not isinstance(card_records, list) or not card_records:
        raise ValueError("card_data provenance is missing")
    card_paths = list(card_data) if card_data is not None else [None] * len(card_records)
    if len(card_paths) != len(card_records):
        raise ValueError("card_data override count does not match manifest")
    for index, record in enumerate(card_records):
        if not isinstance(record, dict):
            raise ValueError(f"card_data[{index}] is malformed")
        path = _workspace_path(
            record.get("relpath"),
            field=f"card_data[{index}].relpath",
            workspace=workspace_path,
            override=card_paths[index],
        )
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"card_data[{index}] hash or byte count mismatch")

    payload_name = manifest.get("payload_file")
    if not isinstance(payload_name, str) or Path(payload_name).name != payload_name:
        raise ValueError("payload_file must be a file name")
    payload_path = directory_path / payload_name
    if sha256_file(payload_path) != manifest.get("payload_sha256"):
        raise ValueError("typed IR payload hash mismatch")
    payload, ir_summary = load_and_validate_official_ir(payload_path)
    if ir_summary.to_dict() != manifest.get("summary"):
        raise ValueError("typed IR summary mismatch")

    exporter_path = workspace_path / "engine_cuda" / "extractor" / "official_rule_exporter.cpp"
    if sha256_file(exporter_path) != manifest.get("exporter_sha256"):
        raise ValueError("exporter source hash mismatch")
    compiler_path = directory_path / "compiler.txt"
    if sha256_file(compiler_path) != manifest.get("compiler_sha256"):
        raise ValueError("compiler identity hash mismatch")
    if verify_docker:
        _verify_docker_image(manifest)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "payload": payload,
        "summary": ir_summary.to_dict(),
        "source": source_path,
        "oracle": oracle_path,
    }


def verify_rule_pack_manifest(
    manifest_path: str | Path,
    *,
    workspace: str | Path,
    source: Path | None = None,
    oracle: Path | None = None,
    card_data: Sequence[Path] | None = None,
    verify_docker: bool = True,
) -> dict[str, Any]:
    binary_manifest_path = Path(manifest_path).resolve()
    binary_manifest = _read_json(binary_manifest_path)
    _private_marker(binary_manifest, binary_manifest_path)
    if binary_manifest.get("schema_version") != RULE_PACK_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported rule-pack manifest schema")
    extraction = verify_extraction_manifest(
        binary_manifest_path.parent,
        workspace=workspace,
        source=source,
        oracle=oracle,
        card_data=card_data,
        verify_docker=verify_docker,
    )
    extraction_manifest_path = extraction["manifest_path"]
    if sha256_file(extraction_manifest_path) != binary_manifest.get("extraction_manifest_sha256"):
        raise ValueError("rule-pack extraction manifest hash mismatch")
    payload_path = binary_manifest_path.with_name(
        binary_manifest_path.name.removesuffix(".manifest.json")
    )
    if sha256_file(payload_path) != binary_manifest.get("output_sha256"):
        raise ValueError("rule-pack output hash mismatch")
    data = payload_path.read_bytes()
    pack_summary = validate_official_rule_pack(data)
    expected_pack = binary_manifest.get("rule_pack")
    if expected_pack != {
        **pack_summary.to_dict(),
        "source_ir_sha256": extraction["summary"]["canonical_sha256"],
    }:
        raise ValueError("rule-pack binary summary mismatch")
    rebuilt, rebuilt_summary, ir_summary = compile_official_rule_pack(extraction["payload"])
    if rebuilt != data:
        raise ValueError("rule-pack is not the deterministic compilation of typed IR")
    if rebuilt_summary.to_dict() != expected_pack or ir_summary.to_dict() != extraction["summary"]:
        raise ValueError("rule-pack rebuild summary mismatch")
    if binary_manifest.get("input_sha256") != extraction["manifest"].get("payload_sha256"):
        raise ValueError("rule-pack input hash does not match extraction manifest")
    return {
        "manifest": binary_manifest,
        "manifest_path": binary_manifest_path,
        "rule_pack": expected_pack,
        "extraction": extraction,
    }
