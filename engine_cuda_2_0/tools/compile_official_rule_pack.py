from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = CUDA_ENGINE_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from ptcg_cuda_engine.official_ir import load_and_validate_official_ir  # noqa: E402
from ptcg_cuda_engine.official_provenance import sha256_file  # noqa: E402
from ptcg_cuda_engine.official_rule_pack import write_official_rule_pack  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile private official IR to a device rule pack.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    payload, ir_summary = load_and_validate_official_ir(input_path)
    summary = write_official_rule_pack(output_path, payload)
    extraction_manifest_path = input_path.with_name("manifest.json")
    if not extraction_manifest_path.is_file():
        raise RuntimeError(
            f"typed IR manifest is required beside input: {extraction_manifest_path}"
        )
    extraction_manifest = json.loads(
        extraction_manifest_path.read_text(encoding="utf-8")
    )
    if extraction_manifest.get("schema_version") != 2:
        raise RuntimeError("typed IR manifest schema 2 is required")
    if extraction_manifest.get("payload_sha256") != sha256_file(input_path):
        raise RuntimeError("typed IR manifest payload hash does not match input")
    manifest = {
        "schema_version": 2,
        "competition_private": True,
        "redistribution_allowed": False,
        "input_sha256": sha256_file(input_path),
        "input_canonical_sha256": ir_summary.canonical_sha256,
        "output_sha256": sha256_file(output_path),
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "official_oracle": extraction_manifest["official_oracle"],
        "source_git_commit": extraction_manifest["source_git_commit"],
        "source_tree_sha256": extraction_manifest["source_tree_sha256"],
        "exporter_sha256": extraction_manifest["exporter_sha256"],
        "extractor_git_commit": extraction_manifest["extractor_git_commit"],
        "card_data": extraction_manifest["card_data"],
        "rule_pack": summary.to_dict(),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), **summary.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
