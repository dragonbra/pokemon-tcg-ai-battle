from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
PYTHON_ROOT = CUDA_ENGINE_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from ptcg_cuda_engine.official_ir import load_and_validate_official_ir  # noqa: E402
from ptcg_cuda_engine.official_provenance import (  # noqa: E402
    canonical_sha256,
    git_value,
    sha256_file,
    source_tree_manifest,
    workspace_relative,
)


DEFAULT_SOURCE = WORKSPACE_ROOT / "engine" / "source" / "ptcgProgram 22"
DEFAULT_OUTPUT = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
DEFAULT_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04"
DEFAULT_ORACLE = WORKSPACE_ROOT / "tmp" / "seeded_cpp_shim_pure_v1" / "libcg_seeded.so"
DEFAULT_CARD_DATA = (
    WORKSPACE_ROOT / "data" / "simulation" / "EN_Card_Data.csv",
    WORKSPACE_ROOT / "data" / "simulation" / "JP_Card_Data.csv",
)
EXTRACTION_MANIFEST_SCHEMA_VERSION = 2
EXPECTED_SOURCE_COMMIT = "3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772"

SEMANTIC_FIELD_COVERAGE: dict[str, Any] = {
    "audited_source_files": ["Card.h", "GameFunction.h", "Skill.h"],
    "CardMaster": {
        "device_fields": [
            "cardId", "cardType", "pokemonType", "evolutionType", "retreatCost",
            "hp", "weakness", "resistance", "energyType", "energyCount", "no",
            "ability", "play", "delay", "attacks",
        ],
        "packed_boolean_fields": 30,
        "normalized_name_fields": ["name", "evolvesFrom", "evolvesFrom2"],
        "excluded_non_runtime_fields": ["code", "nameEn"],
    },
    "Skill": {
        "device_fields": [
            "skillId", "cardId", "skillType", "priority", "firstConditionCount",
            "secondEffectStartIndex", "secondEffectStartIndexEnemy", "triggerStartIndex",
            "areas", "triggers", "effects",
        ],
        "packed_boolean_fields": 8,
        "normalized_name_fields": ["name"],
        "excluded_non_runtime_fields": ["nameEn", "text", "textEn"],
    },
    "Attack": {
        "device_fields": [
            "attackId", "cardId", "damage", "attackFlags", "energies",
            "preEffects", "postEffects", "lastCancelFailAttack",
        ],
        "normalized_name_fields": ["name"],
        "excluded_non_runtime_fields": ["damageText", "nameEn", "text", "textEn"],
    },
    "Effect": {
        "device_fields": [
            "effectType", "effectSelectType", "selectCount", "selectContext", "loopCount",
            "priority", "values", "target", "conditionType", "comparatorType", "failSkip",
            "skillId", "skill", "attack",
        ],
        "packed_boolean_fields": 31,
    },
    "Target": {
        "device_fields": [
            "targetPlayer", "notMe", "skipEnemyTarget", "areas", "conditions",
        ],
    },
    "TargetCondition": {
        "device_fields": ["targetType", "comparatorType", "val", "val2"],
        "normalized_name_fields": ["name"],
        "derived_numeric_sets": [
            "SilcoonOrCascoon", "KoffingOrWeezing", "HonedgeOrDoubladeOrAegislash"
        ],
    },
    "GameFunction": {
        "device_fields": ["functionIndex", "symbol"],
    },
    "unsupported_runtime_fields": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a competition-private typed IR from the read-only official source."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--card-data", type=Path, nargs="*", default=None)
    parser.add_argument("--expected-source-commit", default=EXPECTED_SOURCE_COMMIT)
    parser.add_argument("--expected-cards", type=int, default=1267)
    parser.add_argument("--expected-attacks", type=int, default=1556)
    return parser.parse_args()


def verify_license_boundary(source: Path) -> None:
    license_path = source / "LICENSES" / "LicenseRef-PTCG-ABC-Competition-Use-Only.txt"
    readme_path = source / "README.md"
    if not license_path.is_file() or not readme_path.is_file():
        raise RuntimeError("official competition-only license metadata is missing")
    license_text = license_path.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")
    if "not open-source" not in license_text or "not open-source" not in readme_text:
        raise RuntimeError("official source no longer matches the frozen competition-only boundary")


def docker_image_id(image: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_exporter(source: Path, output: Path, image: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    workspace_mount = f"{WORKSPACE_ROOT.resolve()}:/workspace:ro"
    source_mount = f"{source.resolve()}:/official:ro"
    output_mount = f"{output.resolve()}:/out:rw"
    command = (
        "set -euo pipefail; "
        "g++ -std=c++20 -O2 -Wall -Wextra -rdynamic "
        "-I/official /workspace/engine_cuda/extractor/official_rule_exporter.cpp "
        "-ldl -o /tmp/official_rule_exporter; "
        "/tmp/official_rule_exporter /out/official_rules.json; "
        "g++ --version > /out/compiler.txt"
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            "-v",
            workspace_mount,
            "-v",
            source_mount,
            "-v",
            output_mount,
            image,
            "-lc",
            command,
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    oracle = args.oracle.resolve()
    card_data = [path.resolve() for path in (args.card_data or DEFAULT_CARD_DATA)]
    if not source.is_dir():
        raise SystemExit(f"official source directory does not exist: {source}")
    if not oracle.is_file():
        raise SystemExit(f"frozen official oracle does not exist: {oracle}")
    missing_card_data = [str(path) for path in card_data if not path.is_file()]
    if missing_card_data:
        raise SystemExit(f"card data file(s) do not exist: {missing_card_data}")
    verify_license_boundary(source)
    tree_hash, source_files = source_tree_manifest(source)
    source_commit = git_value(source, "rev-parse", "HEAD")
    if source_commit != args.expected_source_commit:
        raise RuntimeError(
            "official source commit mismatch: "
            f"expected={args.expected_source_commit} actual={source_commit}"
        )
    run_exporter(source, output, args.image)

    rules_path = output / "official_rules.json"
    _, summary = load_and_validate_official_ir(rules_path)
    if summary.cards != args.expected_cards:
        raise RuntimeError(
            f"card count changed: expected {args.expected_cards}, extracted {summary.cards}"
        )
    if summary.attacks != args.expected_attacks:
        raise RuntimeError(
            f"attack count changed: expected {args.expected_attacks}, extracted {summary.attacks}"
        )

    compiler_path = output / "compiler.txt"
    exporter_path = CUDA_ENGINE_ROOT / "extractor" / "official_rule_exporter.cpp"
    audited_source_hashes = {
        relative: next(row["sha256"] for row in source_files if row["path"] == relative)
        for relative in ("Card.h", "Skill.h", "GameFunction.h")
    }
    semantic_field_coverage = dict(SEMANTIC_FIELD_COVERAGE)
    semantic_field_coverage["audited_source_sha256"] = audited_source_hashes
    semantic_field_coverage["audit_contract_sha256"] = canonical_sha256(
        semantic_field_coverage
    )
    manifest = {
        "schema_version": EXTRACTION_MANIFEST_SCHEMA_VERSION,
        "competition_private": True,
        "redistribution_allowed": False,
        "source_relpath": workspace_relative(source, WORKSPACE_ROOT),
        "source_git_commit": source_commit,
        "source_tree_sha256": tree_hash,
        "source_files": source_files,
        "official_oracle": {
            "relpath": workspace_relative(oracle, WORKSPACE_ROOT),
            "bytes": oracle.stat().st_size,
            "sha256": sha256_file(oracle),
        },
        "card_data": [
            {
                "relpath": workspace_relative(path, WORKSPACE_ROOT),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in card_data
        ],
        "exporter_sha256": sha256_file(exporter_path),
        "extractor_git_commit": git_value(WORKSPACE_ROOT, "rev-parse", "HEAD"),
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "compiler_sha256": sha256_file(compiler_path),
        "semantic_field_coverage": semantic_field_coverage,
        "payload_file": rules_path.name,
        "payload_sha256": sha256_file(rules_path),
        "summary": summary.to_dict(),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                "source_tree_sha256": tree_hash,
                "payload_sha256": manifest["payload_sha256"],
                "summary": summary.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
