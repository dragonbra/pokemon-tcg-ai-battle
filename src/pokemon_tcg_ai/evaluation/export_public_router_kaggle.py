"""Export the accepted compact Deck-007 public router for Kaggle."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any

import torch

from ..assets import AssetRegistry, sha256_file
from ..semantic_runtime.deployment.public_meta_memory import (
    DEFAULT_UPDATE,
    PUBLIC_POLICY_ID,
    ROUTED_UPDATES,
    RULE_MANIFEST,
)
from .public_router_candidate import materialize_compact_heads
from .run_public_meta_router_v1_cuda2048 import validate_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "semantic_runtime"
PACKAGE_SCHEMA = "0045_public_meta_router_kaggle_package_v1"
PACKAGE_NAME = "0045_dragapult_ex_007_public_meta_router_v3_compact_fp16_storage_fp32_runtime"
OUTPUT = ROOT / "archive/submission" / PACKAGE_NAME
ARCHIVE = ROOT / "archive/submission/dist" / f"{PACKAGE_NAME}.tar.gz"
V12_ROOT = (
    ROOT / "runs/versions/"
    "V12_dragapult_007_public_meta_router_v3_grass_fold_u200_cuda2048/"
    "artifact/public_router_evaluation"
)
QUALIFYING_REPORT = V12_ROOT / "report.json"
PORTABLE_CANDIDATES = {
    update: V12_ROOT / f"materialization/update-{update:06d}/model.bin"
    for update in ROUTED_UPDATES
}


MAIN = '''"""Kaggle entrypoint for the immutable 0045 public expert router."""
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(globals().get("__file__", Path.cwd())).resolve()
if ROOT.is_file():
    ROOT = ROOT.parent
if not (ROOT / "deck.csv").is_file() and Path("/kaggle_simulations/agent/deck.csv").is_file():
    ROOT = Path("/kaggle_simulations/agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch  # noqa: E402,F401

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _manifest():
    payload = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != "0045_public_meta_router_kaggle_package_v1":
        raise RuntimeError("0045 public-router Kaggle package schema mismatch")
    expected = payload.get("package_file_sha256")
    actual = {
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if not isinstance(expected, dict) or actual != set(expected):
        raise RuntimeError("0045 public-router package file inventory mismatch")
    mismatched = [name for name, digest in expected.items() if _sha256(ROOT / name) != digest]
    if mismatched:
        raise RuntimeError(f"0045 public-router package hash mismatch: {mismatched}")
    return payload

PACKAGE_MANIFEST = _manifest()
from strategy.deployment.public_meta_router import PublicRoutedCompoundPolicy  # noqa: E402
DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
POLICY = PublicRoutedCompoundPolicy.from_compact_checkpoint(
    ROOT / "strategy/model.bin", ROOT / "strategy/router_heads.bin", DECK
)
if POLICY.deployment_identity.get("composite_effective_sha256") != PACKAGE_MANIFEST.get("composite_effective_sha256"):
    raise RuntimeError("0045 public-router runtime/package composite mismatch")

def read_deck_csv():
    return list(DECK)

def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
'''


def _file_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _validated_opponent_id(report: dict[str, Any]) -> str:
    audit = report.get("opponent_policy_identity_audit") or {}
    policy_id = audit.get("requested_policy_id")
    if audit.get("status") != "PASS" or policy_id != "Policy-0809":
        raise RuntimeError("qualifying report does not identify complete Policy-0809")
    return str(policy_id)


def _copy_official_runtime(output: Path) -> None:
    source = next((ROOT / "evaluation/arena/opponents").glob("*/cg"), None)
    if source is None:
        raise FileNotFoundError("official packaged cg runtime is unavailable")
    shutil.copytree(
        source, output / "cg",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _deterministic_archive(source: Path, archive: Path) -> None:
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as handle:
                for path in sorted(source.rglob("*")):
                    if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                        continue
                    info = handle.gettarinfo(path, arcname=str(path.relative_to(source)))
                    info.mtime = 0; info.uid = 0; info.gid = 0
                    info.uname = ""; info.gname = ""
                    with path.open("rb") as stream:
                        handle.addfile(info, stream)
    temporary.replace(archive)


def export(*, output: Path = OUTPUT, archive: Path = ARCHIVE) -> dict[str, Any]:
    if output.exists() or archive.exists():
        raise FileExistsError("formal public-router package output already exists")
    if any(not path.is_file() for path in (*PORTABLE_CANDIDATES.values(), QUALIFYING_REPORT)):
        raise FileNotFoundError("V12 qualifying artifacts are incomplete")
    report = json.loads(QUALIFYING_REPORT.read_text(encoding="utf-8"))
    validate_report(report)
    audit = report["focal_public_router_identity_audit"]
    registry = AssetRegistry.load(PROJECT_ROOT); registry.validate_all()
    asset = next(row for row in registry.decks if row.deck_id == "007")
    deck = tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
    output.mkdir(parents=True)
    try:
        shutil.copytree(
            RUNTIME_ROOT, output / "strategy",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runtime_manifest.json"),
        )
        shutil.copy2(PORTABLE_CANDIDATES[DEFAULT_UPDATE], output / "strategy/model.bin")
        compact_audit = materialize_compact_heads(
            portable_candidates=PORTABLE_CANDIDATES,
            qualifying_report=report,
            output=output / "strategy/router_heads.bin",
        )
        if compact_audit["composite_effective_sha256"] != audit["composite_effective_sha256"]:
            raise RuntimeError("package compact identity differs from qualifying CUDA report")
        _copy_official_runtime(output)
        (output / "deck.csv").write_text(
            "".join(f"{card}\n" for card in deck), encoding="utf-8"
        )
        (output / "main.py").write_text(MAIN, encoding="utf-8")
        manifest = {
            "schema_version": PACKAGE_SCHEMA,
            "project_id": "0045_single_deck_expert_minimal_lora",
            "candidate": output.name,
            "policy_id": PUBLIC_POLICY_ID,
            "policy_kind": "public_observation_routed_specialist",
            "human_decision": "PROMOTE_FOR_KAGGLE_PACKAGE",
            "deck_id": "007",
            "deck_display_name": asset.name,
            "focal_exact_deck_sha256": audit["candidate_materializations"]["282"]["focal_exact_deck_sha256"],
            "default_checkpoint_update": DEFAULT_UPDATE,
            "routed_updates": list(ROUTED_UPDATES),
            "default_for_every_unmatched_observation": DEFAULT_UPDATE,
            "routing_rules": RULE_MANIFEST,
            "routed_modules": audit["routed_modules"],
            "source_checkpoints": {
                str(update): {
                    "source_checkpoint_sha256": audit["candidate_materializations"][str(update)]["source_checkpoint_sha256"],
                    "portable_checkpoint_sha256": audit["candidate_materializations"][str(update)]["portable_checkpoint_sha256"],
                    "deployment_effective_sha256": audit["candidate_materializations"][str(update)]["effective_candidate_sha256"],
                    "head_effective_sha256": audit["source_head_identity"][str(update)],
                } for update in ROUTED_UPDATES
            },
            "compact_deployment_audit": compact_audit,
            "composite_effective_sha256": audit["composite_effective_sha256"],
            "deployment_contract": "kaggle_fp16_storage_fp32_runtime_v1",
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "runtime_framework": "pytorch",
            "native_runtime_load_order": "torch_before_cg",
            "critic_outputs_consumed_by_actor_or_router": False,
            "qualifying_strength_evidence": {
                "status": "PASS",
                "report": str(QUALIFYING_REPORT.relative_to(ROOT)),
                "report_sha256": sha256_file(QUALIFYING_REPORT),
                "benchmark_id": report["benchmark_id"],
                "opponent_policy_id": _validated_opponent_id(report),
                "games": report["summary"]["games"],
                "wins": report["summary"]["wins"],
                "losses": report["summary"]["losses"],
                "draws": report["summary"]["draws"],
                "win_rate": report["summary"]["win_rate"],
                "composite_effective_sha256": audit["composite_effective_sha256"],
                "candidate_identity_status": audit["status"],
                "opponent_identity_status": report["opponent_policy_identity_audit"]["status"],
                "cuda_engine_identity_status": report["cuda_engine_identity_audit"]["status"],
            },
            "semantic_runtime_source": str(RUNTIME_ROOT.relative_to(ROOT)),
        }
        manifest["package_file_sha256"] = _file_inventory(output)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment.update({
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        })
        subprocess.run(
            ["python3", "-c", (
                "import main; assert len(main.read_deck_csv()) == 60; "
                "assert main.POLICY.policy_id == " + repr(PUBLIC_POLICY_ID) + "; "
                "assert main.POLICY.active_update == 282; "
                "assert main.POLICY.deployment_identity['runtime_dtype'] == 'fp32'"
            )], cwd=output, env=environment, check=True,
        )
        _deterministic_archive(output, archive)
        result = dict(manifest)
        result.update({
            "archive": str(archive.relative_to(ROOT)),
            "archive_sha256": sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
            "unpacked_bytes": sum(
                path.stat().st_size for path in output.rglob("*") if path.is_file()
            ),
        })
        return result
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        archive.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    result = export(output=args.output.resolve(), archive=args.archive.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
