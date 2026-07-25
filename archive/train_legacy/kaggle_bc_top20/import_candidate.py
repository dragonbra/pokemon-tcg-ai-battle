"""Safely import one Kaggle BC candidate archive into a new local package directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from train.kaggle_bc_top20.worker import RESULT_SCHEMA


ALLOWED_ROOTS = {"main.py", "deck.csv", "cg", "strategy"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_python_caches(root: Path) -> None:
    for directory in sorted(root.rglob("__pycache__"), reverse=True):
        if directory.is_dir():
            shutil.rmtree(directory)
    for suffix in ("*.pyc", "*.pyo"):
        for path in root.rglob(suffix):
            if path.is_file():
                path.unlink()


def validate_archive_members(members: Iterable[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    rows = list(members)
    if not rows:
        raise ValueError("candidate archive is empty")
    roots: set[str] = set()
    for member in rows:
        path = PurePosixPath(member.name)
        if member.issym() or member.islnk():
            raise ValueError(f"candidate archive links are not allowed: {member.name}")
        if not member.isfile() and not member.isdir():
            raise ValueError(f"unsupported candidate archive member: {member.name}")
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe candidate archive path: {member.name}")
        root = path.parts[0]
        if root not in ALLOWED_ROOTS:
            raise ValueError(f"unexpected candidate archive root: {root}")
        roots.add(root)
    missing = ALLOWED_ROOTS - roots
    if missing:
        raise ValueError(f"candidate archive is missing roots: {sorted(missing)}")
    return rows


def import_candidate(
    *,
    archive: Path,
    result_path: Path,
    output: Path,
    repository_root: Path,
    evaluation_cg_source: Path | None = None,
) -> dict[str, Any]:
    archive = archive.resolve()
    result_path = result_path.resolve()
    output = output.resolve()
    repository_root = repository_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite local package: {output}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != RESULT_SCHEMA or result.get("status") != "candidate_ready":
        raise ValueError("RESULT.json is not a completed Kaggle BC candidate result")
    expected = str((result.get("candidate") or {}).get("archive_sha256", ""))
    actual = _sha256(archive)
    if not expected or actual != expected:
        raise ValueError(f"candidate archive hash mismatch: {actual} != {expected}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        with tarfile.open(archive, "r:gz") as handle:
            members = validate_archive_members(handle.getmembers())
            for member in members:
                target = (output / member.name).resolve()
                if output not in (target, *target.parents):
                    raise ValueError(f"unsafe extraction target: {member.name}")
            handle.extractall(output, members=members)
        from evaluation.runtime.loader import compute_cg_manifest

        cloud_cg_manifest = compute_cg_manifest(output / "cg")
        evaluation_cg_manifest = cloud_cg_manifest
        if evaluation_cg_source is not None:
            evaluation_cg_source = evaluation_cg_source.resolve()
            if (evaluation_cg_source / "cg").is_dir():
                evaluation_cg_source = evaluation_cg_source / "cg"
            evaluation_cg_manifest = compute_cg_manifest(evaluation_cg_source)
            shutil.rmtree(output / "cg")
            shutil.copytree(
                evaluation_cg_source,
                output / "cg",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            (output / "strategy/runtime_normalization.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ptcg_local_evaluation_runtime_normalization_v1",
                        "archive_sha256": actual,
                        "cloud_cg_tree_hash": cloud_cg_manifest["tree_hash"],
                        "evaluation_cg_tree_hash": evaluation_cg_manifest["tree_hash"],
                        "deck_and_model_unchanged": True,
                        "scope": "local evaluation working copy only",
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        validation = subprocess.run(
            [sys.executable, "-m", "evaluation", "validate", str(output)],
            cwd=repository_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if validation.returncode:
            raise RuntimeError(
                "imported package failed local validation: "
                f"{validation.stderr.strip() or validation.stdout.strip()}"
            )
        _remove_python_caches(output)
    except Exception:
        shutil.rmtree(output)
        raise
    return {
        "status": "validated_local_package",
        "output": str(output),
        "archive_sha256": actual,
        "offline_gate_passed": bool(result.get("offline_gate_passed")),
        "official_game_evaluation_performed": False,
        "cloud_cg_tree_hash": cloud_cg_manifest["tree_hash"],
        "evaluation_cg_tree_hash": evaluation_cg_manifest["tree_hash"],
        "evaluation_runtime_normalized": evaluation_cg_source is not None,
        "validation": validation.stdout.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-cg-source",
        type=Path,
        help=(
            "optional canonical local evaluation cg/; the cloud archive hash remains "
            "verified before this runtime-only normalization"
        ),
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    imported = import_candidate(
        archive=args.archive,
        result_path=args.result,
        output=args.output,
        repository_root=repository_root,
        evaluation_cg_source=args.evaluation_cg_source,
    )
    print(json.dumps(imported, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
