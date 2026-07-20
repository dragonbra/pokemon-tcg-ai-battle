#!/usr/bin/env python3
"""Validate and record the immutable V7 AutoIter submission artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class BestStrategy:
    source_submission: str
    best_iteration: int
    label: str
    artifact: Path
    artifact_sha256: str


def _safe_token(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN_RE.fullmatch(value):
        raise ValueError(f"invalid {field}")
    return value


def _artifact_path(value: Any, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("artifact must be relative to the repository")
    resolved = (repo_root / relative).resolve()
    dist_root = (repo_root / "submission" / "dist").resolve()
    try:
        resolved.relative_to(dist_root)
    except ValueError as exc:
        raise ValueError("artifact must be inside submission/dist/") from exc
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"best artifact missing: {path}")
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"invalid best artifact: {path}") from exc

    normalized: set[str] = set()
    for member in members:
        name = member.name.rstrip("/")
        path_parts = Path(name).parts
        if not name or Path(name).is_absolute() or ".." in path_parts:
            raise ValueError("best artifact contains an unsafe member path")
        normalized.add(name)

    required = {"main.py", "deck.csv", "cg"}
    if not required.issubset(normalized):
        raise ValueError("best artifact is missing required archive members")
    if not any(name.startswith("cg/") for name in normalized):
        raise ValueError("best artifact is missing cg runtime files")

    for required_name in ("main.py", "deck.csv"):
        member = next(item for item in members if item.name.rstrip("/") == required_name)
        if not member.isfile():
            raise ValueError(f"best artifact member is not a file: {required_name}")


def validate_best_strategy(marker_path: Path, repo_root: Path) -> BestStrategy:
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read BEST_STRATEGY.json: {marker_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("BEST_STRATEGY.json must contain an object")

    source_submission = _safe_token(payload.get("source_submission"), "source_submission")
    label = _safe_token(payload.get("label"), "label")
    iteration = payload.get("best_iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("best_iteration must be a non-negative integer")
    artifact_sha256 = payload.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or not _SHA256_RE.fullmatch(artifact_sha256):
        raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")

    repo_root = repo_root.resolve()
    artifact = _artifact_path(payload.get("artifact"), repo_root)
    _validate_archive(artifact)
    actual_sha256 = _sha256(artifact)
    if actual_sha256 != artifact_sha256:
        raise ValueError(
            "best artifact SHA-256 mismatch: "
            f"expected {artifact_sha256}, got {actual_sha256}"
        )
    return BestStrategy(source_submission, iteration, label, artifact, artifact_sha256)


def stage_best_for_schedule(
    marker_path: Path,
    repo_root: Path,
    state_dir: Path,
) -> Path:
    """Copy the immutable best artifact to a launchd-safe directory.

    macOS may deny a launchd user agent access to a repository under
    ``~/Documents``.  The staged marker deliberately rewrites only its
    artifact path; the source marker remains the repository's canonical
    record and is never modified by this function.
    """
    best = validate_best_strategy(marker_path, repo_root)
    state_dir = state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    staged_archive = state_dir / best.artifact.name
    temporary_archive = state_dir / f".{best.artifact.name}.tmp"
    shutil.copy2(best.artifact, temporary_archive)
    temporary_archive.replace(staged_archive)

    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["artifact"] = staged_archive.name
    payload["staged_from"] = str(best.artifact.relative_to(repo_root.resolve()))
    payload["artifact_sha256"] = best.artifact_sha256
    temporary_marker = state_dir / ".BEST_STRATEGY.json.tmp"
    temporary_marker.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_marker.replace(state_dir / "BEST_STRATEGY.json")
    return staged_archive


def write_best_strategy(
    marker_path: Path,
    repo_root: Path,
    *,
    source_submission: str,
    best_iteration: int,
    label: str,
    artifact: Path,
    metrics: dict[str, Any] | None = None,
    evidence: str | None = None,
) -> BestStrategy:
    """Atomically record a packaged strategy as the best-known version."""
    _safe_token(source_submission, "source_submission")
    _safe_token(label, "label")
    if (
        isinstance(best_iteration, bool)
        or not isinstance(best_iteration, int)
        or best_iteration < 0
    ):
        raise ValueError("best_iteration must be a non-negative integer")
    artifact = artifact.resolve()
    repo_root = repo_root.resolve()
    try:
        artifact.relative_to(repo_root / "submission" / "dist")
    except ValueError as exc:
        raise ValueError("artifact must be inside submission/dist/") from exc
    _validate_archive(artifact)
    digest = _sha256(artifact)

    previous: dict[str, Any] = {}
    if marker_path.exists():
        previous = json.loads(marker_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "source_submission": source_submission,
        "best_iteration": best_iteration,
        "label": label,
        "status": "best-known",
        "deck_fixed": True,
        "artifact": str(artifact.relative_to(repo_root)),
        "artifact_sha256": digest,
        "metrics": metrics if metrics is not None else previous.get("metrics", {}),
        "evidence": evidence if evidence is not None else previous.get("evidence", ""),
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_name(f".{marker_path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker_path)
    return validate_best_strategy(marker_path, repo_root)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--marker", type=Path, required=True)
    validate_parser.add_argument("--repo-root", type=Path, required=True)

    write_parser = subparsers.add_parser("write-marker")
    write_parser.add_argument("--marker", type=Path, required=True)
    write_parser.add_argument("--repo-root", type=Path, required=True)
    write_parser.add_argument("--source-submission", required=True)
    write_parser.add_argument("--iteration", type=int, required=True)
    write_parser.add_argument("--label", required=True)
    write_parser.add_argument("--artifact", type=Path, required=True)
    write_parser.add_argument("--evidence", default="")
    write_parser.add_argument("--metrics-json", type=Path)

    stage_parser = subparsers.add_parser("stage-schedule")
    stage_parser.add_argument("--marker", type=Path, required=True)
    stage_parser.add_argument("--repo-root", type=Path, required=True)
    stage_parser.add_argument("--state-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "validate":
        result = validate_best_strategy(args.marker, args.repo_root)
        print(
            "\t".join(
                (str(result.artifact), str(result.best_iteration), result.label, result.artifact_sha256)
            )
        )
        return 0
    if args.command == "stage-schedule":
        staged = stage_best_for_schedule(args.marker, args.repo_root, args.state_dir)
        print(staged)
        return 0

    metrics = None
    if args.metrics_json:
        metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    result = write_best_strategy(
        args.marker,
        args.repo_root,
        source_submission=args.source_submission,
        best_iteration=args.iteration,
        label=args.label,
        artifact=args.artifact,
        metrics=metrics,
        evidence=args.evidence,
    )
    print(
        "\t".join(
            (str(result.artifact), str(result.best_iteration), result.label, result.artifact_sha256)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
