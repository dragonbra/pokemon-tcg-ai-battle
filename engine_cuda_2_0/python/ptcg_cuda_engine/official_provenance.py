from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def source_tree_manifest(source: str | Path) -> tuple[str, list[dict[str, Any]]]:
    root = Path(source).resolve()
    rows: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative_path = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        rows.append({"path": relative_path, "bytes": path.stat().st_size, "sha256": digest})
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(digest))
    return combined.hexdigest(), rows


def git_value(path: str | Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(path).resolve()), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def workspace_relative(path: str | Path, workspace: str | Path) -> str:
    resolved = Path(path).resolve()
    root = Path(workspace).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside workspace: {resolved}") from error


def require_workspace_relative(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError(f"{field} must be a workspace-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must not escape the workspace")
    return path
