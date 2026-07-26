"""Fail-closed W&B audit snapshots for immutable version artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Iterable


ALLOWED_NAMES = frozenset(
    {
        "training_summary.json",
        "status.json",
        "checkpoint_selection.json",
        "model_contract.json",
        "dataset_reference.json",
        "metrics_snapshot.json",
    }
)
MANIFEST_NAME = "wandb_snapshot_manifest.json"
SENSITIVE_KEY_PARTS = ("api_key", "password", "secret", "token", "credential")
RAW_DATA_KEY_PARTS = ("raw_data", "rawdata", "dataset_rows", "observations", "replay")


@dataclass(frozen=True)
class SnapshotPolicy:
    """Fixed limits for safe, self-contained W&B audit snapshots."""

    max_file_bytes: int = 1_000_000
    max_total_bytes: int = 4_000_000

    @property
    def allowed_names(self) -> frozenset[str]:
        """Return the fixed, non-configurable audit payload whitelist."""
        return ALLOWED_NAMES


@dataclass(frozen=True)
class _PreparedPayload:
    path: Path
    size: int
    mtime_ns: int
    sha256: str


def validate_snapshot_files(
    artifact_root: str | Path,
    paths: Iterable[str | Path],
    *,
    policy: SnapshotPolicy | None = None,
) -> tuple[Path, ...]:
    """Validate every selected payload before a W&B upload can begin."""
    return tuple(item.path for item in _prepare_payloads(artifact_root, paths, policy=policy))


def upload_snapshot_generation(
    wandb: Any,
    artifact_root: str | Path,
    paths: Iterable[str | Path],
    manifest_path: str | Path,
    *,
    policy: SnapshotPolicy | None = None,
) -> dict[str, str]:
    """Upload immutable payloads followed by their immutable hash manifest."""
    root = Path(artifact_root).resolve(strict=True)
    manifest = _validate_manifest_path(root, manifest_path)
    prepared = _prepare_payloads(root, paths, policy=policy)
    try:
        for item in prepared:
            wandb.save(str(item.path), policy="now", base_path=str(root))
        _ensure_payloads_unchanged(prepared)
        _write_manifest_once(manifest, prepared)
        wandb.save(str(manifest), policy="now", base_path=str(root))
    except Exception as error:
        return {"state": "failed", "reason": str(error)}
    return {"state": "synced", "reason": ""}


def _prepare_payloads(
    artifact_root: str | Path,
    paths: Iterable[str | Path],
    *,
    policy: SnapshotPolicy | None = None,
) -> tuple[_PreparedPayload, ...]:
    root = Path(artifact_root).resolve(strict=True)
    active_policy = policy or SnapshotPolicy()
    prepared: list[_PreparedPayload] = []
    total_bytes = 0
    seen_names: set[str] = set()
    for candidate in paths:
        path = Path(candidate)
        if path.is_symlink():
            raise ValueError(f"snapshot file must not be a symlink: {path}")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"snapshot file is outside artifact root: {path}") from error
        if len(relative.parts) != 1 or relative.name not in ALLOWED_NAMES:
            raise ValueError(f"snapshot filename is not allowed: {relative}")
        if relative.name in seen_names:
            raise ValueError(f"snapshot filename is duplicated: {relative.name}")
        file_stat = resolved.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"snapshot path is not a regular file: {path}")
        if file_stat.st_size > active_policy.max_file_bytes:
            raise ValueError(f"snapshot file is too large: {relative}")
        total_bytes += file_stat.st_size
        if total_bytes > active_policy.max_total_bytes:
            raise ValueError("snapshot payload total is too large")
        contents = resolved.read_bytes()
        if len(contents) != file_stat.st_size:
            raise ValueError(f"snapshot payload changed while being read: {relative}")
        _validate_json_payload(contents, resolved.name)
        prepared.append(
            _PreparedPayload(
                path=resolved,
                size=file_stat.st_size,
                mtime_ns=file_stat.st_mtime_ns,
                sha256=hashlib.sha256(contents).hexdigest(),
            )
        )
        seen_names.add(relative.name)
    return tuple(prepared)


def _validate_manifest_path(root: Path, manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path)
    if manifest.is_symlink():
        raise ValueError("snapshot manifest must not be a symlink")
    resolved = manifest.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("snapshot manifest must be inside artifact root") from error
    if len(relative.parts) != 1 or relative.name != MANIFEST_NAME:
        raise ValueError(f"snapshot manifest must be {MANIFEST_NAME} in artifact root")
    if manifest.exists() or manifest.is_symlink():
        raise ValueError("snapshot manifest already exists and is immutable")
    return resolved


def _validate_json_payload(contents: bytes, name: str) -> None:
    try:
        value = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"snapshot must contain one JSON document, not JSONL: {name}") from error
    _validate_value(value)


def _validate_value(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in SENSITIVE_KEY_PARTS):
                raise ValueError(f"snapshot contains secret key: {key}")
            if any(part in normalized for part in RAW_DATA_KEY_PARTS):
                raise ValueError(f"snapshot contains raw-data key: {key}")
            _validate_value(nested)
    elif isinstance(value, list):
        for item in value:
            _validate_value(item)


def _ensure_payloads_unchanged(payloads: tuple[_PreparedPayload, ...]) -> None:
    for item in payloads:
        file_stat = item.path.stat()
        contents = item.path.read_bytes()
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size != item.size
            or file_stat.st_mtime_ns != item.mtime_ns
            or hashlib.sha256(contents).hexdigest() != item.sha256
        ):
            raise RuntimeError(f"snapshot payload changed before manifest publication: {item.path.name}")


def _write_manifest_once(manifest: Path, payloads: tuple[_PreparedPayload, ...]) -> None:
    contents = {
        "payloads": {
            item.path.name: {"sha256": item.sha256, "size_bytes": item.size}
            for item in payloads
        }
    }
    with manifest.open("x", encoding="utf-8") as output:
        json.dump(contents, output, indent=2, sort_keys=True)
        output.write("\n")
