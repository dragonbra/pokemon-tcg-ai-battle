from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping


COLLECTION_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class KernelMetadata:
    competition: str
    ref: str
    title: str
    author: str
    last_run_time: str | None
    total_votes: int
    raw: dict[str, object]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], *, competition: str) -> "KernelMetadata":
        ref = str(value.get("ref", "")).strip()
        if not ref or "/" not in ref:
            raise ValueError("Kaggle kernel metadata requires ref owner/slug")
        return cls(
            competition=competition,
            ref=ref,
            title=str(value.get("title", ref)),
            author=str(value.get("author", ref.split("/", 1)[0])),
            last_run_time=_optional_text(value.get("lastRunTime")),
            total_votes=_int_value(value.get("totalVotes", 0)),
            raw={str(key): item for key, item in value.items()},
        )

    def source_id(self) -> str:
        owner, slug, *_ = self.ref.split("/")
        return f"{_slug(self.competition)}__{_slug(owner)}__{_slug(slug)}"

    @property
    def url(self) -> str:
        return f"https://www.kaggle.com/code/{self.ref}"


@dataclass(frozen=True)
class CollectionResult:
    source_id: str
    status: str
    directory: Path
    source_hash: str | None
    error: str | None


def collect_leaderboard_snapshot(
    competition: str,
    destination: Path,
    *,
    page_size: int = 200,
    max_pages: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """保存官方 leaderboard 快照；CLI 的分页前缀不属于 JSON，需要先剥离。"""
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    page_token: str | None = None
    page = 0
    while max_pages is None or page < max_pages:
        command = [
            "kaggle",
            "competitions",
            "leaderboard",
            competition,
            "--show",
            "--format",
            "json",
            "--page-size",
            str(page_size),
        ]
        if page_token:
            command.extend(["--page-token", page_token])
        try:
            completed = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=COLLECTION_TIMEOUT_SECONDS,
            )
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise RuntimeError("Kaggle leaderboard timed out") from exc
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "kaggle competitions leaderboard failed")
        payload, next_token = _parse_leaderboard_output(completed.stdout)
        entries.extend(item for item in payload if isinstance(item, Mapping))
        page += 1
        if not next_token or not payload:
            break
        page_token = next_token
    snapshot = {
        "competition": competition,
        "collected_at": datetime.now(UTC).isoformat(),
        "pages": page,
        "entries": entries,
    }
    encoded = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
    immutable = destination / f"leaderboard-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    immutable.write_text(encoded, encoding="utf-8")
    (destination / "leaderboard-latest.json").write_text(encoded, encoding="utf-8")
    return snapshot


def collect_submissions_snapshot(
    competition: str,
    destination: Path,
    *,
    page_size: int = 200,
    max_pages: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """保存当前账号可见的 submission/publicScore 快照，和 leaderboard 分开保留。"""
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    page_token: str | None = None
    page = 0
    while max_pages is None or page < max_pages:
        command = [
            "kaggle",
            "competitions",
            "submissions",
            competition,
            "--format",
            "json",
            "--page-size",
            str(page_size),
        ]
        if page_token:
            command.extend(["--page-token", page_token])
        try:
            completed = runner(command, capture_output=True, text=True, check=False, timeout=COLLECTION_TIMEOUT_SECONDS)
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise RuntimeError("Kaggle submissions timed out") from exc
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "kaggle competitions submissions failed")
        payload, next_token = _parse_cli_array_output(completed.stdout)
        entries.extend(item for item in payload if isinstance(item, Mapping))
        page += 1
        if not next_token or not payload:
            break
        page_token = next_token
    snapshot = {
        "competition": competition,
        "collected_at": datetime.now(UTC).isoformat(),
        "pages": page,
        "entries": entries,
    }
    encoded = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
    immutable = destination / f"submissions-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    immutable.write_text(encoded, encoding="utf-8")
    (destination / "submissions-latest.json").write_text(encoded, encoding="utf-8")
    return snapshot


def collect_kernel_index(
    competition: str,
    destination: Path,
    *,
    page_size: int = 200,
    max_pages: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[KernelMetadata, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    records: list[KernelMetadata] = []
    seen: set[str] = set()
    page = 1
    while max_pages is None or page <= max_pages:
        command = [
            "kaggle",
            "kernels",
            "list",
            "--competition",
            competition,
            "--page",
            str(page),
            "--page-size",
            str(page_size),
            "--format",
            "json",
        ]
        try:
            completed = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=COLLECTION_TIMEOUT_SECONDS,
            )
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise RuntimeError(
                f"Kaggle kernel list timed out on page {page} after {COLLECTION_TIMEOUT_SECONDS:g}s"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "kaggle kernels list failed")
        raw_stdout = completed.stdout.strip()
        if raw_stdout.lower() in {"not found", "no kernels found", "no results found"}:
            break
        try:
            payload = json.loads(raw_stdout or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"unexpected Kaggle kernel list response on page {page}: {raw_stdout[:120]}") from exc
        if not isinstance(payload, list) or not payload:
            break
        page_records = [
            KernelMetadata.from_mapping(item, competition=competition)
            for item in payload
            if isinstance(item, Mapping)
        ]
        for record in page_records:
            if record.ref in seen:
                continue
            seen.add(record.ref)
            records.append(record)
            source_dir = destination / record.source_id()
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "competition": record.competition,
                        "ref": record.ref,
                        "title": record.title,
                        "author": record.author,
                        "last_run_time": record.last_run_time,
                        "total_votes": record.total_votes,
                        "url": record.url,
                        "raw": record.raw,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        page += 1
    return tuple(records)


def collect_kernel_files(
    metadata: KernelMetadata,
    destination: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CollectionResult:
    source_dir = destination / metadata.source_id()
    source_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for label, command in (
        ("pull", ["kaggle", "kernels", "pull", metadata.ref, "--path", str(source_dir), "--metadata"]),
        ("output", ["kaggle", "kernels", "output", metadata.ref, "--path", str(source_dir / "output"), "--quiet"]),
    ):
        try:
            completed = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=COLLECTION_TIMEOUT_SECONDS,
            )
        except (subprocess.TimeoutExpired, TimeoutError):
            errors.append(f"{label}: command timed out after {COLLECTION_TIMEOUT_SECONDS:g}s")
            continue
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"command failed: {' '.join(command[:3])}"
            errors.append(f"{label}: {detail}")
    files = [path for path in source_dir.rglob("*") if path.is_file() and path.name != "metadata.json"]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(source_dir).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return CollectionResult(
        metadata.source_id(),
        "collected" if not errors else "collection_error",
        source_dir,
        digest.hexdigest() if files else None,
        "; ".join(errors) if errors else None,
    )


def safe_extract_archive(archive_path: Path, destination: Path) -> tuple[Path, ...]:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    members: list[tuple[str, bool]] = []
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            members = [(item.filename, _zip_is_link(item)) for item in archive.infolist()]
            _validate_members(destination, members)
            archive.extractall(destination)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            tar_members = archive.getmembers()
            members = [(item.name, item.issym() or item.islnk()) for item in tar_members]
            _validate_members(destination, members)
            try:
                archive.extractall(destination, filter="data")
            except TypeError:  # Python 3.11: 成员路径已在上面完成安全校验
                archive.extractall(destination)
    else:
        raise ValueError(f"unsupported archive format: {archive_path}")
    return tuple(path for path in destination.rglob("*") if path.is_file())


def _validate_members(destination: Path, members: list[tuple[str, bool]]) -> None:
    for name, is_link in members:
        if is_link:
            raise ValueError(f"archive links are not allowed: {name}")
        target = (destination / name).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise ValueError(f"archive member escapes destination: {name}") from exc


def _zip_is_link(item: zipfile.ZipInfo) -> bool:
    mode = (item.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower())
    return value.strip("-._") or "unknown"


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _int_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _parse_leaderboard_output(output: str) -> tuple[list[object], str | None]:
    return _parse_cli_array_output(output)


def _parse_cli_array_output(output: str) -> tuple[list[object], str | None]:
    token_match = re.search(r"^Next Page Token\s*=\s*(.+)$", output, re.MULTILINE)
    json_start = output.find("[")
    if json_start < 0:
        raise RuntimeError(f"unexpected Kaggle leaderboard response: {output[:160]}")
    try:
        payload = json.loads(output[json_start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"unexpected Kaggle leaderboard JSON: {output[:160]}") from exc
    if not isinstance(payload, list):
        raise RuntimeError("Kaggle leaderboard response must be a JSON list")
    return payload, token_match.group(1).strip() if token_match else None
