"""Download and audit official daily Pokemon TCG Episode Datasets.

The Kaggle CLI currently transfers these large archives through one slow stream.
This module obtains the same authenticated, short-lived download response and
uses bounded HTTP byte ranges.  Signed URLs and credentials are never written
to disk or included in the source manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest


DEFAULT_DATES = (
    "2026-07-13",
    "2026-07-14",
    "2026-07-15",
    "2026-07-16",
    "2026-07-17",
    "2026-07-18",
    "2026-07-19",
    "2026-07-20",
    "2026-07-21",
    "2026-07-22",
)
DATASET_PREFIX = "pokemon-tcg-ai-battle-episodes-"
CHUNK_BYTES = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_response(date: str) -> Any:
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiDownloadDatasetRequest()
        request.owner_slug = "kaggle"
        request.dataset_slug = f"{DATASET_PREFIX}{date}"
        return client.datasets.dataset_api_client.download_dataset(request)


def _ranges(start: int, total: int, parts: int) -> list[tuple[int, int]]:
    if start >= total:
        return []
    width = max(1, (total - start + parts - 1) // parts)
    return [
        (offset, min(total - 1, offset + width - 1))
        for offset in range(start, total, width)
    ]


def _download_range(
    url: str,
    start: int,
    end: int,
    output: Path,
    *,
    retries: int,
) -> dict[str, int]:
    expected = end - start + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        completed = output.stat().st_size if output.is_file() else 0
        if completed == expected:
            return {"start": start, "end": end, "bytes": completed}
        if completed > expected:
            raise RuntimeError(f"range part is oversized: {output}")
        range_start = start + completed
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={range_start}-{end}"},
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"server ignored byte range {range_start}-{end}: "
                        f"HTTP {response.status_code}"
                    )
                with output.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        if chunk:
                            handle.write(chunk)
        except (OSError, requests.RequestException, RuntimeError):
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 8))
    completed = output.stat().st_size
    if completed != expected:
        raise RuntimeError(f"incomplete range {start}-{end}: {completed} != {expected}")
    return {"start": start, "end": end, "bytes": completed}


def _parallel_download(
    url: str,
    archive: Path,
    total: int,
    *,
    workers: int,
    retries: int,
) -> None:
    existing = archive.stat().st_size if archive.is_file() else 0
    if existing > total:
        raise RuntimeError(f"existing archive exceeds expected bytes: {archive}")
    ranges = _ranges(existing, total, workers * 2)
    if not ranges:
        return
    parts_root = archive.with_suffix(archive.suffix + ".parts")
    parts_root.mkdir(parents=True, exist_ok=True)
    tasks = {
        (start, end): parts_root / f"{start:012d}-{end:012d}.part"
        for start, end in ranges
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_range,
                url,
                start,
                end,
                path,
                retries=retries,
            ): (start, end)
            for (start, end), path in tasks.items()
        }
        for future in as_completed(futures):
            result = future.result()
            print(
                json.dumps({"event": "range_ready", **result}, sort_keys=True),
                flush=True,
            )
    assembled = archive.with_suffix(archive.suffix + ".assembling")
    with assembled.open("wb") as target:
        if existing:
            with archive.open("rb") as source:
                shutil.copyfileobj(source, target, CHUNK_BYTES)
        for range_key in sorted(tasks):
            with tasks[range_key].open("rb") as source:
                shutil.copyfileobj(source, target, CHUNK_BYTES)
    if assembled.stat().st_size != total:
        raise RuntimeError(
            f"assembled archive has {assembled.stat().st_size} bytes, expected {total}"
        )
    assembled.replace(archive)
    shutil.rmtree(parts_root)


def download_date(
    root: Path,
    date: str,
    *,
    workers: int,
    retries: int,
) -> dict[str, Any]:
    dataset_slug = f"{DATASET_PREFIX}{date}"
    dataset_ref = f"kaggle/{dataset_slug}"
    archive = root / "archives" / f"{dataset_slug}.zip"
    episode_root = root / "episodes" / date
    archive.parent.mkdir(parents=True, exist_ok=True)
    episode_root.mkdir(parents=True, exist_ok=True)
    response = _download_response(date)
    try:
        response.raise_for_status()
        total = int(response.headers["content-length"])
        accepts_ranges = str(response.headers.get("accept-ranges", "")).casefold()
        if accepts_ranges != "bytes":
            raise RuntimeError(f"dataset endpoint does not support byte ranges: {dataset_ref}")
        url = str(response.url)
    finally:
        response.close()
    started = time.monotonic()
    _parallel_download(url, archive, total, workers=workers, retries=retries)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"download is not a valid ZIP archive: {archive}")
    complete_marker = episode_root / ".complete"
    if not complete_marker.is_file():
        with zipfile.ZipFile(archive) as bundle:
            bundle.testzip()
            bundle.extractall(episode_root)
        complete_marker.touch()
    episode_files = sorted(episode_root.rglob("*.json"))
    result = {
        "date": date,
        "dataset_ref": dataset_ref,
        "dataset_url": f"https://www.kaggle.com/datasets/{dataset_ref}",
        "license": "CC0-1.0",
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "episode_root": str(episode_root.resolve()),
        "json_files": len(episode_files),
        "extracted_bytes": sum(path.stat().st_size for path in episode_files),
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps({"event": "date_ready", **result}, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--date", action="append", dest="dates")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        parser.error("KAGGLE_API_TOKEN must be set")
    if args.workers < 1 or args.retries < 1:
        parser.error("--workers and --retries must be positive")
    root = args.output_root.resolve()
    dates = tuple(args.dates or DEFAULT_DATES)
    results = [
        download_date(root, date, workers=args.workers, retries=args.retries)
        for date in dates
    ]
    manifest = {
        "schema_version": "ptcg_official_daily_episode_sources_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_scope": {
            "dates": list(dates),
            "expert_team": "Yushin Ito",
            "winner_only": True,
        },
        "sources": results,
        "totals": {
            "archive_bytes": sum(row["archive_bytes"] for row in results),
            "extracted_bytes": sum(row["extracted_bytes"] for row in results),
            "json_files": sum(row["json_files"] for row in results),
        },
    }
    manifest_path = root / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "download_complete", **manifest["totals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
