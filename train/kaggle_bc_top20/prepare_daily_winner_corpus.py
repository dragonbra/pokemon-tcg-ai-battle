"""Build auditable daily Yushin winner-only shards with bounded disk usage."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train.alakazam_bc_rl.features import UNIVERSAL_FEATURE_SCHEMA, feature_config_for_schema
from train.kaggle_bc_top20.training.build_daily_winner_bc_dataset import (
    DEFAULT_DATES,
    _is_winner,
    _normalize_team,
    build_dataset,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dates(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _archive_for(root: Path, date: str) -> Path:
    canonical = root / "archives" / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
    if canonical.is_file():
        return canonical
    local = root / f"archive-{date[5:7]}{date[8:10]}.zip"
    if local.is_file():
        return local
    raise FileNotFoundError(f"daily archive is missing for {date}")


def _validate_daily_output(dataset: Path, date: str, validation_date: str) -> dict[str, Any]:
    summary_path = dataset.with_suffix(dataset.suffix + ".summary.json")
    manifest_path = dataset.with_suffix(dataset.suffix + ".data_manifest.json")
    if not dataset.is_file() or not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"daily dataset is incomplete: {dataset}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_hash = _sha256(dataset)
    expected_split = "validation" if date == validation_date else "train"
    expected_records = {expected_split: int(summary["records"])}
    if summary.get("run_dates") != [date]:
        raise ValueError(f"daily summary has wrong date: {summary.get('run_dates')}")
    if summary.get("teacher_team") != "Yushin Ito":
        raise ValueError(f"daily summary has wrong expert: {summary.get('teacher_team')}")
    if summary.get("records_by_split") != expected_records:
        raise ValueError(
            f"daily summary has wrong split: {summary.get('records_by_split')}"
        )
    if observed_hash != summary.get("dataset_sha256"):
        raise ValueError(f"daily summary hash mismatch: {date}")
    if observed_hash != manifest.get("dataset_sha256"):
        raise ValueError(f"daily manifest hash mismatch: {date}")
    if manifest.get("source_identity", {}).get("team_name") != "Yushin Ito":
        raise ValueError(f"daily manifest has wrong policy identity: {date}")
    return summary


def _archive_row(archive: Path, date: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as bundle:
        json_entries = [item for item in bundle.infolist() if item.filename.endswith(".json")]
        if not json_entries:
            raise ValueError(f"archive contains no Episode JSON: {archive}")
        extracted_bytes = sum(item.file_size for item in json_entries)
    return {
        "date": date,
        "dataset_ref": f"kaggle/pokemon-tcg-ai-battle-episodes-{date}",
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "json_files": len(json_entries),
        "extracted_bytes": extracted_bytes,
    }


def _header_lists(handle: Any) -> tuple[list[Any], list[Any]]:
    decoder = json.JSONDecoder()
    text_decoder = codecs.getincrementaldecoder("utf-8")()
    markers = {
        "teams": re.compile(r'"TeamNames"\s*:'),
        "rewards": re.compile(r'"rewards"\s*:'),
    }
    values: dict[str, list[Any]] = {}
    buffer = ""
    for _ in range(128):
        chunk = handle.read(64 * 1024)
        if not chunk:
            break
        buffer += text_decoder.decode(chunk)
        for name, marker in markers.items():
            if name in values:
                continue
            match = marker.search(buffer)
            if match is None:
                continue
            start = match.end()
            while start < len(buffer) and buffer[start].isspace():
                start += 1
            try:
                value, _ = decoder.raw_decode(buffer, start)
            except json.JSONDecodeError:
                continue
            if isinstance(value, list):
                values[name] = value
        if len(values) == len(markers):
            return values["teams"], values["rewards"]
    raise ValueError("Episode header does not expose TeamNames and rewards within 8 MiB")


def _extract_winner_files(
    archive: Path, extraction: Path, *, teacher_team: str
) -> dict[str, int]:
    teacher = _normalize_team(teacher_team)
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as bundle:
        entries = [item for item in bundle.infolist() if item.filename.endswith(".json")]
        for index, entry in enumerate(entries, 1):
            counts["episodes_seen"] += 1
            try:
                with bundle.open(entry) as source:
                    teams, rewards = _header_lists(source)
            except (OSError, UnicodeDecodeError, ValueError):
                counts["invalid_headers"] += 1
                continue
            matching = [
                player_index
                for player_index, name in enumerate(teams)
                if _normalize_team(name) == teacher
            ]
            counts["team_matches"] += len(matching)
            winners = [
                player_index for player_index in matching if _is_winner(rewards, player_index)
            ]
            counts["teacher_wins"] += len(winners)
            if winners:
                output = extraction / Path(entry.filename).name
                if output.exists():
                    raise ValueError(f"duplicate archive member basename: {entry.filename}")
                with bundle.open(entry) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            if index == 1 or index % 500 == 0:
                print(
                    json.dumps(
                        {"event": "daily_zip_scan_progress", **counts}, sort_keys=True
                    ),
                    flush=True,
                )
    if not counts["teacher_wins"]:
        raise ValueError(f"archive produced no {teacher_team} winner Episodes: {archive}")
    return dict(counts)


def prepare(
    raw_root: Path,
    dataset_root: Path,
    *,
    dates: list[str],
    validation_date: str,
    storage_path: Path,
    min_free_gib: float,
) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    dataset_root = dataset_root.resolve()
    if validation_date not in dates:
        raise ValueError("validation date must be included in the daily corpus")
    dataset_root.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for date in dates:
        archive = _archive_for(raw_root, date)
        source = _archive_row(archive, date)
        source_rows.append(source)
        day_root = dataset_root / "daily" / date
        dataset = day_root / "dataset.jsonl"
        source_audit_path = day_root / "source_audit.json"
        extraction = raw_root / "episodes" / date
        source_audit: dict[str, int] | None = None
        if dataset.is_file():
            summary = _validate_daily_output(dataset, date, validation_date)
            if source_audit_path.is_file():
                source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
            if extraction.exists():
                shutil.rmtree(extraction)
            print(
                json.dumps(
                    {"event": "daily_shard_reused", "date": date, "records": summary["records"]},
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            occupied = [
                path
                for path in (raw_root / "episodes").iterdir()
                if path.is_dir() and path != extraction and any(path.rglob("*.json"))
            ] if (raw_root / "episodes").is_dir() else []
            if occupied:
                raise RuntimeError(
                    "refusing to extract a second daily archive while another is present: "
                    + ", ".join(str(path) for path in occupied)
                )
            if extraction.exists():
                shutil.rmtree(extraction)
            extraction.mkdir(parents=True, exist_ok=True)
            print(
                json.dumps(
                    {
                        "event": "daily_zip_scan_start",
                        "date": date,
                        "json_files": source["json_files"],
                        "extracted_bytes": source["extracted_bytes"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            day_root.mkdir(parents=True, exist_ok=True)
            source_audit = _extract_winner_files(
                archive, extraction, teacher_team="Yushin Ito"
            )
            source_audit_path.write_text(
                json.dumps(source_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "event": "daily_winner_extract_ready",
                        "date": date,
                        **source_audit,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            summary = build_dataset(
                [extraction],
                dataset,
                teacher_team="Yushin Ito",
                dates={date},
                valid_dates={date} if date == validation_date else set(),
                feature_config=feature_config_for_schema(UNIVERSAL_FEATURE_SCHEMA),
                storage_path=storage_path,
                min_free_gib=min_free_gib,
                require_train_validation=False,
            )
            summary = _validate_daily_output(dataset, date, validation_date)
            shutil.rmtree(extraction)
            print(
                json.dumps(
                    {"event": "daily_shard_ready", "date": date, "records": summary["records"]},
                    sort_keys=True,
                ),
                flush=True,
            )
        daily_rows.append(
            {
                "date": date,
                "dataset": str(dataset.resolve()),
                "dataset_sha256": summary["dataset_sha256"],
                "records": summary["records"],
                "records_by_split": summary["records_by_split"],
                "episodes": summary["episodes"],
                "teacher_wins": (
                    source_audit["teacher_wins"]
                    if source_audit is not None
                    else summary["teacher_wins"]
                ),
                "team_matches": (
                    source_audit["team_matches"]
                    if source_audit is not None
                    else summary["team_matches"]
                ),
                "source_episodes_seen": (
                    source_audit["episodes_seen"]
                    if source_audit is not None
                    else summary["episodes_seen"]
                ),
                "source_header_errors": (
                    source_audit.get("invalid_headers", 0)
                    if source_audit is not None
                    else 0
                ),
            }
        )
        status = {
            "schema_version": "ptcg_daily_winner_prepare_v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "running" if date != dates[-1] else "complete",
            "teacher_team": "Yushin Ito",
            "validation_date": validation_date,
            "completed_dates": [row["date"] for row in daily_rows],
            "sources": source_rows,
            "daily_shards": daily_rows,
        }
        (dataset_root / "daily_prepare_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dates", default=DEFAULT_DATES)
    parser.add_argument("--validation-date", default="2026-07-22")
    parser.add_argument("--storage-path", type=Path, default=Path("/mnt/d"))
    parser.add_argument("--min-free-gib", type=float, default=30.0)
    args = parser.parse_args()
    result = prepare(
        args.raw_root,
        args.dataset_root,
        dates=_dates(args.dates),
        validation_date=args.validation_date,
        storage_path=args.storage_path,
        min_free_gib=args.min_free_gib,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
