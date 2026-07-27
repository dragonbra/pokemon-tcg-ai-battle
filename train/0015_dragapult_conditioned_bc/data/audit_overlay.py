"""Verify the deduplicated 0726 targeted replay overlay and publish its audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _official_ids(archive_path: Path) -> set[int]:
    ids: set[int] = set()
    with zipfile.ZipFile(archive_path) as bundle:
        for item in bundle.infolist():
            if item.is_dir() or not item.filename.endswith(".json"):
                continue
            match = re.fullmatch(r"(?:episode-)?(\d+)(?:-replay)?\.json", item.filename)
            if match is None:
                raise ValueError(f"invalid Episode archive member: {item.filename}")
            episode_id = int(match.group(1))
            if episode_id in ids:
                raise ValueError(f"duplicate official Episode ID: {episode_id}")
            ids.add(episode_id)
    return ids


def build_audit(overlay_root: Path, archive_path: Path) -> dict[str, Any]:
    official_ids = _official_ids(archive_path)
    all_episode_ids: set[int] = set()
    all_content_hashes: dict[str, tuple[str, int]] = {}
    sources: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()

    for source_dir in sorted(path for path in overlay_root.iterdir() if path.is_dir()):
        manifest_path = source_dir / "manifest.json"
        manifest = _load_object(manifest_path)
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"manifest episodes must be a list: {manifest_path}")
        counts: Counter[str] = Counter()
        bytes_by_origin: Counter[str] = Counter()
        for item in episodes:
            if not isinstance(item, dict):
                raise ValueError(f"invalid episode row: {manifest_path}")
            episode_id = int(item["episode_id"])
            if episode_id in all_episode_ids:
                raise ValueError(f"episode {episode_id} occurs in multiple source manifests")
            all_episode_ids.add(episode_id)
            path = source_dir / str(item["file"])
            if not path.exists():
                raise FileNotFoundError(path)
            actual_hash = _sha256(path)
            if actual_hash != item["sha256"]:
                raise ValueError(f"content hash mismatch: {path}")
            previous = all_content_hashes.get(actual_hash)
            if previous is not None:
                raise ValueError(
                    f"duplicate replay content: {previous[0]}/{previous[1]} and "
                    f"{source_dir.name}/{episode_id}"
                )
            all_content_hashes[actual_hash] = (source_dir.name, episode_id)
            in_official = episode_id in official_ids
            expected_origin = "official_base" if in_official else "overlay_download"
            is_symlink = path.is_symlink()
            if in_official and not is_symlink:
                raise ValueError(f"official-base replay is unexpectedly materialized: {path}")
            if not in_official and is_symlink:
                raise ValueError(f"overlay replay unexpectedly uses a symlink: {path}")
            recorded_origin = str(item.get("replay_source"))
            if in_official and recorded_origin != "mounted_daily_dataset":
                raise ValueError(f"official-base provenance mismatch: {path}")
            if not in_official and recorded_origin != "downloaded":
                raise ValueError(f"overlay provenance mismatch: {path}")
            counts[expected_origin] += 1
            bytes_by_origin[expected_origin] += path.stat().st_size
            totals[expected_origin] += 1
            totals["eligible"] += 1

        eligible = len(episodes)
        source_summary = {
            "source_key": source_dir.name,
            "team_name": manifest["team_name"],
            "team_id": manifest["team_id"],
            "submission_id": manifest["submission_id"],
            "deck_sha256": manifest["deck_sha256"],
            "eligible": eligible,
            "official_base": counts["official_base"],
            "overlay_download": counts["overlay_download"],
            "regular_files": sum(
                path.is_file() and not path.is_symlink()
                for path in source_dir.glob("episode-*.json")
            ),
            "symlinks": sum(path.is_symlink() for path in source_dir.glob("episode-*.json")),
            "bytes": dict(sorted(bytes_by_origin.items())),
            "duplicate_episode_ids": 0,
            "duplicate_content_hashes": 0,
            "mismatched_or_quarantined": 0,
        }
        if source_summary["regular_files"] + source_summary["symlinks"] != eligible:
            raise ValueError(f"manifest/file count mismatch: {source_dir}")
        sources.append(source_summary)

    return {
        "schema_version": "0015_targeted_overlay_audit_v1",
        "official_archive": str(archive_path),
        "overlay_root": str(overlay_root),
        "contract": {
            "deduplication_key": "episode_id_then_content_sha256",
            "official_base_precedence": True,
            "official_base_materialization": "symlink_to_isolated_archive_seed_only",
            "overlay_materialization": "regular_file",
            "training_count_rule": "each eligible episode_id exactly once",
        },
        "sources": sources,
        "totals": {
            "eligible": totals["eligible"],
            "official_base": totals["official_base"],
            "overlay_download": totals["overlay_download"],
            "unique_episode_ids": len(all_episode_ids),
            "unique_content_hashes": len(all_content_hashes),
            "duplicate_episode_ids": 0,
            "duplicate_content_hashes": 0,
            "mismatched_or_quarantined": 0,
        },
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.overlay_root, args.official_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
