from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from evaluation.packages.loader import PackageValidationError, SubmissionPackage, load_submission_package

from .adapters import reconstruct_notebook_package
from .catalog import card_image_url, classify_deck, load_card_metadata
from .collect import (
    collect_kernel_files,
    collect_kernel_index,
    collect_leaderboard_snapshot,
    collect_submissions_snapshot,
    safe_extract_archive,
)
from .discussions import collect_discussion_index
from .models import DeckRecord, RunRecord, SourceRecord
from .paths import arena_root, package_root, report_root, source_root
from .reports import build_report_snapshot, write_report_pair, write_reports
from .runner import ArenaRunConfig, ArenaRunner
from .storage import ArenaStore


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_COMPETITION = "pokemon-tcg-ai-battle"
CHALLENGE_COMPETITION = "pokemon-tcg-ai-battle-challenge-strategy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle 公共卡组 Arena")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="收集 Kaggle Code、输出和 Discussion")
    collect.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    collect.add_argument("--competition", default=MAIN_COMPETITION)
    collect.add_argument("--include-challenge-archives", action="store_true")
    collect.add_argument("--max-pages", type=int)
    collect.add_argument("--no-download", action="store_true")
    collect.add_argument("--discussion-pages", type=int, default=1)

    validate = subparsers.add_parser("validate", help="验证 Arena packages")
    validate.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)

    run = subparsers.add_parser("run", help="运行 Arena 对局")
    run.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    run.add_argument("--phase", choices=("smoke", "formal", "continuous"), default="smoke")
    run.add_argument("--games", type=int, default=10)
    run.add_argument("--limit", type=int, default=100)
    run.add_argument("--resume")
    run.add_argument("--include-demoted", action="store_true")
    run.add_argument("--seed", type=int, default=20260722)
    run.add_argument("--max-steps", type=int, default=1000)
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("--checkpoint-every", type=int, default=10)

    report = subparsers.add_parser("report", help="从已有 Arena 数据重绘报告")
    report.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    report.add_argument("--run-id", required=True)
    report.add_argument("--include-demoted", action="store_true")
    report.add_argument("--output", type=Path)

    discussions = subparsers.add_parser("discussions", help="刷新官方 Discussion 快照")
    discussions.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    discussions.add_argument("--competition", default=MAIN_COMPETITION)
    discussions.add_argument("--pages", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            collect_command(args)
        elif args.command == "validate":
            validate_command(args)
        elif args.command == "run":
            run_command(args)
        elif args.command == "report":
            write_report_command(args)
        elif args.command == "discussions":
            discussions_command(args)
        return 0
    except (OSError, PackageValidationError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Arena error: {exc}", file=sys.stderr)
        return 2


def collect_command(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    store = ArenaStore(arena_root(repo_root))
    store.initialize()
    competitions = [args.competition]
    if args.include_challenge_archives and CHALLENGE_COMPETITION not in competitions:
        competitions.append(CHALLENGE_COMPETITION)
    leaderboard, submissions = _collect_public_snapshots(repo_root, args.competition)
    for competition in competitions:
        destination = source_root(repo_root) / competition
        records = collect_kernel_index(competition, destination, max_pages=args.max_pages)
        for metadata in records:
            result = (
                None
                if args.no_download
                else collect_kernel_files(metadata, destination)
            )
            status = "metadata_only" if args.no_download else str(result.status)
            error = None if result is None else result.error
            source_hash = None if result is None else result.source_hash
            source_record = SourceRecord(
                metadata.source_id(),
                competition,
                metadata.ref,
                None,
                metadata.title,
                metadata.author,
                metadata.url,
                {
                    **metadata.raw,
                    "total_votes": metadata.total_votes,
                    "last_run_time": metadata.last_run_time,
                },
                source_hash,
                status,
                error,
                _now(),
            )
            store.upsert_source(source_record)
            if result is not None and _has_source_artifact(result.directory):
                _materialize_source(
                    repo_root,
                    metadata,
                    result.directory,
                    store,
                    leaderboard=_public_score_match(metadata, leaderboard, submissions),
                )
        print(f"collected {competition}: {len(records)} sources")
    collect_discussion_index(
        MAIN_COMPETITION,
        source_root(repo_root) / "discussions",
        pages=max(1, args.discussion_pages),
    )
    _write_source_catalog(repo_root, store)


def validate_command(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    store = ArenaStore(arena_root(repo_root))
    store.initialize()
    card_ids = _official_card_ids(repo_root)
    deck_rows = {str(row["deck_id"]): row for row in store.load_decks()}
    package_dirs = sorted(path for path in package_root(repo_root).iterdir() if path.is_dir()) if package_root(repo_root).is_dir() else []
    valid = invalid = 0
    for package_dir in package_dirs:
        catalog_row = deck_rows.get(package_dir.name)
        if catalog_row is not None and str(catalog_row.get("status")) == "metadata_only":
            print(f"metadata_only {package_dir.name}: excluded from validation")
            continue
        validation_error: str | None = None
        try:
            package = load_submission_package(package_dir, card_ids)
        except (OSError, PackageValidationError, ValueError) as exc:
            invalid += 1
            validation_error = str(exc)
            print(f"invalid {package_dir.name}: {exc}")
            _update_deck_validation(store, package_dir.name, "invalid", validation_error)
        else:
            valid += 1
            print(f"valid {package.name}: {package.package_hash}")
            _update_deck_validation(store, package.name, "valid", None)
    _write_source_catalog(repo_root, store)
    print(f"validated packages: valid={valid} invalid={invalid}")


def run_command(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    store = ArenaStore(arena_root(repo_root))
    store.initialize()
    packages = _load_arena_packages(repo_root)
    if len(packages) < 2:
        raise ValueError("Arena requires at least two valid packages")
    _ensure_runtime_deck_records(repo_root, store, packages)
    run_id = args.resume or f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{args.phase}"
    previous = store.load_run(run_id) if args.resume else None
    phase_transition = (
        previous is not None
        and str(previous.get("phase")) in {"smoke", "formal"}
        and args.phase == "continuous"
    )
    if previous is not None and str(previous.get("phase")) != args.phase and not phase_transition:
        raise ValueError(f"resume run phase mismatch: stored={previous.get('phase')} requested={args.phase}")
    run_config = _json_args(args)
    store.create_run(
        RunRecord(
            run_id,
            args.phase,
            run_config,
            str(previous.get("started_at", _now())) if previous else _now(),
            None,
            "running",
        )
    )
    runner = ArenaRunner(
        store,
        packages,
        ArenaRunConfig(
            run_id,
            args.phase,
            games_per_pair=args.games,
            max_steps=args.max_steps,
            worker_timeout_seconds=args.timeout,
            seed=args.seed,
            include_demoted=args.include_demoted,
            checkpoint_every=max(1, args.checkpoint_every),
        ),
        checkpoint_callback=lambda: write_report_pair(
            build_report_snapshot(store, run_id, include_demoted=True),
            report_root(repo_root),
        ),
    )
    try:
        if args.phase in ("smoke", "formal"):
            runner.run_initial_round(tuple(packages))
        else:
            runner.run_continuous_round(max(1, args.limit))
    except KeyboardInterrupt:
        store.create_run(
            RunRecord(
                run_id,
                args.phase,
                run_config,
                str(previous.get("started_at", _now())) if previous else _now(),
                _now(),
                "interrupted",
            )
        )
        raise
    except Exception:
        store.create_run(RunRecord(run_id, args.phase, run_config, str(previous.get("started_at", _now())) if previous else _now(), _now(), "error"))
        raise
    store.create_run(RunRecord(run_id, args.phase, run_config, str(previous.get("started_at", _now())) if previous else _now(), _now(), "finished"))
    snapshot = build_report_snapshot(store, run_id, include_demoted=True)
    paths = write_report_pair(snapshot, report_root(repo_root))
    print(f"run_id: {run_id}")
    print(f"all_report: {paths['all_html']}")
    print(f"eligible_report: {paths['eligible_html']}")


def write_report_command(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    store = ArenaStore(arena_root(repo_root))
    store.initialize()
    destination = args.output.resolve() if args.output else report_root(repo_root)
    # 双版本报告必须从同一份 All 事实构建；--include-demoted 保留为兼容参数。
    snapshot = build_report_snapshot(store, args.run_id, include_demoted=True)
    paths = write_report_pair(snapshot, destination)
    print(f"all_report: {paths['all_html']}")
    print(f"eligible_report: {paths['eligible_html']}")


def discussions_command(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    destination = source_root(repo_root) / "discussions"
    records = collect_discussion_index(args.competition, destination, pages=max(1, args.pages))
    print(f"discussions: {len(records)}")


def _load_arena_packages(repo_root: Path) -> dict[str, SubmissionPackage]:
    card_ids = _official_card_ids(repo_root)
    store = ArenaStore(arena_root(repo_root))
    store.initialize()
    catalog_status = {
        str(row["deck_id"]): str(row["status"])
        for row in store.load_decks()
    }
    packages: dict[str, SubmissionPackage] = {}
    if package_root(repo_root).is_dir():
        for package_dir in sorted(path for path in package_root(repo_root).iterdir() if path.is_dir()):
            if catalog_status.get(package_dir.name) != "valid":
                continue
            try:
                package = load_submission_package(package_dir, card_ids)
            except (OSError, PackageValidationError, ValueError):
                continue
            packages[package.name] = package
    internal_candidates = (
        repo_root / "work" / "alakazam_v8_current",
        repo_root / "submission" / "alakazam_v8_luna_deck_opt",
        repo_root / "submission" / "alakazam_v8",
    )
    for internal in internal_candidates:
        if not internal.is_dir():
            continue
        try:
            packages["internal_alakazam_v8_current"] = load_submission_package(
                internal,
                card_ids,
                name="internal_alakazam_v8_current",
            )
        except (OSError, PackageValidationError, ValueError):
            continue
        break
    return packages


def _materialize_source(
    repo_root: Path,
    metadata: object,
    source_dir: Path,
    store: ArenaStore,
    *,
    leaderboard: Mapping[str, object] | None = None,
) -> None:
    source_id = metadata.source_id()
    deck_id = source_id
    target = package_root(repo_root) / deck_id
    archives = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and (
            path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
            or path.suffix in {".tgz", ".zip", ".tar"}
        )
    )
    package_candidate: Path | None = None
    for archive in archives:
        extracted = source_dir / ".extracted" / archive.stem
        try:
            safe_extract_archive(archive, extracted)
        except (OSError, ValueError):
            continue
        package_candidate = _find_package(extracted)
        if package_candidate is not None:
            break
    baseline_cg = _baseline_cg(repo_root)
    original_package_hash: str | None = None
    original_cg_hash: str | None = None
    if package_candidate is not None:
        original_package_hash = _hash_tree(package_candidate)
        original_cg_hash = _hash_tree(package_candidate / "cg") if (package_candidate / "cg").is_dir() else None
        _copy_strategy_without_cg(package_candidate, target, baseline_cg)
        status = "exact_submission"
        deck = _read_deck_file(target / "deck.csv")
        adapter_error = None
    elif getattr(metadata, "competition", "") == CHALLENGE_COMPETITION:
        status = "metadata_only"
        deck = []
        adapter_error = "challenge-strategy source did not contain a submission archive"
    else:
        result = reconstruct_notebook_package(
            source_dir,
            target,
            baseline_cg,
        )
        status = result.status
        deck = list(result.deck)
        adapter_error = result.error
    card_metadata = load_card_metadata(repo_root / "data" / "official" / "EN_Card_Data.csv")
    override = _load_override(repo_root, source_id)
    archetype, primary_ids, display_name = classify_deck(deck, card_metadata, override)
    primary_pokemon = [
        {
            "card_id": card_id,
            "name": card_metadata.get(card_id, {}).get("Card Name", "Unknown Pokémon"),
            "count": deck.count(card_id),
            "image_url": card_image_url(card_metadata.get(card_id, {})),
        }
        for card_id in primary_ids
    ]
    metadata_payload = {
        "source_title": metadata.title,
        "source_url": metadata.url,
        "author": metadata.author,
        "votes": metadata.total_votes,
        "primary_pokemon": primary_pokemon,
        "adapter_status": status,
        "adapter_error": adapter_error,
        "original_package_hash": original_package_hash,
        "original_cg_hash": original_cg_hash,
    }
    if leaderboard:
        metadata_payload.update({f"public_{key}": value for key, value in leaderboard.items()})
    store.upsert_deck(
        DeckRecord(
            deck_id,
            source_id,
            display_name,
            archetype,
            primary_ids,
            str(target.relative_to(repo_root)),
            _hash_tree(target) if target.is_dir() else None,
            "public",
            status,
            metadata_payload,
        )
    )


def _collect_public_snapshots(
    repo_root: Path,
    competition: str,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    destination = source_root(repo_root) / "_meta" / competition
    snapshot: Mapping[str, object] = {"entries": ()}
    submissions: Mapping[str, object] = {"entries": ()}
    try:
        snapshot = collect_leaderboard_snapshot(competition, destination)
        submissions = collect_submissions_snapshot(competition, destination)
    except RuntimeError as exc:
        print(f"public score snapshot unavailable: {exc}", file=sys.stderr)
        submissions = {"entries": ()}
    entries = snapshot.get("entries", ())
    submission_entries = submissions.get("entries", ()) if isinstance(submissions, Mapping) else ()
    return (
        tuple(item for item in entries if isinstance(item, dict)),
        tuple(item for item in submission_entries if isinstance(item, dict)),
    )


def _has_source_artifact(source_dir: Path) -> bool:
    return any(
        path.is_file()
        and (
            path.suffix == ".ipynb"
            or path.name in {"main.py", "deck.csv"}
            or path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar", ".zip"))
        )
        for path in source_dir.rglob("*")
    )


def _public_score_match(
    metadata: object,
    leaderboard: tuple[dict[str, object], ...],
    submissions: tuple[dict[str, object], ...],
) -> Mapping[str, object] | None:
    raw = getattr(metadata, "raw", {})
    for key in ("publicScore", "public_score", "score"):
        if isinstance(raw, Mapping) and raw.get(key) not in (None, ""):
            return {"score": raw[key], "match": "kernel_metadata"}
    author = _normalize_text(str(getattr(metadata, "author", "")))
    title = _normalize_text(str(getattr(metadata, "title", "")))
    for entry in submissions:
        submitted_by = _normalize_text(str(entry.get("submittedBy", entry.get("submitted_by", ""))))
        if submitted_by and (submitted_by == author or submitted_by in title or author in submitted_by):
            return {
                "score": entry.get("publicScore", entry.get("public_score")),
                "submission_ref": entry.get("ref", entry.get("fileName", "")),
                "match": "submission_author",
            }
    for rank, entry in enumerate(leaderboard, start=1):
        team = _normalize_text(str(entry.get("teamName", "")))
        if not team:
            continue
        if team == author or team in title or author in team:
            return {
                "score": entry.get("score"),
                "rank": entry.get("rank", rank),
                "team_name": entry.get("teamName", ""),
                "match": "author_or_title",
            }
    return None


def _normalize_text(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _load_override(repo_root: Path, source_id: str) -> Mapping[str, object]:
    path = repo_root / "arena" / "catalog" / "overrides.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    value = payload.get(source_id, {}) if isinstance(payload, dict) else {}
    return value if isinstance(value, Mapping) else {}


def _read_deck_file(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _copy_strategy_without_cg(source: Path, target: Path, baseline_cg: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        for child in source.iterdir():
            if child.name in {"cg", "__pycache__"} or child.is_symlink():
                continue
            destination = temporary / child.name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)
        shutil.copytree(baseline_cg, temporary / "cg")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _replace_directory(temporary, target)


def _replace_directory(temporary: Path, target: Path) -> None:
    backup: Path | None = None
    try:
        if target.exists():
            backup = target.with_name(f".{target.name}.old-{os.getpid()}")
            if backup.exists():
                shutil.rmtree(backup)
            target.rename(backup)
        temporary.rename(target)
    except Exception:
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
        if backup is not None and backup.exists():
            backup.rename(target)
        raise
    finally:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _baseline_cg(repo_root: Path) -> Path:
    candidates = (
        repo_root / "work" / "alakazam_v8_current" / "cg",
        repo_root / "submission" / "alakazam_v8_luna_deck_opt" / "cg",
        repo_root / "submission" / "alakazam_v8" / "cg",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ValueError("official baseline cg runtime was not found")


def _ensure_runtime_deck_records(
    repo_root: Path,
    store: ArenaStore,
    packages: Mapping[str, SubmissionPackage],
) -> None:
    card_metadata = load_card_metadata(repo_root / "data" / "official" / "EN_Card_Data.csv")
    for name, package in packages.items():
        if any(str(row["deck_id"]) == name for row in store.load_decks()):
            continue
        archetype, primary_ids, display_name = classify_deck(package.deck, card_metadata, {})
        store.upsert_deck(
            DeckRecord(
                name,
                name,
                display_name,
                archetype,
                primary_ids,
                str(package.root.relative_to(repo_root)),
                package.package_hash,
                "internal_reference" if name.startswith("internal_") else "public",
                "valid",
                {"primary_pokemon": [{"card_id": card_id, "name": card_metadata[card_id].get("Card Name", "")} for card_id in primary_ids]},
            )
        )


def _update_deck_validation(
    store: ArenaStore,
    deck_id: str,
    status: str,
    error: str | None,
) -> None:
    row = next((item for item in store.load_decks() if str(item["deck_id"]) == deck_id), None)
    if row is None:
        return
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    if error:
        metadata["validation_error"] = error
    else:
        metadata.pop("validation_error", None)
    store.upsert_deck(
        DeckRecord(
            deck_id,
            str(row["source_id"]),
            str(row["display_name"]),
            str(row["archetype"]),
            tuple(int(value) for value in row.get("primary_pokemon_ids", ())),
            str(row["package_path"]),
            str(row["package_hash"]) if row.get("package_hash") else None,
            str(row["role"]),
            status,
            metadata,
        )
    )


def _write_source_catalog(repo_root: Path, store: ArenaStore) -> None:
    """把所有来源（包括没有可运行 package 的来源）写成可审计 Markdown。"""
    sources = store.load_sources()
    decks = {str(row["source_id"]): row for row in store.load_decks()}
    lines = [
        "# Kaggle 公共 Code 卡组目录",
        "",
        f"来源数：{len(sources)}。每次收集保留原始 metadata；public score 只记录可追溯的 snapshot 映射。",
        "",
        "| 来源状态 | 卡组状态 | 标志卡组 | 标题 | 作者 | Votes | Public score | 链接 |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    json_rows: list[dict[str, object]] = []
    for source in sources:
        source_id = str(source["source_id"])
        deck = decks.get(source_id, {})
        metadata = deck.get("metadata", {}) if isinstance(deck, Mapping) else {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        title = str(source.get("title", source_id)).replace("|", "/")
        author = str(source.get("author", "")).replace("|", "/")
        display = str(deck.get("display_name", "未重建"))
        score = metadata.get("public_score", "")
        lines.append(
            f"| {source.get('status', '')} | {deck.get('status', '未发现 package')} | {display} | "
            f"{title} | {author} | {source.get('metadata', {}).get('total_votes', '') if isinstance(source.get('metadata'), Mapping) else ''} | "
            f"{score} | [{source_id}]({source.get('url', '#')}) |"
        )
        json_rows.append({**source, "deck": deck})
    catalog_root = repo_root / "arena" / "catalog"
    catalog_root.mkdir(parents=True, exist_ok=True)
    (catalog_root / "kaggle-sources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (catalog_root / "kaggle-sources.json").write_text(
        json.dumps(json_rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _find_package(root: Path) -> Path | None:
    candidates = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    return next(
        (
            path
            for path in candidates
            if (path / "main.py").is_file() and (path / "deck.csv").is_file() and (path / "cg").is_dir()
        ),
        None,
    )


def _official_card_ids(repo_root: Path) -> set[int]:
    path = repo_root / "data" / "official" / "EN_Card_Data.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {int(row["Card ID"]) for row in csv.DictReader(handle)}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_args(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result
