"""Build the small private Kaggle Dataset used by the BC worker Notebook.

The generated directory contains code, frozen source manifests, exact decks,
and the shared model config.  It does not contain replay payloads, processed
datasets, checkpoints, secrets, official competition data, or Competition Use
Only cg binaries.  The Notebook copies the card CSV and cg from its attached
official competition input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from train.kaggle_bc_top20.worker import INPUT_SCHEMA, JOB_SCHEMA


DEFAULT_CAMPAIGN = Path("rl_runs/top20_bc_campaign_20260723.json")
DEFAULT_SHARED_CONFIG = Path("train/alakazam_bc_rl/shared_model_config.json")
DEFAULT_DAILY_WINNER_DECK = Path("work/yushin_ito_exact_bc_v2/deck.csv")
def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_campaign_jobs(
    sources: Iterable[dict[str, Any]],
    *,
    include_orders: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Return unfinished jobs, preserving frozen roster order."""
    rows = []
    seen_orders: set[int] = set()
    for source in sources:
        order = int(source["selected_roster_order"])
        if order in seen_orders:
            raise ValueError(f"duplicate selected_roster_order: {order}")
        seen_orders.add(order)
        if source.get("v1_complete"):
            continue
        if include_orders is not None and order not in include_orders:
            continue
        rows.append(dict(source))
    rows.sort(key=lambda row: int(row["selected_roster_order"]))
    if include_orders is not None:
        selected = {int(row["selected_roster_order"]) for row in rows}
        missing = sorted(include_orders - selected)
        if missing:
            raise ValueError(f"requested job orders are unavailable or already complete: {missing}")
    return rows


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_python_tree(
    source: Path,
    destination: Path,
    *,
    excluded_top_level: set[str] | None = None,
) -> None:
    excluded = excluded_top_level or set()
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0] in excluded:
            continue
        _copy_file(path, destination / relative)


def _copy_runtime_source(repository_root: Path, bundle_repo: Path) -> None:
    _copy_file(repository_root / "pyproject.toml", bundle_repo / "pyproject.toml")
    _copy_python_tree(repository_root / "rl_environment", bundle_repo / "rl_environment")
    _copy_python_tree(repository_root / "train/alakazam_bc_rl", bundle_repo / "train/alakazam_bc_rl")
    _copy_python_tree(repository_root / "train/kaggle_bc_top20", bundle_repo / "train/kaggle_bc_top20")
    _copy_python_tree(
        repository_root / "evaluation",
        bundle_repo / "evaluation",
        excluded_top_level={"opponents"},
    )


def _git_value(repository_root: Path, command: list[str]) -> str | None:
    result = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _bundle_hashes(output: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        if path.name == "ptcg_kaggle_bc_input.json":
            continue
        result[path.relative_to(output).as_posix()] = _sha256(path)
    return result


def build_input_bundle(
    *,
    repository_root: Path,
    output: Path,
    campaign_path: Path,
    shared_config_path: Path,
    owner: str,
    dataset_slug: str,
    include_orders: set[int] | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output = output.resolve()
    campaign_path = (repository_root / campaign_path).resolve()
    shared_config_path = (repository_root / shared_config_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle directory: {output}")
    campaign = _read_json(campaign_path)
    shared = _read_json(shared_config_path)
    if shared.get("read_only") is not True:
        raise ValueError("shared model config is not frozen read-only")
    jobs = select_campaign_jobs(
        campaign.get("sources") or [],
        include_orders=include_orders,
    )
    if not jobs:
        raise ValueError("campaign selection produced no unfinished jobs")

    output.mkdir(parents=True)
    bundle_repo = output / "repo"
    _copy_runtime_source(repository_root, bundle_repo)
    _copy_file(shared_config_path, output / "config/shared_model_config.json")

    job_rows: list[dict[str, Any]] = []
    for source in jobs:
        order = int(source["selected_roster_order"])
        package_name = str(source["package_name"])
        run_root = (repository_root / str(source["run_root"])).resolve()
        source_manifest = run_root / "source_manifest.json"
        deck = run_root / "source_deck.csv"
        frozen = _read_json(source_manifest)
        identity = frozen.get("source_identity") or {}
        if identity != source.get("source_identity"):
            raise ValueError(f"campaign/source identity mismatch for {package_name}")
        job_root = output / "jobs" / f"{order:02d}-{package_name}"
        _copy_file(source_manifest, job_root / "source_manifest.json")
        _copy_file(deck, job_root / "source_deck.csv")
        job_rows.append(
            {
                "schema_version": JOB_SCHEMA,
                "selected_roster_order": order,
                "experiment_id": str(source["experiment_id"]),
                "package_name": package_name,
                "team_name": str(source["team_name"]),
                "source_identity": identity,
                "source_manifest": (
                    job_root / "source_manifest.json"
                ).relative_to(output).as_posix(),
                "deck": (job_root / "source_deck.csv").relative_to(output).as_posix(),
            }
        )

    jobs_path = output / "jobs.json"
    _write_json(
        jobs_path,
        {
            "schema_version": "ptcg_kaggle_bc_jobs_v1",
            "created_at": _timestamp(),
            "campaign_sha256": _sha256(campaign_path),
            "excluded_completed_orders": sorted(
                int(row["selected_roster_order"])
                for row in campaign.get("sources") or []
                if row.get("v1_complete")
            ),
            "jobs": job_rows,
        },
    )
    _write_json(
        output / "dataset-metadata.json",
        {
            "title": "Pokemon TCG private BC cloud input",
            "id": f"{owner}/{dataset_slug}",
            "licenses": [{"name": "other"}],
        },
    )
    manifest = {
        "schema_version": INPUT_SCHEMA,
        "created_at": _timestamp(),
        "private_dataset_required": True,
        "contains_secrets": False,
        "contains_raw_replays": False,
        "repository_commit": _git_value(repository_root, ["git", "rev-parse", "HEAD"]),
        "repository_status_sha256": hashlib.sha256(
            (_git_value(repository_root, ["git", "status", "--porcelain=v1"]) or "").encode(
                "utf-8"
            )
        ).hexdigest(),
        "campaign_source": str(campaign_path),
        "campaign_sha256": _sha256(campaign_path),
        "shared_model_config": "config/shared_model_config.json",
        "shared_model_config_sha256": _sha256(shared_config_path),
        "repo_dir": "repo",
        "cg_source": "runtime/cg",
        "cg_source_contract": (
            "copied at runtime from the attached pokemon-tcg-ai-battle "
            "competition sample_submission"
        ),
        "jobs_file": jobs_path.relative_to(output).as_posix(),
        "job_orders": [int(row["selected_roster_order"]) for row in job_rows],
        "files_sha256": _bundle_hashes(output),
    }
    _write_json(output / "ptcg_kaggle_bc_input.json", manifest)
    return manifest


def build_upload_directory(
    *,
    repository_root: Path,
    output: Path,
    campaign_path: Path,
    shared_config_path: Path,
    owner: str,
    dataset_slug: str,
    include_orders: set[int] | None = None,
) -> dict[str, Any]:
    """Create a Kaggle Dataset upload directory with one hash-audited bundle archive."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite upload directory: {output}")
    output.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="ptcg-kaggle-bc-input-") as temporary:
            bundle_root = Path(temporary) / "bundle"
            input_manifest = build_input_bundle(
                repository_root=repository_root,
                output=bundle_root,
                campaign_path=campaign_path,
                shared_config_path=shared_config_path,
                owner=owner,
                dataset_slug=dataset_slug,
                include_orders=include_orders,
            )
            archive = output / "ptcg_kaggle_bc_input.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for path in sorted(bundle_root.rglob("*")):
                    handle.add(
                        path,
                        arcname=path.relative_to(bundle_root).as_posix(),
                        recursive=False,
                    )
        _write_json(
            output / "dataset-metadata.json",
            {
                "title": "Pokemon TCG private BC cloud input",
                "id": f"{owner}/{dataset_slug}",
                "licenses": [{"name": "other"}],
            },
        )
        upload_manifest = {
            "schema_version": "ptcg_kaggle_bc_upload_v1",
            "created_at": _timestamp(),
            "archive": archive.name,
            "archive_sha256": _sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "job_orders": input_manifest["job_orders"],
            "private_dataset_required": True,
        }
        _write_json(output / "upload_manifest.json", upload_manifest)
        return upload_manifest
    except Exception:
        shutil.rmtree(output)
        raise


def build_daily_winner_upload_directory(
    *,
    repository_root: Path,
    output: Path,
    shared_config_path: Path,
    deck_path: Path,
    owner: str,
    dataset_slug: str,
    teacher_team: str,
) -> dict[str, Any]:
    """Create the code-only input for the reference-corpus model comparison."""
    repository_root = repository_root.resolve()
    output = output.resolve()
    shared_config_path = (repository_root / shared_config_path).resolve()
    deck_path = (repository_root / deck_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite upload directory: {output}")
    shared = _read_json(shared_config_path)
    if shared.get("read_only") is not True:
        raise ValueError("shared model config is not frozen read-only")
    cards = [
        int(line.strip())
        for line in deck_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError("daily winner runtime deck must contain exactly 60 card IDs")
    canonical_deck = hashlib.sha256(
        ",".join(str(value) for value in sorted(cards)).encode("ascii")
    ).hexdigest()
    output.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="ptcg-daily-winner-input-") as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            bundle_repo = bundle / "repo"
            _copy_runtime_source(repository_root, bundle_repo)
            _copy_file(shared_config_path, bundle / "config/shared_model_config.json")
            job_root = bundle / "jobs/01-yushin_daily_winner_our_model"
            _copy_file(deck_path, job_root / "source_deck.csv")
            source_manifest = {
                "schema_version": "ptcg_daily_team_winner_source_v1",
                "source_identity": {
                    "selection": "daily_team_name_winner_only",
                    "team_name": teacher_team,
                    "run_dates": [f"2026-07-{day:02d}" for day in range(13, 23)],
                    "deck_sha256": canonical_deck,
                },
            }
            source_path = job_root / "source_manifest.json"
            _write_json(source_path, source_manifest)
            job = {
                "schema_version": JOB_SCHEMA,
                "selected_roster_order": 1,
                "experiment_id": "0025-yushin_daily_winner_our_model",
                "package_name": "yushin_daily_winner_our_model",
                "team_name": teacher_team,
                "source_identity": source_manifest["source_identity"],
                "source_manifest": source_path.relative_to(bundle).as_posix(),
                "deck": (job_root / "source_deck.csv").relative_to(bundle).as_posix(),
            }
            jobs_path = bundle / "jobs.json"
            _write_json(
                jobs_path,
                {
                    "schema_version": "ptcg_kaggle_bc_jobs_v1",
                    "created_at": _timestamp(),
                    "jobs": [job],
                },
            )
            manifest = {
                "schema_version": INPUT_SCHEMA,
                "created_at": _timestamp(),
                "private_dataset_required": True,
                "contains_secrets": False,
                "contains_raw_replays": False,
                "corpus_mode": "daily_team_winners",
                "teacher_team": teacher_team,
                "repository_commit": _git_value(
                    repository_root, ["git", "rev-parse", "HEAD"]
                ),
                "shared_model_config": "config/shared_model_config.json",
                "shared_model_config_sha256": _sha256(shared_config_path),
                "repo_dir": "repo",
                "cg_source": "runtime/cg",
                "cg_source_contract": (
                    "copied at runtime from the attached pokemon-tcg-ai-battle "
                    "competition sample_submission"
                ),
                "jobs_file": jobs_path.relative_to(bundle).as_posix(),
                "job_orders": [1],
                "files_sha256": _bundle_hashes(bundle),
            }
            _write_json(bundle / "ptcg_kaggle_bc_input.json", manifest)
            archive = output / "ptcg_kaggle_bc_input.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for path in sorted(bundle.rglob("*")):
                    handle.add(
                        path,
                        arcname=path.relative_to(bundle).as_posix(),
                        recursive=False,
                    )
        _write_json(
            output / "dataset-metadata.json",
            {
                "title": "Pokemon TCG Yushin daily winner model comparison input",
                "id": f"{owner}/{dataset_slug}",
                "licenses": [{"name": "other"}],
            },
        )
        upload_manifest = {
            "schema_version": "ptcg_kaggle_bc_upload_v1",
            "created_at": _timestamp(),
            "archive": archive.name,
            "archive_sha256": _sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "job_orders": [1],
            "corpus_mode": "daily_team_winners",
            "teacher_team": teacher_team,
            "private_dataset_required": True,
        }
        _write_json(output / "upload_manifest.json", upload_manifest)
        return upload_manifest
    except Exception:
        shutil.rmtree(output)
        raise


def _parse_orders(value: str | None) -> set[int] | None:
    if value is None:
        return None
    orders = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not orders or any(order < 1 for order in orders):
        raise ValueError("--orders must be a comma-separated list of positive integers")
    return orders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--shared-config", type=Path, default=DEFAULT_SHARED_CONFIG)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="pokemon-tcg-bc-cloud-input")
    parser.add_argument("--orders", help="optional comma-separated unfinished roster orders")
    parser.add_argument(
        "--daily-winner-team",
        help="build a standalone daily TeamNames + winner-only comparison input",
    )
    parser.add_argument("--deck", type=Path, default=DEFAULT_DAILY_WINNER_DECK)
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    if args.daily_winner_team:
        if args.orders:
            parser.error("--orders cannot be combined with --daily-winner-team")
        manifest = build_daily_winner_upload_directory(
            repository_root=repository_root,
            output=args.output,
            shared_config_path=args.shared_config,
            deck_path=args.deck,
            owner=args.owner,
            dataset_slug=args.dataset_slug,
            teacher_team=args.daily_winner_team,
        )
    else:
        manifest = build_upload_directory(
            repository_root=repository_root,
            output=args.output,
            campaign_path=args.campaign,
            shared_config_path=args.shared_config,
            owner=args.owner,
            dataset_slug=args.dataset_slug,
            include_orders=_parse_orders(args.orders),
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
