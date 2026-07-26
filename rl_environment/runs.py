"""Helpers for allocating tracked, globally numbered RL experiments."""

from __future__ import annotations

import re
import argparse
import hashlib
from html import escape
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPOSITORY_ROOT / "rl_runs"
EXPERIMENT_ROOT = RUNS_ROOT / "artifact"
CHECKPOINT_ROOT = RUNS_ROOT / "checkpoint"
NUMBERED_ARTIFACT = re.compile(r"^(?P<number>\d{4})-(?P<label>.+)$")
EXPERIMENT_DIR = re.compile(r"^(?P<number>\d{4})-(?P<label>[a-z0-9][a-z0-9_-]*)$")
VERSIONED_ATTEMPT = re.compile(r"^V(?P<version>[1-9]\d*)_(?P<tag>[a-z0-9][a-z0-9_]*)$")
PROJECT_ID = re.compile(r"^(?P<number>\d{4})_(?P<slug>[a-z0-9]+(?:_[a-z0-9]+)*)$")
WANDB_PROJECT = "pokemon-tcg-policy-learning"
WANDB_METRIC_SCHEMA = "ptcg_tracking_v1"


@dataclass(frozen=True)
class ProjectPaths:
    """Top-level directories allocated for one experiment project."""

    project_id: str
    train: Path
    archive: Path
    runs: Path
    manifest: Path


@dataclass(frozen=True)
class VersionPaths:
    project_id: str
    version_name: str
    project_archive: Path
    run_root: Path
    artifact: Path
    checkpoints: Path
    tensorboard: Path
    wandb: Path
    evaluation: Path
    config: Path
    metrics: Path
    summary: Path
    status: Path
    checkpoint_selection: Path
    model_contract: Path
    dataset_reference: Path
    metrics_snapshot: Path
    wandb_snapshot_manifest: Path


def project_version_paths(project_id: str, version_name: str) -> VersionPaths:
    if PROJECT_ID.fullmatch(project_id) is None:
        raise ValueError(f"invalid project ID: {project_id}")
    if VERSIONED_ATTEMPT.fullmatch(version_name) is None:
        raise ValueError(f"invalid version name: {version_name}")
    project_archive = REPOSITORY_ROOT / "experiments" / project_id
    run_root = REPOSITORY_ROOT / "rl_runs" / project_id / "versions" / version_name
    artifact = run_root / "artifact"
    return VersionPaths(
        project_id=project_id,
        version_name=version_name,
        project_archive=project_archive,
        run_root=run_root,
        artifact=artifact,
        checkpoints=run_root / "checkpoint",
        tensorboard=run_root / "tensorboard",
        wandb=run_root / "wandb",
        evaluation=project_archive / "evaluation" / f"{version_name}.html",
        config=artifact / "training_config.json",
        metrics=artifact / "training_metrics.jsonl",
        summary=artifact / "training_summary.json",
        status=artifact / "status.json",
        checkpoint_selection=artifact / "checkpoint_selection.json",
        model_contract=artifact / "model_contract.json",
        dataset_reference=artifact / "dataset_reference.json",
        metrics_snapshot=artifact / "metrics_snapshot.json",
        wandb_snapshot_manifest=artifact / "wandb_snapshot_manifest.json",
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_version_status(paths: VersionPaths, updates: dict[str, object]) -> None:
    """Atomically merge lifecycle updates into an allocated version's status."""
    try:
        current = json.loads(paths.status.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable version status: {paths.status}") from error
    if not isinstance(current, dict):
        raise ValueError(f"invalid version status: {paths.status}")
    current.update(updates)
    _write_json(paths.status, current)


def initialize_version(project_id: str, version_name: str) -> VersionPaths:
    """Allocate the next immutable runtime version for an initialized project."""
    paths = project_version_paths(project_id, version_name)
    if not (paths.project_archive / "manifest.json").is_file():
        raise FileNotFoundError(f"project manifest does not exist: {paths.project_archive / 'manifest.json'}")

    versions_root = paths.run_root.parent
    existing_versions = [
        int(match.group("version"))
        for child in versions_root.iterdir()
        if child.is_dir() and (match := VERSIONED_ATTEMPT.fullmatch(child.name))
    ] if versions_root.is_dir() else []
    requested = int(VERSIONED_ATTEMPT.fullmatch(version_name).group("version"))
    if paths.run_root.exists() or paths.evaluation.exists():
        raise FileExistsError(f"version assets already exist: {paths.run_root} or {paths.evaluation}")
    expected = max(existing_versions, default=0) + 1
    if requested != expected:
        raise ValueError(f"version must be V{expected}_<tag>, not {version_name}")

    try:
        for path in (paths.artifact, paths.checkpoints, paths.tensorboard, paths.wandb):
            path.mkdir(parents=True, exist_ok=False)
        _write_json(paths.status, {"state": "allocated", "version": version_name})
    except Exception:
        shutil.rmtree(paths.run_root, ignore_errors=True)
        raise
    return paths


def record_evaluation(paths: VersionPaths, *, run_id: str, report_sha256: str) -> None:
    """Create one immutable backlink to an existing archived evaluation report."""
    if not paths.evaluation.is_file():
        raise FileNotFoundError(f"evaluation report does not exist: {paths.evaluation}")
    record = paths.artifact / "evaluation.json"
    if record.exists():
        raise FileExistsError(f"evaluation provenance already exists: {record}")
    _write_json(
        record,
        {
            "report": str(paths.evaluation.relative_to(REPOSITORY_ROOT)),
            "report_sha256": report_sha256,
            "run_id": run_id,
            "version": paths.version_name,
        },
    )


@dataclass(frozen=True)
class TrainingPaths:
    """Separated paths for one training run."""

    run: Path
    checkpoints: Path
    tensorboard: Path
    config: Path
    metrics: Path
    summary: Path


def wandb_run_id(experiment_id: str, version_name: str) -> str:
    """Return a stable W&B-safe ID for one immutable repository version."""
    identity = f"{experiment_id}--{version_name}"
    normalized = re.sub(r"[^a-z0-9_-]+", "-", identity.lower()).strip("-_")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    prefix_length = 64 - len(digest) - 1
    prefix = normalized[:prefix_length].rstrip("-_") or "run"
    return f"{prefix}-{digest}"


def training_paths(output: Path) -> TrainingPaths:
    """Resolve one immutable version inside an experiment or a local smoke run."""
    resolved = output.resolve()
    runs_root = RUNS_ROOT.resolve()
    experiment_root = EXPERIMENT_ROOT.resolve()
    if resolved.parent == experiment_root:
        raise ValueError(
            "training output must be a versioned experiment child such as "
            f"{output / 'V1_initial_contract'}"
        )
    if resolved.parent.parent == experiment_root:
        experiment = resolved.parent
        if EXPERIMENT_DIR.fullmatch(experiment.name) is None:
            raise ValueError(f"invalid experiment directory name: {experiment.name}")
        if VERSIONED_ATTEMPT.fullmatch(resolved.name) is None:
            raise ValueError(
                "training attempt must match V<n>_<snake_case_tag>: "
                f"{resolved.name}"
            )
        paths = TrainingPaths(
            run=output,
            checkpoints=CHECKPOINT_ROOT / experiment.name / resolved.name,
            tensorboard=RUNS_ROOT / "tensorboard" / experiment.name / resolved.name,
            config=output / "training_config.json",
            metrics=output / "training_metrics.jsonl",
            summary=output / "training_summary.json",
        )
        occupied = [
            path
            for path in (paths.run, paths.checkpoints, paths.tensorboard)
            if path.is_file() or (path.is_dir() and any(child.is_file() for child in path.rglob("*")))
        ]
        if occupied:
            raise FileExistsError(
                "training attempt paths are already in use; allocate the next version: "
                + ", ".join(str(path) for path in occupied)
            )
        return paths
    return TrainingPaths(
        run=output,
        checkpoints=output / "checkpoints",
        tensorboard=output / "tensorboard",
        config=output / "config.json",
        metrics=output / "metrics.jsonl",
        summary=output / "summary.json",
    )


def numbered_artifact_path(requested: Path, group: str) -> Path:
    """Map legacy ``rl_runs/<label>/evaluation`` into the separated archive tree."""
    if group != "evaluation":
        raise ValueError(f"unknown RL artifact group: {group}")
    resolved = requested.resolve()
    runs_root = RUNS_ROOT.resolve()
    if resolved.name != "evaluation" or resolved.parent.parent != runs_root:
        return requested
    experiment = resolved.parent
    if NUMBERED_ARTIFACT.match(experiment.name):
        return RUNS_ROOT / "evaluation" / experiment.name
    normalized = re.sub(r"[^a-z0-9_-]+", "-", experiment.name.lower()).strip("-")
    if EXPERIMENT_ROOT.is_dir():
        for child in EXPERIMENT_ROOT.iterdir():
            match = EXPERIMENT_DIR.match(child.name)
            if child.is_dir() and match is not None and match.group("label") == normalized:
                return RUNS_ROOT / "evaluation" / child.name
    allocated = next_experiment_path(experiment.name, EXPERIMENT_ROOT)
    return RUNS_ROOT / "evaluation" / allocated.name


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _next_project_id() -> int:
    roots = (
        REPOSITORY_ROOT / "train",
        REPOSITORY_ROOT / "experiments",
        REPOSITORY_ROOT / "rl_runs",
    )
    ids: set[int] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            match = PROJECT_ID.fullmatch(child.name)
            if child.is_dir() and match is not None:
                ids.add(int(match.group("number")))
    return max(ids, default=12) + 1


def _required_value(name: str, value: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _project_templates(project_id: str, objective: str) -> dict[str, str]:
    return {
        "README.md": f"""# {project_id}\n\nInitialized outline for **{objective}**. Complete this archive before starting a training version; it does not assert experiment results.\n\n- [Design](DESIGN.md) · [Manifest](manifest.json) · [Data audit](data_audit/) · [Decisions](decisions/)\n- [Version decisions](versions/README.md) · [Evaluation index](evaluation/index.html)\n- Training implementation: `../../train/{project_id}/` · Runtime records: `../../rl_runs/{project_id}/`\n""",
        "DESIGN.md": f"""# {project_id} design\n\nInitialized outline requiring completion before a training version starts.\n\n## Current Stage\n\n## Input and Action Contract\n\n## Model and Objective\n\n## Data Contract\n\n## Evaluation Contract\n\n## Next Decision\n""",
        "DESIGN.html": f"""<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>{project_id} design</title></head>\n<body><h1>{project_id} design</h1><p>Initialized outline requiring completion before a training version starts.</p>\n<ul><li><a href=\"DESIGN.md\">Design outline</a></li><li><a href=\"README.md\">Project README</a></li><li><a href=\"manifest.json\">Manifest</a></li><li><a href=\"evaluation/index.html\">Evaluation index</a></li></ul></body></html>\n""",
        "versions/README.md": "# Version decisions\n\nInitialized outline: record one Markdown decision document per immutable `V<n>_<tag>` version before that version starts.\n",
        "evaluation/index.html": """<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Evaluation index</title></head>\n<body><h1>Evaluation index</h1><p>Initialized outline; no evaluation facts have been recorded.</p>\n<table><thead><tr><th>Version</th><th>Candidate</th><th>Run ID</th><th>Games</th><th>W/L/error</th><th>Win rate</th><th>Completion rate</th><th>Report</th></tr></thead><tbody></tbody></table></body></html>\n""",
    }


def _project_evaluation_link(root: Path, project_id: str) -> str:
    """Return the canonical project link or an existing legacy evaluation index."""
    match = PROJECT_ID.fullmatch(project_id)
    if match is None:
        raise ValueError(f"invalid project ID: {project_id}")
    if int(match.group("number")) >= 13:
        return f"{project_id}/evaluation/index.html"
    canonical = root / project_id / "evaluation" / "index.html"
    legacy_id = project_id.replace("_", "-", 1)
    legacy = root.parent / "rl_runs" / "evaluation" / legacy_id / "index.html"
    if canonical.is_file():
        return f"{project_id}/evaluation/index.html"
    if legacy.is_file():
        return f"../rl_runs/evaluation/{legacy_id}/index.html"
    return f"{project_id}/evaluation/index.html"


def _render_project_cards(project_rows: list[dict[str, str]]) -> str:
    cards = []
    for row in project_rows:
        project_id = escape(row["project_id"])
        objective = escape(row["objective"])
        evaluation = escape(row["evaluation"], quote=True)
        cards.append(
            '<article class="project">'
            f'<span class="project-id">{project_id}</span>'
            f'<h3>{project_id}</h3><p>{objective}</p><div class="links">'
            f'<a href="{project_id}/README.md">档案</a>'
            f'<a href="{project_id}/DESIGN.html">设计</a>'
            f'<a href="{evaluation}">评测</a></div></article>'
        )
    return "".join(cards)


def refresh_experiment_index() -> None:
    """Render discovery data while retaining an existing curated HTML shell."""
    root = REPOSITORY_ROOT / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    projects: list[dict[str, object]] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name == "legacy":
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            match = PROJECT_ID.fullmatch(child.name)
            if match is not None and int(match.group("number")) < 13:
                continue
            raise ValueError(f"experiment archive is missing manifest: {child}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"unreadable experiment manifest: {manifest_path}") from error
        if not isinstance(manifest, dict):
            raise ValueError(f"invalid experiment manifest: {manifest_path}")
        manifest_project_id = manifest.get("project_id")
        if manifest_project_id is None:
            match = PROJECT_ID.fullmatch(child.name)
            if match is None or int(match.group("number")) >= 13:
                raise ValueError(f"experiment manifest is missing project_id: {manifest_path}")
            indexed_manifest = {**manifest, "project_id": child.name}
        elif not isinstance(manifest_project_id, str):
            raise ValueError(f"invalid experiment manifest: {manifest_path}")
        elif manifest_project_id != child.name:
            raise ValueError(
                f"manifest project_id {manifest_project_id!r} does not match directory {child.name!r}"
            )
        else:
            indexed_manifest = manifest
        projects.append(indexed_manifest)
    projects.sort(key=lambda manifest: str(manifest["project_id"]))
    rows = [
        {
            "project_id": str(manifest["project_id"]),
            "objective": str(manifest.get("objective", "")),
            "status": str(manifest.get("status", "")),
            "evaluation": _project_evaluation_link(root, str(manifest["project_id"])),
        }
        for manifest in projects
    ]
    markdown = [
        "# Experiment projects",
        "",
        "| Project | Objective | Status | Archive | Design | Evaluation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        project_id = row["project_id"]
        markdown.append(
            f'| {project_id} | {row["objective"]} | {row["status"]} | '
            f'[{project_id}]({project_id}/README.md) | '
            f'[Design]({project_id}/DESIGN.html) | '
            f'[Evaluation]({row["evaluation"]}) |'
        )
    (root / "INDEX.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    html_path = root / "INDEX.html"
    existing = html_path.read_text(encoding="utf-8") if html_path.is_file() else ""
    cards = _render_project_cards(rows)
    project_section = re.compile(r'(<section class="projects">).*?(</section>)', re.DOTALL)
    if project_section.search(existing):
        rendered = project_section.sub(r"\1" + cards + r"\2", existing, count=1)
    else:
        rendered = (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<title>Experiment projects</title></head>\n<body><h1>Experiment projects</h1>'
            f'<section class="projects">{cards}</section></body></html>\n'
        )
    html_path.write_text(rendered, encoding="utf-8")


def initialize_project(
    slug: str,
    *,
    objective: str,
    deck: str,
    expert_source: str,
    dataset_contract: str,
    engine_revision: str,
    opponent_pool_snapshot: str,
) -> ProjectPaths:
    """Atomically allocate the durable roots and manifest for one RL project."""
    normalized_slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    if not normalized_slug or re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", normalized_slug) is None:
        raise ValueError("project slug must contain an ASCII letter or number")

    contract = {
        "objective": _required_value("objective", objective),
        "deck": _required_value("deck", deck),
        "expert_source": _required_value("expert_source", expert_source),
        "dataset_contract": _required_value("dataset_contract", dataset_contract),
        "engine_revision": _required_value("engine_revision", engine_revision),
        "opponent_pool_snapshot": _required_value("opponent_pool_snapshot", opponent_pool_snapshot),
    }
    project_id = f"{_next_project_id():04d}_{normalized_slug}"
    train = REPOSITORY_ROOT / "train" / project_id
    archive = REPOSITORY_ROOT / "experiments" / project_id
    runs = REPOSITORY_ROOT / "rl_runs" / project_id
    targets = (train, archive, runs)
    occupied = [path for path in targets if path.exists()]
    if occupied:
        raise FileExistsError("project paths already exist: " + ", ".join(map(str, occupied)))

    created: list[Path] = []
    manifest_path = archive / "manifest.json"
    temporary_manifest = archive / "manifest.json.tmp"
    try:
        for path in (train, archive, runs):
            path.mkdir(parents=True)
            created.append(path)
        for path in (
            archive / "data_audit",
            archive / "decisions",
            archive / "versions",
            archive / "evaluation",
            runs / "dataset",
            runs / "versions",
        ):
            path.mkdir()
        manifest = {
            "schema_version": "ptcg_experiment_project_v1",
            "project_id": project_id,
            **contract,
            "status": "initialized",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_status_porcelain": _git_value("status", "--short"),
            "paths": {
                "training": str(train.relative_to(REPOSITORY_ROOT)),
                "archive": str(archive.relative_to(REPOSITORY_ROOT)),
                "runs": str(runs.relative_to(REPOSITORY_ROOT)),
            },
        }
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
        for relative_path, contents in _project_templates(project_id, contract["objective"]).items():
            (archive / relative_path).write_text(contents, encoding="utf-8")
        refresh_experiment_index()
    except Exception:
        for path in reversed(created):
            shutil.rmtree(path, ignore_errors=True)
        raise

    return ProjectPaths(
        project_id=project_id,
        train=train,
        archive=archive,
        runs=runs,
        manifest=manifest_path,
    )


def next_experiment_path(label: str, runs_root: Path = EXPERIMENT_ROOT) -> Path:
    """Return the next globally numbered experiment directory."""
    normalized = re.sub(r"[^a-z0-9_-]+", "-", label.lower()).strip("-")
    if not normalized:
        raise ValueError("experiment label must contain an ASCII letter or number")
    numbers = [
        int(match.group("number"))
        for child in runs_root.iterdir()
        if child.is_dir() and (match := EXPERIMENT_DIR.match(child.name))
    ] if runs_root.is_dir() else []
    return runs_root / f"{max(numbers, default=0) + 1:04d}-{normalized}"


def initialize_experiment(
    label: str,
    *,
    objective: str,
    runs_root: Path = EXPERIMENT_ROOT,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Create a tracked run record without creating a supervised dataset."""
    root = next_experiment_path(label, runs_root)
    root.mkdir(parents=True)
    evaluation = RUNS_ROOT / "evaluation" / root.name
    evaluation.mkdir(parents=True)
    tensorboard = RUNS_ROOT / "tensorboard" / root.name
    tensorboard.mkdir(parents=True)
    checkpoint = CHECKPOINT_ROOT / root.name
    checkpoint.mkdir(parents=True)
    manifest = {
        "schema_version": "ptcg_experiment_v2",
        "experiment_id": root.name,
        "label": root.name.split("-", 1)[1],
        "objective": objective,
        "status": "initialized",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain": _git_value("status", "--short"),
        "paths": {
            "run": str(root.relative_to(RUNS_ROOT.parent)),
            "tensorboard": str(tensorboard.relative_to(RUNS_ROOT.parent)),
            "checkpoint": str(checkpoint.relative_to(RUNS_ROOT.parent)),
            "evaluation": str(evaluation.relative_to(RUNS_ROOT.parent)),
            "dataset": None,
        },
        "tracking": {
            "metric_schema": WANDB_METRIC_SCHEMA,
            "wandb_mode": "disabled",
            "wandb_project": WANDB_PROJECT,
        },
        **(metadata or {}),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _main() -> None:
    parser = argparse.ArgumentParser(description="Create a numbered RL experiment directory")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("label")
    create.add_argument("--objective", required=True)
    create_project = sub.add_parser("create-project")
    create_project.add_argument("slug")
    create_project.add_argument("--objective", required=True)
    create_project.add_argument("--deck", required=True)
    create_project.add_argument("--expert-source", required=True)
    create_project.add_argument("--dataset-contract", required=True)
    create_project.add_argument("--engine-revision", required=True)
    create_project.add_argument("--opponent-pool-snapshot", required=True)
    args = parser.parse_args()
    if args.command == "create":
        print(initialize_experiment(args.label, objective=args.objective))
    elif args.command == "create-project":
        paths = initialize_project(
            args.slug,
            objective=args.objective,
            deck=args.deck,
            expert_source=args.expert_source,
            dataset_contract=args.dataset_contract,
            engine_revision=args.engine_revision,
            opponent_pool_snapshot=args.opponent_pool_snapshot,
        )
        print(paths.archive)


if __name__ == "__main__":
    _main()
