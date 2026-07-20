from __future__ import annotations

from pathlib import Path


def is_complete_submission(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "main.py").is_file()
        and (path / "deck.csv").is_file()
        and (path / "cg").is_dir()
    )


def historical_submission_dirs(root: Path) -> list[Path]:
    submission_root = root / "submission"
    if not submission_root.is_dir():
        return []
    return sorted(
        path
        for path in submission_root.iterdir()
        if path.name != "dist" and is_complete_submission(path)
    )


def work_submission_dirs(root: Path) -> list[Path]:
    work_root = root / "work"
    if not work_root.is_dir():
        return []
    excluded_names = {"docs", "auto-iteration"}
    return sorted(
        path
        for path in work_root.iterdir()
        if path.name not in excluded_names and is_complete_submission(path)
    )


def resolve_submission(name: str, root: Path, prefer_work: bool = True) -> Path:
    directories = (
        work_submission_dirs(root) + historical_submission_dirs(root)
        if prefer_work
        else historical_submission_dirs(root) + work_submission_dirs(root)
    )
    for path in directories:
        if path.name == name:
            return path
    raise ValueError(f"unknown submission directory: {name}")
