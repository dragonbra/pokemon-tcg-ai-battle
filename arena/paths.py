from __future__ import annotations

from pathlib import Path


def arena_root(repo_root: Path) -> Path:
    return repo_root.resolve() / "arena"


def source_root(repo_root: Path) -> Path:
    return arena_root(repo_root) / "sources"


def package_root(repo_root: Path) -> Path:
    return arena_root(repo_root) / "packages"


def run_root(repo_root: Path) -> Path:
    return arena_root(repo_root) / "runs"


def report_root(repo_root: Path) -> Path:
    return arena_root(repo_root) / "reports"
