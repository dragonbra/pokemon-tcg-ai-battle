from __future__ import annotations

from pathlib import Path

from .loader import SubmissionPackage, load_submission_package


def validate_submission_package(
    package_root: Path,
    official_card_ids: set[int],
) -> SubmissionPackage:
    """统一的提交 package 校验入口。"""
    return load_submission_package(package_root, official_card_ids)
