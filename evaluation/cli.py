from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from evaluation.packages.loader import (
    PackageValidationError,
    SubmissionPackage,
    load_submission_package,
)
from evaluation.cards import card_image_url, load_card_catalog
from evaluation.metrics.profiles import available_metric_profiles
from evaluation.runner.batch import BatchConfig, default_worker_count, run_batch
from evaluation.runtime import assert_cg_compatible
from rl_environment.runs import numbered_artifact_path


EXPECTED_FIELDS = frozenset(
    {"name", "package", "display_name", "representative_card_ids", "enabled", "tags"}
)
DEFAULT_CATALOG = Path(__file__).resolve().parent / "configs" / "opponents.json"
ARENA_OPPONENTS_RELATIVE = Path("arena") / "opponents"
DEFAULT_MAX_STEPS = 1_000
MIN_RESEARCH_GAMES = 10
RESEARCH_EVALUATION_ROOT = Path(__file__).resolve().parents[1] / "rl_runs"
_NUMBERED_OUTPUT = re.compile(r"^(?P<number>\d{4})-(?P<label>.+)$")


def _read_catalog_entries(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageValidationError(f"could not read opponent catalog: {exc}") from exc

    if not isinstance(payload, dict) or set(payload) != {"opponents"}:
        raise PackageValidationError("opponent catalog must contain only an opponents list")
    entries = payload["opponents"]
    if not isinstance(entries, list):
        raise PackageValidationError("opponents must be a list")
    result: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PackageValidationError(f"catalog entry {index} must be an object")
        unknown = sorted(set(entry) - EXPECTED_FIELDS)
        if unknown:
            raise PackageValidationError(
                f"catalog entry {index} has unknown field: {unknown[0]}"
            )
        if set(entry) != EXPECTED_FIELDS:
            missing = sorted(EXPECTED_FIELDS - set(entry))
            raise PackageValidationError(
                f"catalog entry {index} is missing field: {missing[0]}"
            )
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise PackageValidationError(f"catalog entry {index} name must be a non-empty string")
        if not isinstance(entry["package"], str):
            raise PackageValidationError(f"catalog entry {index} package must be a string")
        if not isinstance(entry["display_name"], str) or not entry["display_name"]:
            raise PackageValidationError(
                f"catalog entry {index} display_name must be a non-empty string"
            )
        representative_card_ids = entry["representative_card_ids"]
        if (
            not isinstance(representative_card_ids, list)
            or not 1 <= len(representative_card_ids) <= 2
            or not all(type(card_id) is int and card_id > 0 for card_id in representative_card_ids)
            or len(set(representative_card_ids)) != len(representative_card_ids)
        ):
            raise PackageValidationError(
                f"catalog entry {index} representative_card_ids must contain 1-2 unique card IDs"
            )
        if type(entry["enabled"]) is not bool:
            raise PackageValidationError(f"catalog entry {index} enabled must be a boolean")
        if not isinstance(entry["tags"], list) or not all(
            isinstance(tag, str) for tag in entry["tags"]
        ):
            raise PackageValidationError(f"catalog entry {index} tags must be a list of strings")
        result.append(entry)
    return result


def _package_path(package_value: str, evaluation_root: Path) -> Path:
    relative_package = Path(package_value)
    if relative_package.is_absolute():
        raise PackageValidationError("catalog package path must be inside evaluation root")
    package_root = (evaluation_root / relative_package).resolve()
    resolved_evaluation_root = evaluation_root.resolve()
    arena_opponents_root = (resolved_evaluation_root / ARENA_OPPONENTS_RELATIVE).resolve()
    try:
        package_root.relative_to(arena_opponents_root)
    except ValueError as exc:
        raise PackageValidationError(
            "catalog package path must be inside evaluation/arena/opponents"
        ) from exc
    if not package_root.is_dir():
        raise PackageValidationError(f"catalog package does not exist: {package_root}")
    return package_root


def _load_official_card_ids(evaluation_root: Path) -> set[int]:
    card_path = evaluation_root.parent / "data" / "official" / "EN_Card_Data.csv"
    if not card_path.is_file():
        raise PackageValidationError(f"official card data does not exist: {card_path}")
    card_ids: set[int] = set()
    with card_path.open(newline="", encoding="utf-8-sig") as card_file:
        for row in csv.reader(card_file):
            if row and row[0] != "Card ID":
                card_ids.add(int(row[0].split(":")[-1]))
    return card_ids


def _validate_catalog_entries(
    entries: list[dict[str, Any]],
    evaluation_root: Path,
) -> list[tuple[dict[str, Any], Path]]:
    validated_entries: list[tuple[dict[str, Any], Path]] = []
    seen_names: set[str] = set()
    for entry in entries:
        name = entry["name"]
        if name in seen_names:
            raise PackageValidationError(f"duplicate opponent name: {name}")
        seen_names.add(name)
        validated_entries.append((entry, _package_path(entry["package"], evaluation_root)))
    return validated_entries


def _load_catalog(
    path: Path,
    evaluation_root: Path,
    official_card_ids: set[int],
) -> list[SubmissionPackage]:
    packages: list[SubmissionPackage] = []
    entries = _validate_catalog_entries(_read_catalog_entries(path), evaluation_root)
    card_catalog = load_card_catalog(
        evaluation_root.parent / "data" / "official" / "EN_Card_Data.csv"
    )
    for entry, package_root in entries:
        if not entry["enabled"]:
            continue
        package = load_submission_package(package_root, official_card_ids, name=entry["name"])
        representative_cards = []
        for card_id in entry["representative_card_ids"]:
            if card_id not in package.deck:
                raise PackageValidationError(
                    f"representative card ID {card_id} is not in opponent deck: {entry['name']}"
                )
            metadata = card_catalog.get(card_id)
            if metadata is None or "Pokémon" not in metadata["stage_or_type"]:
                raise PackageValidationError(
                    f"representative card ID {card_id} must be an official Pokémon card"
                )
            representative_cards.append(
                {
                    "card_id": card_id,
                    "name": metadata["name"],
                    "image_url": card_image_url(
                        metadata["expansion"], metadata["collection_number"]
                    ),
                }
            )
        packages.append(
            replace(
                package,
                display_name=entry["display_name"],
                representative_cards=tuple(representative_cards),
            )
        )
    return packages


def load_opponent_catalog(path: Path, evaluation_root: Path) -> list[SubmissionPackage]:
    """读取启用的 opponent，并验证其标准 Submission Package 契约。"""
    return _load_catalog(path, evaluation_root, _load_official_card_ids(evaluation_root))


def list_enabled_opponents(path: Path) -> list[str]:
    """按 catalog 顺序返回启用 opponent 的名称。"""
    evaluation_root = path.resolve().parent.parent
    entries = _validate_catalog_entries(_read_catalog_entries(path), evaluation_root)
    return [entry["name"] for entry, _ in entries if entry["enabled"]]


def validate_catalog(path: Path, official_card_ids: set[int]) -> list[SubmissionPackage]:
    """使用仓库官方卡 ID 验证 catalog 中所有启用 package。"""
    return _load_catalog(path, path.resolve().parent.parent, official_card_ids)


def _selected_opponents(
    requested: str,
    available: Iterable[SubmissionPackage],
) -> tuple[SubmissionPackage, ...]:
    packages = tuple(available)
    by_name = {package.name: package for package in packages}
    if requested.strip() == "all":
        return packages

    names = [name.strip() for name in requested.split(",") if name.strip()]
    if not names:
        raise PackageValidationError("--opponents must be all or a comma-separated opponent list")
    duplicate_names = {name for name in names if names.count(name) > 1}
    if duplicate_names:
        raise PackageValidationError(f"duplicate opponent: {sorted(duplicate_names)[0]}")
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise PackageValidationError(f"unknown opponent: {unknown[0]}")
    return tuple(by_name[name] for name in names)


def _metric_modules(values: Iterable[str]) -> tuple[str, ...]:
    modules = tuple(
        module.strip()
        for value in values
        for module in value.split(",")
        if module.strip()
    )
    if len(set(modules)) != len(modules):
        raise PackageValidationError("duplicate metric module")
    return modules


def _validate_positive(value: int, argument: str) -> None:
    if value < 1:
        raise PackageValidationError(f"{argument} must be at least one")


def _numbered_research_output_root(requested: Path) -> Path:
    """Allocate a globally numbered report below ``rl_runs``."""
    return numbered_artifact_path(requested, "evaluation")


def _validate_research_coverage(
    requested_opponents: str,
    games: int,
    opponents: tuple[SubmissionPackage, ...],
) -> None:
    """Enforce full-catalog evaluation with at least ten games per opponent."""
    minimum_total_games = len(opponents) * MIN_RESEARCH_GAMES
    if games < MIN_RESEARCH_GAMES:
        raise PackageValidationError(
            f"repo evaluation requires at least {MIN_RESEARCH_GAMES} games per opponent "
            f"({len(opponents)}×{MIN_RESEARCH_GAMES}="
            f"{minimum_total_games}); got {games}"
        )
    if requested_opponents.strip() != "all":
        raise PackageValidationError("repo evaluation requires --opponents all")


def _write_validation(package: SubmissionPackage) -> None:
    print(f"name: {package.name}")
    print(f"deck_hash: {package.deck_hash}")
    print(f"cg_tree_hash: {package.cg_manifest['tree_hash']}")
    print("60-card valid: true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="宝可梦 TCG 提交 package 评测")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-opponents", help="列出启用的 opponent")

    validate = subparsers.add_parser("validate", help="预检一个标准 Submission Package")
    validate.add_argument("package", type=Path)

    run = subparsers.add_parser("run", help="运行候选与 catalog opponent 的批量评测")
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--opponents", required=True, help="all 或以逗号分隔的 opponent 名称")
    run.add_argument(
        "--games",
        type=int,
        default=10,
        help="每个 opponent 的对局数（默认且至少 10；固定 18 opponent catalog）",
    )
    run.add_argument("--output", type=Path, required=True, help="评测报告根目录；RL 路径自动编号")
    run.add_argument("--control", type=Path, help="仅用于报告对比展示的标准 package")
    run.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="调试时在临时 trace 中生成可视化帧（默认关闭）",
    )
    run.add_argument("--no-visualize", dest="visualize", action="store_false")
    run.add_argument("--keep-temp", action="store_true")
    run.add_argument(
        "--save-traces",
        dest="keep_temp",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    run.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行对局 worker 数；默认按 CPU affinity 自动选择，最多 8",
    )
    run.add_argument(
        "--worker-cpu-threads",
        type=int,
        default=1,
        help="限制每个 worker 的 OMP/MKL CPU 线程数",
    )
    run.add_argument(
        "--metric-profile",
        choices=available_metric_profiles(),
        default="core",
        help="选择内置 metric profile",
    )
    run.add_argument(
        "--metric-module",
        action="append",
        default=[],
        help="额外指标插件模块路径，可写 MODULE[:Class]；可重复或以逗号分隔",
    )
    run.set_defaults(visualize=False)
    return parser


def _run(args: argparse.Namespace) -> str:
    evaluation_root = args.catalog.resolve().parent.parent
    official_card_ids = _load_official_card_ids(evaluation_root)
    _validate_positive(args.games, "--games")
    _validate_positive(args.max_steps, "--max-steps")
    if args.workers is not None:
        _validate_positive(args.workers, "--workers")
    if args.worker_cpu_threads is not None:
        _validate_positive(args.worker_cpu_threads, "--worker-cpu-threads")

    candidate = load_submission_package(args.candidate, official_card_ids)
    control = (
        load_submission_package(args.control, official_card_ids) if args.control is not None else None
    )
    opponents = _selected_opponents(
        args.opponents,
        load_opponent_catalog(args.catalog, evaluation_root),
    )
    _validate_research_coverage(args.opponents, args.games, opponents)
    for opponent in opponents:
        assert_cg_compatible(candidate, opponent)

    args.output = _numbered_research_output_root(args.output)

    result = run_batch(
        BatchConfig(
            candidate=candidate,
            opponents=opponents,
            games_per_opponent=args.games,
            output_root=args.output,
            visualize=args.visualize,
            max_steps=args.max_steps,
            control=control,
            metric_module_paths=_metric_modules(args.metric_module),
            metric_profile_id=args.metric_profile,
            keep_temp=args.keep_temp,
            workers=args.workers if args.workers is not None else default_worker_count(),
            worker_cpu_threads=args.worker_cpu_threads,
        )
    )
    return result.run_id


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-opponents":
            print("\n".join(list_enabled_opponents(args.catalog)))
            return 0
        if args.command == "validate":
            evaluation_root = args.catalog.resolve().parent.parent
            package = load_submission_package(
                args.package,
                _load_official_card_ids(evaluation_root),
            )
            _write_validation(package)
            return 0
        run_id = _run(args)
    except (PackageValidationError, ValueError) as exc:
        parser.error(str(exc))
    else:
        print(f"run_id: {run_id}")
        print(f"report: {(args.output / run_id).resolve()}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
