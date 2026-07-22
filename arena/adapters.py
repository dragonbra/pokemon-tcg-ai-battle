from __future__ import annotations

import ast
import json
import os
import shutil
import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path


DECK_NAMES = {"deck", "my_deck", "mydeck", "my-deck"}


@dataclass(frozen=True)
class AdapterResult:
    status: str
    package_path: Path
    deck: tuple[int, ...]
    error: str | None


def reconstruct_notebook_package(
    source_dir: Path,
    package_dir: Path,
    baseline_cg: Path,
) -> AdapterResult:
    static_source = _find_static_source(source_dir)
    if static_source is not None:
        main_path, deck_path = static_source
        try:
            source = main_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(main_path))
            deck = _read_deck_csv(deck_path)
            if len(deck) != 60:
                raise ValueError(f"deck.csv must contain 60 cards, got {len(deck)}")
            if not _defines_agent(tree):
                raise ValueError("main.py must define agent")
            compile(source, str(main_path), "exec")
        except (OSError, SyntaxError, ValueError, UnicodeError) as exc:
            return AdapterResult("invalid", package_dir, (), str(exc))
        _write_package(package_dir, source, deck, baseline_cg, companion_root=main_path.parent)
        return AdapterResult("source_reconstructed", package_dir, tuple(deck), None)

    notebooks = sorted(source_dir.rglob("*.ipynb"))
    if not notebooks:
        return AdapterResult("invalid", package_dir, (), "Notebook source was not found")
    try:
        source = _notebook_source(notebooks[0])
        tree = ast.parse(source, filename=str(notebooks[0]))
        deck = _find_deck(tree)
        if len(deck) != 60:
            raise ValueError(f"literal deck must contain 60 cards, got {len(deck)}")
        if not _defines_agent(tree):
            raise ValueError("Notebook source must define agent")
        compile(source, str(notebooks[0]), "exec")
    except (OSError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
        return AdapterResult("invalid", package_dir, (), str(exc))

    _write_package(package_dir, source, deck, baseline_cg)
    return AdapterResult("notebook_reconstructed", package_dir, tuple(deck), None)


def _find_static_source(source_dir: Path) -> tuple[Path, Path] | None:
    candidates = [
        (path / "main.py", path / "deck.csv")
        for path in sorted(
            (candidate for candidate in source_dir.rglob("*") if candidate.is_dir()),
            key=lambda item: (
                0 if item.name in {"submission", "selected_agent_build"} else 1,
                len(item.parts),
                item.as_posix(),
            ),
        )
        if (path / "main.py").is_file() and (path / "deck.csv").is_file()
    ]
    if (source_dir / "main.py").is_file() and (source_dir / "deck.csv").is_file():
        return source_dir / "main.py", source_dir / "deck.csv"
    return candidates[0] if candidates else None


def _read_deck_csv(path: Path) -> list[int]:
    deck: list[int] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                continue
            if len(row) != 1:
                raise ValueError("deck.csv must contain one card ID per line")
            try:
                deck.append(int(row[0].strip()))
            except ValueError as exc:
                raise ValueError(f"deck.csv contains a non-integer card ID: {row[0]!r}") from exc
    return deck


def _write_package(
    package_dir: Path,
    source: str,
    deck: list[int],
    baseline_cg: Path,
    *,
    companion_root: Path | None = None,
) -> None:
    if not baseline_cg.is_dir():
        raise ValueError(f"baseline cg directory was not found: {baseline_cg}")
    package_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{package_dir.name}.tmp-", dir=package_dir.parent))
    try:
        (temporary / "main.py").write_text(source, encoding="utf-8")
        (temporary / "deck.csv").write_text(
            "".join(f"{card_id}\n" for card_id in deck),
            encoding="utf-8",
        )
        if companion_root is not None:
            for child in companion_root.iterdir():
                if child.name in {"main.py", "deck.csv", "cg", "__pycache__"} or child.is_symlink():
                    continue
                if child.is_dir() and any(item.suffix == ".py" for item in child.rglob("*.py")):
                    shutil.copytree(child, temporary / child.name)
                    continue
                if child.is_file() and child.suffix.lower() in {".py", ".pt", ".bin", ".json", ".pkl", ".pickle"}:
                    shutil.copy2(child, temporary / child.name)
        shutil.copytree(baseline_cg, temporary / "cg")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    backup: Path | None = None
    try:
        if package_dir.exists():
            backup = package_dir.with_name(f".{package_dir.name}.old-{os.getpid()}")
            if backup.exists():
                shutil.rmtree(backup)
            package_dir.rename(backup)
        temporary.rename(package_dir)
    except Exception:
        if package_dir.exists():
            shutil.rmtree(package_dir)
        if backup is not None and backup.exists():
            backup.rename(package_dir)
        raise
    finally:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        chunks.extend(
            line for line in text.splitlines(keepends=True)
            if not line.lstrip().startswith(("%", "!", "%%"))
        )
    return "".join(chunks)


def _find_deck(tree: ast.AST) -> list[int]:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        names = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(_normalized_name(name) in DECK_NAMES for name in names):
            continue
        value = node.value
        parsed = _literal_sequence(value)
        if parsed is not None:
            return parsed
    raise ValueError("a literal 60-card deck list named DECK, my_deck or deck was not found")


def _literal_sequence(node: ast.AST) -> list[int] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [ast.literal_eval(item) for item in node.elts]
        return [int(value) for value in values]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
            values = _literal_sequence(node.left)
            if values is not None:
                return values * node.right.value
    return None


def _normalized_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id.lower().replace("_", "-")
    return ""


def _defines_agent(tree: ast.AST) -> bool:
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "agent" for node in ast.walk(tree))
