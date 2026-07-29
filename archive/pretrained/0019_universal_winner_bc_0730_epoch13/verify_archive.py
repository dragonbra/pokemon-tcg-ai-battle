"""Verify file identity and strict model loading for this archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loader import load_model


ARCHIVE_ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((ARCHIVE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for relative_path, expected in manifest["file_sha256"].items():
        actual = _sha256(ARCHIVE_ROOT / relative_path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative_path}: {actual}")
    source_sha256 = _tree_sha256(ARCHIVE_ROOT / "model_source")
    if source_sha256 != manifest["model"]["model_source_bundle_sha256"]:
        raise ValueError(f"model source bundle SHA-256 mismatch: {source_sha256}")
    model = load_model("cpu")
    print(
        f"verified {manifest['asset_id']}: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,} parameters"
    )


if __name__ == "__main__":
    main()
