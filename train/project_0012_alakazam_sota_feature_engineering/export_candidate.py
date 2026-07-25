from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch


MAIN = '''from pathlib import Path
import sys

def _root():
    source = globals().get("__file__")
    if isinstance(source, str) and (Path(source).resolve().parent / "deck.csv").is_file():
        return Path(source).resolve().parent
    kaggle = Path("/kaggle_simulations/agent")
    return kaggle if (kaggle / "deck.csv").is_file() else Path.cwd()

ROOT = _root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strategy.inference import FeatureEngineeringInference

DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
_POLICY = FeatureEngineeringInference.from_checkpoint(ROOT / "strategy" / "model.bin")

def read_deck_csv():
    return list(DECK)

def agent(observation):
    if observation.get("select") is None:
        _POLICY.reset()
        return list(DECK)
    return _POLICY.select(observation)
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_candidate(checkpoint: Path, source_package: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    for required in (source_package / "deck.csv", source_package / "cg"):
        if not required.exists():
            raise FileNotFoundError(required)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    if metadata.get("model_version") != "alakazam_sota_feature_engineering_bc_v1":
        raise ValueError("wrong checkpoint model version")
    strategy = output / "strategy"
    strategy.mkdir(parents=True)
    shutil.copy2(source_package / "deck.csv", output / "deck.csv")
    shutil.copytree(source_package / "cg", output / "cg")
    (output / "main.py").write_text(MAIN, encoding="utf-8")
    (strategy / "__init__.py").write_text("", encoding="utf-8")
    project = Path(__file__).resolve().parent
    base = project.parent / "project_0010_alakazam_sota_model" / "model.py"
    for source, name in (
        (base, "base_model.py"),
        (project / "model.py", "model.py"),
        (project / "codec.py", "codec.py"),
        (project / "inference.py", "inference.py"),
    ):
        shutil.copy2(source, strategy / name)
    portable = {
        "model": payload["model"],
        "metadata": {
            **metadata,
            "source_checkpoint_sha256": _sha256(checkpoint),
            "portable_checkpoint": True,
        },
    }
    model_path = strategy / "model.bin"
    torch.save(portable, model_path)
    manifest = {
        "schema_version": "alakazam_sota_feature_engineering_candidate_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_sha256": _sha256(model_path),
        "deck_sha256": _sha256(output / "deck.csv"),
        "source_package": str(source_package.resolve()),
    }
    (strategy / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a self-contained 0012 candidate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_candidate(args.checkpoint, args.source_package, args.output), indent=2))


if __name__ == "__main__":
    main()
