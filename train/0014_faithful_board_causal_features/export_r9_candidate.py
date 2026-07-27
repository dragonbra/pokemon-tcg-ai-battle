"""Export an R9 checkpoint as a self-contained arena candidate."""
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


def _submission_root():
    source_file = globals().get("__file__")
    if isinstance(source_file, str):
        root = Path(source_file).resolve().parent
        if (root / "deck.csv").is_file():
            return root
    kaggle_root = Path("/kaggle_simulations/agent")
    if (kaggle_root / "deck.csv").is_file():
        return kaggle_root
    return Path.cwd()


ROOT = _submission_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.r9_inference import R9Policy

DECK = [
    int(line)
    for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
_POLICY = R9Policy.from_checkpoint(
    ROOT / "strategy" / "model.bin",
    ROOT / "strategy" / "card_ontology.json",
    DECK,
)


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


def export_candidate(
    checkpoint: Path,
    ontology: Path,
    source_package: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    for required in (checkpoint, ontology, source_package / "deck.csv", source_package / "cg"):
        if not required.exists():
            raise FileNotFoundError(required)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    if metadata.get("schema_version") != "0014_r9_training_v1":
        raise ValueError("checkpoint is not a 0014 R9 policy")

    strategy = output / "strategy"
    (strategy / "features").mkdir(parents=True)
    (strategy / "knowledge").mkdir(parents=True)
    shutil.copy2(source_package / "deck.csv", output / "deck.csv")
    shutil.copytree(source_package / "cg", output / "cg")
    (output / "main.py").write_text(MAIN, encoding="utf-8")
    (strategy / "__init__.py").write_text(
        'PROJECT_ID = "0014_faithful_board_causal_features"\n', encoding="utf-8"
    )
    (strategy / "features/__init__.py").write_text("", encoding="utf-8")
    (strategy / "knowledge/__init__.py").write_text("", encoding="utf-8")
    project = Path(__file__).resolve().parent
    copies = {
        "base_model.py": "base_model.py",
        "model.py": "model.py",
        "ac_model.py": "ac_model.py",
        "r5_model.py": "r5_model.py",
        "r7_model.py": "r7_model.py",
        "r9_model.py": "r9_model.py",
        "r9_inference.py": "r9_inference.py",
        "card_features.py": "card_features.py",
        "card_semantics.py": "card_semantics.py",
        "config.py": "config.py",
        "online_runtime.py": "online_runtime.py",
        "features/compiler.py": "features/compiler.py",
        "knowledge/state.py": "knowledge/state.py",
        "knowledge/ledger.py": "knowledge/ledger.py",
    }
    for source, target in copies.items():
        shutil.copy2(project / source, strategy / target)
    shutil.copy2(ontology, strategy / "card_ontology.json")
    portable = {
        "model": payload["model"],
        "metadata": {
            **metadata,
            "portable_checkpoint": True,
            "source_checkpoint_sha256": _sha256(checkpoint),
        },
    }
    model_path = strategy / "model.bin"
    torch.save(portable, model_path)
    manifest = {
        "schema_version": "0014_r9_candidate_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "ontology_sha256": _sha256(strategy / "card_ontology.json"),
        "deck_sha256": _sha256(output / "deck.csv"),
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "source_package": str(source_package.resolve()),
    }
    (strategy / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_candidate(**vars(args)), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
