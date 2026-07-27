"""Export an R2 checkpoint as a self-contained arena candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from . import PROJECT_ID, TARGET_DECK_PATH


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

from strategy.source_inference import SourceR2Policy

DECK = [int(line) for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines() if line.strip()]
_POLICY = SourceR2Policy.from_checkpoint(
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
    deployment_source_id: int = 0,
    deck: Path = TARGET_DECK_PATH,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    for required in (checkpoint, ontology, deck, source_package / "cg"):
        if not required.exists():
            raise FileNotFoundError(required)
    deck_cards = [line.strip() for line in deck.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck_cards) != 60 or any(not card_id.isdigit() for card_id in deck_cards):
        raise ValueError("deployment deck must contain exactly 60 numeric card IDs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    if metadata.get("schema_version") != "0016_source_r2_training_v1":
        raise ValueError("checkpoint is not an 0016 source-conditioned R2 policy")
    source_model = metadata.get("source_model") or {}
    vocabulary_size = source_model.get("vocabulary_size")
    if (
        isinstance(deployment_source_id, bool)
        or not isinstance(deployment_source_id, int)
        or isinstance(vocabulary_size, bool)
        or not isinstance(vocabulary_size, int)
        or not 0 <= deployment_source_id <= vocabulary_size
    ):
        raise ValueError("deployment source ID is outside the checkpoint vocabulary")

    strategy = output / "strategy"
    (strategy / "features").mkdir(parents=True)
    (strategy / "knowledge").mkdir(parents=True)
    shutil.copy2(deck, output / "deck.csv")
    shutil.copytree(source_package / "cg", output / "cg")
    (output / "main.py").write_text(MAIN, encoding="utf-8")
    (strategy / "__init__.py").write_text(
        f'PROJECT_ID = "{PROJECT_ID}"\n', encoding="utf-8"
    )
    (strategy / "features/__init__.py").write_text("", encoding="utf-8")
    (strategy / "knowledge/__init__.py").write_text("", encoding="utf-8")
    project = Path(__file__).resolve().parent
    copies = {
        "base_model.py": "base_model.py",
        "model.py": "model.py",
        "ac_model.py": "ac_model.py",
        "r2_model.py": "r2_model.py",
        "r2_inference.py": "r2_inference.py",
        "source_model.py": "source_model.py",
        "source_inference.py": "source_inference.py",
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
            "deployment_source_id": deployment_source_id,
            "portable_checkpoint": True,
            "source_checkpoint_sha256": _sha256(checkpoint),
        },
    }
    model_path = strategy / "model.bin"
    torch.save(portable, model_path)
    manifest = {
        "schema_version": "0016_source_r2_candidate_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "ontology_sha256": _sha256(strategy / "card_ontology.json"),
        "deck_sha256": _sha256(output / "deck.csv"),
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "source_package": str(source_package.resolve()),
        "deployment_deck": str(deck.resolve()),
        "deployment_source_id": deployment_source_id,
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
    parser.add_argument("--deployment-source-id", type=int, default=0)
    parser.add_argument("--deck", type=Path, default=TARGET_DECK_PATH)
    args = parser.parse_args()
    print(json.dumps(export_candidate(**vars(args)), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
