"""Export a self-contained 0031 Arena candidate from a model-only checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch


PROJECT_ID = "0031_rule_faithful_semantic_foundation_pretraining"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deck_hash(deck: list[int]) -> str:
    encoded = ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _read_deck(path: Path) -> list[int]:
    deck = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60 or any(card_id <= 0 for card_id in deck):
        raise ValueError("deck must contain exactly 60 positive card IDs")
    return deck


def _main_source() -> str:
    return '''import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(globals().get("__file__", Path.cwd())).resolve()
if ROOT.is_file():
    ROOT = ROOT.parent
if not (ROOT / "deck.csv").is_file() and Path("/kaggle_simulations/agent/deck.csv").is_file():
    ROOT = Path("/kaggle_simulations/agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strategy.inference import PortableSemanticPolicy
DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
POLICY = PortableSemanticPolicy.from_checkpoint(ROOT / "strategy/model.bin", DECK)
def read_deck_csv():
    return list(DECK)
def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
'''


def _copy_runtime(source_root: Path, strategy: Path) -> None:
    for package in ("contracts", "domain", "features", "knowledge", "model", "deployment"):
        shutil.copytree(
            source_root / package,
            strategy / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
        )
    shutil.copytree(source_root / "assets", strategy / "assets")
    (strategy / "inference.py").write_text(
        "from .deployment.inference import PortableSemanticPolicy, legal_fallback\n\n"
        "__all__ = ['PortableSemanticPolicy', 'legal_fallback']\n",
        encoding="ascii",
    )
    (strategy / "online_runtime.py").write_text(
        "from .deployment.online_runtime import OnlineCausalEncoder\n\n"
        "__all__ = ['OnlineCausalEncoder']\n",
        encoding="ascii",
    )


def export_candidate(
    *, checkpoint: Path, deck_path: Path, cg_source: Path, output: Path, deck_id: str
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "state_dict", "metadata"}
        or payload.get("schema_version") != "0031_model_only_checkpoint_v1"
    ):
        raise ValueError("checkpoint is not the 0031 model-only contract")
    metadata = payload["metadata"]
    if (
        not isinstance(metadata, dict)
        or metadata.get("project_id") != PROJECT_ID
        or metadata.get("arm") != "rule_faithful_semantic"
        or (metadata.get("model_config") or {}).get("model") != "SemanticPolicy"
    ):
        raise ValueError("checkpoint does not match the 0031 semantic contract")
    if not cg_source.is_dir():
        raise FileNotFoundError(cg_source)
    deck = _read_deck(deck_path)
    strategy = output / "strategy"
    try:
        strategy.mkdir(parents=True)
        (output / "deck.csv").write_text(
            "".join(f"{card_id}\n" for card_id in deck), encoding="ascii"
        )
        shutil.copytree(
            cg_source,
            output / "cg",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (output / "main.py").write_text(_main_source(), encoding="ascii")
        (strategy / "__init__.py").write_text("", encoding="ascii")
        _copy_runtime(Path(__file__).resolve().parent, strategy)
        shutil.copy2(checkpoint, strategy / "model.bin")
        manifest = {
            "schema_version": "0031_rule_faithful_semantic_candidate_v1",
            "candidate": output.name,
            "project_id": PROJECT_ID,
            "version": metadata.get("version"),
            "arm": metadata.get("arm"),
            "deck_id": deck_id,
            "deck_sha256": _deck_hash(deck),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_schema_version": payload["schema_version"],
            "checkpoint_selection": checkpoint.stem,
            "checkpoint_epoch": metadata.get("epoch"),
            "checkpoint_global_step": metadata.get("global_step"),
            "model_only": True,
            "optimizer_state_saved": False,
            "validation": metadata.get("validation"),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if any(path.is_symlink() for path in output.rglob("*")):
            raise ValueError("candidate package contains a symlink")
        return manifest
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", dest="deck_path", type=Path, required=True)
    parser.add_argument("--cg-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deck-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_candidate(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
