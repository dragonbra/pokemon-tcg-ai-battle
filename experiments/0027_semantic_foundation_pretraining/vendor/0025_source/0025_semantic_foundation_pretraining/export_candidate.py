"""Export a self-contained 0025 Arena candidate from a model-only checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch


PROJECT_ID = "0025_semantic_foundation_pretraining"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deck_hash(deck: list[int]) -> str:
    canonical = ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _read_deck(path: Path) -> list[int]:
    deck = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60 or any(card_id <= 0 for card_id in deck):
        raise ValueError("deck must contain exactly 60 positive card IDs")
    return deck


def _main_source(arm: str) -> str:
    policy_import = (
        "from strategy.deployment.canonical_inference import PortableCanonicalPolicy\n"
        if arm == "canonical_semantic"
        else "from strategy.portable_inference import PortablePolicy\n"
    )
    policy_load = (
        "PortableCanonicalPolicy.from_checkpoint(ROOT / \"strategy/model.bin\", DECK)"
        if arm == "canonical_semantic"
        else "PortablePolicy.from_checkpoint(ROOT / \"strategy/model.bin\", DECK)"
    )
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
''' + policy_import + '''
DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
POLICY = ''' + policy_load + '''
def read_deck_csv():
    return list(DECK)
def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
'''


def _copy_canonical_runtime(source_root: Path, strategy: Path) -> None:
    for relative in (
        "features/__init__.py",
        "features/prototypes.py",
        "features/canonical/__init__.py",
        "features/canonical/batching.py",
        "features/canonical/compiler.py",
        "features/canonical/resolver.py",
        "features/canonical/schema.py",
        "knowledge/__init__.py",
        "knowledge/ledger.py",
        "knowledge/state.py",
        "model/__init__.py",
        "model/canonical/__init__.py",
        "model/canonical/config.py",
        "model/canonical/decoder.py",
        "model/canonical/options.py",
        "model/canonical/policy.py",
        "model/canonical/prototypes.py",
        "model/canonical/state.py",
        "model/canonical/typed.py",
        "deployment/__init__.py",
        "deployment/canonical_inference.py",
        "deployment/canonical_online_runtime.py",
        "deployment/inference.py",
    ):
        source = source_root / relative
        target = strategy / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    assets = source_root / "assets"
    deployment = strategy / "deployment"
    for name in (
        "official_public_prototypes_v1.json",
        "official_full_engine_prototypes_v1.json",
    ):
        shutil.copy2(assets / name, deployment / name)
    (strategy / "inference.py").write_text(
        "from .deployment.inference import legal_fallback\n\n__all__ = ['legal_fallback']\n",
        encoding="ascii",
    )
    (strategy / "online_runtime.py").write_text(
        "from .deployment.canonical_online_runtime import OnlineCausalEncoder\n\n"
        "__all__ = ['OnlineCausalEncoder']\n",
        encoding="ascii",
    )


def export_candidate(
    *,
    checkpoint: Path,
    deck_path: Path,
    cg_source: Path,
    output: Path,
    deck_id: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("checkpoint is not model-only")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or metadata.get("arm") not in {
        "legacy_default",
        "canonical_semantic",
    }:
        raise ValueError("unsupported 0025 checkpoint arm")
    arm = metadata["arm"]
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
        (output / "main.py").write_text(_main_source(arm), encoding="ascii")
        (strategy / "__init__.py").write_text("", encoding="ascii")
        source_root = Path(__file__).resolve().parent
        if arm == "canonical_semantic":
            _copy_canonical_runtime(source_root, strategy)
        else:
            copies = {
                source_root / "legacy/base_model.py": strategy / "base_model.py",
                source_root / "deployment/inference.py": strategy / "inference.py",
                source_root / "deployment/online_runtime.py": strategy / "online_runtime.py",
                source_root / "deployment/portable_inference.py": strategy / "portable_inference.py",
            }
            for source, target in copies.items():
                shutil.copy2(source, target)
        shutil.copy2(checkpoint, strategy / "model.bin")
        manifest = {
            "schema_version": f"0025_{arm}_candidate_v1",
            "candidate": output.name,
            "project_id": PROJECT_ID,
            "deck_id": deck_id,
            "deck_sha256": _deck_hash(deck),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_schema_version": payload["schema_version"],
            "arm": arm,
            "epoch": metadata.get("epoch"),
            "validation": metadata.get("validation"),
            "model_only": True,
            "optimizer_state_saved": False,
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
