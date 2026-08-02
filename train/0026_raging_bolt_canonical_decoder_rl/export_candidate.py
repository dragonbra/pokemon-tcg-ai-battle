"""Export a self-contained Arena package from a 0026 decoder-only checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from . import FOCAL_DECK_ID, PROJECT_ID
from .checkpoint import load_model_checkpoint
from .focal.contract import CHECKPOINT_PATH as INITIAL_CHECKPOINT_PATH
from .policy import load_actor_critic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_runtime(output: Path) -> None:
    source = Path(__file__).resolve().parent / "focal"
    strategy = output / "strategy"
    files = (
        "features/__init__.py", "features/prototypes.py",
        "features/canonical/__init__.py", "features/canonical/batching.py",
        "features/canonical/compiler.py", "features/canonical/resolver.py",
        "features/canonical/schema.py", "knowledge/__init__.py",
        "knowledge/ledger.py", "knowledge/state.py", "model/__init__.py",
        "model/canonical/__init__.py", "model/canonical/config.py",
        "model/canonical/decoder.py", "model/canonical/options.py",
        "model/canonical/policy.py", "model/canonical/prototypes.py",
        "model/canonical/state.py", "model/canonical/typed.py",
        "deployment/__init__.py", "deployment/canonical_inference.py",
        "deployment/canonical_online_runtime.py", "deployment/inference.py",
    )
    for relative in files:
        target = strategy / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    for name in (
        "official_public_prototypes_v1.json", "official_full_engine_prototypes_v1.json"
    ):
        shutil.copy2(source / "assets" / name, strategy / "deployment" / name)
    (strategy / "__init__.py").write_text("", encoding="ascii")
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
    *, checkpoint: Path, output: Path, cg_source: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    model, identity = load_actor_critic("cpu")
    decoder_identity = load_model_checkpoint(checkpoint, model)
    initial = torch.load(INITIAL_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if set(initial) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("initial 0025 checkpoint schema mismatch")
    payload = {
        "schema_version": initial["schema_version"],
        "state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.actor.state_dict().items()
        },
        "metadata": {
            **initial["metadata"],
            "source_project": PROJECT_ID,
            "source_decoder_checkpoint_sha256": decoder_identity["checkpoint_sha256"],
            "source_decoder_update": decoder_identity["update"],
        },
    }
    deck_source = (
        Path(__file__).resolve().parent / "league" / "decks" / FOCAL_DECK_ID / "deck.csv"
    )
    deck = [int(line) for line in deck_source.read_text().splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError("focal deck is not exact 60")
    try:
        output.mkdir(parents=True)
        shutil.copy2(deck_source, output / "deck.csv")
        shutil.copytree(cg_source, output / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        _copy_runtime(output)
        torch.save(payload, output / "strategy" / "model.bin")
        (output / "main.py").write_text(
            """import os
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
from strategy.deployment.canonical_inference import PortableCanonicalPolicy
DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
POLICY = PortableCanonicalPolicy.from_checkpoint(ROOT / "strategy/model.bin", DECK)
def read_deck_csv():
    return list(DECK)
def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
""",
            encoding="ascii",
        )
        canonical = ",".join(str(card) for card in sorted(deck)).encode("ascii")
        manifest = {
            "schema_version": "0026_decoder_only_candidate_v1",
            "project_id": PROJECT_ID,
            "candidate": output.name,
            "deck_id": FOCAL_DECK_ID,
            "deck_sha256": hashlib.sha256(canonical).hexdigest(),
            "initial_0025_checkpoint_sha256": identity.checkpoint_sha256,
            "decoder_checkpoint": str(checkpoint),
            "decoder_checkpoint_sha256": decoder_identity["checkpoint_sha256"],
            "decoder_update": decoder_identity["update"],
            "representation_sha256": model.representation_sha256(),
            "model_only_source": True,
            "optimizer_state_saved": False,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if any(path.is_symlink() for path in output.rglob("*")):
            raise ValueError("candidate contains a symlink")
        return manifest
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cg-source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_candidate(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
