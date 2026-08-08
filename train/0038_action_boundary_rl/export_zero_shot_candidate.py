"""Export the audited 0806 checkpoint as a self-contained 0034 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "0038_action_boundary_rl"
SOURCE_SHA256 = "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8"
SOURCE_SCHEMA = "0031_model_only_checkpoint_v1"
OUTPUT_SCHEMA = "0031_shared_prototype_fp16_storage_fp32_runtime_candidate_checkpoint_v1"
_PROTOTYPE_ALIASES = ("state_encoder.prototypes.", "option_encoder.prototypes.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deck_hash(deck: list[int]) -> str:
    payload = ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _repo_relative(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


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


def _portable_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload["state_dict"]
    canonical = {
        name: value.half() if torch.is_floating_point(value) else value
        for name, value in source.items()
        if not name.startswith(_PROTOTYPE_ALIASES)
    }
    prototype_keys = {
        name.removeprefix("prototype_encoder.")
        for name in canonical
        if name.startswith("prototype_encoder.")
    }
    for alias in _PROTOTYPE_ALIASES:
        alias_keys = {name.removeprefix(alias) for name in source if name.startswith(alias)}
        if alias_keys != prototype_keys:
            raise ValueError(f"prototype alias inventory differs: {alias}")
        for suffix in prototype_keys:
            if not torch.equal(
                source[f"prototype_encoder.{suffix}"], source[f"{alias}{suffix}"]
            ):
                raise ValueError(f"prototype alias tensor differs: {alias}{suffix}")
    return {
        "schema_version": OUTPUT_SCHEMA,
        "state_dict": canonical,
        "metadata": dict(payload["metadata"]),
    }


def export_candidate(
    *, checkpoint: Path, deck_path: Path, cg_source: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if _sha256(checkpoint) != SOURCE_SHA256:
        raise ValueError("source checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata")
    if (
        payload.get("schema_version") != SOURCE_SCHEMA
        or len(payload.get("state_dict", {})) != 293
        or not isinstance(metadata, dict)
        or metadata.get("project_id") != "0031_rule_faithful_semantic_foundation_pretraining"
        or metadata.get("version") != "V2_full_winners_bs1024_20260616_20260803"
        or metadata.get("epoch") != 11
        or metadata.get("global_step") != 141878
    ):
        raise ValueError("source checkpoint does not match the audited 0806 contract")
    deck = [int(line) for line in deck_path.read_text().splitlines() if line.strip()]
    if len(deck) != 60 or any(card_id <= 0 for card_id in deck):
        raise ValueError("deck must contain exactly 60 positive card IDs")
    source_root = Path(__file__).resolve().parent / "semantic_policy"
    strategy = output / "strategy"
    try:
        strategy.mkdir(parents=True)
        (output / "deck.csv").write_text(
            "".join(f"{card_id}\n" for card_id in deck), encoding="ascii"
        )
        shutil.copytree(cg_source, output / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (output / "main.py").write_text(_main_source(), encoding="ascii")
        (strategy / "__init__.py").write_text("", encoding="ascii")
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
        portable = _portable_checkpoint(payload)
        torch.save(portable, strategy / "model.bin")
        manifest = {
            "schema_version": "0034_large_model_zero_shot_candidate_v1",
            "candidate": output.name,
            "project_id": PROJECT_ID,
            "version": "V1_large_model_zero_shot_frozen0019",
            "deck_id": "dragapult_third_ptcg_club",
            "deck_sha256": _deck_hash(deck),
            "source_checkpoint": str(_repo_relative(checkpoint)),
            "source_checkpoint_sha256": SOURCE_SHA256,
            "source_archive": "large-model-0806.tar.gz",
            "source_archive_sha256": "940d2ffeaf49e23ae2b387375c600150a49d06009c86451f03e0b195539f73c6",
            "checkpoint_epoch": 11,
            "checkpoint_global_step": 141878,
            "portable_checkpoint_schema_version": OUTPUT_SCHEMA,
            "portable_checkpoint_sha256": _sha256(strategy / "model.bin"),
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "model_only": True,
            "optimizer_state_saved": False,
            "validation": metadata.get("validation"),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", dest="deck_path", type=Path, required=True)
    parser.add_argument("--cg-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_candidate(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
