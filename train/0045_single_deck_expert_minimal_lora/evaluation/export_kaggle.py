"""Export one 0044 checkpoint as a self-contained Kaggle submission package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any

import torch

from ..assets import AssetRegistry, sha256_file
from ..own_archetype import OwnArchetypeVocabulary
from .candidate import materialize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "semantic_runtime"
BASE_PORTABLE = (
    PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin"
)
PACKAGE_SCHEMA = "0044_champion_league_kaggle_package_v1"

MAIN = '''"""Kaggle entrypoint for one immutable 0044 deployment identity."""
import hashlib
import json
import os
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
import torch  # noqa: E402,F401

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _manifest():
    payload = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != "0044_champion_league_kaggle_package_v1":
        raise RuntimeError("0044 Kaggle package schema mismatch")
    expected = payload.get("package_file_sha256")
    actual = {
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if not isinstance(expected, dict) or actual != set(expected):
        raise RuntimeError("0044 Kaggle package file inventory mismatch")
    mismatched = [name for name, digest in expected.items() if _sha256(ROOT / name) != digest]
    if mismatched:
        raise RuntimeError(f"0044 Kaggle package hash mismatch: {mismatched}")
    return payload

PACKAGE_MANIFEST = _manifest()
from strategy.deployment.compound_inference import PortableCompoundSemanticPolicy  # noqa: E402
DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
POLICY = PortableCompoundSemanticPolicy.from_checkpoint(ROOT / "strategy/model.bin", DECK)

def read_deck_csv():
    return list(DECK)

def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
'''


def _file_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _copy_official_runtime(output: Path) -> None:
    source = next((ROOT / "evaluation/arena/opponents").glob("*/cg"), None)
    if source is None:
        raise FileNotFoundError("official packaged cg runtime is unavailable")
    shutil.copytree(source, output / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def export(
    *, checkpoint: Path, deck_id: str, output: Path, archive: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    if output.exists() or archive.exists():
        raise FileExistsError("0044 Kaggle output/package archive already exists")
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    asset = next(row for row in registry.decks if row.deck_id == deck_id)
    deck = tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    own_id = next(row.archetype_id for row in vocabulary.mappings if row.deck_id == deck_id)
    output.mkdir(parents=True)
    try:
        shutil.copytree(
            RUNTIME_ROOT, output / "strategy",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runtime_manifest.json"),
        )
        model_path = output / "strategy/model.bin"
        _, audit = materialize(
            checkpoint=checkpoint, base_portable=BASE_PORTABLE,
            deck=deck, deck_id=deck_id, own_archetype_id=own_id,
            output=model_path, device=torch.device("cpu"),
        )
        formal_evidence = selection.get("formal_strength_evidence")
        if formal_evidence is not None:
            expected = {
                "checkpoint_update": audit.checkpoint_update,
                "source_checkpoint_sha256": audit.source_checkpoint_sha256,
                "portable_checkpoint_sha256": audit.portable_checkpoint_sha256,
                "deployment_effective_sha256": audit.effective_candidate_sha256,
                "deck_id": deck_id,
                "focal_exact_deck_sha256": audit.focal_exact_deck_sha256,
            }
            mismatched = {
                key: {"expected": value, "actual": formal_evidence.get(key)}
                for key, value in expected.items()
                if formal_evidence.get(key) != value
            }
            if formal_evidence.get("status") != "PASS" or mismatched:
                raise RuntimeError(
                    f"formal strength evidence does not identify this package: {mismatched}"
                )
        _copy_official_runtime(output)
        (output / "deck.csv").write_text(
            "".join(f"{card}\n" for card in deck), encoding="utf-8"
        )
        (output / "main.py").write_text(MAIN, encoding="utf-8")
        manifest = {
            "schema_version": PACKAGE_SCHEMA,
            "project_id": "0044_g2_dragapult_policy_option_lora",
            "candidate": output.name,
            "checkpoint_update": audit.checkpoint_update,
            "source_checkpoint": str(checkpoint.relative_to(ROOT)),
            "source_checkpoint_sha256": audit.source_checkpoint_sha256,
            "deck_id": deck_id,
            "deck_display_name": asset.name,
            "focal_exact_deck_sha256": audit.focal_exact_deck_sha256,
            "own_archetype_id": own_id,
            "portable_checkpoint_sha256": audit.portable_checkpoint_sha256,
            "deployment_effective_sha256": audit.effective_candidate_sha256,
            "deployment_contract": audit.to_manifest(),
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "runtime_framework": "pytorch",
            "native_runtime_load_order": "torch_before_cg",
            "model_only": True,
            "optimizer_state_saved": False,
            "selection": selection,
            "strength_evidence": formal_evidence or "rollout_diagnostic_only_no_cuda2048",
            "semantic_runtime_source": str(RUNTIME_ROOT.relative_to(ROOT)),
        }
        manifest["package_file_sha256"] = _file_inventory(output)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        subprocess.run(
            ["python3", "-c", "import main; assert len(main.read_deck_csv()) == 60"],
            cwd=output, check=True,
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        with tarfile.open(temporary, "w:gz") as handle:
            for path in sorted(output.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    handle.add(path, arcname=str(path.relative_to(output)))
        temporary.replace(archive)
        manifest["archive"] = str(archive.relative_to(ROOT))
        manifest["archive_sha256"] = sha256_file(archive)
        return manifest
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        archive.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    args = parser.parse_args()
    result = export(
        checkpoint=args.checkpoint.resolve(), deck_id=args.deck_id,
        output=args.output.resolve(), archive=args.archive.resolve(),
        selection=json.loads(args.selection_json.read_text()),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
