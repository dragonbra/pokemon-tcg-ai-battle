"""Kaggle-compatible entrypoint for the self-contained 0038 compound policy."""

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


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_package_manifest():
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "0038_compound_kaggle_candidate_v4":
        raise RuntimeError("0038 package manifest schema mismatch")
    expected = manifest.get("package_file_sha256")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("0038 package has no immutable file inventory")
    actual_paths = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        unexpected = sorted(actual_paths - set(expected))
        raise RuntimeError(
            f"0038 package file inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    mismatched = [
        relative for relative, digest in expected.items()
        if _sha256(ROOT / relative) != digest
    ]
    if mismatched:
        raise RuntimeError(f"0038 package file hash mismatch: {mismatched}")
    return manifest


PACKAGE_MANIFEST = _validate_package_manifest()

from strategy.deployment.compound_inference import PortableCompoundSemanticPolicy

DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
POLICY = PortableCompoundSemanticPolicy.from_checkpoint(
    ROOT / "strategy/model.bin", DECK
)


def read_deck_csv():
    return list(DECK)


def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
