from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .constants import ONTOLOGY_PATH, PROJECT_ID, TARGET_DECK


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _main_source() -> str:
    return '''from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.portable_inference import PortablePolicy

DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
_POLICY = PortablePolicy.from_checkpoint(
    ROOT / "strategy/model.bin", ROOT / "strategy/card_ontology.json", DECK
)

def agent(observation):
    if observation.get("select") is None:
        _POLICY.reset()
        return list(DECK)
    return _POLICY.select(observation)
'''


def write_candidate_model(source: Path, output: Path) -> dict[str, bool]:
    """Write an inference-only checkpoint even when the BC source is resumable."""
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError("checkpoint has no model state")
    resume_keys = {"optimizer", "scheduler", "scaler", "rng_state", "rollout"}
    source_progress = {
        key: int(payload[key])
        for key in ("epoch", "global_step", "update")
        if isinstance(payload.get(key), int)
    }
    model_only = {
        "schema_version": "0018_candidate_model_only_v1",
        "model": payload["model"],
        "metadata": payload.get("metadata") or {},
        "source_progress": source_progress,
    }
    torch.save(model_only, output)
    packaged = torch.load(output, map_location="cpu", weights_only=False)
    return {
        "source_had_optimizer_state": bool(resume_keys.intersection(payload)),
        "packaged_optimizer_state_saved": bool(resume_keys.intersection(packaged)),
    }


def export_candidate(checkpoint: Path, cg_source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    strategy = output / "strategy"
    (strategy / "features").mkdir(parents=True)
    (strategy / "knowledge").mkdir(parents=True)
    (output / "deck.csv").write_text(
        "".join(f"{card_id}\n" for card_id in TARGET_DECK), encoding="ascii"
    )
    shutil.copytree(cg_source, output / "cg")
    (output / "main.py").write_text(_main_source(), encoding="utf-8")
    (strategy / "__init__.py").write_text(
        f'PROJECT_ID = "{PROJECT_ID}"\n', encoding="ascii"
    )
    (strategy / "features/__init__.py").write_text("", encoding="ascii")
    (strategy / "knowledge/__init__.py").write_text("", encoding="ascii")
    policy_root = Path(__file__).resolve().parent / "policy"
    copies = (
        "base_model.py",
        "model.py",
        "ac_model.py",
        "r2_model.py",
        "r15_model.py",
        "card_features.py",
        "card_semantics.py",
        "config.py",
        "online_runtime.py",
        "portable_inference.py",
        "features/compiler.py",
        "knowledge/state.py",
        "knowledge/ledger.py",
    )
    for relative in copies:
        shutil.copy2(policy_root / relative, strategy / relative)
    shutil.copy2(ONTOLOGY_PATH, strategy / "card_ontology.json")
    checkpoint_audit = write_candidate_model(checkpoint, strategy / "model.bin")
    manifest = {
        "schema_version": "0018_alakazam_terminal_rl_candidate_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "deck_sha256": _sha256(output / "deck.csv"),
        "model_sha256": _sha256(strategy / "model.bin"),
        "source_id": 1,
        **checkpoint_audit,
        "optimizer_state_saved": checkpoint_audit["packaged_optimizer_state_saved"],
    }
    if manifest["optimizer_state_saved"]:
        raise ValueError("candidate checkpoint contains optimizer state")
    (strategy / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="export an 0018 arena candidate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cg-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            export_candidate(args.checkpoint, args.cg_source, args.output),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
