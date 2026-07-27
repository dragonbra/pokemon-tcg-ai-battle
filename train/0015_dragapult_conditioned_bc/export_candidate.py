"""Export an audited 0015 checkpoint as a self-contained arena candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_checkpoint(version_root: Path, criterion: str) -> Path:
    metric = {
        "best_greedy_exact": "bc/validation/exact_action",
        "best_validation_loss": "bc/validation/loss",
    }.get(criterion)
    if metric is None:
        raise ValueError(f"unsupported checkpoint criterion: {criterion}")
    rows = []
    for sidecar in sorted((version_root / "checkpoint").glob("epoch-*.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        value = float(payload["metadata"]["metrics"][metric])
        rows.append((value, int(payload["epoch"]), sidecar.with_suffix(".pt")))
    if not rows:
        raise ValueError(f"version has no checkpoints: {version_root}")
    if criterion == "best_greedy_exact":
        selected = max(rows, key=lambda item: (item[0], -item[1]))
    else:
        selected = min(rows, key=lambda item: (item[0], item[1]))
    return selected[2]


def _main_source(source_id: int, no_deck: bool) -> str:
    return f'''from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strategy.source_inference import SourceR15Policy

DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
_POLICY = SourceR15Policy.from_checkpoint(
    ROOT / "strategy/model.bin", ROOT / "strategy/card_ontology.json", DECK,
    source_id={source_id}, no_deck={no_deck!r},
)

def read_deck_csv():
    return list(DECK)

def agent(observation):
    if observation.get("select") is None:
        _POLICY.reset()
        return list(DECK)
    return _POLICY.select(observation)
'''


def export_candidate(
    version_root: Path,
    criterion: str,
    ontology: Path,
    deck: Path,
    cg_source: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    checkpoint = select_checkpoint(version_root, criterion)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    supported_schemas = {
        "0015_r15_source_conditioned_training_v1",
        "0015_r15_rule_contract_training_v1",
    }
    if metadata.get("schema_version") not in supported_schemas:
        raise ValueError("checkpoint is not a supported 0015 R15-family policy")
    model_family = metadata.get("model_family", "r15")
    if model_family not in {"r15", "r15_rule_contract"}:
        raise ValueError(f"unsupported 0015 model family: {model_family}")
    model_contract = json.loads(
        (version_root / "artifact/model_contract.json").read_text(encoding="utf-8")
    )
    source_id = int(model_contract["target_source_id"])
    no_deck = bool(metadata["no_deck"])
    cards = [int(line) for line in deck.read_text().splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError("target deck must contain exactly 60 cards")

    strategy = output / "strategy"
    (strategy / "features").mkdir(parents=True)
    (strategy / "knowledge").mkdir(parents=True)
    shutil.copy2(deck, output / "deck.csv")
    shutil.copytree(cg_source, output / "cg")
    (output / "main.py").write_text(_main_source(source_id, no_deck), encoding="utf-8")
    (strategy / "__init__.py").write_text('PROJECT_ID = "0015_dragapult_conditioned_bc"\n')
    (strategy / "features/__init__.py").write_text("")
    (strategy / "knowledge/__init__.py").write_text("")
    core = Path("train/0014_faithful_board_causal_features")
    copies = {
        "base_model.py": "base_model.py",
        "model.py": "model.py",
        "ac_model.py": "ac_model.py",
        "r2_model.py": "r2_model.py",
        "r15_model.py": "r15_model.py",
        "card_features.py": "card_features.py",
        "card_semantics.py": "card_semantics.py",
        "config.py": "config.py",
        "online_runtime.py": "online_runtime.py",
        "features/compiler.py": "features/compiler.py",
        "knowledge/state.py": "knowledge/state.py",
        "knowledge/ledger.py": "knowledge/ledger.py",
    }
    for source, target in copies.items():
        shutil.copy2(core / source, strategy / target)
    portable = Path(__file__).parent / "portable"
    shutil.copy2(portable / "source_model.py", strategy / "source_model.py")
    shutil.copy2(portable / "source_inference.py", strategy / "source_inference.py")
    if model_family == "r15_rule_contract":
        shutil.copy2(portable / "rule_contract_model.py", strategy / "rule_contract_model.py")
    shutil.copy2(Path(__file__).parent / "ablation.py", strategy / "ablation.py")
    shutil.copy2(ontology, strategy / "card_ontology.json")
    model_path = strategy / "model.bin"
    torch.save(
        {
            "model": payload["model"],
            "metadata": {
                **metadata,
                "portable_checkpoint": True,
                "source_checkpoint_sha256": _sha256(checkpoint),
            },
        },
        model_path,
    )
    manifest = {
        "schema_version": "0015_source_conditioned_r15_candidate_v1",
        "version": version_root.name,
        "model_family": model_family,
        "criterion": criterion,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "target_source_id": source_id,
        "no_deck": no_deck,
        "deck_sha256": _sha256(output / "deck.csv"),
        "model_sha256": _sha256(model_path),
    }
    (strategy / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-root", type=Path, required=True)
    parser.add_argument("--criterion", default="best_greedy_exact")
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--cg-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_candidate(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
