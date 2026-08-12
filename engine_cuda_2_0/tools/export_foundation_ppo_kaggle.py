"""Export a CUDA foundation PPO checkpoint as a Kaggle-ready agent archive."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PROJECT = ROOT / "train" / "0020_pluggable_deck_rl"
CG_SOURCE = ROOT / "evaluation" / "arena" / "opponents" / "dragapult_ex_03_v20260729_rl" / "cg"
DECK_SOURCES = {
    "lucario": ROOT / "evaluation" / "arena" / "opponents" / "mega_lucario_ex_solrock_07_bc" / "deck.csv",
}
RUNTIME_FILES = (
    "__init__.py",
    "ac_model.py",
    "base_model.py",
    "card_features.py",
    "card_semantics.py",
    "inference.py",
    "model.py",
    "online_runtime.py",
    "r2_model.py",
    "r15_model.py",
    "source_model.py",
    "source_r15_model.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def export_policy_checkpoint(source: Path, target: Path) -> dict[str, Any]:
    import torch

    ppo = torch.load(source, map_location="cpu", weights_only=False)
    model_state = ppo.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError("PPO checkpoint is missing model_state_dict")
    policy_state = {
        name.removeprefix("policy."): tensor.detach().cpu()
        for name, tensor in model_state.items()
        if name.startswith("policy.")
    }
    base_path = (
        ROOT
        / "rl_runs/0020_pluggable_deck_rl/versions/V1_frozen_0019_epoch13"
        / "checkpoint/epoch-0013-da9b13d6f82d19d4.pt"
    )
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    base_state = base.get("model")
    if not isinstance(base_state, dict):
        raise KeyError("foundation checkpoint is missing model")
    if set(policy_state) != set(base_state):
        raise ValueError("PPO policy keys do not match the foundation model")
    mismatched = [
        name
        for name in base_state
        if tuple(policy_state[name].shape) != tuple(base_state[name].shape)
    ]
    if mismatched:
        raise ValueError(f"PPO policy shape mismatch: {mismatched[:8]}")

    metadata = dict(base["metadata"])
    metadata["rl_export"] = {
        "source_checkpoint_sha256": sha256_file(source),
        "source_iteration": int(ppo["iteration"]),
        "best_metric": ppo.get("best_metric"),
        "best_metric_used": ppo.get("best_metric_used"),
        "best_metric_value": ppo.get("best_metric_value"),
        "train_scope": (ppo.get("config") or {}).get("train_scope"),
    }
    payload = {
        "model": policy_state,
        "epoch": base["epoch"],
        "global_step": base["global_step"],
        "metadata": metadata,
    }
    torch.save(payload, target)
    return metadata["rl_export"]


def replace_expected_checkpoint_hash(inference_path: Path, checkpoint_hash: str) -> None:
    source = inference_path.read_text(encoding="utf-8")
    marker = "EXPECTED_CHECKPOINT_SHA256 = ("
    start = source.index(marker)
    end = source.index(")", start) + 1
    replacement = f'{marker}\n    "{checkpoint_hash}"\n)'
    inference_path.write_text(source[:start] + replacement + source[end:], encoding="utf-8")


def validate_agent_last(main_path: Path) -> None:
    tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions or functions[-1] != "agent":
        raise ValueError("agent() must be the final top-level function")


def build_bundle(checkpoint: Path, deck_key: str, output: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    deck_source = DECK_SOURCES[deck_key]
    deck = read_deck(deck_source)
    if len(deck) != 60:
        raise ValueError(f"expected a 60-card deck, got {len(deck)}")

    strategy = output / "strategy"
    strategy.mkdir(parents=True)
    shutil.copy2(PACKAGE_PROJECT / "package_main.py", output / "main.py")
    shutil.copy2(deck_source, output / "deck.csv")
    shutil.copytree(CG_SOURCE, output / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    for name in RUNTIME_FILES:
        shutil.copy2(PACKAGE_PROJECT / name, strategy / name)
    shutil.copytree(PACKAGE_PROJECT / "features", strategy / "features", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copytree(PACKAGE_PROJECT / "knowledge", strategy / "knowledge", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(
        ROOT / "rl_runs/0020_pluggable_deck_rl/versions/V1_frozen_0019_epoch13/artifact/card_ontology.json",
        strategy / "card_ontology.json",
    )
    rl_export = export_policy_checkpoint(checkpoint, strategy / "model.bin")
    model_hash = sha256_file(strategy / "model.bin")
    replace_expected_checkpoint_hash(strategy / "inference.py", model_hash)
    validate_agent_last(output / "main.py")

    manifest = {
        "schema_version": "0020_cuda_ppo_kaggle_v1",
        "deck": deck_key,
        "deck_source": str(deck_source.relative_to(ROOT)),
        "source_checkpoint": checkpoint.name,
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "model_sha256": model_hash,
        **rl_export,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    archive = output / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("main.py", "deck.csv", "manifest.json", "strategy", "cg"):
            tar.add(output / name, arcname=name)
    manifest["archive"] = str(archive)
    manifest["archive_sha256"] = sha256_file(archive)
    manifest["archive_bytes"] = archive.stat().st_size
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", choices=tuple(DECK_SOURCES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.checkpoint, args.deck, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
