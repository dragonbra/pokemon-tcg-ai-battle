"""Build a self-contained work/submission package from a full-action checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch


MAIN_TEMPLATE = r'''
from pathlib import Path
import sys


def _submission_root():
    source_file = globals().get("__file__")
    if isinstance(source_file, str):
        source_root = Path(source_file).resolve().parent
        if (source_root / "deck.csv").is_file():
            return source_root
    kaggle_root = Path("/kaggle_simulations/agent")
    if (kaggle_root / "deck.csv").is_file():
        return kaggle_root
    return Path.cwd()


ROOT = _submission_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.full_action_inference import FullActionPolicy


DECK = [
    int(line)
    for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
_POLICY = FullActionPolicy.from_checkpoint(ROOT / "strategy" / "model.bin")


def read_deck_csv():
    return list(DECK)


def agent(observation):
    # select=None is the simulator deck/reset protocol, not a strategy fallback.
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


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload or "metadata" not in payload:
        raise ValueError(f"checkpoint is missing model metadata: {path}")
    return payload


def build(
    output: Path,
    checkpoint: Path,
    source_package: Path,
    *,
    experiment_id: str,
    name: str = "Alakazam BC v1",
) -> Path:
    output = output.resolve()
    checkpoint = checkpoint.resolve()
    source_package = source_package.resolve()
    if output.exists():
        raise FileExistsError(f"submission package already exists: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (source_package / "deck.csv").is_file() or not (source_package / "cg").is_dir():
        raise FileNotFoundError(f"source package is missing deck.csv or cg: {source_package}")

    payload = _load_checkpoint(checkpoint)
    output.mkdir(parents=True)
    strategy = output / "strategy"
    strategy.mkdir()
    shutil.copy2(source_package / "deck.csv", output / "deck.csv")
    shutil.copytree(
        source_package / "cg",
        output / "cg",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    (output / "main.py").write_text(MAIN_TEMPLATE.lstrip(), encoding="utf-8")
    (strategy / "__init__.py").write_text("", encoding="utf-8")

    repository_root = Path(__file__).resolve().parents[2]
    shutil.copy2(
        repository_root / "rl" / "model" / "card_metadata.py",
        strategy / "card_metadata.py",
    )
    shutil.copy2(repository_root / "rl" / "model" / "features.py", strategy / "features.py")
    shutil.copy2(repository_root / "rl" / "core" / "model.py", strategy / "core_model.py")
    full_action_model = (repository_root / "rl" / "model" / "full_action_model.py").read_text(
        encoding="utf-8"
    ).replace(
        "from rl.core.model import CandidatePolicyValueNet, ModelConfig",
        "from .core_model import CandidatePolicyValueNet, ModelConfig",
    )
    (strategy / "full_action_model.py").write_text(full_action_model, encoding="utf-8")
    inference = (repository_root / "rl" / "model" / "full_action_inference.py").read_text(
        encoding="utf-8"
    ).replace(
        "from rl.core.model import ModelConfig",
        "from .core_model import ModelConfig",
    )
    (strategy / "full_action_inference.py").write_text(inference, encoding="utf-8")

    model_path = strategy / "model.bin"
    torch.save(
        {
            "step": int(payload.get("step", 0)),
            "model": payload["model"],
            "metadata": payload["metadata"],
        },
        model_path,
    )
    manifest = {
        "schema_version": "ptcg_pure_model_submission_v1",
        "name": name,
        "experiment_id": experiment_id,
        "checkpoint_step": int(payload.get("step", 0)),
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "feature_schema": (payload.get("metadata") or {}).get("feature_config", {}).get(
            "schema_version"
        ),
        "action_contract": "full_action_set_v1",
        "fallback": None,
    }
    (strategy / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, default=Path("work/alakazam_v9"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--name", default="Alakazam BC v1")
    args = parser.parse_args()
    print(
        build(
            args.output,
            args.checkpoint,
            args.source_package,
            experiment_id=args.experiment_id,
            name=args.name,
        )
    )


if __name__ == "__main__":
    main()
