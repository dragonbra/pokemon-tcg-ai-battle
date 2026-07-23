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


def _canonical_deck_sha256(deck: list[int]) -> str:
    normalized = ",".join(str(card_id) for card_id in sorted(deck))
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _cg_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"cg source contains no files: {root}")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"cg source may not contain symlinks: {path}")
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build(
    output: Path,
    checkpoint: Path,
    source_package: Path | None = None,
    *,
    experiment_id: str,
    name: str = "Alakazam BC v1",
    deck: Path | None = None,
    cg_source: Path | None = None,
    source_identity: dict[str, Any] | None = None,
    shared_model_config: Path | None = None,
) -> Path:
    output = output.resolve()
    checkpoint = checkpoint.resolve()
    source_package = source_package.resolve() if source_package is not None else None
    if output.exists():
        raise FileExistsError(f"submission package already exists: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if source_package is not None:
        deck = deck or source_package / "deck.csv"
        cg_source = cg_source or source_package / "cg"
    if deck is None or cg_source is None:
        raise ValueError("deck and cg_source are required when source_package is not provided")
    deck = deck.resolve()
    cg_source = cg_source.resolve()
    if (cg_source / "cg").is_dir():
        cg_source = cg_source / "cg"
    if not deck.is_file() or not cg_source.is_dir():
        raise FileNotFoundError(f"deck or cg source is missing: {deck}, {cg_source}")
    deck_cards = [
        int(line.strip())
        for line in deck.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck_cards) != 60:
        raise ValueError(f"deck must contain exactly 60 card IDs: {deck}")
    deck_sha256 = _canonical_deck_sha256(deck_cards)
    if source_identity and str(source_identity.get("deck_sha256")) != deck_sha256:
        raise ValueError(
            "source identity deck hash does not match --deck: "
            f"{source_identity.get('deck_sha256')} != {deck_sha256}"
        )

    payload = _load_checkpoint(checkpoint)
    metadata = payload.get("metadata") or {}
    checkpoint_deck = [int(card_id) for card_id in metadata.get("deck") or []]
    if checkpoint_deck and (
        len(checkpoint_deck) != 60
        or _canonical_deck_sha256(checkpoint_deck) != deck_sha256
    ):
        raise ValueError("checkpoint deck metadata does not match --deck")
    if (source_identity or shared_model_config) and len(checkpoint_deck) != 60:
        raise ValueError("audited package checkpoint is missing exact deck metadata")
    feature_schema = (metadata.get("feature_config") or {}).get("schema_version")
    if shared_model_config is not None and feature_schema != "ptcg_features_universal":
        raise ValueError(f"checkpoint feature schema is not universal: {feature_schema!r}")
    if shared_model_config is not None:
        shared = json.loads(shared_model_config.read_text(encoding="utf-8"))
        model_config = metadata.get("model_config") or {}
        expected = {
            "d_model": int(shared["d_model"]),
            "hidden_dim": int(shared["hidden_dim"]),
            "num_heads": int(shared["num_heads"]),
            "num_transformer_layers": int(shared["transformer_layers"]),
            "dropout": float(shared["dropout"]),
        }
        observed = {key: model_config.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"checkpoint model config does not match shared config: {observed} != {expected}"
            )
    output.mkdir(parents=True)
    strategy = output / "strategy"
    strategy.mkdir()
    shutil.copy2(deck, output / "deck.csv")
    shutil.copytree(
        cg_source,
        output / "cg",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    (output / "main.py").write_text(MAIN_TEMPLATE.lstrip(), encoding="utf-8")
    (strategy / "__init__.py").write_text("", encoding="utf-8")

    repository_root = Path(__file__).resolve().parents[3]
    shutil.copy2(
        repository_root / "train" / "alakazam_bc_rl" / "card_metadata.py",
        strategy / "card_metadata.py",
    )
    shutil.copy2(
        repository_root / "train" / "alakazam_bc_rl" / "features.py",
        strategy / "features.py",
    )
    shutil.copy2(repository_root / "rl_environment" / "model.py", strategy / "core_model.py")
    full_action_model = (
        repository_root / "train" / "alakazam_bc_rl" / "full_action_model.py"
    ).read_text(
        encoding="utf-8"
    ).replace(
        "from rl_environment.model import CandidatePolicyValueNet, ModelConfig",
        "from .core_model import CandidatePolicyValueNet, ModelConfig",
    )
    (strategy / "full_action_model.py").write_text(full_action_model, encoding="utf-8")
    inference = (
        repository_root / "train" / "alakazam_bc_rl" / "full_action_inference.py"
    ).read_text(
        encoding="utf-8"
    ).replace(
        "from rl_environment.model import ModelConfig",
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
    source_files = {
        path.name: _sha256(path)
        for path in sorted(strategy.glob("*.py"))
    }
    manifest = {
        "schema_version": "ptcg_pure_model_submission_v1",
        "name": name,
        "experiment_id": experiment_id,
        "checkpoint_step": int(payload.get("step", 0)),
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "feature_schema": feature_schema,
        "action_contract": "full_action_set_v1",
        "selection_contract": metadata.get("selection_contract"),
        "model_config": metadata.get("model_config"),
        "source_identity": source_identity,
        "deck_sha256": deck_sha256,
        "deck_file_sha256": _sha256(output / "deck.csv"),
        "cg_tree_sha256": _cg_tree_hash(output / "cg"),
        "source_files_sha256": source_files,
        "shared_model_config": (
            str(shared_model_config.resolve()) if shared_model_config is not None else None
        ),
        "shared_model_config_sha256": (
            _sha256(shared_model_config) if shared_model_config is not None else None
        ),
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
    parser.add_argument("--source-package", type=Path)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--cg-source", type=Path)
    parser.add_argument("--source-identity", type=Path)
    parser.add_argument("--shared-model-config", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--name", default="Alakazam BC v1")
    args = parser.parse_args()
    source_identity = (
        json.loads(args.source_identity.read_text(encoding="utf-8"))
        if args.source_identity is not None
        else None
    )
    if source_identity and "source_identity" in source_identity:
        source_identity = source_identity["source_identity"]
    print(
        build(
            args.output,
            args.checkpoint,
            args.source_package,
            experiment_id=args.experiment_id,
            name=args.name,
            deck=args.deck,
            cg_source=args.cg_source,
            source_identity=source_identity,
            shared_model_config=args.shared_model_config,
        )
    )


if __name__ == "__main__":
    main()
