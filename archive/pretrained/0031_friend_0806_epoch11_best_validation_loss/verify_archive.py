"""Verify hashes, model-only boundaries, strict loading, and a minimal forward."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from loader import ARCHIVE_ROOT, load_checkpoint, load_model
from model_source.contracts.fields import WIDTHS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _smoke_batch() -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        "global_cat": torch.zeros(1, WIDTHS.global_cat, dtype=torch.long),
        "global_num": torch.zeros(1, WIDTHS.global_num),
        "global_state": torch.ones(1, WIDTHS.global_state, dtype=torch.long),
        "min_count": torch.zeros(1, dtype=torch.long),
        "max_count": torch.ones(1, dtype=torch.long),
        "targets": torch.ones(1, 1, dtype=torch.long),
    }
    for prefix, cat_width, num_width, valid in (
        ("card", WIDTHS.card_cat, WIDTHS.card_num, False),
        ("resource", WIDTHS.resource_cat, WIDTHS.resource_num, False),
        ("event", WIDTHS.event_cat, WIDTHS.event_num, False),
        ("option", WIDTHS.option_cat, WIDTHS.option_num, True),
    ):
        batch[f"{prefix}_cat"] = torch.zeros(1, 1, cat_width, dtype=torch.long)
        batch[f"{prefix}_num"] = torch.zeros(1, 1, num_width)
        batch[f"{prefix}_mask"] = torch.full((1, 1), valid, dtype=torch.bool)
        if prefix != "option":
            batch[f"{prefix}_state"] = torch.ones(
                1, 1, getattr(WIDTHS, f"{prefix}_state"), dtype=torch.long
            )
    batch["card_parent"] = torch.zeros(1, 1, dtype=torch.long)
    for relation in ("source", "target", "before", "after"):
        batch[f"event_{relation}"] = torch.zeros(1, 1, dtype=torch.long)
    batch["option_state"] = torch.zeros(1, 1, WIDTHS.option_state, dtype=torch.long)
    batch["option_source"] = torch.zeros(1, 1, dtype=torch.long)
    batch["option_target"] = torch.zeros(1, 1, dtype=torch.long)
    batch["option_context"] = torch.zeros(1, 1, dtype=torch.long)
    batch["option_effect_card"] = torch.zeros(1, 1, dtype=torch.long)
    for prefix in ("option_skill", "option_effect"):
        batch[f"{prefix}_id"] = torch.zeros(1, 1, dtype=torch.long)
        batch[f"{prefix}_role"] = torch.zeros(1, 1, dtype=torch.long)
        batch[f"{prefix}_parent"] = torch.zeros(1, 1, dtype=torch.long)
        batch[f"{prefix}_mask"] = torch.zeros(1, 1, dtype=torch.bool)
    return batch


def main() -> None:
    manifest = json.loads((ARCHIVE_ROOT / "manifest.json").read_text(encoding="ascii"))
    payload = load_checkpoint()
    forbidden = {"optimizer", "scheduler", "scaler", "rng", "rollout", "replay"}
    if forbidden & set(payload):
        raise ValueError("archive contains forbidden resumable training state")
    for relative_path, expected in manifest["file_sha256"].items():
        actual = _sha256(ARCHIVE_ROOT / relative_path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative_path}: {actual}")
    source_sha256 = _tree_sha256(ARCHIVE_ROOT / "model_source")
    if source_sha256 != manifest["model"]["model_source_bundle_sha256"]:
        raise ValueError(f"model source bundle SHA-256 mismatch: {source_sha256}")
    if any(path.is_symlink() for path in ARCHIVE_ROOT.rglob("*")):
        raise ValueError("pretrained archive must not contain symlinks")
    model = load_model("cpu")
    with torch.inference_mode():
        logits = model(_smoke_batch())
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError("minimal 0031 forward smoke failed")
    print(
        f"verified {manifest['asset_id']}: epoch={payload['metadata']['epoch']}, "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )


if __name__ == "__main__":
    main()
