from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hints(directory: Path) -> dict[str, object]:
    source = directory / "idonly_policy.py"
    if not source.is_file():
        return {"adapter_hint": "independent_or_rule_runtime", "codec_hint": "audit_required"}
    text = source.read_text(encoding="utf-8", errors="replace")
    decoder_match = re.search(r"decoder_layers:\s*int\s*=\s*(\d+)", text)
    if "teacher_logits_and_need" in text:
        adapter = "goal_or_auxiliary_pointer"
    elif decoder_match and int(decoder_match.group(1)) > 1:
        adapter = "stacked_recurrent_pointer"
    elif "mean_pool" in text or "meanpool" in directory.name.lower():
        adapter = "meanpool_pointer"
    else:
        adapter = "idonly_pointer"
    return {
        "adapter_hint": adapter,
        "codec_hint": "idonly_codec_family_audit_exact_version",
        "source": str(source),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory local BC checkpoints without importing PyTorch.")
    parser.add_argument("--root", type=Path, default=Path("bc_models"))
    parser.add_argument("--hash", action="store_true", help="Compute checkpoint SHA256 values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
            continue
        if "archive" in path.parts:
            continue
        model_dir = next(
            (parent for parent in path.parents if parent.parent == root),
            path.parent,
        )
        row: dict[str, object] = {
            "model_dir": model_dir.name,
            "checkpoint": str(path),
            "mib": round(path.stat().st_size / 1024**2, 3),
        }
        row.update(source_hints(model_dir))
        if args.hash:
            row["sha256"] = sha256(path)
        rows.append(row)
    print(json.dumps({"root": str(root), "checkpoint_count": len(rows), "checkpoints": rows}, indent=2))


if __name__ == "__main__":
    main()
