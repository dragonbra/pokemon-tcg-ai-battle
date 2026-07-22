"""Build a standard candidate package backed only by a full-action checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MAIN_TEMPLATE = r'''
from pathlib import Path

from rl.model.full_action_inference import FullActionPolicy


CHECKPOINT = Path(__CHECKPOINT__)
_POLICY = FullActionPolicy.from_checkpoint(CHECKPOINT, map_location="cpu")
PACKAGE_ROOT = Path(__file__).resolve().parent
DECK = [
    int(line)
    for line in (PACKAGE_ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def agent(observation):
    # ``select=None`` is the submission deck/reset protocol, not a strategy fallback.
    if observation.get("select") is None:
        _POLICY.reset()
        return list(DECK)
    return _POLICY.select(observation)
'''


def build(output: Path, checkpoint: Path, source_package: Path) -> Path:
    output = output.resolve()
    checkpoint = checkpoint.resolve()
    source_package = source_package.resolve()
    if output.exists():
        raise FileExistsError(f"candidate already exists: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (source_package / "deck.csv").is_file() or not (source_package / "cg").is_dir():
        raise FileNotFoundError(f"source package is missing deck.csv or cg: {source_package}")
    output.mkdir(parents=True)
    shutil.copy2(source_package / "deck.csv", output / "deck.csv")
    shutil.copytree(source_package / "cg", output / "cg")
    (output / "main.py").write_text(
        MAIN_TEMPLATE.replace("__CHECKPOINT__", repr(str(checkpoint))).lstrip(),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "ptcg_pure_model_candidate_v1",
        "checkpoint": str(checkpoint),
        "source_package": str(source_package),
        "fallback": None,
        "action_contract": "full_action_set_v1",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, default=Path("work/alakazam_v9"))
    args = parser.parse_args()
    print(build(args.output, args.checkpoint, args.source_package))


if __name__ == "__main__":
    main()
