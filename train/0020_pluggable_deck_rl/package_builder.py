"""Build self-contained zero-shot candidate packages from the frozen foundation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent
FOUNDATION_ROOT = (
    REPOSITORY_ROOT
    / "rl_runs/0020_pluggable_deck_rl/versions/V1_frozen_0019_epoch13"
)
CANDIDATE_ROOT = REPOSITORY_ROOT / "evaluation/arena/candidates"
CG_SOURCE = (
    REPOSITORY_ROOT
    / "evaluation/arena/opponents/dragapult_ex_03_v20260729_rl/cg"
)
DECK_SOURCES = {
    "alakazam": "alakazam_dudunsparce_04_sota",
    "dragapult": "dragapult_ex_03_v20260729_rl",
    "marnie": "marnies_grimmsnarl_ex_froslass_05_bc",
    "lucario": "mega_lucario_ex_solrock_07_bc",
    "kangaskhan": "mega_kangaskhan_ex_crustle_01_v20260729_bc",
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


def build_candidate(deck_key: str, *, overwrite: bool = False) -> Path:
    source_name = DECK_SOURCES[deck_key]
    target = CANDIDATE_ROOT / f"0020_zero_shot_{deck_key}"
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"candidate already exists: {target}")
        shutil.rmtree(target)
    strategy = target / "strategy"
    strategy.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "package_main.py", target / "main.py")
    shutil.copy2(
        REPOSITORY_ROOT / f"evaluation/arena/opponents/{source_name}/deck.csv",
        target / "deck.csv",
    )
    shutil.copytree(CG_SOURCE, target / "cg")
    for name in RUNTIME_FILES:
        shutil.copy2(PROJECT_ROOT / name, strategy / name)
    shutil.copytree(
        PROJECT_ROOT / "features",
        strategy / "features",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        PROJECT_ROOT / "knowledge",
        strategy / "knowledge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(
        FOUNDATION_ROOT / "checkpoint/epoch-0013-da9b13d6f82d19d4.pt",
        strategy / "model.bin",
    )
    shutil.copy2(
        FOUNDATION_ROOT / "artifact/card_ontology.json",
        strategy / "card_ontology.json",
    )
    manifest = {
        "schema_version": "0020_zero_shot_candidate_v1",
        "candidate": target.name,
        "deck_source": source_name,
        "checkpoint_sha256": (
            "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"
        ),
        "foundation_version": "V1_frozen_0019_epoch13",
        "source_id": 0,
        "training_updates": 0,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("decks", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.decks) - set(DECK_SOURCES))
    if unknown:
        parser.error(f"unknown deck key: {unknown[0]}")
    for deck_key in args.decks or DECK_SOURCES:
        print(build_candidate(deck_key, overwrite=args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
