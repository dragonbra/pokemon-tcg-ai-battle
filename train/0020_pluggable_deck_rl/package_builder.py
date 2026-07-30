"""Build self-contained zero-shot candidate packages from the frozen foundation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


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
    "region_bot": "kaggle_submission_54954310_episode_88830387_player_1",
}
LOCAL_DECKS = {
    "region_bot": PROJECT_ROOT / "decks/region_bot_raging_bolt.csv",
}
PERSONA_POCS = {
    "dragapult": {
        "target_name": "0020_persona226_dragapult_poc",
        "source_id": 226,
        "team_name": "THIRD PTCG Club",
        "comparison_version": "V2_zero_shot_dragapult",
        "evidence": {
            "deck_sha256": (
                "d7381d44ef13b77909fa867204b8e6e6940aab53ee4fb18175c7d800a134ff34"
            ),
            "0015_role": "target_dragapult_dusknoir",
            "0019_exact_deck_winning_episodes": 0,
            "0019_closest_deck_sha256": (
                "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
            ),
            "0019_closest_deck_shared_card_slots": 49,
            "0019_closest_deck_different_card_slots": 11,
            "0019_closest_deck_winning_episodes": 1,
        },
    },
    "region_bot": {
        "target_name": "0020_persona98_raging_bolt_poc",
        "source_id": 98,
        "team_name": "James Cox & Henry Chao",
        "comparison_version": "V8_zero_shot_region_bot",
    },
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
RL_PORTABLE_INFERENCE = PROJECT_ROOT / "rl/policy/portable_inference.py"
RL_PACKAGE_MAIN = PROJECT_ROOT / "rl_package_main.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_candidate(
    deck_key: str,
    *,
    overwrite: bool = False,
    persona_poc: bool = False,
) -> Path:
    if persona_poc and deck_key not in PERSONA_POCS:
        raise ValueError(f"persona_poc is not registered for {deck_key}")
    source_name = DECK_SOURCES[deck_key]
    persona = PERSONA_POCS.get(deck_key) if persona_poc else None
    target_name = persona["target_name"] if persona else f"0020_zero_shot_{deck_key}"
    target = CANDIDATE_ROOT / target_name
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"candidate already exists: {target}")
        shutil.rmtree(target)
    strategy = target / "strategy"
    strategy.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "package_main.py", target / "main.py")
    deck_source = LOCAL_DECKS.get(deck_key)
    if deck_source is None:
        deck_source = (
            REPOSITORY_ROOT / f"evaluation/arena/opponents/{source_name}/deck.csv"
        )
    shutil.copy2(deck_source, target / "deck.csv")
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
        "source_id": int(persona["source_id"]) if persona else 0,
        "training_updates": 0,
    }
    if persona:
        manifest.update(
            {
                "schema_version": "0020_persona_poc_candidate_v1",
                "audit_only": True,
                "persona_team_name": persona["team_name"],
                "comparison_version": persona["comparison_version"],
            }
        )
        if "evidence" in persona:
            manifest["persona_evidence"] = persona["evidence"]
    if deck_key == "region_bot":
        manifest["kaggle_provenance"] = {
            "leaderboard_date": "2026-07-30",
            "rank": 1,
            "team_name": "James Cox & Henry Chao",
            "leaderboard_score": 1180.3,
            "submission_id": 54954310,
            "episode_id": 88830387,
            "episode_player_index": 1,
            "deck_sha256": (
                "f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a"
            ),
        }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def build_rl_candidate(
    deck_key: str,
    *,
    checkpoint: Path,
    target_name: str,
    overwrite: bool = False,
) -> Path:
    """Build a self-contained PPO candidate from an audited model-only checkpoint."""
    checkpoint = checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "0020_dragapult_actor_critic_model_only_v1":
        raise ValueError("RL candidate requires the 0020 model-only checkpoint schema")
    update = payload.get("update")
    metadata = payload.get("metadata")
    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise ValueError("RL checkpoint has no valid positive update")
    if not isinstance(metadata, dict) or metadata.get("stage") != "ppo":
        raise ValueError("RL checkpoint is not from the PPO stage")
    forbidden = {"optimizer", "scheduler", "rng", "rollout_buffer", "replay"}
    if forbidden.intersection(payload):
        raise ValueError("RL checkpoint contains forbidden resumable state")

    source_name = DECK_SOURCES[deck_key]
    target = CANDIDATE_ROOT / target_name
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"candidate already exists: {target}")
        shutil.rmtree(target)
    strategy = target / "strategy"
    strategy.mkdir(parents=True)
    shutil.copy2(RL_PACKAGE_MAIN, target / "main.py")
    deck_source = LOCAL_DECKS.get(deck_key)
    if deck_source is None:
        deck_source = REPOSITORY_ROOT / f"evaluation/arena/opponents/{source_name}/deck.csv"
    shutil.copy2(deck_source, target / "deck.csv")
    shutil.copytree(CG_SOURCE, target / "cg")
    for name in RUNTIME_FILES:
        shutil.copy2(PROJECT_ROOT / name, strategy / name)
    shutil.copy2(RL_PORTABLE_INFERENCE, strategy / "portable_inference.py")
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
    shutil.copy2(checkpoint, strategy / "model.bin")
    shutil.copy2(
        FOUNDATION_ROOT / "artifact/card_ontology.json",
        strategy / "card_ontology.json",
    )
    manifest = {
        "schema_version": "0020_ppo_candidate_v1",
        "candidate": target.name,
        "deck_source": source_name,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_training_version": metadata.get("version"),
        "training_updates": update,
        "source_id": 0,
        "optimizer_state_saved": False,
        "rollout_saved": False,
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
    parser.add_argument("--persona-poc", action="store_true")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.decks) - set(DECK_SOURCES))
    if unknown:
        parser.error(f"unknown deck key: {unknown[0]}")
    deck_keys = args.decks or DECK_SOURCES
    if args.persona_poc and (
        len(deck_keys) != 1 or list(deck_keys)[0] not in PERSONA_POCS
    ):
        parser.error(
            "--persona-poc requires exactly one registered deck key: "
            + ", ".join(PERSONA_POCS)
        )
    for deck_key in deck_keys:
        print(
            build_candidate(
                deck_key,
                overwrite=args.overwrite,
                persona_poc=args.persona_poc,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
