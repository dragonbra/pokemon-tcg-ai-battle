"""V5 memory-safe restart of generalist 001-067 training from U208."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from .run_v1 import PROJECT_ROOT, ROOT, RULES, run


VERSION = "V5_generalist_focal_001_067_u208_micro512"
START_UPDATE = 208
FOCAL_DECK_IDS = tuple(f"{index:03d}" for index in range(1, 68))
PARENT_CHECKPOINT = ROOT / "rl_runs/0043_champion_league_rl/versions/V3_generalist_focal_001_067/checkpoint/update-000208.pt"
PARENT_PFSP_STATE = ROOT / "rl_runs/0043_champion_league_rl/versions/V2_mixed_focal_cohorts/artifact/pfsp_state.json"
VERSION_ROOT = ROOT / "rl_runs/0043_champion_league_rl/versions" / VERSION
WANDB_RUN_ID = "0043-v5-generalist-focal-001-067-u208-micro512"
FORWARD_MICROBATCH_SIZE = 512


def readiness(*, version_root: Path = VERSION_ROOT) -> dict[str, object]:
    registry = AssetRegistry.load(PROJECT_ROOT); audit = registry.validate_all()
    decks = tuple(d.deck_id for d in registry.decks if "training" in d.roles)
    if decks != FOCAL_DECK_IDS:
        raise RuntimeError("V5 focal pool is not exact 001-067")
    required = (PARENT_CHECKPOINT, PARENT_PFSP_STATE, RULES, DEFAULT_BUILD_DIR / "_ptcg_cuda.so")
    missing = [str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError(f"V5 artifacts missing: {missing}")
    if version_root.exists() and any(version_root.rglob("*")):
        raise FileExistsError(f"V5 formal version path is already used: {version_root}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if parent.get("schema_version") != "0043_focal_v1_model_only_v1" or parent.get("update") != 208:
        raise RuntimeError("V5 parent is not the complete U208 model-only checkpoint")
    return {
        "schema_version": "0043_v5_memory_safe_readiness_v1", "status": "READY_AWAITING_USER_LAUNCH",
        "version": VERSION, "asset_audit": asdict(audit),
        "parent": {"checkpoint_update": 208, "path": str(PARENT_CHECKPOINT.relative_to(ROOT)), "sha256": sha256_file(PARENT_CHECKPOINT), "pfsp_state": str(PARENT_PFSP_STATE.relative_to(ROOT)), "pfsp_state_sha256": sha256_file(PARENT_PFSP_STATE), "pfsp_boundary": "V2 final committed state; conservatively excludes V3 U207 and failed U208 observations"},
        "focal": {"deck_ids": list(FOCAL_DECK_IDS), "schedule": "seeded_frequency_balanced_random_v1"},
        "optimizer": {"initialization": "fresh", "logical_minibatch_size": 2048, "forward_microbatch_size": FORWARD_MICROBATCH_SIZE, "epochs": 3, "semantic_change": False},
        "reference_anchor_update": 0,
        "cuda_engine": {"version": "2.0", "extension_sha256": sha256_file(DEFAULT_BUILD_DIR / "_ptcg_cuda.so")},
        "wandb": {"entity": "dragon_bra", "project": "pokemon-tcg-policy-learning", "run_id": WANDB_RUN_ID},
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--launch-formal",action="store_true");p.add_argument("--updates",type=int,default=None);p.add_argument("--wandb-mode",choices=("online","offline"),default="online");p.add_argument("--readiness-output",type=Path);a=p.parse_args();result=readiness()
    if not a.launch_formal:
        if a.readiness_output: a.readiness_output.parent.mkdir(parents=True,exist_ok=True);a.readiness_output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps(result,indent=2,sort_keys=True));return 0
    run(updates=a.updates,wandb_mode=a.wandb_mode,launch_formal=True,version=VERSION,start_update=START_UPDATE,parent_checkpoint=PARENT_CHECKPOINT,parent_pfsp_state=PARENT_PFSP_STATE,wandb_run_id=WANDB_RUN_ID,wandb_name="0043 · V5 generalist 001-067 · U208 micro512",focal_deck_ids=FOCAL_DECK_IDS,forward_microbatch_size=FORWARD_MICROBATCH_SIZE)
    return 0


if __name__ == "__main__": raise SystemExit(main())
