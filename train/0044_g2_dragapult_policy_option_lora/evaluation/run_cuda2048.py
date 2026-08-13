"""Formal 0044 U20 CUDA Engine 2.0 evaluation against frozen Policy-0809."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from evaluation.runtime.seeded import build_seeded_runtime
from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..cuda_engine_2.identity import CudaEngineIdentity
from ..own_archetype import OwnArchetypeVocabulary
from ..policy_identity import materialize_policy_bundle
from ..rollout import ChunkedCudaRolloutCollector, RolloutJob
from ..training.run_v1 import RULES, _load_opponent, _runtime_root
from .candidate import CONTRACT_ID, materialize
from .preflight import require_formal_evaluation_manifest
from .schedule import MASTER_SEED, materialize as materialize_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions/V1_focal_002_007"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _wilson(wins: int, games: int) -> list[float]:
    z = 1.959963984540054; p = wins / games; d = 1 + z*z/games
    c = (p + z*z/(2*games))/d
    s = z*math.sqrt(p*(1-p)/games + z*z/(4*games*games))/d
    return [c-s, c+s]


def run(*, checkpoint: Path, deck_id: str, output_root: Path, replicas: int) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    registry = AssetRegistry.load(PROJECT_ROOT); registry.validate_all()
    deck_asset = next(row for row in registry.decks if row.deck_id == deck_id)
    deck = tuple(map(int, (PROJECT_ROOT / deck_asset.deck_path).read_text().splitlines()))
    vocabulary = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=PROJECT_ROOT)
    own_id = next(row.archetype_id for row in vocabulary.mappings if row.deck_id == deck_id)
    device = torch.device("cuda:0")
    output_root.mkdir(parents=True)
    candidate, candidate_audit = materialize(
        checkpoint=checkpoint,
        base_portable=VERSION_ROOT / "artifact/focal_seed/model.bin",
        deck=deck, deck_id=deck_id, own_archetype_id=own_id,
        output=output_root / "materialization/model.bin", device=device,
    )
    opponent, opponent_audit = _load_opponent("Policy-0809", "001", device)
    focal_ptr = {p.untyped_storage().data_ptr() for p in candidate.parameters()}
    opponent_module = opponent.model if hasattr(opponent, "model") else opponent.actor
    shared = focal_ptr & {p.untyped_storage().data_ptr() for p in opponent_module.parameters()}
    if shared:
        raise RuntimeError("FATAL: focal/opponent parameter storage alias")
    schedule = materialize_schedule(
        PROJECT_ROOT, focal_deck_id=deck_id,
        focal_deployment_identity=candidate_audit.effective_candidate_sha256,
        opponent_policy_id="Policy-0809", replicas=replicas,
    )
    decks = {row.deck_id: tuple(map(int, (PROJECT_ROOT / row.deck_path).read_text().splitlines())) for row in registry.decks}
    runtime = build_seeded_runtime()
    jobs = [RolloutJob(
        game_id=row["game_id"], opponent_id=row["opponent_deck_id"],
        focal_first=row["focal_won_toss"], focal_won_toss=row["focal_won_toss"],
        seed=row["engine_seed"], search_seed=row["search_seed"], policy_seed=row["policy_seed"],
        source_policy_update=candidate_audit.checkpoint_update,
        focal_deck=deck, opponent_deck=decks[row["opponent_deck_id"]],
        runtime_root=_runtime_root(), opponent_policy_id="Policy-0809",
        focal_deck_id=deck_id, focal_own_archetype_id=own_id,
        engine_library=runtime.library_path, action_boundary_mode="enabled",
        trace_policy="errors_and_sample",
    ) for row in schedule["jobs"]]
    collector = ChunkedCudaRolloutCollector(
        candidate, opponent, rollout_batch_size=256, trajectory_games_per_update=None,
        device=device, rules_path=RULES, extension_dir=DEFAULT_BUILD_DIR,
        lane_count=256, mode="greedy", check_interval=8, record_trajectory=False,
        agent_selects_first_player=True, opponent_policy_id="Policy-0809",
        opponent_identity_audit=opponent_audit,
    )
    started=time.perf_counter(); episodes=collector.collect(jobs); elapsed=time.perf_counter()-started
    metrics=collector.metrics()
    entries=[]
    for episode in episodes:
        d=episode.diagnostics
        choice=d.get("first_player_choice")
        if not isinstance(choice, dict) or type(choice.get("focal_first")) is not bool:
            raise RuntimeError(
                f"Frozen episode {episode.job.game_id} lacks Agent first-player evidence"
            )
        focal_first = bool(choice["focal_first"])
        entries.append({
            "game_id":episode.job.game_id,"opponent_id":episode.job.opponent_id,
            "outcome":1 if episode.reward==1 else -1 if episode.reward==-1 else 0,
            "turns":episode.turns,"valid":episode.valid,"error":episode.error,
            "focal_won_toss":episode.job.focal_won_toss,
            "first_player_choice":choice,"focal_first":focal_first,
            "lane_routing_audit_status":d.get("lane_routing_audit_status"),
        })
    games=256*replicas; wins=sum(r["outcome"]==1 for r in entries); losses=sum(r["outcome"]==-1 for r in entries); draws=games-wins-losses
    first=[r for r in entries if r["focal_first"]]; second=[r for r in entries if not r["focal_first"]]
    cuda=CudaEngineIdentity.resolve(
        ROOT, rule_pack=RULES,
        binary=DEFAULT_BUILD_DIR / "ptcg_cuda_smoke",
        extension=DEFAULT_BUILD_DIR / "_ptcg_cuda.so",
        require_gpu=True, require_extension=True,
    ).to_manifest()
    manifest={
        "candidate_deployment_identity_audit":candidate_audit.to_manifest(),
        "opponent_policy_identity_audit":asdict(opponent_audit),
        "schedule":{"evaluation_id":"FrozenMeta2048-V1" if replicas==8 else "FrozenMeta256-V1","master_seed":MASTER_SEED,"games":games,"replicas":replicas,"schedule_sha256":schedule["schedule_sha256"],"seat_semantics":"seeded_toss_winner_agent_context_41_choice","exact_deck_pool_hash":schedule["base_schedule_sha256"]},
        "selection_mode":"greedy","official_engine":True,"cuda_engine_identity_audit":cuda,
    }
    require_formal_evaluation_manifest(manifest)
    if len(entries)!=games or any(not r["valid"] or r["error"] for r in entries):
        raise RuntimeError("formal evaluation has error/unfinished games")
    if metrics.get("rollout/lane_routing_audit_failures",0) or metrics.get("rollout/cuda_feature_d2h_bytes",0):
        raise RuntimeError("formal evaluation routing/residency gate failed")
    summary={"games":games,"wins":wins,"losses":losses,"draws":draws,"win_rate":wins/games,"wilson_95":_wilson(wins,games),"focal_first_games":len(first),"focal_first_win_rate":sum(r["outcome"]==1 for r in first)/len(first),"focal_second_games":len(second),"focal_second_win_rate":sum(r["outcome"]==1 for r in second)/len(second),"elapsed_seconds":elapsed,"games_per_second":games/elapsed}
    report={"schema_version":"0044_u20_frozen_policy0809_cuda_v1","created_at":datetime.now(UTC).isoformat(),"status":"PASS","checkpoint":str(checkpoint),"checkpoint_sha256":sha256_file(checkpoint),"checkpoint_update":candidate_audit.checkpoint_update,"focal_deck_id":deck_id,"focal_deck_display_name":deck_asset.name,"focal_exact_deck_sha256":candidate_audit.focal_exact_deck_sha256,"focal_deck_cards":list(deck),**manifest,"summary":summary,"collector_metrics":metrics,"entries":entries,"promotion_status":"NOT_PROMOTED_EVIDENCE_ONLY"}
    _atomic_json(output_root/"report.json",report)
    return report


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--deck-id",required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--replicas",type=int,choices=(1,8),default=8);a=p.parse_args()
    r=run(checkpoint=a.checkpoint.resolve(),deck_id=a.deck_id,output_root=a.output_root.resolve(),replicas=a.replicas)
    print(json.dumps(r["summary"],indent=2));return 0


if __name__=="__main__": raise SystemExit(main())
