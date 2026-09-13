"""Restart-safe U407 G2-candidate cross-deck CUDA-2048 assessment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import torch

from evaluation.runtime.seeded import build_seeded_runtime

from ..assets import AssetRegistry, sha256_file
from ..cuda_engine_2.build import DEFAULT_BUILD_DIR
from ..cuda_engine_2.identity import CudaEngineIdentity
from ..own_archetype import OwnArchetypeVocabulary
from ..policy.actor_critic import SemanticActorCritic
from ..policy.strategy_adapters import PolicyStrategyAdapter, ValueResidualAdapter
from ..policy_identity import materialize_policy_bundle
from ..rollout import ChunkedCudaRolloutCollector, RolloutJob
from ..runtime import load_policy
from ..training.run_v1 import RULES, _load_opponent, _runtime_root
from .candidate import CONTRACT_ID, CandidateAudit, _tensor_hash, materialize
from .preflight import require_formal_evaluation_manifest
from .schedule import MASTER_SEED, materialize as materialize_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
VERSION = "V7_u407_g2_candidate_cross_deck_cuda2048"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
ARTIFACT_ROOT = VERSION_ROOT / "artifact"
REPORT_ROOT = ARTIFACT_ROOT / "reports"
STATE_PATH = ARTIFACT_ROOT / "state.json"
V6_ROOT = ROOT / (
    "rl_runs/0044_g2_dragapult_policy_option_lora/versions/"
    "V6_generalist_focal_001_067_u233_restart"
)
U407_CHECKPOINT = V6_ROOT / "checkpoint/update-000407.pt"
BASE_PORTABLE = ROOT / (
    "rl_runs/0044_g2_dragapult_policy_option_lora/versions/V1_focal_002_007/"
    "artifact/focal_seed/model.bin"
)
RANDOM_SAMPLE_SEED = 430_043_407
RANDOM_DECK_IDS = tuple(
    f"{value:03d}"
    for value in sorted(random.Random(RANDOM_SAMPLE_SEED).sample(range(30, 68), 10))
)
FIXED_DECK_IDS = tuple(f"{value:03d}" for value in range(1, 11))
EVALUATION_DECK_IDS = FIXED_DECK_IDS + RANDOM_DECK_IDS
EXECUTION_DECK_IDS = RANDOM_DECK_IDS + FIXED_DECK_IDS
ARMS = ("g1", "g2_candidate")
MAX_PARALLEL_ARMS = 2
EXPECTED_OPPONENT_EFFECTIVE = (
    "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _exact_hash(cards: tuple[int, ...]) -> str:
    return hashlib.sha256("\n".join(map(str, cards)).encode("ascii")).hexdigest()


def audit_deck_scope() -> dict[str, Any]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    by_id = {row.deck_id: row for row in registry.decks}
    decks = []
    for deck_id in EVALUATION_DECK_IDS:
        asset = by_id[deck_id]
        cards = tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
        if len(cards) != 60:
            raise RuntimeError(f"deck {deck_id} is not exact 60")
        decks.append({
            "deck_id": deck_id,
            "display_name": asset.name,
            "card_count": len(cards),
            "exact_deck_sha256": asset.content_sha256,
            "file_sha256": asset.file_sha256,
        })
    return {
        "status": "PASS",
        "random_sample_seed": RANDOM_SAMPLE_SEED,
        "fixed_deck_ids": list(FIXED_DECK_IDS),
        "random_deck_ids": list(RANDOM_DECK_IDS),
        "deck_ids": list(EVALUATION_DECK_IDS),
        "decks": decks,
    }


def validate_u407_checkpoint(path: Path = U407_CHECKPOINT) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("U407 checkpoint is missing")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "0044_focal_v1_model_only_v1"
        or payload.get("update") != 407
        or payload.get("metadata", {}).get("version")
        != "V6_generalist_focal_001_067_u233_restart"
    ):
        raise RuntimeError("checkpoint is not the complete V6 U407 model-only identity")
    state = payload.get("state_dict", {})
    if (
        state.get("value_adapter.own_embedding.weight") is None
        or tuple(state["value_adapter.own_embedding.weight"].shape) != (29, 16)
        or tuple(state["value_head.heads.archetype.1.weight"].shape) != (15, 320)
    ):
        raise RuntimeError("U407 model shape contract mismatch")
    return {"update": 407, "sha256": sha256_file(path), "status": "PASS"}


def validate_terminal_boundary() -> dict[str, Any]:
    checkpoint = validate_u407_checkpoint()
    if (V6_ROOT / "checkpoint/update-000408.pt").exists():
        raise RuntimeError("V6 escaped the U407 terminal boundary")
    rows = [
        json.loads(line) for line in
        (V6_ROOT / "artifact/training_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if (
        int(rows[-1].get("trainer/update", -1)) != 407
        or int(rows[-1].get("checkpoint/update", -1)) != 407
        or int(rows[-1].get("rollout/source_policy_update", -1)) != 406
    ):
        raise RuntimeError("V6 canonical metrics did not stop at U407/source-U406")
    pfsp = json.loads((V6_ROOT / "artifact/pfsp_state.json").read_text())
    records = [
        *pfsp["deck_records"].values(),
        *pfsp["policy_records"].values(),
        *pfsp["joint_records"].values(),
    ]
    if max(int(row["last_updated"]) for row in records) != 406:
        raise RuntimeError("V6 PFSP state did not stop at committed source-U406")
    return {
        **checkpoint,
        "metric_update": 407,
        "rollout_source_policy_update": 406,
        "pfsp_max_source_update": 406,
    }


def initial_state() -> dict[str, Any]:
    return {
        "schema_version": "0044_u407_g2_candidate_gate_state_v1",
        "status": "WAITING_FOR_U407",
        "version": VERSION,
        "random_sample_seed": RANDOM_SAMPLE_SEED,
        "fixed_deck_ids": list(FIXED_DECK_IDS),
        "random_deck_ids": list(RANDOM_DECK_IDS),
        "evaluation_deck_ids": list(EVALUATION_DECK_IDS),
        "completed_arms": [],
        "human_promotion_decision": "REQUIRED",
    }


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    state = json.loads(path.read_text()) if path.is_file() else initial_state()
    completed = state.get("completed_arms")
    if not isinstance(completed, list) or len(completed) != len(set(completed)):
        raise RuntimeError("state has duplicate completed arms")
    allowed = {f"{deck}:{arm}" for deck in EVALUATION_DECK_IDS for arm in ARMS}
    if any(key not in allowed for key in completed):
        raise RuntimeError("state has an unknown completed arm")
    if (
        state.get("random_sample_seed") != RANDOM_SAMPLE_SEED
        or state.get("evaluation_deck_ids") != list(EVALUATION_DECK_IDS)
    ):
        raise RuntimeError("state cohort identity changed")
    return state


def _wilson(wins: int, games: int) -> list[float]:
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return [center - half, center + half]


def _g1_candidate(deck: tuple[int, ...], deck_id: str, own_id: int, output: Path,
                  device: torch.device) -> tuple[SemanticActorCritic, CandidateAudit]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    policy = next(row for row in registry.policies if row.policy_id == "Champion-G1")
    portable_asset = next(row for row in policy.artifacts if row.purpose == "portable_fp16_artifact")
    source_asset = next(row for row in policy.artifacts if row.purpose == "model_only_delta")
    source_path = PROJECT_ROOT / source_asset.path
    portable_path = PROJECT_ROOT / portable_asset.path
    strict = torch.load(portable_path, map_location="cpu", weights_only=True)
    copied = dict(strict)
    copied["metadata"] = {
        **strict["metadata"], "focal_deck_id": deck_id,
        "own_archetype_id": own_id, "deployment_contract": CONTRACT_ID,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(copied, temporary)
    os.replace(temporary, output)
    loaded = load_policy("Champion-G1", deck_id=deck_id)
    loaded.metadata["own_archetype_id"] = own_id
    model = SemanticActorCritic(
        loaded.actor, loaded.value_head, own_id, 15
    )
    model.allocation_head = loaded.allocation_head
    # The sealed G1 portable runtime carries the historical adapter classes.
    # Strict-load their identical tensors into 0044-native adapters so the
    # collector gets current diagnostic methods without changing inference.
    value_adapter = ValueResidualAdapter(320, own_archetype_classes=15)
    value_adapter.load_state_dict(loaded.value_adapter.state_dict(), strict=True)
    strategy_adapter = PolicyStrategyAdapter(320, own_archetype_classes=15)
    strategy_adapter.load_state_dict(
        loaded.policy_strategy_adapter.state_dict(), strict=True
    )
    model.value_adapter = value_adapter
    model.policy_strategy_adapter = strategy_adapter
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    deck_hash = next(row.content_sha256 for row in registry.decks if row.deck_id == deck_id)
    audit = CandidateAudit(
        source_checkpoint_sha256=sha256_file(source_path),
        portable_checkpoint_sha256=sha256_file(output),
        effective_candidate_sha256=_tensor_hash(strict, deck_hash),
        checkpoint_update=10,
        focal_exact_deck_sha256=deck_hash,
    )
    return model, audit


def validate_arm_report(report: dict[str, Any], *, arm: str, deck_id: str) -> None:
    candidate = report.get("candidate_deployment_identity_audit") or {}
    opponent = report.get("opponent_policy_identity_audit") or {}
    entries = report.get("entries")
    metrics = report.get("collector_metrics") or {}
    expected_update = 10 if arm == "g1" else 407
    if report.get("checkpoint_update") != expected_update:
        raise RuntimeError(f"{deck_id}:{arm} checkpoint update mismatch")
    if (
        report.get("status") != "PASS"
        or report.get("summary", {}).get("games") != 2048
        or not isinstance(entries, list)
        or len(entries) != 2048
        or any(row.get("valid") is not True or row.get("error") is not None for row in entries)
    ):
        raise RuntimeError(f"{deck_id}:{arm} is not 2048 valid terminal games")
    if (
        candidate.get("status") != "PASS"
        or candidate.get("contract_id") != CONTRACT_ID
        or candidate.get("storage_dtype") != "fp16"
        or candidate.get("runtime_dtype") != "fp32"
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != "Policy-0809"
    ):
        raise RuntimeError(f"{deck_id}:{arm} identity audit failed")
    if report.get("focal_opponent_shared_parameter_storages") != 0:
        raise RuntimeError(f"{deck_id}:{arm} focal/opponent storage alias")
    if (
        metrics.get("rollout/lane_routing_audit_failures", 0) != 0
        or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
    ):
        raise RuntimeError(f"{deck_id}:{arm} routing/residency gate failed")


def run_arm(*, deck_id: str, arm: str, output_root: Path) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    asset = next(row for row in registry.decks if row.deck_id == deck_id)
    deck = tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    mapped_id = next(row.archetype_id for row in vocabulary.mappings if row.deck_id == deck_id)
    device = torch.device("cuda:0")
    output_root.mkdir(parents=True, exist_ok=False)
    if arm == "g2_candidate":
        candidate, audit = materialize(
            checkpoint=U407_CHECKPOINT, base_portable=BASE_PORTABLE,
            deck=deck, deck_id=deck_id, own_archetype_id=mapped_id,
            output=output_root / "materialization/model.bin", device=device,
        )
        policy_label = "U407-G2-Candidate"
    else:
        parent_id = vocabulary.classes[mapped_id].embedding_init_from
        candidate, audit = _g1_candidate(
            deck, deck_id, parent_id, output_root / "materialization/model.bin", device
        )
        policy_label = "Champion-G1"

    opponent, opponent_audit = _load_opponent("Policy-0809", "001", device)
    focal_ptr = {p.untyped_storage().data_ptr() for p in candidate.parameters()}
    opponent_module = opponent.model if hasattr(opponent, "model") else opponent.actor
    shared = focal_ptr & {
        p.untyped_storage().data_ptr() for p in opponent_module.parameters()
    }
    schedule = materialize_schedule(
        PROJECT_ROOT, focal_deck_id=deck_id,
        focal_deployment_identity=audit.effective_candidate_sha256,
        opponent_policy_id="Policy-0809", replicas=8,
    )
    decks = {
        row.deck_id: tuple(map(int, (PROJECT_ROOT / row.deck_path).read_text().splitlines()))
        for row in registry.decks
    }
    runtime = build_seeded_runtime()
    jobs = [RolloutJob(
        game_id=row["game_id"], opponent_id=row["opponent_deck_id"],
        focal_first=row["focal_won_toss"], focal_won_toss=row["focal_won_toss"],
        coin_winner_seed=row["coin_winner_seed"],
        seed=row["engine_seed"], search_seed=row["search_seed"],
        policy_seed=row["policy_seed"], source_policy_update=audit.checkpoint_update,
        focal_deck=deck, opponent_deck=decks[row["opponent_deck_id"]],
        runtime_root=_runtime_root(), opponent_policy_id="Policy-0809",
        focal_deck_id=deck_id,
        focal_own_archetype_id=(mapped_id if arm == "g2_candidate" else parent_id),
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
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    elapsed = time.perf_counter() - started
    metrics = collector.metrics()
    entries = []
    for episode in episodes:
        choice = episode.diagnostics.get("first_player_choice")
        if not isinstance(choice, dict) or type(choice.get("focal_first")) is not bool:
            raise RuntimeError("Frozen episode lacks Agent first-player evidence")
        entries.append({
            "game_id": episode.job.game_id,
            "opponent_id": episode.job.opponent_id,
            "outcome": 1 if episode.reward == 1 else -1 if episode.reward == -1 else 0,
            "turns": episode.turns, "valid": episode.valid, "error": episode.error,
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": choice, "focal_first": bool(choice["focal_first"]),
        })
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    draws = 2048 - wins - losses
    first = [row for row in entries if row["focal_first"]]
    second = [row for row in entries if not row["focal_first"]]
    cuda = CudaEngineIdentity.resolve(
        ROOT, rule_pack=RULES, binary=DEFAULT_BUILD_DIR / "ptcg_cuda_smoke",
        extension=DEFAULT_BUILD_DIR / "_ptcg_cuda.so", require_gpu=True,
        require_extension=True,
    ).to_manifest()
    manifest = {
        "candidate_deployment_identity_audit": audit.to_manifest(),
        "opponent_policy_identity_audit": asdict(opponent_audit),
        "schedule": {
            "evaluation_id": "FrozenMeta2048-V1", "master_seed": MASTER_SEED,
            "games": 2048, "replicas": list(range(8)),
            "schedule_sha256": schedule["schedule_sha256"],
            "seat_semantics": "seeded_toss_winner_agent_context_41_choice",
            "exact_deck_pool_hash": schedule["base_schedule_sha256"],
        },
        "selection_mode": "greedy", "official_engine": True,
        "cuda_engine_identity_audit": cuda,
    }
    require_formal_evaluation_manifest(manifest)
    report = {
        "schema_version": "0044_u407_g2_candidate_arm_v1",
        "created_at": datetime.now(UTC).isoformat(), "status": "PASS",
        "arm": arm, "focal_policy_label": policy_label,
        "checkpoint_update": audit.checkpoint_update,
        "checkpoint_sha256": audit.source_checkpoint_sha256,
        "focal_deck_id": deck_id, "focal_deck_display_name": asset.name,
        "focal_exact_deck_sha256": asset.content_sha256,
        "focal_deck_cards": list(deck),
        "focal_opponent_shared_parameter_storages": len(shared),
        **manifest,
        "summary": {
            "games": 2048, "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / 2048, "wilson_95": _wilson(wins, 2048),
            "focal_first_games": len(first),
            "focal_first_win_rate": sum(row["outcome"] == 1 for row in first) / len(first),
            "focal_second_games": len(second),
            "focal_second_win_rate": sum(row["outcome"] == 1 for row in second) / len(second),
            "elapsed_seconds": elapsed, "games_per_second": 2048 / elapsed,
        },
        "collector_metrics": metrics, "entries": entries,
        "promotion_status": "HUMAN_DECISION_REQUIRED",
    }
    validate_arm_report(report, arm=arm, deck_id=deck_id)
    _atomic_json(output_root / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-arm", action="store_true")
    parser.add_argument("--deck-id")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.run_arm:
        if not args.deck_id or not args.arm or not args.output_root:
            parser.error("--run-arm requires --deck-id, --arm and --output-root")
        run_arm(deck_id=args.deck_id, arm=args.arm, output_root=args.output_root.resolve())
        return 0

    boundary = validate_terminal_boundary()
    scope = audit_deck_scope()
    state = load_state()
    state.update({
        "status": "RUNNING", "u407": boundary, "deck_scope": scope,
        "started_at": state.get("started_at") or datetime.now(UTC).isoformat(),
    })
    _atomic_json(STATE_PATH, state)
    pending = [
        (deck_id, arm) for deck_id in EXECUTION_DECK_IDS for arm in ARMS
        if f"{deck_id}:{arm}" not in state["completed_arms"]
    ]
    for offset in range(0, len(pending), MAX_PARALLEL_ARMS):
        batch = pending[offset:offset + MAX_PARALLEL_ARMS]
        processes = []
        for deck_id, arm in batch:
            output = REPORT_ROOT / deck_id / arm
            if output.exists():
                attempts = ARTIFACT_ROOT / "incomplete_attempts"
                attempts.mkdir(parents=True, exist_ok=True)
                destination = attempts / f"{deck_id}-{arm}-{time.time_ns()}"
                os.replace(output, destination)
            command = [
                sys.executable, "-m", __package__ + ".g2_candidate_gate",
                "--run-arm", "--deck-id", deck_id, "--arm", arm,
                "--output-root", str(output),
            ]
            processes.append((deck_id, arm, output, subprocess.Popen(command, cwd=ROOT)))
        state.update({
            "current_deck": None, "current_arm": None,
            "current_arms": [f"{deck_id}:{arm}" for deck_id, arm in batch],
            "max_parallel_arms": MAX_PARALLEL_ARMS,
            "execution_order": list(EXECUTION_DECK_IDS),
        })
        _atomic_json(STATE_PATH, state)
        for deck_id, arm, output, process in processes:
            returncode = process.wait()
            if returncode:
                raise RuntimeError(f"{deck_id}:{arm} worker exited {returncode}")
            report = json.loads((output / "report.json").read_text())
            validate_arm_report(report, arm=arm, deck_id=deck_id)
            state["completed_arms"].append(f"{deck_id}:{arm}")
            _atomic_json(STATE_PATH, state)
    state.update({
        "status": "EVALUATION_COMPLETE_REPORT_PENDING",
        "finished_at": datetime.now(UTC).isoformat(),
        "current_deck": None, "current_arm": None, "current_arms": [],
    })
    _atomic_json(STATE_PATH, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
