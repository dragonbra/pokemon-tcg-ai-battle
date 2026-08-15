"""Restart-safe Full-67 Benchmark V2: Champion-G3 versus G4 candidate U57."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from ..assets import sha256_file
from ..training.run_v1 import ROOT
from . import benchmark_v2_g2_vs_g3_u110_full67 as previous
from .run_benchmark_v2 import validate_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "V16_g3_vs_g4_u57_full67_benchmark_v2"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
ARTIFACT_ROOT = VERSION_ROOT / "artifact"
REPORT_ROOT = ARTIFACT_ROOT / "reports"
STATE_PATH = ARTIFACT_ROOT / "state.json"

G3_VERSION = previous.G3_VERSION
G3_CHECKPOINT = previous.G3_CHECKPOINT
G3_CHECKPOINT_SHA256 = previous.G3_CHECKPOINT_SHA256
G4_VERSION = "V15_g4_v14_u2_rollout_only_cuda_recovery"
G4_CHECKPOINT = ROOT / (
    "rl_runs/0044_g2_dragapult_policy_option_lora/versions/"
    f"{G4_VERSION}/checkpoint/update-000057.pt"
)
G4_CHECKPOINT_SHA256 = "dff296e81046b0882b0a61f89e926a6a1d401c7fcb04eefd8841ec133200fe50"
ARMS = ("g3", "g4_u57")
CHECKPOINTS = {
    "g3": (G3_CHECKPOINT, 110),
    "g4_u57": (G4_CHECKPOINT, 57),
}

# Preserve the actual order in which the prior report acquired its reusable
# Champion-G3 baselines.  The remaining decks retain that run's seeded order.
REUSABLE_G3_DECK_IDS = (
    "007", "003", "002", "066", "067", "065", "006", "043", "023", "050",
    "011", "026", "038", "046", "053", "010", "013", "054", "058", "039",
)
EXECUTION_DECK_IDS = REUSABLE_G3_DECK_IDS + tuple(
    deck_id for deck_id in previous.EXECUTION_DECK_IDS
    if deck_id not in REUSABLE_G3_DECK_IDS
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _new_report(deck_id: str, arm: str) -> Path:
    return REPORT_ROOT / deck_id / arm / "report.json"


def _previous_g3_report(deck_id: str) -> Path:
    return previous.REPORT_ROOT / deck_id / "g3_u110" / "report.json"


def report_path(deck_id: str, arm: str) -> Path:
    if arm == "g3" and _previous_g3_report(deck_id).is_file():
        return _previous_g3_report(deck_id)
    return _new_report(deck_id, arm)


def preflight(*, require_unused: bool = False) -> dict[str, Any]:
    expected = {f"{value:03d}" for value in range(1, 68)}
    if (
        len(EXECUTION_DECK_IDS) != 67
        or set(EXECUTION_DECK_IDS) != expected
        or EXECUTION_DECK_IDS[:len(REUSABLE_G3_DECK_IDS)] != REUSABLE_G3_DECK_IDS
    ):
        raise RuntimeError("Full-67 G3/G4 execution order contract changed")
    if require_unused and VERSION_ROOT.exists() and any(VERSION_ROOT.rglob("*")):
        raise FileExistsError(f"formal V16 path is already used: {VERSION_ROOT}")
    arms = []
    for arm, checkpoint, expected_hash, update, version in (
        ("g3", G3_CHECKPOINT, G3_CHECKPOINT_SHA256, 110, G3_VERSION),
        ("g4_u57", G4_CHECKPOINT, G4_CHECKPOINT_SHA256, 57, G4_VERSION),
    ):
        if not checkpoint.is_file() or sha256_file(checkpoint) != expected_hash:
            raise RuntimeError(f"{arm} checkpoint identity changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != "0044_focal_v1_model_only_v1"
            or payload.get("update") != update
            or payload.get("metadata", {}).get("version") != version
        ):
            raise RuntimeError(f"{arm} checkpoint metadata mismatch")
        arms.append({
            "arm": arm, "update": update, "version": version,
            "path": str(checkpoint), "sha256": expected_hash,
        })
    reusable = []
    for deck_id in REUSABLE_G3_DECK_IDS:
        path = _previous_g3_report(deck_id)
        if not path.is_file():
            raise RuntimeError(f"missing reusable Champion-G3 baseline: {deck_id}")
        report = json.loads(path.read_text(encoding="utf-8"))
        previous.validate_arm_report(report, arm="g3_u110", deck_id=deck_id)
        reusable.append({"deck_id": deck_id, "report": str(path)})
    return {
        "schema_version": "0044_g3_vs_g4_u57_full67_preflight_v1",
        "status": "PASS", "version": VERSION,
        "execution_deck_ids": list(EXECUTION_DECK_IDS),
        "reused_g3_deck_ids": list(REUSABLE_G3_DECK_IDS),
        "reused_g3_reports": reusable, "arms": arms,
        "games_per_arm_per_deck": 2048, "total_games": 67 * 2 * 2048,
    }


def validate_arm_report(report: dict[str, Any], *, arm: str, deck_id: str) -> None:
    validate_report(report)
    checkpoint, update = CHECKPOINTS[arm]
    audit = report["focal_policy_identity_audit"]
    if (
        report.get("focal_deck_id") != deck_id
        or report.get("focal_checkpoint_update") != update
        or audit.get("source_checkpoint_sha256") != sha256_file(checkpoint)
        or report.get("opponent_policy_identity_audit", {}).get("requested_policy_id")
        != "Policy-0809"
    ):
        raise RuntimeError(f"{deck_id}:{arm} Benchmark V2 identity mismatch")


def validate_pair(g3: dict[str, Any], g4: dict[str, Any], *, deck_id: str) -> None:
    validate_arm_report(g3, arm="g3", deck_id=deck_id)
    validate_arm_report(g4, arm="g4_u57", deck_id=deck_id)
    if (
        g3["focal_exact_deck_sha256"] != g4["focal_exact_deck_sha256"]
        or g3["opponent_policy_identity_audit"]["effective_policy_sha256"]
        != g4["opponent_policy_identity_audit"]["effective_policy_sha256"]
        or g3["schedule"]["common_random_schedule_sha256"]
        != g4["schedule"]["common_random_schedule_sha256"]
    ):
        raise RuntimeError(f"{deck_id}: paired identity mismatch")
    fields = (
        "game_id", "opponent_id", "opponent_meta_archetype_id", "engine_seed",
        "search_seed", "policy_seed", "coin_winner_seed", "focal_won_toss",
    )
    for left, right in zip(g3["entries"], g4["entries"], strict=True):
        if tuple(left[key] for key in fields) != tuple(right[key] for key in fields):
            raise RuntimeError(f"{deck_id}: paired job identity mismatch")


def _read_pair(deck_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    reports = tuple(
        json.loads(report_path(deck_id, arm).read_text(encoding="utf-8"))
        for arm in ARMS
    )
    validate_pair(reports[0], reports[1], deck_id=deck_id)
    return reports


def _validated_completed() -> list[str]:
    completed = []
    for deck_id in EXECUTION_DECK_IDS:
        if not all(report_path(deck_id, arm).is_file() for arm in ARMS):
            continue
        _read_pair(deck_id)
        completed.append(deck_id)
    return completed


def _run_arm(deck_id: str, arm: str) -> subprocess.Popen[Any]:
    checkpoint, update = CHECKPOINTS[arm]
    output = REPORT_ROOT / deck_id / arm
    command = [
        sys.executable, "-m",
        "train.0047_meta_routed_moe_rl.evaluation.run_benchmark_v2",
        "--deck-id", deck_id, "--output-root", str(output),
        "--checkpoint", str(checkpoint), "--checkpoint-update", str(update),
    ]
    return subprocess.Popen(command, cwd=ROOT)


def _render(*, partial: bool) -> None:
    command = [
        sys.executable, "-m",
        "train.0047_meta_routed_moe_rl.evaluation."
        "render_benchmark_v2_g3_vs_g4_u57_full67",
    ]
    if partial:
        command.append("--allow-partial")
    subprocess.run(command, cwd=ROOT, check=True)


def _run_controller() -> int:
    before = not STATE_PATH.exists()
    readiness = preflight(require_unused=before)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (ARTIFACT_ROOT / "controller.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another G3/G4 Benchmark V2 controller is active") from exc
    completed = _validated_completed()
    state = {
        **readiness,
        "schema_version": "0044_g3_vs_g4_u57_full67_state_v1",
        "status": "RUNNING", "completed_deck_ids": completed,
        "current_deck_id": None, "current_arms": [],
        "started_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(STATE_PATH, state)
    for deck_id in EXECUTION_DECK_IDS:
        if deck_id in completed:
            continue
        arms_to_run = [arm for arm in ARMS if not report_path(deck_id, arm).is_file()]
        processes = []
        for arm in arms_to_run:
            output = REPORT_ROOT / deck_id / arm
            if output.exists():
                attempts = ARTIFACT_ROOT / "incomplete_attempts"
                attempts.mkdir(parents=True, exist_ok=True)
                os.replace(output, attempts / f"{deck_id}-{arm}-{time.time_ns()}")
            processes.append((arm, _run_arm(deck_id, arm)))
        state["current_deck_id"] = deck_id
        state["current_arms"] = arms_to_run
        _atomic_json(STATE_PATH, state)
        for arm, process in processes:
            if process.wait() != 0:
                raise RuntimeError(f"{deck_id}:{arm} Benchmark V2 worker failed")
        _read_pair(deck_id)
        completed.append(deck_id)
        state["completed_deck_ids"] = completed
        state["current_deck_id"] = None
        state["current_arms"] = []
        _atomic_json(STATE_PATH, state)
        _render(partial=True)
    state.update({
        "status": "EVALUATION_COMPLETE", "current_deck_id": None,
        "current_arms": [], "finished_at": datetime.now(UTC).isoformat(),
    })
    _atomic_json(STATE_PATH, state)
    _render(partial=False)
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    lock_stream.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if not args.launch_formal:
        result = preflight(require_unused=False)
        if args.preflight_output:
            _atomic_json(args.preflight_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return _run_controller()


if __name__ == "__main__":
    raise SystemExit(main())
