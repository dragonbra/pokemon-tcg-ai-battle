"""Restart-safe Full-67 Benchmark V2: Champion-G2 versus G3 U110."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import fcntl
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import torch

from ..assets import sha256_file
from .run_benchmark_v2 import validate_report
from ..training.run_v1 import ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "V11_g2_vs_g3_u110_full67_benchmark_v2"
VERSION_ROOT = ROOT / "rl_runs/0044_g2_dragapult_policy_option_lora/versions" / VERSION
ARTIFACT_ROOT = VERSION_ROOT / "artifact"
REPORT_ROOT = ARTIFACT_ROOT / "reports"
STATE_PATH = ARTIFACT_ROOT / "state.json"
G2_VERSION = "V5_g2_uniform_0042_v1_lr_benchmark_v2"
G2_CHECKPOINT = ROOT / f"rl_runs/0044_g2_dragapult_policy_option_lora/versions/{G2_VERSION}/checkpoint/update-000000.pt"
G2_CHECKPOINT_SHA256 = "3ee103ed721a5f5771d0c07c13598bd19b1f4ead71765740442c85891600be06"
G3_VERSION = "V10_g3_v9_u50_generalist_001_067_core16_benchmark_v2"
G3_CHECKPOINT = ROOT / f"rl_runs/0044_g2_dragapult_policy_option_lora/versions/{G3_VERSION}/checkpoint/update-000110.pt"
G3_CHECKPOINT_SHA256 = "e1101f1712f2e4751aa56ba983b1a1165a9852b4cb1ab18913fbcd6108beac79"
ARMS = ("g2", "g3_u110")
ORIGINAL_PRIORITY_DECK_IDS = ("007", "003", "002")
EXPEDITED_DECK_IDS = ("066", "067")
PRIORITY_DECK_IDS = ORIGINAL_PRIORITY_DECK_IDS + EXPEDITED_DECK_IDS
REMAINDER_ORDER_SEED = 44_110_067
_remainder = [
    f"{value:03d}" for value in range(1, 68)
    if f"{value:03d}" not in ORIGINAL_PRIORITY_DECK_IDS
]
random.Random(REMAINDER_ORDER_SEED).shuffle(_remainder)
EXECUTION_DECK_IDS = (
    PRIORITY_DECK_IDS
    + tuple(deck_id for deck_id in _remainder if deck_id not in EXPEDITED_DECK_IDS)
)
CHECKPOINTS = {"g2": (G2_CHECKPOINT, 0), "g3_u110": (G3_CHECKPOINT, 110)}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def preflight(*, require_unused: bool = False) -> dict[str, Any]:
    if len(EXECUTION_DECK_IDS) != 67 or set(EXECUTION_DECK_IDS) != {
        f"{value:03d}" for value in range(1, 68)
    } or EXECUTION_DECK_IDS[:len(PRIORITY_DECK_IDS)] != PRIORITY_DECK_IDS:
        raise RuntimeError("Full-67 execution order contract changed")
    if require_unused and VERSION_ROOT.exists() and any(VERSION_ROOT.rglob("*")):
        raise FileExistsError(f"formal V11 path is already used: {VERSION_ROOT}")
    rows = []
    for arm, checkpoint, expected_hash, update, version in (
        ("g2", G2_CHECKPOINT, G2_CHECKPOINT_SHA256, 0, G2_VERSION),
        ("g3_u110", G3_CHECKPOINT, G3_CHECKPOINT_SHA256, 110, G3_VERSION),
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
        rows.append({"arm": arm, "update": update, "version": version, "path": str(checkpoint), "sha256": expected_hash})
    if (G3_CHECKPOINT.parent / "update-000111.pt").exists():
        raise RuntimeError("G3 training escaped the exact U110 terminal boundary")
    return {
        "schema_version": "0044_g2_vs_g3_u110_full67_preflight_v1",
        "status": "PASS", "version": VERSION,
        "priority_deck_ids": list(PRIORITY_DECK_IDS),
        "expedited_deck_ids": list(EXPEDITED_DECK_IDS),
        "remainder_order_seed": REMAINDER_ORDER_SEED,
        "execution_deck_ids": list(EXECUTION_DECK_IDS),
        "arms": rows, "games_per_arm_per_deck": 2048,
        "total_games": 67 * 2 * 2048,
    }


def validate_arm_report(report: dict[str, Any], *, arm: str, deck_id: str) -> None:
    validate_report(report)
    checkpoint, update = CHECKPOINTS[arm]
    audit = report["focal_policy_identity_audit"]
    if (
        report.get("focal_deck_id") != deck_id
        or report.get("focal_checkpoint_update") != update
        or audit.get("source_checkpoint_sha256") != sha256_file(checkpoint)
        or report.get("opponent_policy_identity_audit", {}).get("requested_policy_id") != "Policy-0809"
    ):
        raise RuntimeError(f"{deck_id}:{arm} Benchmark V2 identity mismatch")


def validate_pair(g2: dict[str, Any], g3: dict[str, Any], *, deck_id: str) -> None:
    validate_arm_report(g2, arm="g2", deck_id=deck_id)
    validate_arm_report(g3, arm="g3_u110", deck_id=deck_id)
    if (
        g2["focal_exact_deck_sha256"] != g3["focal_exact_deck_sha256"]
        or g2["opponent_policy_identity_audit"]["effective_policy_sha256"]
        != g3["opponent_policy_identity_audit"]["effective_policy_sha256"]
        or g2["schedule"]["common_random_schedule_sha256"]
        != g3["schedule"]["common_random_schedule_sha256"]
    ):
        raise RuntimeError(f"{deck_id}: paired identity mismatch")
    fields = (
        "game_id", "opponent_id", "opponent_meta_archetype_id",
        "engine_seed", "search_seed", "policy_seed", "coin_winner_seed",
        "focal_won_toss",
    )
    for left, right in zip(g2["entries"], g3["entries"], strict=True):
        if tuple(left[key] for key in fields) != tuple(right[key] for key in fields):
            raise RuntimeError(f"{deck_id}: paired job identity mismatch")


def _report(deck_id: str, arm: str) -> Path:
    return REPORT_ROOT / deck_id / arm / "report.json"


def _validated_completed() -> list[str]:
    completed = []
    for deck_id in EXECUTION_DECK_IDS:
        paths = [_report(deck_id, arm) for arm in ARMS]
        if not all(path.is_file() for path in paths):
            continue
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        validate_pair(reports[0], reports[1], deck_id=deck_id)
        completed.append(deck_id)
    return completed


def _run_arm(deck_id: str, arm: str, output: Path) -> subprocess.Popen[Any]:
    checkpoint, update = CHECKPOINTS[arm]
    command = [
        sys.executable, "-m",
        "train.0044_g2_dragapult_policy_option_lora.evaluation.run_benchmark_v2",
        "--deck-id", deck_id, "--output-root", str(output),
        "--checkpoint", str(checkpoint), "--checkpoint-update", str(update),
    ]
    return subprocess.Popen(command, cwd=ROOT)


def _run_controller() -> int:
    before = not STATE_PATH.exists()
    readiness = preflight(require_unused=before)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (ARTIFACT_ROOT / "controller.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another Full-67 Benchmark V2 controller is already active") from exc
    completed = _validated_completed()
    state = {
        **readiness, "schema_version": "0044_g2_vs_g3_u110_full67_state_v1",
        "status": "RUNNING", "completed_deck_ids": completed,
        "current_deck_id": None,
        "started_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(STATE_PATH, state)
    for deck_id in EXECUTION_DECK_IDS:
        if deck_id in completed:
            continue
        processes = []
        for arm in ARMS:
            output = REPORT_ROOT / deck_id / arm
            if output.exists():
                attempts = ARTIFACT_ROOT / "incomplete_attempts"
                attempts.mkdir(parents=True, exist_ok=True)
                os.replace(output, attempts / f"{deck_id}-{arm}-{time.time_ns()}")
            processes.append((arm, output, _run_arm(deck_id, arm, output)))
        state["current_deck_id"] = deck_id
        state["current_arms"] = list(ARMS)
        _atomic_json(STATE_PATH, state)
        for arm, output, process in processes:
            if process.wait() != 0:
                raise RuntimeError(f"{deck_id}:{arm} Benchmark V2 worker failed")
        reports = [json.loads(_report(deck_id, arm).read_text(encoding="utf-8")) for arm in ARMS]
        validate_pair(reports[0], reports[1], deck_id=deck_id)
        completed.append(deck_id)
        state["completed_deck_ids"] = completed
        state["current_deck_id"] = None
        state["current_arms"] = []
        _atomic_json(STATE_PATH, state)
        subprocess.run([
            sys.executable, "-m",
            "train.0044_g2_dragapult_policy_option_lora.evaluation.render_benchmark_v2_g2_vs_g3_u110_full67",
            "--allow-partial",
        ], cwd=ROOT, check=True)
    state.update({
        "status": "EVALUATION_COMPLETE", "current_deck_id": None,
        "current_arms": [], "finished_at": datetime.now(UTC).isoformat(),
    })
    _atomic_json(STATE_PATH, state)
    subprocess.run([
        sys.executable, "-m",
        "train.0044_g2_dragapult_policy_option_lora.evaluation.render_benchmark_v2_g2_vs_g3_u110_full67",
    ], cwd=ROOT, check=True)
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
