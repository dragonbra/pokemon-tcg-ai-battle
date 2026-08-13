"""Reverse-order Full-67 extension of the U407 G2 candidate gate."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import g2_candidate_gate as gate


VERSION = "V8_u407_g2_candidate_full67_cuda2048"
VERSION_ROOT = gate.ROOT / "rl_runs/0043_champion_league_rl/versions" / VERSION
ARTIFACT_ROOT = VERSION_ROOT / "artifact"
REPORT_ROOT = ARTIFACT_ROOT / "reports"
STATE_PATH = ARTIFACT_ROOT / "state.json"
DECK_IDS = tuple(f"{value:03d}" for value in range(67, 0, -1))
MAX_PARALLEL_ARMS = 2


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _v7_report(deck_id: str, arm: str) -> Path:
    return gate.REPORT_ROOT / deck_id / arm / "report.json"


def source_report(deck_id: str, arm: str) -> Path | None:
    local = REPORT_ROOT / deck_id / arm / "report.json"
    if local.is_file():
        return local
    inherited = _v7_report(deck_id, arm)
    return inherited if inherited.is_file() else None


def wait_for_v7() -> None:
    while True:
        if gate.STATE_PATH.is_file():
            state = json.loads(gate.STATE_PATH.read_text(encoding="utf-8"))
            if state.get("status") == "EVALUATION_COMPLETE_REPORT_PENDING":
                subprocess.run(
                    [sys.executable, "-m", gate.__package__ + ".render_g2_candidate_gate"],
                    cwd=gate.ROOT, check=True,
                )
                return
        time.sleep(30)


def main() -> int:
    wait_for_v7()
    gate.validate_terminal_boundary()
    completed: list[str] = []
    inherited: list[str] = []
    for deck_id in DECK_IDS:
        for arm in gate.ARMS:
            report_path = source_report(deck_id, arm)
            if report_path is None:
                continue
            gate.validate_arm_report(
                json.loads(report_path.read_text(encoding="utf-8")),
                arm=arm, deck_id=deck_id,
            )
            key = f"{deck_id}:{arm}"
            completed.append(key)
            if report_path.is_relative_to(gate.REPORT_ROOT):
                inherited.append(key)
    state = {
        "schema_version": "0043_u407_g2_candidate_full67_state_v1",
        "status": "RUNNING", "version": VERSION,
        "execution_order": list(DECK_IDS), "max_parallel_arms": MAX_PARALLEL_ARMS,
        "completed_arms": completed, "inherited_v7_arms": inherited,
        "started_at": datetime.now(UTC).isoformat(),
        "human_promotion_decision": "REQUIRED",
    }
    _atomic_json(STATE_PATH, state)
    pending = [
        (deck_id, arm) for deck_id in DECK_IDS for arm in gate.ARMS
        if f"{deck_id}:{arm}" not in completed
    ]
    for offset in range(0, len(pending), MAX_PARALLEL_ARMS):
        batch = pending[offset:offset + MAX_PARALLEL_ARMS]
        processes = []
        for deck_id, arm in batch:
            output = REPORT_ROOT / deck_id / arm
            if output.exists():
                attempts = ARTIFACT_ROOT / "incomplete_attempts"
                attempts.mkdir(parents=True, exist_ok=True)
                os.replace(output, attempts / f"{deck_id}-{arm}-{time.time_ns()}")
            command = [
                sys.executable, "-m", gate.__package__ + ".g2_candidate_gate",
                "--run-arm", "--deck-id", deck_id, "--arm", arm,
                "--output-root", str(output),
            ]
            processes.append((deck_id, arm, output, subprocess.Popen(command, cwd=gate.ROOT)))
        state["current_arms"] = [f"{deck_id}:{arm}" for deck_id, arm in batch]
        _atomic_json(STATE_PATH, state)
        for deck_id, arm, output, process in processes:
            if process.wait():
                raise RuntimeError(f"{deck_id}:{arm} Full-67 worker failed")
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            gate.validate_arm_report(report, arm=arm, deck_id=deck_id)
            state["completed_arms"].append(f"{deck_id}:{arm}")
            _atomic_json(STATE_PATH, state)
    state.update({
        "status": "EVALUATION_COMPLETE_REPORT_PENDING",
        "current_arms": [], "finished_at": datetime.now(UTC).isoformat(),
    })
    _atomic_json(STATE_PATH, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
