"""Recovered serialized V14-U50 → V16–V17 focal-deck PPO curriculum."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ..policy.own_archetype import OwnArchetypeVocabulary
from ..rollout.deck_routing import exact_deck_sha256


ROOT = Path(__file__).resolve().parents[3]
VERSIONS = ROOT / "rl_runs/0042_full_model_design/versions"
STATE = ROOT / ".tmp/training_monitor/0042_full_model_design/deck_curriculum_recovery_v16_v17/state.json"
OPPONENT_EFFECTIVE = "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"
SEGMENTS = (
    {
        "version": "V16_festival_lead_dipplin_010_numeric_gate_removed",
        "deck": "festival_lead_dipplin_010",
        "class_id": 6,
    },
    {
        "version": "V17_mega_kangaskhan_ex_crustle_011",
        "deck": "mega_kangaskhan_ex_crustle_011",
        "class_id": 5,
    },
)
INITIAL_PREDECESSOR = "V14_mega_lucario_ex_solrock_009"
LEGACY_STOPPED_PREDECESSOR = "V11_marnies_grimmsnarl_ex_froslass_001"


def _atomic_state(payload: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _pid_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def validate_u50(version: str) -> Path:
    root = VERSIONS / version
    status_path = root / "artifact/status.json"
    summary_path = root / "artifact/training_summary.json"
    checkpoint = root / "checkpoint/update-000050.pt"
    sidecar = checkpoint.with_suffix(".pt.sha256")
    frozen_path = root / "artifact/frozen_results/core-update-000050.json"
    if not all(path.is_file() for path in (status_path, summary_path, checkpoint, sidecar, frozen_path)):
        raise FileNotFoundError(f"{version} U50 terminal evidence is incomplete")
    status, summary, frozen = _json(status_path), _json(summary_path), _json(frozen_path)
    allowed_state = (
        "stopped_by_request" if version == LEGACY_STOPPED_PREDECESSOR else "complete"
    )
    if (
        status.get("state") != allowed_state
        or summary.get("state") != allowed_state
        or int(status.get("checkpoint_update", -1)) != 50
        or int(summary.get("updates", -1)) != 50
        or status.get("error") is not None
    ):
        raise RuntimeError(f"{version} did not terminate cleanly at U50")
    expected_sha = sidecar.read_text().strip()
    if len(expected_sha) != 64 or _sha256(checkpoint) != expected_sha:
        raise RuntimeError(f"{version} U50 checkpoint sidecar mismatch")
    entries = frozen.get("entries")
    candidate = frozen.get("candidate_deployment_identity_audit") or {}
    opponent = frozen.get("policy_identity_audit") or {}
    if (
        frozen.get("benchmark_kind") != "cuda_2048"
        or frozen.get("expected_games") != 2048
        or frozen.get("checkpoint_update") != 50
        or not isinstance(entries, list)
        or len(entries) != 2048
        or len({row.get("game_id") for row in entries}) != 2048
        or len({row.get("seed") for row in entries}) != 2048
        or any(
            row.get("valid") is not True
            or row.get("error") is not None
            or row.get("semantic_fallback") is not False
            for row in entries
        )
        or candidate.get("status") != "PASS"
        or candidate.get("checkpoint_update") != 50
        or candidate.get("source_checkpoint_sha256") != expected_sha
        or candidate.get("storage_dtype") != "fp16"
        or candidate.get("runtime_dtype") != "fp32"
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != "Policy-0809"
        or opponent.get("effective_policy_sha256") != OPPONENT_EFFECTIVE
    ):
        raise RuntimeError(f"{version} U50 Frozen-0809 CUDA-2048 evidence failed")
    return checkpoint


def validate_deck(segment: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = ROOT / "train/0042_full_model_design/focal_decks" / str(segment["deck"])
    deck_path, manifest_path = root / "deck.csv", root / "manifest.json"
    manifest = _json(manifest_path)
    cards = tuple(int(line) for line in deck_path.read_text().splitlines())
    vocabulary = OwnArchetypeVocabulary.load()
    own = vocabulary.classify_own_deck(cards)
    if (
        len(cards) != 60
        or exact_deck_sha256(cards) != manifest.get("exact_deck_sha256")
        or own.value != segment["class_id"]
        or manifest.get("own_archetype_id") != own.value
        or manifest.get("own_archetype_name") != vocabulary.classes[own.value].name
    ):
        raise RuntimeError(f"focal deck/archetype identity mismatch: {root}")
    source = ROOT / manifest["provenance"]["source_deck"]
    if deck_path.read_bytes() != source.read_bytes():
        raise RuntimeError(f"focal deck differs from immutable catalog source: {root}")
    return deck_path, manifest


def training_command(segment: dict[str, Any], checkpoint: Path) -> list[str]:
    deck_path, manifest = validate_deck(segment)
    version = str(segment["version"])
    runner = [
        sys.executable, "-m", "train.0042_full_model_design.training.run_full_semantic",
        "--version", version, "--updates", "50",
        "--worker-processes", "16", "--engines-per-worker", "8",
        "--inference-channels-per-role", "8", "--coalesce-ms", "5.0",
        "--games-per-update", "256", "--engine-backend", "accelerated:cuda_resident",
        "--cuda-lane-count", "256", "--rollout-batch-size", "256",
        "--trajectory-games-per-update", "256", "--ppo-minibatch-size", "2048",
        "--ppo-forward-microbatch-size", "1024", "--decoder-lr", "2e-5",
        "--policy-adapter-lr", "4e-5", "--allocation-lr", "2e-5",
        "--ppo-gradient-accumulation", "1", "--ppo-epochs", "3",
        "--eval-every", "10", "--adaptation-arm", "strategy", "--preset", "FULL_MODEL",
        "--gae-lambda", "0.95", "--credit-clock", "turn",
        "--loss-weighting", "episode_equal_decisions", "--wandb-mode", "online",
        "--launch-formal", "--initial-model-checkpoint", str(checkpoint.relative_to(ROOT)),
        "--allow-fp16-deployment-numeric-drift",
        "--focal-deck-path", str(deck_path.relative_to(ROOT)),
        "--focal-deck-id", str(manifest["deck_id"]),
        "--focal-exact-deck-sha256", str(manifest["exact_deck_sha256"]),
        "--focal-deck-display-name", str(manifest["display_name"]),
        "--focal-deck-source", str(manifest["source"]),
    ]
    return [
        sys.executable, "-m", "train.0042_full_model_design.monitor_training",
        "--version", version, "--interval-seconds", "30",
        "--first-metrics-grace-seconds", "600", "--metrics-stale-seconds", "900",
        "--minimum-free-gib", "10", "--", *runner,
    ]


def wait_for_predecessor(version: str) -> Path:
    while True:
        try:
            checkpoint = validate_u50(version)
        except (FileNotFoundError, json.JSONDecodeError):
            _atomic_state({"state": "waiting_for_u50", "predecessor": version, "checked_at": time.time()})
            time.sleep(15)
            continue
        status = _json(VERSIONS / version / "artifact/status.json")
        if _pid_alive(status.get("pid")):
            _atomic_state({"state": "waiting_for_process_exit", "predecessor": version, "checked_at": time.time()})
            time.sleep(5)
            continue
        return checkpoint


def main() -> int:
    predecessor = INITIAL_PREDECESSOR
    try:
        for segment in SEGMENTS:
            checkpoint = wait_for_predecessor(predecessor)
            version = str(segment["version"])
            target = VERSIONS / version
            if target.exists():
                raise FileExistsError(f"target version already exists: {target}")
            command = training_command(segment, checkpoint)
            _atomic_state({
                "state": "launching", "predecessor": predecessor, "version": version,
                "source_checkpoint": str(checkpoint.relative_to(ROOT)),
                "own_archetype_id": segment["class_id"], "started_at": time.time(),
            })
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"{version} monitor exited {completed.returncode}")
            validate_u50(version)
            predecessor = version
        _atomic_state({"state": "complete", "final_version": predecessor, "finished_at": time.time()})
        return 0
    except Exception as error:
        _atomic_state({
            "state": "blocked", "predecessor": predecessor,
            "error": f"{type(error).__name__}: {error}", "failed_at": time.time(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
