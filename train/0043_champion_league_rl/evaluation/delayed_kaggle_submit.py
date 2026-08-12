"""One-shot delayed deck-007 selection, package export, and Kaggle submission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import wandb

from ..assets import sha256_file
from .export_kaggle import export
from .select_rollout_checkpoint import select


ROOT = Path(__file__).resolve().parents[3]
COMPETITION = "pokemon-tcg-ai-battle"
RUN_PATH = "dragon_bra/pokemon-tcg-policy-learning/0043-v2-mixed-focal-cohorts"
VERSION_ROOT = ROOT / "rl_runs/0043_champion_league_rl/versions/V2_mixed_focal_cohorts"
METRICS = VERSION_ROOT / "artifact/training_metrics.jsonl"
CHECKPOINTS = VERSION_ROOT / "checkpoint"
SKILL_QUOTA = Path("/home/cyd/.codex/skills/nvidia-kaggle-skill/scripts/submission_quota.py")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=True)


def _submissions() -> list[dict[str, str]]:
    output = _command([
        "kaggle", "competitions", "submissions", "-c", COMPETITION, "-v"
    ]).stdout
    import csv
    from io import StringIO
    return list(csv.DictReader(StringIO(output)))


def _wandb_complete_updates() -> set[int]:
    run = wandb.Api(timeout=60).run(RUN_PATH)
    rows = run.scan_history(
        keys=["trainer/update", "rollout/source_policy_update", "rollout/focal_deck/007/win_rate"],
        page_size=1000,
    )
    return {
        int(row["rollout/source_policy_update"])
        for row in rows
        if isinstance(row.get("rollout/source_policy_update"), (int, float))
        and isinstance(row.get("rollout/focal_deck/007/win_rate"), (int, float))
    }


def run(*, output_root: Path, scheduled_epoch: float) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "single_submission.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = output_root / "submission_receipt.json"
        if receipt_path.exists():
            raise FileExistsError("one-shot Kaggle receipt already exists")
        if time.time() < scheduled_epoch:
            time.sleep(scheduled_epoch - time.time())
        started = datetime.now(timezone.utc).isoformat()
        # No process signal, CUDA command, or training mutation exists below.
        wandb_updates = _wandb_complete_updates()
        selection = select(
            METRICS, CHECKPOINTS, allowed_source_updates=wandb_updates
        )
        selected = selection["selected"]
        update = int(selected["source_policy_update"])
        if update not in wandb_updates:
            raise RuntimeError("selected canonical metric row is not mirrored to W&B")
        selection["wandb_run"] = RUN_PATH
        selection["wandb_mirror_verified"] = True
        selection_path = output_root / "selection.json"
        _atomic_json(selection_path, selection)
        name = f"0043_dragapult_ex_007_v2_u{update:03d}_fp16_storage_fp32_runtime"
        package = ROOT / "archive/submission" / name
        archive = ROOT / "archive/submission/dist" / f"{name}.tar.gz"
        manifest = export(
            checkpoint=Path(selected["checkpoint"]), deck_id="007",
            output=package, archive=archive, selection=selection,
        )
        _atomic_json(output_root / "package_manifest.json", manifest)
        quota = json.loads(_command([
            "python", str(SKILL_QUOTA), COMPETITION, "--by-user", "--as-json"
        ]).stdout)
        _atomic_json(output_root / "quota_before_submit.json", quota)
        if quota.get("exhausted") or not isinstance(quota.get("remaining"), int) or quota["remaining"] < 1:
            raise RuntimeError(f"Kaggle submission quota unavailable: {quota}")
        before = _submissions()
        before_refs = {row["ref"] for row in before}
        message = f"0043 V2 U{update} · 007 Dragapult · rollout-selected"
        attempt = {
            "authorized_attempts": 1, "attempt_number": 1,
            "competition": COMPETITION, "archive": str(archive),
            "archive_sha256": sha256_file(archive), "message": message,
            "started_at": started,
        }
        _atomic_json(output_root / "submission_attempt.json", attempt)
        # Exactly one externally visible submit command. Never wrap this in a retry.
        submitted = subprocess.run([
            "kaggle", "competitions", "submit", "-c", COMPETITION,
            "-f", str(archive), "-m", message,
        ], text=True, capture_output=True)
        attempt.update({
            "returncode": submitted.returncode,
            "stdout": submitted.stdout,
            "stderr": submitted.stderr,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        _atomic_json(output_root / "submission_attempt.json", attempt)
        # Even on ambiguous CLI output, inspect existing submissions only; never resubmit.
        found = None
        deadline = time.monotonic() + 90 * 60
        while time.monotonic() < deadline:
            try:
                candidates = [row for row in _submissions() if row["ref"] not in before_refs]
                exact = [row for row in candidates if row.get("fileName") == archive.name]
                if exact:
                    found = exact[0]
                    if "COMPLETE" in found.get("status", "") or "ERROR" in found.get("status", ""):
                        break
            except Exception as error:
                attempt["poll_error"] = repr(error)
                _atomic_json(output_root / "submission_attempt.json", attempt)
            time.sleep(30)
        receipt = {
            "schema_version": "0043_deck007_single_kaggle_submission_receipt_v1",
            "competition": COMPETITION,
            "submit_commands_issued": 1,
            "automatic_retry": False,
            "selected_checkpoint_update": update,
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "portable_checkpoint_sha256": manifest["portable_checkpoint_sha256"],
            "deployment_effective_sha256": manifest["deployment_effective_sha256"],
            "archive": str(archive),
            "archive_sha256": manifest["archive_sha256"],
            "kaggle": found,
            "cli_returncode": submitted.returncode,
            "status": (
                "SUBMISSION_OBSERVED" if found is not None
                else "UNCERTAIN_NO_RETRY"
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(receipt_path, receipt)
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scheduled-epoch", type=float, required=True)
    args = parser.parse_args()
    result = run(output_root=args.output_root.resolve(), scheduled_epoch=args.scheduled_epoch)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
