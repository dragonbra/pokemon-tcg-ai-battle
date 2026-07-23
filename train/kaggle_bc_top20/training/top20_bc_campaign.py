"""Run the frozen Top-20 single-expert BC campaign as resumable transactions."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
from html import escape
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from kaggle.api.kaggle_api_extended import KaggleApi

from train.kaggle_bc_top20.training.download_expert_replays import (
    _deck_from_replay,
    _download_one,
    _wait_for_request_slot,
    canonical_deck_sha256,
    eligible_episode_rows,
)
from train.kaggle_bc_top20.training.download_top100_exact_replays import (
    _model_value,
    select_leaderboard_submission,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPOSITORY_ROOT / "rl_runs"
ROSTER_CSV = REPOSITORY_ROOT / "docs/reports/rl/top100_static_rank_index-20260723.csv"
RESEARCH_ROOT = REPOSITORY_ROOT / "rl_runs/dataset/top100_research_20260723"
SHARED_CONFIG = REPOSITORY_ROOT / "train/alakazam_bc_rl/shared_model_config.json"
CAMPAIGN_STATE = RUNS_ROOT / "top20_bc_campaign_20260723.json"
RAW_ROOT = REPOSITORY_ROOT / "data/replays/top20_single_expert_bc_20260723"
REPORT_PATH = (
    REPOSITORY_ROOT / "docs/reports/rl/top20_bc_training_evaluation-20260723.html"
)
CG_SOURCE = REPOSITORY_ROOT / "evaluation/opponents/romanrozen_v9/cg"
OBJECTIVE = "Train one exact Top-100 expert BC surrogate"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def _source_staging(static_rank: int) -> Path:
    matches = sorted(RESEARCH_ROOT.glob(f"{static_rank:04d}-rank-{static_rank:02d}-*"))
    if len(matches) != 1:
        raise ValueError(f"static rank {static_rank} has {len(matches)} source staging dirs")
    return matches[0]


def load_roster() -> list[dict[str, Any]]:
    with ROSTER_CSV.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["selected_roster_order"]]
    rows.sort(key=lambda row: int(row["selected_roster_order"]))
    if len(rows) != 20:
        raise ValueError(f"static roster has {len(rows)} selected sources instead of 20")
    if rows[0]["team_name"] != "LumenLiquidity":
        raise ValueError("LumenLiquidity is not selected_roster_order=1")
    if any(
        row["team_name"] != "LumenLiquidity"
        and row["meets_200_episode_floor"] != "True"
        for row in rows
    ):
        raise ValueError("ordinary roster source violates the frozen 200-Episode floor")
    for row in rows:
        row["static_rank"] = int(row["static_rank"])
        row["selected_roster_order"] = int(row["selected_roster_order"])
        row["team_id"] = int(row["team_id"])
        row["submission_id"] = int(row["submission_id"])
        row["episodes"] = int(row["episodes"])
        row["selection_score"] = float(row["selection_score_65_lcb_35_volume"])
        row["source_staging"] = str(_source_staging(row["static_rank"]))
    return rows


def _new_experiment(package_name: str) -> Path:
    command = [
        sys.executable,
        "-m",
        "rl_environment.runs",
        "create",
        package_name,
        "--objective",
        OBJECTIVE,
    ]
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    path = Path(result.stdout.strip()).resolve()
    if path.parent != RUNS_ROOT.resolve() or not path.is_dir():
        raise RuntimeError(f"run allocator returned an invalid path: {result.stdout!r}")
    return path


def _append_command(run_root: Path, command: Iterable[str], note: str = "") -> None:
    command_text = " ".join(str(value) for value in command)
    with (run_root / "commands.md").open("a", encoding="utf-8") as handle:
        handle.write(f"- `{command_text}`")
        if note:
            handle.write(f" — {note}")
        handle.write("\n")


def _run_command(
    run_root: Path,
    command: list[str],
    log_name: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    _append_command(run_root, command)
    log_path = run_root / log_name
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        f"command: {' '.join(command)}\n"
        f"exit_code: {result.returncode}\n"
        f"wall_seconds: {time.monotonic() - started:.3f}\n\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n",
        encoding="utf-8",
    )
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")
    return result


def _campaign() -> dict[str, Any]:
    if CAMPAIGN_STATE.is_file():
        return _read_json(CAMPAIGN_STATE)
    shared = _read_json(SHARED_CONFIG)
    campaign = {
        "schema_version": "ptcg_top20_single_expert_bc_campaign_v1",
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "status": "initialized",
        "roster_csv": _relative(ROSTER_CSV),
        "roster_csv_sha256": _sha256(ROSTER_CSV),
        "shared_model_config": _relative(SHARED_CONFIG),
        "shared_model_config_sha256": _sha256(SHARED_CONFIG),
        "sota_trial_id": shared["sota_trial_id"],
        "sources": [],
    }
    _write_json(CAMPAIGN_STATE, campaign)
    return campaign


def _save_campaign(campaign: dict[str, Any]) -> None:
    campaign["updated_at"] = _timestamp()
    _write_json(CAMPAIGN_STATE, campaign)


def _current_leaderboard(api: KaggleApi) -> dict[int, Any]:
    _wait_for_request_slot(1.0)
    rows = api.competition_leaderboard_view(
        "pokemon-tcg-ai-battle", page_size=100
    ) or []
    return {
        int(_model_value(row, "team_id", "teamId")): row
        for row in rows
    }


def _resolve_current_submission(api: KaggleApi, row: dict[str, Any], leaderboard: dict[int, Any]) -> int:
    leaderboard_row = leaderboard.get(int(row["team_id"]))
    if leaderboard_row is None:
        return int(row["submission_id"])
    _wait_for_request_slot(1.0)
    submissions = list(api.competition_team_submissions(int(row["team_id"])) or [])
    selected, _ = select_leaderboard_submission(leaderboard_row, submissions)
    return int(_model_value(selected, "id"))


def _current_deck_hash(
    api: KaggleApi,
    submission_id: int,
    probe_root: Path,
) -> str | None:
    _wait_for_request_slot(1.0)
    rows = eligible_episode_rows(
        list(api.competition_list_episodes(submission_id) or []), submission_id
    )
    if not rows:
        return None
    episode = rows[-1]
    episode_id = int(episode["episode_id"])
    _download_one(episode_id, probe_root, 12, 1.0, 60.0)
    payload = _read_json(probe_root / f"episode-{episode_id}-replay.json")
    deck = _deck_from_replay(payload, int(episode["player_index"]))
    return canonical_deck_sha256(deck)


def refresh_sources() -> dict[str, Any]:
    campaign = _campaign()
    roster = load_roster()
    by_package = {row["package_name"]: row for row in campaign.get("sources") or []}
    api = KaggleApi()
    api.authenticate()
    leaderboard = _current_leaderboard(api)
    probe_root = Path(tempfile.mkdtemp(prefix="top20-current-deck-probes-"))
    try:
        for roster_row in roster:
            package_name = str(roster_row["package_name"])
            entry = by_package.get(package_name)
            if entry is None:
                run_root = _new_experiment(package_name)
                entry = {
                    "package_name": package_name,
                    "experiment_id": run_root.name,
                    "run_root": _relative(run_root),
                    "selected_roster_order": roster_row["selected_roster_order"],
                    "static_rank": roster_row["static_rank"],
                    "team_name": roster_row["team_name"],
                    "team_id": roster_row["team_id"],
                    "original_submission_id": roster_row["submission_id"],
                    "selection_score": roster_row["selection_score"],
                    "status": "allocated",
                }
                campaign["sources"].append(entry)
                by_package[package_name] = entry
                _save_campaign(campaign)
            run_root = REPOSITORY_ROOT / entry["run_root"]
            source_path = Path(str(roster_row["source_staging"])) / "source_manifest.json"
            sampled = _read_json(source_path)
            deck_source = source_path.parent / str(sampled["deck_file"])
            deck = [
                int(line.strip())
                for line in deck_source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            deck_hash = canonical_deck_sha256(deck)
            if deck_hash != roster_row["deck_sha256"]:
                raise ValueError(f"static roster deck hash changed for {package_name}")

            _wait_for_request_slot(1.0)
            episode_rows = eligible_episode_rows(
                list(api.competition_list_episodes(roster_row["submission_id"]) or []),
                roster_row["submission_id"],
            )
            if roster_row["team_name"] != "LumenLiquidity" and len(episode_rows) < 200:
                raise RuntimeError(
                    f"ordinary source {package_name} now has only {len(episode_rows)} Episodes"
                )
            current_submission_id = _resolve_current_submission(api, roster_row, leaderboard)
            current_hash = deck_hash
            if current_submission_id != roster_row["submission_id"]:
                current_hash = _current_deck_hash(api, current_submission_id, probe_root)

            original_metadata = _read_json(Path(sampled["metadata_file"]))
            original_ids = {
                int(item["episode_id"]) for item in original_metadata.get("episodes") or []
            }
            refreshed_ids = {int(item["episode_id"]) for item in episode_rows}
            deck_destination = run_root / "source_deck.csv"
            if not deck_destination.exists():
                shutil.copy2(deck_source, deck_destination)
            exact_source = {
                "schema_version": "ptcg_single_expert_bc_source_v2",
                "status": "refreshed_exact_train_source",
                "single_policy_constraint": True,
                "source_id": sampled["source_id"],
                "package_name": package_name,
                "selected_roster_order": roster_row["selected_roster_order"],
                "static_rank": roster_row["static_rank"],
                "source_policy": sampled["source_policy"],
                "source_identity": {
                    "team_id": roster_row["team_id"],
                    "submission_id": roster_row["submission_id"],
                    "deck_sha256": deck_hash,
                },
                "deck_file": deck_destination.name,
                "deck_profile": sampled["deck_profile"],
                "episode_contract": "PUBLIC + COMPLETED + exact submission_id player index",
                "episodes": episode_rows,
                "refreshed_at": _timestamp(),
            }
            source_destination = run_root / "source_manifest.json"
            if source_destination.exists():
                existing = _read_json(source_destination)
                if existing.get("source_identity") != exact_source["source_identity"]:
                    raise ValueError(f"frozen source identity changed in {source_destination}")
            else:
                _write_json(source_destination, exact_source)
                source_destination.chmod(0o444)
            refresh = {
                "schema_version": "ptcg_pretrain_refresh_v1",
                "refreshed_at": _timestamp(),
                "original_submission_id": roster_row["submission_id"],
                "current_submission_id": current_submission_id,
                "original_episode_count": roster_row["episodes"],
                "refreshed_episode_count": len(episode_rows),
                "added_episode_ids": sorted(refreshed_ids - original_ids),
                "removed_from_current_api_window_episode_ids": sorted(original_ids - refreshed_ids),
                "original_deck_sha256": deck_hash,
                "current_deck_sha256": current_hash,
                "submission_changed": current_submission_id != roster_row["submission_id"],
                "deck_changed": current_hash is not None and current_hash != deck_hash,
                "source_choice": "continue frozen original exact submission",
                "final_source_identity": exact_source["source_identity"],
            }
            _write_json(run_root / "pretrain_refresh.json", refresh)
            _append_command(
                run_root,
                ["kaggle-api", "list-episodes", str(roster_row["submission_id"])],
                "metadata refresh; PUBLIC/COMPLETED filter and exact submission_id lookup",
            )
            entry.update(
                {
                    "status": "refreshed",
                    "source_identity": exact_source["source_identity"],
                    "refreshed_episode_count": len(episode_rows),
                    "current_submission_id": current_submission_id,
                    "submission_changed": refresh["submission_changed"],
                    "deck_changed": refresh["deck_changed"],
                }
            )
            entry.pop("failure", None)
            _save_campaign(campaign)
            print(
                f"REFRESH {roster_row['selected_roster_order']:02d}/20 "
                f"{package_name}: {len(episode_rows)} episodes "
                f"(+{len(refresh['added_episode_ids'])})",
                flush=True,
            )
    finally:
        shutil.rmtree(probe_root)
    campaign["status"] = "sources_refreshed"
    campaign["sources"].sort(key=lambda row: int(row["selected_roster_order"]))
    _save_campaign(campaign)
    return campaign


def _attempt_gate(run_root: Path, attempt_id: str) -> dict[str, Any]:
    summary_path = run_root / attempt_id / "training_summary.json"
    summary = _read_json(summary_path)
    best = summary.get("best_epoch_metrics") or {}
    test = summary.get("final_test") or {}
    test_evaluated = bool(test)
    metrics = {
        "validation_exact_action_rate": float(
            best.get("validation/exact_action_rate", summary["best_validation_exact_action_rate"])
        ),
        "validation_multi_action_exact_rate": float(
            best.get("validation/multi_action_exact_rate", 0.0)
        ),
        "validation_selection_count_accuracy": float(
            best.get("validation/selection_count_accuracy", 0.0)
        ),
        "validation_legal_action_rate": float(
            best.get("validation/legal_action_rate", 0.0)
        ),
        "validation_policy_loss": float(best.get("validation/policy_loss", float("inf"))),
        "train_exact_action_rate": float(best.get("train/exact_action_rate", 0.0)),
        "test_exact_action_rate": float(test.get("exact_action_rate", 0.0)),
        "test_multi_action_exact_rate": float(test.get("multi_action_exact_rate", 0.0)),
        "test_selection_count_accuracy": float(test.get("selection_count_accuracy", 0.0)),
        "test_legal_action_rate": (
            float(test["legal_action_rate"]) if "legal_action_rate" in test else None
        ),
    }
    validation_checks = {
        "validation_exact_action_rate": metrics["validation_exact_action_rate"] >= 0.75,
        "validation_multi_action_exact_rate": (
            metrics["validation_multi_action_exact_rate"] >= 0.65
        ),
        "validation_selection_count_accuracy": (
            metrics["validation_selection_count_accuracy"] >= 0.98
        ),
        "validation_legal_action_rate": metrics["validation_legal_action_rate"] == 1.0,
    }
    checks = {
        **validation_checks,
        "test_legal_action_rate": (
            metrics["test_legal_action_rate"] == 1.0 if test_evaluated else None
        ),
    }
    checkpoint = Path(summary["checkpoint"])
    gate = {
        "schema_version": "ptcg_single_expert_bc_gate_v2",
        "attempt_id": attempt_id,
        "decided_at": _timestamp(),
        "selection_inputs": ["train", "validation"],
        "test_used_for_lr_selection": False,
        "metrics": metrics,
        "checks": checks,
        "validation_gate_passed": all(validation_checks.values()),
        "test_evaluated": test_evaluated,
        "offline_gate_passed": test_evaluated and all(checks.values()),
        "best_epoch": int(summary["best_epoch"]),
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "wall_seconds": float(summary["runtime"]["train_function_seconds"]),
        "peak_gpu_memory_bytes": int(summary["runtime"]["peak_gpu_memory_bytes"]),
        "parameter_count": int(summary["runtime"]["parameter_count"]),
    }
    _write_json(run_root / attempt_id / "gate_decision.json", gate)
    return gate


def _train_attempt(
    run_root: Path,
    dataset: Path,
    attempt_id: str,
    learning_rate: float,
) -> dict[str, Any]:
    attempt_root = run_root / attempt_id
    if (attempt_root / "gate_decision.json").is_file():
        # Recompute from the immutable summary so pre-isolation V1 gates are
        # migrated to the explicit validation/test protocol.
        return _attempt_gate(run_root, attempt_id)
    shared = _read_json(SHARED_CONFIG)
    command = [
        sys.executable,
        "-m",
        "train.alakazam_bc_rl.training.train_full_action_bc",
        str(dataset),
        "--output",
        str(attempt_root),
        "--epochs",
        str(shared["epochs"]),
        "--batch-size",
        str(shared["batch_size"]),
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(shared["seed"]),
        "--d-model",
        str(shared["d_model"]),
        "--hidden-dim",
        str(shared["hidden_dim"]),
        "--num-heads",
        str(shared["num_heads"]),
        "--transformer-layers",
        str(shared["transformer_layers"]),
        "--dropout",
        str(shared["dropout"]),
        "--device",
        "cuda",
        "--skip-test",
    ]
    print(f"TRAIN START {run_root.name}/{attempt_id} lr={learning_rate:g}", flush=True)
    _run_command(run_root, command, f"{attempt_id}_train.log")
    gate = _attempt_gate(run_root, attempt_id)
    print(
        f"TRAIN DONE {run_root.name}/{attempt_id}: "
        f"val={gate['metrics']['validation_exact_action_rate']:.4f} "
        f"validation_gate={gate['validation_gate_passed']}",
        flush=True,
    )
    return gate


def _evaluate_frozen_test(
    run_root: Path,
    dataset: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    attempt_id = str(selection["selected_attempt_id"])
    attempt_root = run_root / attempt_id
    summary_path = attempt_root / "training_summary.json"
    summary = _read_json(summary_path)
    if summary.get("final_test"):
        gate = _attempt_gate(run_root, attempt_id)
        test_audit = {
            "evaluated_at": _timestamp(),
            "attempt_id": attempt_id,
            "checkpoint": selection["selected_checkpoint"],
            "checkpoint_sha256": selection["selected_checkpoint_sha256"],
            "test_reused_from_pre_isolation_attempt": True,
            "metrics": summary["final_test"],
        }
        _write_json(run_root / "frozen_test_evaluation.json", test_audit)
        return gate
    shared = _read_json(SHARED_CONFIG)
    checkpoint = REPOSITORY_ROOT / str(selection["selected_checkpoint"])
    command = [
        sys.executable,
        "-m",
        "train.alakazam_bc_rl.training.train_full_action_bc",
        str(dataset),
        "--evaluate-only-checkpoint",
        str(checkpoint),
        "--evaluation-split",
        "test",
        "--batch-size",
        str(shared["batch_size"]),
        "--device",
        "cuda",
    ]
    result = _run_command(run_root, command, "frozen_test.log")
    metrics = json.loads(result.stdout)
    summary["final_test"] = metrics
    summary["test_evaluation_protocol"] = "after validation-only attempt selection"
    _write_json(summary_path, summary)
    gate = _attempt_gate(run_root, attempt_id)
    _write_json(
        run_root / "frozen_test_evaluation.json",
        {
            "evaluated_at": _timestamp(),
            "attempt_id": attempt_id,
            "checkpoint": selection["selected_checkpoint"],
            "checkpoint_sha256": selection["selected_checkpoint_sha256"],
            "test_reused_from_pre_isolation_attempt": False,
            "metrics": metrics,
        },
    )
    return gate


def _prepare_dataset(entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    run_root = REPOSITORY_ROOT / str(entry["run_root"])
    raw_root = RAW_ROOT / str(entry["package_name"])
    raw_manifest = raw_root / "manifest.json"
    _download_source(entry)
    dataset_root = REPOSITORY_ROOT / "rl_runs/dataset" / str(entry["experiment_id"])
    dataset = dataset_root / "dataset.jsonl"
    summary_path = dataset.with_suffix(dataset.suffix + ".summary.json")
    if not summary_path.is_file():
        command = [
            sys.executable,
            "-m",
            "train.kaggle_bc_top20.training.build_kaggle_bc_dataset",
            str(raw_root),
            "--manifest",
            str(raw_manifest),
            "--output",
            str(dataset),
            "--feature-schema",
            "ptcg_features_universal",
        ]
        print(f"DATASET START {entry['package_name']}", flush=True)
        _run_command(run_root, command, "dataset.log")
        print(f"DATASET DONE {entry['package_name']}", flush=True)
    summary = _read_json(summary_path)
    data_manifest_source = Path(summary["data_manifest"])
    data_manifest = run_root / "data_manifest.json"
    if not data_manifest.is_file():
        shutil.copy2(data_manifest_source, data_manifest)
    audit_path = run_root / "audit.json"
    audit_required = not audit_path.is_file()
    if audit_path.is_file():
        existing_audit = _read_json(audit_path)
        if existing_audit.get("status") != "passed":
            archive = (
                run_root / "audit_failures" / f"audit-{_sha256(audit_path)[:12]}.json"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.is_file():
                shutil.copy2(audit_path, archive)
            audit_required = True
    if audit_required:
        command = [
            sys.executable,
            "-m",
            "train.kaggle_bc_top20.training.audit_kaggle_bc_dataset",
            str(dataset),
            "--manifest",
            str(data_manifest),
            "--output",
            str(audit_path),
        ]
        _run_command(run_root, command, "audit.log")
    audit = _read_json(audit_path)
    if audit.get("status") != "passed":
        raise RuntimeError(f"dataset audit failed: {audit_path}")
    return dataset, {"summary": summary, "audit": audit, "raw_manifest": _relative(raw_manifest)}


def _seed_replay_cache(entry: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    """Reuse already retained Episode payloads before making Kaggle requests."""
    run_root = REPOSITORY_ROOT / str(entry["run_root"])
    source = _read_json(run_root / "source_manifest.json")
    expected = {int(row["episode_id"]) for row in source["episodes"]}
    raw_root.mkdir(parents=True, exist_ok=True)
    existing = {
        int(path.name.split("-", 2)[1])
        for path in raw_root.glob("episode-*-replay.json")
    }
    missing = expected - existing
    linked: list[dict[str, Any]] = []
    if missing:
        for candidate in (REPOSITORY_ROOT / "data/replays").rglob(
            "episode-*-replay.json"
        ):
            if raw_root.resolve() in candidate.resolve().parents:
                continue
            try:
                episode_id = int(candidate.name.split("-", 2)[1])
            except (IndexError, ValueError):
                continue
            if episode_id not in missing:
                continue
            destination = raw_root / candidate.name
            if destination.exists():
                continue
            try:
                os.link(candidate, destination)
                method = "hardlink"
            except OSError:
                shutil.copy2(candidate, destination)
                method = "copy"
            linked.append(
                {
                    "episode_id": episode_id,
                    "source": _relative(candidate),
                    "method": method,
                }
            )
            missing.remove(episode_id)
            if not missing:
                break
    result = {
        "seeded_at": _timestamp(),
        "expected_episode_count": len(expected),
        "already_present": len(existing),
        "reused_episode_count": len(linked),
        "remaining_download_count": len(missing),
        "reused": linked,
    }
    _write_json(run_root / "replay_cache_seed.json", result)
    return result


def _download_source(entry: dict[str, Any]) -> Path:
    run_root = REPOSITORY_ROOT / str(entry["run_root"])
    raw_root = RAW_ROOT / str(entry["package_name"])
    raw_manifest = raw_root / "manifest.json"
    if raw_manifest.is_file():
        return raw_manifest
    seeded = _seed_replay_cache(entry, raw_root)
    command = [
        sys.executable,
        "-m",
        "train.kaggle_bc_top20.training.download_expert_replays",
        "--source-manifest",
        str(run_root / "source_manifest.json"),
        "--output",
        str(raw_root),
        "--workers",
        "2",
        "--retries",
        "6",
        "--request-interval",
        "2.0",
    ]
    print(
        f"DOWNLOAD START {entry['package_name']} "
        f"remaining={seeded['remaining_download_count']}",
        flush=True,
    )
    _run_command(run_root, command, "download.log")
    print(f"DOWNLOAD DONE {entry['package_name']}", flush=True)
    return raw_manifest


def run_all_v1(
    limit: int | None = None,
    *,
    retry_failed: bool = False,
    start_order: int = 1,
) -> dict[str, Any]:
    campaign = _campaign()
    shared = _read_json(SHARED_CONFIG)
    pending = [
        entry
        for entry in campaign["sources"]
        if not entry.get("v1_complete")
        and int(entry["selected_roster_order"]) >= start_order
        and (retry_failed or entry.get("status") != "source_failed")
    ]
    if limit is not None:
        pending = pending[:limit]
    prefetched_entry: dict[str, Any] | None = None
    prefetch_future: concurrent.futures.Future[Path] | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        for pending_index, entry in enumerate(pending):
            index = int(entry["selected_roster_order"])
            run_root = REPOSITORY_ROOT / str(entry["run_root"])
            try:
                if prefetched_entry is entry and prefetch_future is not None:
                    prefetch_future.result()
                    prefetched_entry = None
                    prefetch_future = None
                dataset, data = _prepare_dataset(entry)
                if pending_index + 1 < len(pending):
                    next_entry = pending[pending_index + 1]
                    next_manifest = (
                        RAW_ROOT / str(next_entry["package_name"]) / "manifest.json"
                    )
                    if not next_manifest.is_file():
                        prefetched_entry = next_entry
                        prefetch_future = prefetch_pool.submit(_download_source, next_entry)
                gate = _train_attempt(
                    run_root,
                    dataset,
                    "V1_shared_config",
                    float(shared["base_learning_rate"]),
                )
                entry.update(
                    {
                        "v1_complete": True,
                        "status": "v1_complete",
                        "dataset": _relative(dataset),
                        "raw_manifest": data["raw_manifest"],
                        "records": data["audit"]["records"],
                        "records_by_split": data["audit"]["records_by_split"],
                        "risk_flags": data["audit"]["risk_flags"],
                        "v1_validation_gate_passed": gate["validation_gate_passed"],
                        "v1_validation_exact": gate["metrics"][
                            "validation_exact_action_rate"
                        ],
                    }
                )
                entry.pop("failure", None)
            except Exception as exc:  # noqa: BLE001 - preserve campaign progress
                entry.update(
                    {
                        "v1_complete": False,
                        "status": "source_failed",
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
                (run_root / "failure.json").write_text(
                    json.dumps(
                        {"failed_at": _timestamp(), "error": entry["failure"]},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"SOURCE FAILED {entry['package_name']}: {entry['failure']}", flush=True)
            _save_campaign(campaign)
            print(f"V1 PROGRESS {index}/20", flush=True)
    if all(entry.get("v1_complete") for entry in campaign["sources"]):
        campaign["status"] = "v1_phase_complete"
    _save_campaign(campaign)
    return campaign


def _attempt_curve(run_root: Path, attempt_id: str) -> list[dict[str, Any]]:
    rows = []
    with (run_root / attempt_id / "training_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _first_rescue_side(run_root: Path) -> tuple[str, dict[str, Any]]:
    gate = _read_json(run_root / "V1_shared_config/gate_decision.json")
    curves = _attempt_curve(run_root, "V1_shared_config")
    metrics = gate["metrics"]
    gap = metrics["train_exact_action_rate"] - metrics["validation_exact_action_rate"]
    validation = [float(row["validation/exact_action_rate"]) for row in curves]
    train = [float(row["train/exact_action_rate"]) for row in curves]
    late_drop = max(validation) - validation[-1] > 0.01
    oscillation = len(validation) >= 5 and statistics.pstdev(validation[-5:]) > 0.004
    improving = (
        len(validation) >= 3
        and validation[-1] > validation[-3]
        and train[-1] > train[-3]
    )
    if gap > 0.12 or late_drop or oscillation:
        side = "lower"
        reason = "overfit gap, late validation drop, or oscillation"
    elif metrics["train_exact_action_rate"] < 0.75 and improving:
        side = "upper"
        reason = "train and validation are both low while the last three epochs improve"
    else:
        side = "lower"
        reason = "evidence is ambiguous; protocol defaults to lower LR"
    return side, {
        "train_validation_gap": gap,
        "validation_late_drop": late_drop,
        "validation_last5_std": statistics.pstdev(validation[-5:]),
        "last3_train_validation_improving": improving,
        "reason": reason,
    }


def _select_attempt(run_root: Path) -> dict[str, Any]:
    gates = [
        _read_json(path)
        for path in sorted(run_root.glob("V*_*/gate_decision.json"))
    ]
    # A checkpoint that fails another validation guard must not displace a
    # passing checkpoint merely because exact action rate falls inside the
    # 0.3-point tie window.  All gate inputs are train/validation metrics.
    eligible = [gate for gate in gates if gate["validation_gate_passed"]] or gates
    highest = max(gate["metrics"]["validation_exact_action_rate"] for gate in eligible)
    near = [
        gate
        for gate in eligible
        if highest - gate["metrics"]["validation_exact_action_rate"] < 0.003
    ]
    selected = min(
        near,
        key=lambda gate: (
            gate["metrics"]["validation_policy_loss"],
            -gate["metrics"]["validation_exact_action_rate"],
            gate["attempt_id"],
        ),
    )
    decision = {
        "schema_version": "ptcg_single_expert_attempt_selection_v1",
        "selected_at": _timestamp(),
        "selection_inputs": ["train", "validation"],
        "test_or_evaluation_used": False,
        "tie_window": 0.003,
        "passing_attempt_preferred": any(gate["validation_gate_passed"] for gate in gates),
        "attempts": gates,
        "selected_attempt_id": selected["attempt_id"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "validation_gate_passed": selected["validation_gate_passed"],
        "offline_gate_passed": selected["offline_gate_passed"],
    }
    _write_json(run_root / "attempt_selection.json", decision)
    return decision


def run_rescues() -> dict[str, Any]:
    campaign = _campaign()
    shared = _read_json(SHARED_CONFIG)
    queue = [entry for entry in campaign["sources"] if entry.get("v1_complete")]
    queue.sort(
        key=lambda entry: (
            0 if entry["team_name"] == "LumenLiquidity" else 1,
            -float(entry["selection_score"]),
        )
    )
    for entry in queue:
        run_root = REPOSITORY_ROOT / str(entry["run_root"])
        v1 = _attempt_gate(run_root, "V1_shared_config")
        if (
            not v1["offline_gate_passed"]
            and v1["metrics"]["validation_exact_action_rate"] < 0.75
        ):
            side, evidence = _first_rescue_side(run_root)
            lr = float(shared[f"{side}_rescue_learning_rate"])
            attempt_id = f"V2_lr_{side}"
            _write_json(run_root / "rescue_decision.json", {"side": side, **evidence})
            v2 = _train_attempt(run_root, Path(entry["dataset"]), attempt_id, lr)
            if not v2["validation_gate_passed"]:
                other = "upper" if side == "lower" else "lower"
                _train_attempt(
                    run_root,
                    Path(entry["dataset"]),
                    "V3_lr_other_side",
                    float(shared[f"{other}_rescue_learning_rate"]),
                )
        selection = _select_attempt(run_root)
        selected_gate = _evaluate_frozen_test(run_root, Path(entry["dataset"]), selection)
        selection["offline_gate_passed"] = selected_gate["offline_gate_passed"]
        selection["test_evaluated_after_selection"] = True
        _write_json(run_root / "attempt_selection.json", selection)
        entry.update(
            {
                "status": "attempts_complete",
                "attempts_complete": True,
                "selected_attempt_id": selection["selected_attempt_id"],
                "offline_gate_passed": selected_gate["offline_gate_passed"],
                "risk_flags": sorted(
                    set(entry.get("risk_flags") or [])
                    | (
                        {"imitation_below_gate"}
                        if not selected_gate["offline_gate_passed"]
                        else set()
                    )
                ),
            }
        )
        _save_campaign(campaign)
    campaign["status"] = "rescue_phase_complete"
    _save_campaign(campaign)
    return campaign


def _evaluation_summary(run_root: Path, attempt_id: str) -> dict[str, Any]:
    output_root = run_root / "evaluation" / attempt_id
    runs = sorted(output_root.glob("run-*"), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"evaluation created no run under {output_root}")
    evaluation_root = runs[-1]
    summary = _read_json(evaluation_root / "summary.json")
    metrics = _read_json(evaluation_root / "metrics.json")
    outcome = metrics.get("outcome") or {}
    outcome_payload = outcome.get("payload") or {}
    turn_order = outcome_payload.get("by_turn_order") or {}
    correctness = metrics.get("correctness") or {}
    failure_classes = (correctness.get("diagnostics") or {}).get("failure_classes") or {}
    powerful_hand = metrics.get("powerful_hand") or {}
    setup_relay = metrics.get("setup_relay") or {}
    setup_payload = setup_relay.get("payload") or {}
    second_turn_draws = setup_payload.get("second_turn_draws") or {}
    post_ko = metrics.get("post_ko_relay") or {}
    post_ko_payload = post_ko.get("payload") or {}
    attack_quality = metrics.get("attack_quality") or {}
    library_pressure = metrics.get("library_pressure") or {}
    summary.update(
        {
            "run_root": _relative(evaluation_root),
            "metrics": _relative(evaluation_root / "metrics.json"),
            "report_html": _relative(evaluation_root / "report.html"),
            "report_markdown": _relative(evaluation_root / "report.md"),
            "semantic_metrics": {
                "first_player_win_rate": (turn_order.get("first") or {}).get("value"),
                "second_player_win_rate": (turn_order.get("second") or {}).get("value"),
                "illegal_action_count": int(failure_classes.get("illegal_action", 0)),
                "candidate_error_count": int(failure_classes.get("candidate_error", 0)),
                "correctness_error_count": int(correctness.get("numerator", 0)),
                "powerful_hand_rate": powerful_hand.get("value"),
                "second_turn_setup_rate": setup_relay.get("value"),
                "second_turn_ability_draws_per_game": (
                    (second_turn_draws.get("all_games") or {}).get("average")
                ),
                # Revision 2 defines semantic relay success in payload.success_rate;
                # the legacy top-level value has the opposite proxy interpretation.
                "post_ko_relay_success_rate": post_ko_payload.get("success_rate"),
                "attack_non_prize_rate": attack_quality.get("value"),
                "library_pressure_per_game": library_pressure.get("value"),
            },
        }
    )
    return summary


def package_and_evaluate() -> dict[str, Any]:
    campaign = _campaign()
    for entry in campaign["sources"]:
        if not entry.get("attempts_complete") or not entry.get("offline_gate_passed"):
            continue
        if entry.get("evaluation_complete"):
            continue
        run_root = REPOSITORY_ROOT / str(entry["run_root"])
        selection = _read_json(run_root / "attempt_selection.json")
        attempt_id = str(selection["selected_attempt_id"])
        checkpoint = REPOSITORY_ROOT / str(selection["selected_checkpoint"])
        package = REPOSITORY_ROOT / "submission/top_player" / str(entry["package_name"])
        if not package.is_dir():
            command = [
                sys.executable,
                "-m",
                "train.kaggle_bc_top20.training.build_full_action_submission",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(package),
                "--deck",
                str(run_root / "source_deck.csv"),
                "--cg-source",
                str(CG_SOURCE),
                "--source-identity",
                str(run_root / "source_manifest.json"),
                "--shared-model-config",
                str(SHARED_CONFIG),
                "--experiment-id",
                str(entry["experiment_id"]),
                "--name",
                str(entry["package_name"]),
            ]
            _run_command(run_root, command, "package.log")
        validate = [
            sys.executable,
            "-m",
            "evaluation",
            "validate",
            str(package),
        ]
        validation = _run_command(run_root, validate, "package_validation.log", check=False)
        _write_json(
            run_root / "package_validation.json",
            {
                "validated_at": _timestamp(),
                "package": _relative(package),
                "exit_code": validation.returncode,
                "passed": validation.returncode == 0,
            },
        )
        if validation.returncode:
            entry.update({"status": "package_validation_failed"})
            _save_campaign(campaign)
            continue
        evaluation = [
            sys.executable,
            "-m",
            "evaluation",
            "run",
            "--candidate",
            str(package),
            "--opponents",
            "all",
            "--games",
            "10",
            "--metric-profile",
            "auto_iteration_v8_setup_relay",
            "--no-visualize",
            "--output",
            str(run_root / "evaluation" / attempt_id),
        ]
        print(f"EVALUATION START {entry['package_name']}", flush=True)
        _run_command(run_root, evaluation, "evaluation.log", check=False)
        summary = _evaluation_summary(run_root, attempt_id)
        completed = int(summary.get("completed_games", summary.get("games", 0)))
        errors = int(summary.get("errors", 0))
        unfinished = int(summary.get("unfinished", 0))
        hard_guard = completed == 180 and errors == 0 and unfinished == 0
        _write_json(
            run_root / "evaluation_summary.json",
            {**summary, "hard_guard_passed": hard_guard},
        )
        entry.update(
            {
                "status": "evaluation_complete" if hard_guard else "evaluation_guard_failed",
                "package": _relative(package),
                "package_validated": True,
                "evaluation_complete": True,
                "evaluation_hard_guard_passed": hard_guard,
                "evaluation": _relative(run_root / "evaluation_summary.json"),
            }
        )
        _save_campaign(campaign)
        print(f"EVALUATION DONE {entry['package_name']} guard={hard_guard}", flush=True)
    campaign["status"] = "evaluation_phase_complete"
    _save_campaign(campaign)
    return campaign


def cleanup_datasets() -> dict[str, Any]:
    campaign = _campaign()
    for entry in campaign["sources"]:
        if not entry.get("attempts_complete"):
            continue
        run_root = REPOSITORY_ROOT / str(entry["run_root"])
        dataset = Path(str(entry.get("dataset", "")))
        if not dataset.is_absolute():
            dataset = REPOSITORY_ROOT / dataset
        cleanup_path = run_root / "cleanup.json"
        if cleanup_path.is_file():
            continue
        removed: list[dict[str, Any]] = []
        if dataset.is_file():
            artifact_root = (REPOSITORY_ROOT / "rl_runs/dataset").resolve()
            if artifact_root not in dataset.resolve().parents or dataset.name != "dataset.jsonl":
                raise ValueError(f"refusing to remove unexpected dataset path: {dataset}")
            size = dataset.stat().st_size
            dataset.unlink()
            removed.append({"path": _relative(dataset), "bytes": size})
        cleanup = {
            "schema_version": "ptcg_single_expert_cleanup_v1",
            "cleaned_at": _timestamp(),
            "removed": removed,
            "raw_replays_retained": True,
            "irreversible_boundary": (
                "Derived JSONL can be rebuilt from retained exact raw replays and manifests; "
                "raw replays were not deleted."
            ),
            "free_bytes_after": shutil.disk_usage(REPOSITORY_ROOT).free,
        }
        _write_json(cleanup_path, cleanup)
        entry["cleanup_complete"] = True
        _save_campaign(campaign)
    campaign["status"] = "cleanup_complete"
    _save_campaign(campaign)
    return campaign


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def render_report() -> Path:
    campaign = _campaign()
    rows: list[str] = []
    for entry in campaign["sources"]:
        run_root = REPOSITORY_ROOT / str(entry["run_root"])
        attempts: list[str] = []
        gates: dict[str, dict[str, Any]] = {}
        for gate_path in sorted(run_root.glob("V*_*/gate_decision.json")):
            gate = _read_json(gate_path)
            gates[str(gate["attempt_id"])] = gate
            attempts.append(
                f"{escape(str(gate['attempt_id']))}: "
                f"exact {_pct(gate['metrics']['validation_exact_action_rate'])}, "
                f"multi {_pct(gate['metrics']['validation_multi_action_exact_rate'])}"
            )
        selection = (
            _read_json(run_root / "attempt_selection.json")
            if (run_root / "attempt_selection.json").is_file()
            else {}
        )
        selected_attempt = str(selection.get("selected_attempt_id", ""))
        selected_gate = gates.get(selected_attempt) or {}
        offline = selected_gate.get("metrics") or {}
        evaluation = (
            _read_json(REPOSITORY_ROOT / str(entry["evaluation"]))
            if entry.get("evaluation")
            else {}
        )
        semantic = evaluation.get("semantic_metrics") or {}
        wins = evaluation.get("wins")
        losses = evaluation.get("losses")
        draws = evaluation.get("draws")
        result = (
            "—"
            if wins is None
            else (
                f"{wins}-{losses}-{draws} · {_pct(evaluation.get('win_rate'))}<br>"
                f"先/后 {_pct(semantic.get('first_player_win_rate'))} / "
                f"{_pct(semantic.get('second_player_win_rate'))}<br>"
                f"errors {evaluation.get('errors', '—')} · "
                f"unfinished {evaluation.get('unfinished', '—')} · "
                f"illegal {semantic.get('illegal_action_count', '—')}"
            )
        )
        by_opponent = evaluation.get("by_opponent") or {}
        opponent_details = " · ".join(
            f"{escape(str(name))} {_pct(values.get('win_rate'))}"
            for name, values in sorted(by_opponent.items())
        )
        if opponent_details:
            result += f"<details><summary>18 opponents</summary>{opponent_details}</details>"
        report_link = ""
        if evaluation.get("report_html"):
            report_path = (REPOSITORY_ROOT / str(evaluation["report_html"])).resolve()
            report_link = f'<br><a href="{escape(str(report_path))}">完整语义报告</a>'
        result += report_link

        records_by_split = entry.get("records_by_split") or {}
        record_text = (
            f"{entry.get('records', '—')}<br>"
            f"T/V/Test {records_by_split.get('train', '—')} / "
            f"{records_by_split.get('validation', '—')} / "
            f"{records_by_split.get('test', '—')}"
        )
        offline_text = "—"
        if offline:
            offline_text = (
                f"exact {_pct(offline.get('validation_exact_action_rate'))}<br>"
                f"multi {_pct(offline.get('validation_multi_action_exact_rate'))}<br>"
                f"count {_pct(offline.get('validation_selection_count_accuracy'))}<br>"
                f"legal V/Test {_pct(offline.get('validation_legal_action_rate'))} / "
                f"{_pct(offline.get('test_legal_action_rate'))}"
            )
        checkpoint_text = "—"
        if selection:
            checkpoint_text = (
                f"{escape(selected_attempt)} · rescue {max(0, len(gates) - 1)}<br>"
                f"<code>{escape(str(selection.get('selected_checkpoint', '—')))}</code><br>"
                f"<code>{escape(str(selection.get('selected_checkpoint_sha256', ''))[:16])}</code>"
            )
        semantic_text = "—"
        if semantic:
            semantic_text = (
                f"Powerful Hand {_pct(semantic.get('powerful_hand_rate'))}<br>"
                f"T2 setup {_pct(semantic.get('second_turn_setup_rate'))}<br>"
                f"Post-KO {_pct(semantic.get('post_ko_relay_success_rate'))}<br>"
                f"non-prize atk {_pct(semantic.get('attack_non_prize_rate'))}<br>"
                f"library {semantic.get('library_pressure_per_game', '—')}"
            )
        if entry.get("evaluation_hard_guard_passed"):
            review = "eligible for user review"
        elif entry.get("offline_gate_passed") is False and entry.get("attempts_complete"):
            review = "reject: imitation_below_gate"
        else:
            review = "pending"
        rows.append(
            "<tr>"
            f"<td>{entry['selected_roster_order']}</td>"
            f"<td><strong>{escape(str(entry['package_name']))}</strong><br>"
            f"Rank {entry['static_rank']} · {escape(str(entry['team_name']))}</td>"
            f"<td>{entry.get('refreshed_episode_count','—')}</td>"
            f"<td>{record_text}</td>"
            f"<td>{'<br>'.join(attempts) or '—'}</td>"
            f"<td>{offline_text}</td>"
            f"<td>{checkpoint_text}</td>"
            f"<td>{'validated' if entry.get('package_validated') else '—'}<br>{review}</td>"
            f"<td>{result}</td>"
            f"<td>{semantic_text}</td>"
            f"<td>{', '.join(entry.get('risk_flags') or []) or 'none'}</td>"
            "</tr>"
        )
    completed_v1 = sum(bool(entry.get("v1_complete")) for entry in campaign["sources"])
    passed = sum(bool(entry.get("offline_gate_passed")) for entry in campaign["sources"])
    packaged = sum(bool(entry.get("package_validated")) for entry in campaign["sources"])
    evaluated = sum(
        bool(entry.get("evaluation_hard_guard_passed")) for entry in campaign["sources"]
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top 20 单专家 BC 训练与评测 · 2026-07-23</title>
<style>
body{{font:14px/1.55 system-ui,sans-serif;margin:0;background:#f4f6f2;color:#17201b}}main{{max-width:1500px;margin:auto;padding:36px}}
h1{{font-size:34px;margin:0 0 8px}}.summary{{display:flex;gap:16px;flex-wrap:wrap;margin:22px 0}}.card{{background:white;border:1px solid #ccd5ce;border-radius:12px;padding:14px 18px;min-width:180px}}.card b{{font-size:24px;display:block}}
table{{width:100%;border-collapse:collapse;background:white;font-size:12px}}th,td{{padding:10px;border:1px solid #d8dfda;text-align:left;vertical-align:top}}th{{background:#e8eee9;position:sticky;top:0}}code{{font-family:ui-monospace,monospace;overflow-wrap:anywhere}}.note{{background:#fff7df;border-left:4px solid #c69122;padding:14px 18px;margin:20px 0}}.table-wrap{{overflow-x:auto}}details{{margin-top:6px;max-width:420px}}
</style></head><body><main><h1>Top 20 单专家 BC 训练与 Repository Evaluation</h1>
<p>冻结静态 roster 与 V4 通用结构；每位专家独立 source identity、dataset、随机初始化、optimizer、checkpoint、run 和 package。报告不自动修改 opponent pool。</p>
<div class="summary"><div class="card"><span>V1 完成</span><b>{completed_v1}/20</b></div><div class="card"><span>离线 gate 通过</span><b>{passed}/20</b></div><div class="card"><span>Package validated</span><b>{packaged}/20</b></div><div class="card"><span>180 局硬护栏通过</span><b>{evaluated}/20</b></div><div class="card"><span>Campaign 状态</span><b style="font-size:15px">{campaign['status']}</b></div></div>
<div class="note">offline gate：validation exact ≥75%、multi-action exact ≥65%、selection-count ≥98%、validation/test legal 100%。Test、repository evaluation 与 Kaggle 结果均不参与救援 LR 选择。</div>
<div class="table-wrap"><table><thead><tr><th>顺序</th><th>专家 / 静态 Rank</th><th>Episodes</th><th>Decision records</th><th>V1/V2/V3</th><th>最终离线指标</th><th>最终 checkpoint / rescue</th><th>Package / 审核建议</th><th>18×10 evaluation</th><th>语义指标</th><th>风险</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Shared config: <code>{campaign['shared_model_config']}</code> · SHA-256 <code>{campaign['shared_model_config_sha256']}</code><br>Campaign state: <code>{_relative(CAMPAIGN_STATE)}</code></p>
</main></body></html>"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    return REPORT_PATH


def run_all() -> None:
    refresh_sources()
    run_all_v1()
    run_rescues()
    package_and_evaluate()
    cleanup_datasets()
    render_report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("refresh", "v1", "rescue", "evaluate", "cleanup", "report", "all"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--start-order", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "refresh":
        refresh_sources()
    elif args.phase == "v1":
        run_all_v1(
            limit=args.limit,
            retry_failed=args.retry_failed,
            start_order=args.start_order,
        )
    elif args.phase == "rescue":
        run_rescues()
    elif args.phase == "evaluate":
        package_and_evaluate()
    elif args.phase == "cleanup":
        cleanup_datasets()
    elif args.phase == "report":
        print(render_report())
    else:
        run_all()


if __name__ == "__main__":
    main()
