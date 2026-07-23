"""Run the frozen 210-minute BC capacity campaign one transaction at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from evaluation.runtime.loader import compute_cg_manifest
from train.alakazam_bc_rl.training.build_full_action_candidate import build as build_candidate
from train.kaggle_bc_top20.training.render_bc_capacity_search import render_campaign


TRAINING_BUDGET_SECONDS = 12_600.0
STOP_LAUNCH_SECONDS = 11_700.0
INITIAL_ESTIMATES = {
    "S": 720.0,
    "B": 900.0,
    "W": 1_800.0,
    "D": 1_800.0,
    "L": 2_880.0,
}
LEARNING_RATES = (1e-4, 3e-4, 5e-4)


@dataclass(frozen=True)
class TrialSpec:
    version: str
    phase: str
    architecture: str
    d_model: int
    layers: int
    hidden_dim: int
    learning_rate: float
    seed: int = 7
    dropout: float = 0.0
    heads: int = 4
    epochs: int = 20
    batch_size: int = 256


ARCHITECTURES = {
    "S": (192, 2, 384),
    "B": (256, 2, 512),
    "W": (384, 2, 768),
    "D": (256, 4, 512),
    "L": (384, 4, 768),
}


def phase_a_specs() -> list[TrialSpec]:
    d_model, layers, hidden_dim = ARCHITECTURES["B"]
    return [
        TrialSpec(
            version=f"V{index}_b_d256_l2_lr{_lr_tag(lr)}_s7",
            phase="A",
            architecture="B",
            d_model=d_model,
            layers=layers,
            hidden_dim=hidden_dim,
            learning_rate=lr,
        )
        for index, lr in enumerate(LEARNING_RATES, 1)
    ]


def phase_b_specs(learning_rate: float) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for index, architecture in enumerate(("S", "W", "D", "L"), 4):
        d_model, layers, hidden_dim = ARCHITECTURES[architecture]
        specs.append(
            TrialSpec(
                version=(
                    f"V{index}_{architecture.lower()}_d{d_model}_l{layers}_"
                    f"lr{_lr_tag(learning_rate)}_s7"
                ),
                phase="B",
                architecture=architecture,
                d_model=d_model,
                layers=layers,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
            )
        )
    return specs


def choose_baseline_learning_rate(results: list[dict[str, Any]]) -> float:
    """Choose from Phase A using validation evidence only."""
    completed = [result for result in results if result.get("status") == "completed"]
    if not completed:
        raise ValueError("Phase A has no completed baseline trial")
    highest_exact = max(_offline(result)["validation_exact"] for result in completed)
    equivalent = [
        result
        for result in completed
        if highest_exact - _offline(result)["validation_exact"] < 0.003
    ]
    minimum_loss = min(_offline(result)["validation_policy_loss"] for result in equivalent)
    loss_equivalent = [
        result
        for result in equivalent
        if _offline(result)["validation_policy_loss"] <= minimum_loss + 0.001
    ]
    minimum_stability = min(_offline(result)["validation_last5_std"] for result in loss_equivalent)
    stable = [
        result
        for result in loss_equivalent
        if _offline(result)["validation_last5_std"] <= minimum_stability + 0.0005
    ]
    default = next(
        (
            result
            for result in stable
            if abs(float(result["spec"]["learning_rate"]) - 3e-4) < 1e-12
        ),
        None,
    )
    selected = default or min(
        stable,
        key=lambda result: (
            _offline(result)["validation_policy_loss"],
            _offline(result)["validation_last5_std"],
            abs(float(result["spec"]["learning_rate"]) - 3e-4),
        ),
    )
    return float(selected["spec"]["learning_rate"])


def choose_phase_c_spec(
    results: list[dict[str, Any]],
    baseline_learning_rate: float,
) -> tuple[TrialSpec | None, str]:
    """Choose at most one rescue using only train/validation curves."""
    by_architecture = {
        str(result["spec"]["architecture"]): result
        for result in results
        if result.get("status") == "completed"
    }
    baseline = next(
        (
            result
            for result in results
            if result.get("status") == "completed"
            and result["spec"]["architecture"] == "B"
            and abs(float(result["spec"]["learning_rate"]) - baseline_learning_rate) < 1e-12
        ),
        None,
    )
    ordered = [by_architecture[name] for name in ("L", "W", "D") if name in by_architecture]
    if baseline is None or not ordered:
        return None, "capacity evidence is incomplete; no rescue launched"

    baseline_offline = _offline(baseline)
    for result in ordered:
        offline = _offline(result)
        spec = result["spec"]
        architecture = str(spec["architecture"])
        if (
            offline["train_exact"] > baseline_offline["train_exact"] + 0.003
            and offline["validation_exact"] < baseline_offline["validation_exact"] - 0.001
            and offline["exact_gap"] > baseline_offline["exact_gap"] + 0.005
        ):
            return (
                _derived_spec(
                    architecture,
                    version=8,
                    phase="C",
                    learning_rate=float(spec["learning_rate"]),
                    dropout=0.05,
                ),
                f"{architecture} increased train fit while validation and gap worsened; "
                "testing dropout=0.05",
            )

    for result in ordered:
        offline = _offline(result)
        spec = result["spec"]
        learning_rate = float(spec["learning_rate"])
        architecture = str(spec["architecture"])
        if offline["validation_last5_std"] > 0.004 and learning_rate > LEARNING_RATES[0]:
            lower = LEARNING_RATES[LEARNING_RATES.index(learning_rate) - 1]
            return (
                _derived_spec(
                    architecture,
                    version=8,
                    phase="C",
                    learning_rate=lower,
                ),
                f"{architecture} validation oscillated late; testing adjacent lower LR",
            )
        if (
            offline["train_last5_slope"] > 0.0005
            and offline["validation_last5_slope"] > 0.0005
            and learning_rate < LEARNING_RATES[-1]
        ):
            higher = LEARNING_RATES[LEARNING_RATES.index(learning_rate) + 1]
            return (
                _derived_spec(
                    architecture,
                    version=8,
                    phase="C",
                    learning_rate=higher,
                ),
                f"{architecture} train and validation still rose together; "
                "testing adjacent higher LR",
            )
    return None, "capacity trials were stable enough that an LR/dropout rescue was not justified"


def choose_phase_d_spec(
    results: list[dict[str, Any]],
    version: int,
    baseline_learning_rate: float,
) -> tuple[TrialSpec, str]:
    """Select one seed check from capacity evidence, without evaluation/test data."""
    capacity = [
        result
        for result in results
        if result.get("status") == "completed"
        and result["spec"]["architecture"] in ARCHITECTURES
        and int(result["spec"]["seed"]) == 7
        and (
            result["spec"]["architecture"] != "B"
            or abs(float(result["spec"]["learning_rate"]) - baseline_learning_rate) < 1e-12
        )
    ]
    highest_exact = max(_offline(result)["validation_exact"] for result in capacity)
    near = [
        result
        for result in capacity
        if highest_exact - _offline(result)["validation_exact"] < 0.003
    ]
    selected = min(
        near,
        key=lambda result: (
            int(result["model"]["parameter_count"]),
            -_offline(result)["validation_exact"],
            _offline(result)["validation_policy_loss"],
            str(result["spec"]["architecture"]),
        ),
    )
    spec = selected["spec"]
    architecture = str(spec["architecture"])
    return (
        _derived_spec(
            architecture,
            version=version,
            phase="D",
            learning_rate=float(spec["learning_rate"]),
            dropout=float(spec["dropout"]),
            seed=17,
        ),
        f"{architecture} is within 0.3pp of the highest validation exact at the "
        "smallest observed parameter count in that interval",
    )


def _derived_spec(
    architecture: str,
    *,
    version: int,
    phase: str,
    learning_rate: float,
    dropout: float = 0.0,
    seed: int = 7,
) -> TrialSpec:
    d_model, layers, hidden_dim = ARCHITECTURES[architecture]
    dropout_tag = f"_do{str(dropout).replace('.', 'p')}" if dropout else ""
    return TrialSpec(
        version=(
            f"V{version}_{architecture.lower()}_d{d_model}_l{layers}_"
            f"lr{_lr_tag(learning_rate)}{dropout_tag}_s{seed}"
        ),
        phase=phase,
        architecture=architecture,
        d_model=d_model,
        layers=layers,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        dropout=dropout,
        seed=seed,
    )


def run_campaign(
    repository_root: Path,
    experiment_root: Path,
    dataset: Path,
    source_package: Path,
    design_document: Path,
    cxx_runtime_lib: Path,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    experiment_root = experiment_root.resolve()
    dataset = dataset.resolve()
    source_package = source_package.resolve()
    design_document = design_document.resolve()
    _freeze_campaign(
        repository_root,
        experiment_root,
        dataset,
        source_package,
        design_document,
    )
    comparison_path = experiment_root / "comparison_manifest.json"
    comparison = _load_json(comparison_path, default=_new_comparison(experiment_root.name))
    decisions: list[str] = list(comparison.get("decisions", []))
    _write_json(comparison_path, comparison)
    render_campaign(repository_root, experiment_root)

    for spec in phase_a_specs():
        comparison = _run_or_resume(
            repository_root,
            experiment_root,
            dataset,
            source_package,
            cxx_runtime_lib,
            spec,
            comparison,
        )
    completed_a = _results_for_phase(comparison, "A")
    if not any(result.get("status") == "completed" for result in completed_a):
        decisions.append("Phase A produced no completed trial; capacity search cannot continue.")
        return _finalize_campaign(
            repository_root,
            experiment_root,
            comparison,
            decisions,
            status="invalid_incomplete_mandatory_matrix",
        )
    selected_lr = choose_baseline_learning_rate(completed_a)
    decision = f"Phase A selected lr={selected_lr:g} using validation-only tie rules."
    if decision not in decisions:
        decisions.append(decision)
    comparison["selected_baseline_learning_rate"] = selected_lr
    comparison["decisions"] = _deduplicate(decisions)
    comparison["updated_at"] = _timestamp()
    _write_json(comparison_path, comparison)
    render_campaign(repository_root, experiment_root)

    for spec in phase_b_specs(selected_lr):
        comparison = _run_or_resume(
            repository_root,
            experiment_root,
            dataset,
            source_package,
            cxx_runtime_lib,
            spec,
            comparison,
        )

    mandatory_specs = [*phase_a_specs(), *phase_b_specs(selected_lr)]
    mandatory_complete = all(
        any(
            result.get("status") == "completed"
            and result["spec"]["version"] == spec.version
            for result in comparison["trials"]
        )
        for spec in mandatory_specs
    )
    if not mandatory_complete:
        decisions.append(
            "The mandatory Phase A/B matrix is incomplete; optional rescue and seed trials "
            "were not launched."
        )
        return _finalize_campaign(
            repository_root,
            experiment_root,
            comparison,
            decisions,
            status="invalid_incomplete_mandatory_matrix",
            selected_lr=selected_lr,
        )

    phase_c, phase_c_reason = choose_phase_c_spec(list(comparison["trials"]), selected_lr)
    decisions.append(f"Phase C: {phase_c_reason}.")
    if phase_c is not None:
        comparison = _run_or_resume(
            repository_root,
            experiment_root,
            dataset,
            source_package,
            cxx_runtime_lib,
            phase_c,
            comparison,
        )

    next_version = max(
        int(str(result["spec"]["version"]).split("_", 1)[0][1:])
        for result in comparison["trials"]
    ) + 1
    phase_d, phase_d_reason = choose_phase_d_spec(
        list(comparison["trials"]), next_version, selected_lr
    )
    decisions.append(f"Phase D candidate: {phase_d_reason}.")
    comparison = _run_or_resume(
        repository_root,
        experiment_root,
        dataset,
        source_package,
        cxx_runtime_lib,
        phase_d,
        comparison,
    )
    return _finalize_campaign(
        repository_root,
        experiment_root,
        comparison,
        decisions,
        status="complete_waiting_user_selection",
        selected_lr=selected_lr,
    )


def _finalize_campaign(
    repository_root: Path,
    experiment_root: Path,
    comparison: dict[str, Any],
    decisions: list[str],
    *,
    status: str,
    selected_lr: float | None = None,
) -> dict[str, Any]:
    if _training_seconds(comparison) > TRAINING_BUDGET_SECONDS:
        status = "invalid_training_budget_exceeded"
        decisions.append("The hard 12,600-second cumulative training budget was exceeded.")
    comparison["status"] = status
    if selected_lr is not None:
        comparison["selected_baseline_learning_rate"] = selected_lr
    comparison["decisions"] = _deduplicate(decisions)
    comparison["training_seconds"] = _training_seconds(comparison)
    comparison["remaining_training_seconds"] = max(
        0.0, TRAINING_BUDGET_SECONDS - comparison["training_seconds"]
    )
    comparison["completed_at"] = _timestamp()
    _write_json(experiment_root / "comparison_manifest.json", comparison)
    _write_decisions(experiment_root, comparison)
    _update_experiment_manifest(experiment_root, comparison)
    render_campaign(repository_root, experiment_root)
    return comparison


def _run_or_resume(
    repository_root: Path,
    experiment_root: Path,
    dataset: Path,
    source_package: Path,
    cxx_runtime_lib: Path,
    spec: TrialSpec,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    existing = next(
        (
            result
            for result in comparison["trials"]
            if result["spec"]["version"] == spec.version
        ),
        None,
    )
    if existing is not None:
        print(
            f"[resume] {spec.version} already recorded as {existing.get('status')}",
            flush=True,
        )
        return comparison
    if not _can_launch(comparison, spec, mandatory=spec.phase in {"A", "B"}):
        result = {
            "schema_version": "ptcg_capacity_trial_result_v1",
            "status": "skipped",
            "spec": asdict(spec),
            "failure": "11,700-second launch threshold did not leave enough training budget",
            "training": {
                "wall_seconds": 0.0,
                "predicted_seconds": _predicted_seconds(comparison, spec),
            },
            "completed_at": _timestamp(),
        }
        _persist_trial_result(experiment_root, spec, result)
        return _append_trial_result(repository_root, experiment_root, comparison, result)

    print(f"[trial] starting {spec.version}", flush=True)
    try:
        result = _run_trial(
            repository_root,
            experiment_root,
            dataset,
            source_package,
            cxx_runtime_lib,
            spec,
        )
    except Exception as error:
        run_root = experiment_root / spec.version
        prior_status = _load_json(run_root / "status.json", default={})
        result = {
            "schema_version": "ptcg_capacity_trial_result_v1",
            "status": "failed",
            "spec": asdict(spec),
            "failure": f"{type(error).__name__}: {error}",
            "failure_stage": prior_status.get("stage", "trial_orchestration"),
            "traceback": traceback.format_exc(),
            "training": prior_status.get("training", {"wall_seconds": 0.0}),
            "completed_at": _timestamp(),
        }
        _persist_trial_result(experiment_root, spec, result)
        print(f"[trial] failed {spec.version}: {result['failure']}", flush=True)
    return _append_trial_result(repository_root, experiment_root, comparison, result)


def _append_trial_result(
    repository_root: Path,
    experiment_root: Path,
    comparison: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    comparison["trials"].append(result)
    comparison["training_seconds"] = _training_seconds(comparison)
    comparison["updated_at"] = _timestamp()
    _write_json(experiment_root / "comparison_manifest.json", comparison)
    render_campaign(repository_root, experiment_root)
    return comparison


def _persist_trial_result(
    experiment_root: Path,
    spec: TrialSpec,
    result: dict[str, Any],
) -> None:
    run_root = experiment_root / spec.version
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "status.json", result)
    _write_json(run_root / "trial_result.json", result)


def _run_trial(
    repository_root: Path,
    experiment_root: Path,
    dataset: Path,
    source_package: Path,
    cxx_runtime_lib: Path,
    spec: TrialSpec,
) -> dict[str, Any]:
    run_root = experiment_root / spec.version
    checkpoint_root = (
        repository_root
        / "rl"
        / "artifact"
        / "checkpoint"
        / experiment_root.name
        / spec.version
    )
    tensorboard_root = (
        repository_root
        / "rl"
        / "_runs"
        / "tensorboard"
        / experiment_root.name
        / spec.version
    )
    candidate_root = Path("/tmp") / f"{experiment_root.name}-candidates" / spec.version
    evaluation_root = experiment_root / "evaluation" / spec.version
    for path in (run_root, checkpoint_root, tensorboard_root, candidate_root, evaluation_root):
        if path.exists() and any(path.rglob("*")):
            raise FileExistsError(f"trial artifact path is already in use: {path}")

    control_status_path = experiment_root / "_campaign_status" / f"{spec.version}.json"
    _write_json(
        control_status_path,
        {
            "schema_version": "ptcg_capacity_trial_v1",
            "status": "training",
            "stage": "training",
            "spec": asdict(spec),
            "started_at": _timestamp(),
        },
    )
    train_command = [
        sys.executable,
        "-m",
        "train.alakazam_bc_rl.training.train_full_action_bc",
        str(dataset),
        "--output",
        str(run_root),
        "--epochs",
        str(spec.epochs),
        "--batch-size",
        str(spec.batch_size),
        "--learning-rate",
        str(spec.learning_rate),
        "--seed",
        str(spec.seed),
        "--d-model",
        str(spec.d_model),
        "--hidden-dim",
        str(spec.hidden_dim),
        "--num-heads",
        str(spec.heads),
        "--transformer-layers",
        str(spec.layers),
        "--dropout",
        str(spec.dropout),
        "--device",
        "cuda",
        "--storage-path",
        "/mnt/c",
        "--min-free-gib",
        "10",
    ]
    staged_train_log = experiment_root / "_campaign_logs" / f"{spec.version}.train.log"
    train_wall, train_code = _run_logged_process(
        train_command,
        repository_root,
        staged_train_log,
        heartbeat_metrics=run_root / "training_metrics.jsonl",
    )
    run_root.mkdir(parents=True, exist_ok=True)
    if staged_train_log.is_file():
        shutil.move(str(staged_train_log), run_root / "train.log")
    status_path = run_root / "status.json"
    _write_json(
        status_path,
        {
            "schema_version": "ptcg_capacity_trial_v1",
            "status": "running",
            "stage": "training_artifact_validation",
            "spec": asdict(spec),
            "training": {"wall_seconds": train_wall},
            "started_at": _timestamp(),
        },
    )
    if train_code != 0:
        result = {
            "schema_version": "ptcg_capacity_trial_result_v1",
            "status": "failed",
            "spec": asdict(spec),
            "failure": f"training process exited with code {train_code}",
            "training": {"wall_seconds": train_wall},
            "completed_at": _timestamp(),
        }
        _persist_trial_result(experiment_root, spec, result)
        return result

    metrics_path = run_root / "training_metrics.jsonl"
    summary_path = run_root / "training_summary.json"
    metrics = _read_jsonl(metrics_path)
    summary = _load_json(summary_path)
    checkpoint = checkpoint_root / "best_validation.pt"
    latest = checkpoint_root / "latest.pt"
    events = list(tensorboard_root.glob("events.out.tfevents.*"))
    if len(metrics) != spec.epochs or not checkpoint.is_file() or not latest.is_file() or not events:
        raise RuntimeError(f"{spec.version} is missing required 20-epoch artifacts")
    if _contains_nonfinite(metrics):
        raise RuntimeError(f"{spec.version} contains NaN or Inf metrics")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = payload["model"]
    parameter_count = sum(tensor.numel() for tensor in model_state.values())
    checkpoint_sha256 = _sha256(checkpoint)
    candidate_root.parent.mkdir(parents=True, exist_ok=True)
    build_candidate(candidate_root, checkpoint, source_package)
    environment = _runtime_environment(cxx_runtime_lib)
    _write_json(
        status_path,
        {
            "schema_version": "ptcg_capacity_trial_v1",
            "status": "running",
            "stage": "candidate_validation",
            "spec": asdict(spec),
            "training": {"wall_seconds": train_wall},
        },
    )
    validation_wall, validation_code = _run_logged_process(
        [sys.executable, "-m", "evaluation", "validate", str(candidate_root)],
        repository_root,
        run_root / "candidate_validation.log",
        environment=environment,
    )
    if validation_code != 0:
        raise RuntimeError(f"candidate validation failed for {spec.version}")

    before_runs = set(evaluation_root.glob("run-*")) if evaluation_root.exists() else set()
    evaluation_command = [
        sys.executable,
        "-m",
        "evaluation",
        "run",
        "--candidate",
        str(candidate_root),
        "--opponents",
        "all",
        "--games",
        "10",
        "--no-visualize",
        "--metric-profile",
        "auto_iteration_v8_setup_relay",
        "--output",
        str(evaluation_root),
    ]
    _write_json(
        status_path,
        {
            "schema_version": "ptcg_capacity_trial_v1",
            "status": "running",
            "stage": "official_evaluation",
            "spec": asdict(spec),
            "training": {"wall_seconds": train_wall},
        },
    )
    evaluation_wall, evaluation_code = _run_logged_process(
        evaluation_command,
        repository_root,
        run_root / "evaluation.log",
        environment=environment,
    )
    if evaluation_code != 0:
        raise RuntimeError(f"official-engine evaluation failed for {spec.version}")
    new_runs = set(evaluation_root.glob("run-*")) - before_runs
    if len(new_runs) != 1:
        raise RuntimeError(f"expected one new evaluation run for {spec.version}, got {new_runs}")
    report_root = new_runs.pop()
    evaluation_summary = _load_json(report_root / "summary.json")
    evaluation_metrics = _load_json(report_root / "metrics.json")
    evaluation_manifest = _load_json(report_root / "manifest.json")
    offline = _offline_summary(summary, metrics)
    runtime = summary.get("runtime") or {}
    result = {
        "schema_version": "ptcg_capacity_trial_result_v1",
        "status": "completed",
        "spec": asdict(spec),
        "training": {
            "wall_seconds": train_wall,
            "epoch_median_seconds": statistics.median(
                float(row["runtime/epoch_seconds"]) for row in metrics
            ),
            "peak_gpu_memory_bytes": int(runtime.get("peak_gpu_memory_bytes", 0)),
            "validation_control_only": True,
        },
        "offline": offline,
        "model": {
            "parameter_count": parameter_count,
            "state_dict_tensors": len(model_state),
            "checkpoint": str(checkpoint.relative_to(repository_root)),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_step": int(payload.get("step", 0)),
            "feature_schema": payload["metadata"]["feature_config"]["schema_version"],
            "model_config": payload["metadata"]["model_config"],
        },
        "candidate": {
            "path": str(candidate_root),
            "manifest": _load_json(candidate_root / "manifest.json"),
            "validation_wall_seconds": validation_wall,
            "package_hash": evaluation_manifest["candidate"]["package_hash"],
        },
        "evaluation": _evaluation_summary(
            repository_root,
            report_root,
            evaluation_wall,
            evaluation_summary,
            evaluation_metrics,
            evaluation_manifest,
        ),
        "completed_at": _timestamp(),
    }
    _persist_trial_result(experiment_root, spec, result)
    if control_status_path.is_file():
        control_status_path.unlink()
    print(
        f"[trial] completed {spec.version}: "
        f"val={offline['validation_exact']:.4f}, "
        f"eval={evaluation_summary['wins']}/{evaluation_summary['total_games']}",
        flush=True,
    )
    return result


def _evaluation_summary(
    repository_root: Path,
    report_root: Path,
    wall_seconds: float,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    outcome_payload = metrics["outcome"].get("payload") or {}
    return {
        "run_id": manifest["run_id"],
        "report_root": str(report_root.relative_to(repository_root)),
        "report_html": str((report_root / "report.html").relative_to(repository_root)),
        "wall_seconds": wall_seconds,
        "profile_id": manifest["metric_profile"]["id"],
        "profile_revision": manifest["metric_profile"]["revision"],
        "catalog_packages": [item["name"] for item in manifest["opponents"]],
        "catalog_hash": _hash_json(manifest["opponents"]),
        "candidate_package_hash": manifest["candidate"]["package_hash"],
        **summary,
        "by_turn_order": outcome_payload.get("by_turn_order", {}),
        "metrics": {
            metric_id: {
                "value": metrics[metric_id].get("value"),
                "numerator": metrics[metric_id].get("numerator"),
                "denominator": metrics[metric_id].get("denominator"),
                "payload": metrics[metric_id].get("payload", {}),
            }
            for metric_id in (
                "correctness",
                "powerful_hand",
                "setup_relay",
                "post_ko_relay",
                "attack_quality",
                "rare_candy",
                "run_away_draw",
                "library_pressure",
            )
        },
    }


def _offline_summary(summary: dict[str, Any], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    best_epoch = int(summary["best_epoch"])
    best = next(row for row in metrics if int(row["step"]) == best_epoch)
    contexts = best["validation/by_selection"]
    exact_values = [float(value["exact_action_rate"]) for value in contexts.values()]
    last_five = metrics[-5:]
    return {
        "best_epoch": best_epoch,
        "validation_exact": float(summary["best_validation_exact_action_rate"]),
        "train_exact": float(best["train/exact_action_rate"]),
        "exact_gap": float(best["train/exact_action_rate"])
        - float(best["validation/exact_action_rate"]),
        "train_policy_loss": float(best["train/policy_loss"]),
        "validation_policy_loss": float(best["validation/policy_loss"]),
        "train_loss": float(best["train/loss"]),
        "validation_count_loss": float(best["validation/count_loss"]),
        "validation_single_accuracy": float(best["validation/single_action_accuracy"]),
        "validation_multi_exact": float(best["validation/multi_action_exact_rate"]),
        "validation_count_accuracy": float(best["validation/selection_count_accuracy"]),
        "validation_context_macro_exact": statistics.mean(exact_values),
        "validation_contexts": contexts,
        "validation_last5_std": statistics.pstdev(
            float(row["validation/exact_action_rate"]) for row in last_five
        ),
        "validation_last5_slope": _slope(
            [float(row["validation/exact_action_rate"]) for row in last_five]
        ),
        "train_last5_slope": _slope(
            [float(row["train/exact_action_rate"]) for row in last_five]
        ),
        "curves": [
            {
                "epoch": int(row["step"]),
                "train_exact": float(row["train/exact_action_rate"]),
                "validation_exact": float(row["validation/exact_action_rate"]),
                "train_policy_loss": float(row["train/policy_loss"]),
                "validation_policy_loss": float(row["validation/policy_loss"]),
            }
            for row in metrics
        ],
    }


def _can_launch(
    comparison: dict[str, Any],
    spec: TrialSpec,
    *,
    mandatory: bool = False,
) -> bool:
    predicted = _predicted_seconds(comparison, spec)
    allowed = _training_seconds(comparison) + predicted <= STOP_LAUNCH_SECONDS
    if not allowed and mandatory:
        print(
            f"[budget] mandatory {spec.version} cannot launch: "
            f"used={_training_seconds(comparison):.1f}s predicted={predicted:.1f}s",
            flush=True,
        )
    return allowed


def _predicted_seconds(comparison: dict[str, Any], spec: TrialSpec) -> float:
    matching = [
        max(
            float(result["training"]["wall_seconds"]),
            float(result["training"].get("epoch_median_seconds", 0.0)) * spec.epochs,
        )
        for result in comparison.get("trials", [])
        if result.get("status") == "completed"
        and result["spec"]["architecture"] == spec.architecture
    ]
    estimate = max(matching, default=INITIAL_ESTIMATES[spec.architecture])
    return estimate * 1.15


def _run_logged_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    *,
    environment: dict[str, str] | None = None,
    heartbeat_metrics: Path | None = None,
) -> tuple[float, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_heartbeat = started
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            now = time.monotonic()
            if now - last_heartbeat >= 30:
                epochs = len(_read_jsonl(heartbeat_metrics)) if heartbeat_metrics else None
                suffix = f" epochs={epochs}" if epochs is not None else ""
                print(
                    f"[process] elapsed={now - started:.0f}s{suffix}: {' '.join(command[:4])}",
                    flush=True,
                )
                last_heartbeat = now
            time.sleep(2)
    return time.monotonic() - started, int(process.returncode or 0)


def _freeze_campaign(
    repository_root: Path,
    experiment_root: Path,
    dataset: Path,
    source_package: Path,
    design_document: Path,
) -> None:
    manifest_path = experiment_root / "search_manifest.json"
    if manifest_path.is_file():
        current = _load_json(manifest_path)
        if current["dataset"]["sha256"] != _sha256(dataset):
            raise RuntimeError("frozen dataset changed after campaign initialization")
        metadata_path = dataset.with_suffix(dataset.suffix + ".card_metadata.json")
        summary_path = dataset.with_suffix(dataset.suffix + ".summary.json")
        if current["dataset"]["metadata_sha256"] != _sha256(metadata_path):
            raise RuntimeError("frozen card metadata changed after campaign initialization")
        if current["dataset"]["summary_sha256"] != _sha256(summary_path):
            raise RuntimeError("frozen dataset summary changed after campaign initialization")
        if current["source"]["combined_sha256"] != _source_hash(repository_root)[0]:
            raise RuntimeError("frozen model/training source changed after campaign initialization")
        if current["design_document"]["sha256"] != _sha256(design_document):
            raise RuntimeError("frozen campaign design changed after campaign initialization")
        if current["source_package"]["deck_sha256"] != _sha256(source_package / "deck.csv"):
            raise RuntimeError("frozen source package deck changed after campaign initialization")
        current_cg = compute_cg_manifest(source_package / "cg")
        if current["source_package"]["cg_tree_hash"] != current_cg["tree_hash"]:
            raise RuntimeError("frozen source package cg runtime changed after initialization")
        return

    metadata_path = dataset.with_suffix(dataset.suffix + ".card_metadata.json")
    summary_path = dataset.with_suffix(dataset.suffix + ".summary.json")
    data_manifest = repository_root / "rl_runs/0003-yushin_ito_exact_bc_v2/data_manifest.json"
    data_audit = repository_root / "rl_runs/0003-yushin_ito_exact_bc_v2/audit.json"
    source_hash, source_files = _source_hash(repository_root)
    git_status = _git(repository_root, "status", "--porcelain")
    source_patch = _git(
        repository_root,
        "diff",
        "--",
        "train/alakazam_bc_rl",
        "train/alakazam_bc_rl/training/train_full_action_bc.py",
        "train/alakazam_bc_rl/training/build_full_action_candidate.py",
        "train/kaggle_bc_top20/training/bc_capacity_search.py",
        "train/kaggle_bc_top20/training/render_bc_capacity_search.py",
        "rl_environment",
        "evaluation/cli.py",
        "evaluation/runner",
        "evaluation/metrics",
        "evaluation/packages",
        "evaluation/runtime",
        "evaluation/configs/opponents.json",
    )
    (experiment_root / "source_revision.patch").write_text(source_patch, encoding="utf-8")
    source_snapshot = {
        path: (repository_root / path).read_text(encoding="utf-8") for path in source_files
    }
    _write_json(experiment_root / "source_snapshot.json", source_snapshot)
    cg_manifest = compute_cg_manifest(source_package / "cg")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        ),
    }
    manifest = {
        "schema_version": "ptcg_bc_capacity_search_v1",
        "experiment_id": experiment_root.name,
        "created_at": _timestamp(),
        "design_document": {
            "path": str(design_document.relative_to(repository_root)),
            "sha256": _sha256(design_document),
            "revision": "210-minute-v1",
        },
        "budget": {
            "training_seconds": TRAINING_BUDGET_SECONDS,
            "stop_launch_seconds": STOP_LAUNCH_SECONDS,
            "single_gpu_serial": True,
        },
        "dataset": {
            "path": str(dataset),
            "bytes": dataset.stat().st_size,
            "sha256": _sha256(dataset),
            "metadata_path": str(metadata_path),
            "metadata_bytes": metadata_path.stat().st_size,
            "metadata_sha256": _sha256(metadata_path),
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "summary": _load_json(summary_path),
            "source_experiment": "0003-yushin_ito_exact_bc_v2",
            "expert_team": "Yushin Ito",
            "submission_id": 54773249,
        },
        "evidence": {
            "data_manifest": str(data_manifest.relative_to(repository_root)),
            "data_manifest_sha256": _sha256(data_manifest),
            "data_audit": str(data_audit.relative_to(repository_root)),
            "data_audit_sha256": _sha256(data_audit),
        },
        "source": {
            "git_commit": _git(repository_root, "rev-parse", "HEAD").strip(),
            "git_status_porcelain": git_status,
            "source_revision_patch": "source_revision.patch",
            "source_snapshot": "source_snapshot.json",
            "source_snapshot_sha256": _sha256(experiment_root / "source_snapshot.json"),
            "combined_sha256": source_hash,
            "files": source_files,
        },
        "environment": environment,
        "protocol": {
            "epochs": 20,
            "batch_size": 256,
            "optimizer": "AdamW",
            "heads": 4,
            "main_seed": 7,
            "confirmation_seed": 17,
            "feature_schema": "ptcg_features_universal",
            "action_contract": "full_action_set_v1",
            "metric_profile": "auto_iteration_v8_setup_relay",
            "metric_profile_revision": 2,
            "opponents": 18,
            "games_per_opponent": 10,
            "search_uses_test_or_evaluation": False,
        },
        "source_package": {
            "path": str(source_package),
            "deck_sha256": _sha256(source_package / "deck.csv"),
            "cg_tree_hash": cg_manifest["tree_hash"],
        },
    }
    _write_json(manifest_path, manifest)


def _source_hash(repository_root: Path) -> tuple[str, dict[str, str]]:
    roots = [
        repository_root / "train/alakazam_bc_rl",
        repository_root / "rl_environment",
        repository_root / "train/alakazam_bc_rl/training/train_full_action_bc.py",
        repository_root / "train/alakazam_bc_rl/training/build_full_action_candidate.py",
        repository_root / "train/kaggle_bc_top20/training/bc_capacity_search.py",
        repository_root / "train/kaggle_bc_top20/training/render_bc_capacity_search.py",
        repository_root / "evaluation/cli.py",
        repository_root / "evaluation/runner",
        repository_root / "evaluation/metrics",
        repository_root / "evaluation/packages",
        repository_root / "evaluation/runtime",
        repository_root / "evaluation/configs/opponents.json",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(
                path
                for path in root.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    hashes = {
        str(path.relative_to(repository_root)): _sha256(path)
        for path in sorted(set(files))
    }
    digest = hashlib.sha256()
    for path, file_hash in hashes.items():
        digest.update(f"{path}:{file_hash}\n".encode("utf-8"))
    return digest.hexdigest(), hashes


def _update_experiment_manifest(
    experiment_root: Path,
    comparison: dict[str, Any],
) -> None:
    path = experiment_root / "manifest.json"
    manifest = _load_json(path)
    manifest.update(
        {
            "status": comparison["status"],
            "completed_at": comparison.get("completed_at"),
            "campaign": "bc_capacity_search_210m",
            "training_seconds": comparison["training_seconds"],
            "trial_count": len(comparison["trials"]),
            "completed_trials": sum(
                result.get("status") == "completed" for result in comparison["trials"]
            ),
            "selection_status": "waiting_for_user",
        }
    )
    _write_json(path, manifest)


def _write_decisions(experiment_root: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# BC Capacity Search Decisions",
        "",
        "Search decisions below use train/validation evidence only. Test and official-engine ",
        "evaluation results were recorded after each trial but never fed into branching.",
        "",
    ]
    lines.extend(f"- {decision}" for decision in comparison.get("decisions", []))
    lines.extend(
        (
            "",
            f"- Training wall time: {comparison['training_seconds']:.1f} seconds.",
            f"- Final status: `{comparison['status']}`.",
            "- No model was promoted or selected; the campaign is waiting for the user.",
            "",
        )
    )
    (experiment_root / "decisions.md").write_text("\n".join(lines), encoding="utf-8")


def _new_comparison(experiment_id: str) -> dict[str, Any]:
    return {
        "schema_version": "ptcg_bc_capacity_comparison_v1",
        "experiment_id": experiment_id,
        "status": "running",
        "created_at": _timestamp(),
        "training_budget_seconds": TRAINING_BUDGET_SECONDS,
        "stop_launch_seconds": STOP_LAUNCH_SECONDS,
        "training_seconds": 0.0,
        "search_control_inputs": ["train", "validation"],
        "test_or_evaluation_used_for_branching": False,
        "trials": [],
        "decisions": [],
    }


def _results_for_phase(comparison: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    return [result for result in comparison["trials"] if result["spec"]["phase"] == phase]


def _training_seconds(comparison: dict[str, Any]) -> float:
    return sum(
        float(result.get("training", {}).get("wall_seconds", 0.0))
        for result in comparison.get("trials", [])
    )


def _runtime_environment(cxx_runtime_lib: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{cxx_runtime_lib}:{existing}" if existing else str(cxx_runtime_lib)
    )
    return environment


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return value != value or value in (float("inf"), float("-inf"))
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _slope(values: list[float]) -> float:
    return (values[-1] - values[0]) / max(1, len(values) - 1)


def _offline(result: dict[str, Any]) -> dict[str, Any]:
    return result["offline"]


def _lr_tag(value: float) -> str:
    return {1e-4: "1e4", 3e-4: "3e4", 5e-4: "5e4"}[value]


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _git(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("rl_runs/dataset/0003-yushin_ito_exact_bc_v2/dataset.jsonl"),
    )
    parser.add_argument("--source-package", type=Path, default=Path("work/alakazam_bc_v1"))
    parser.add_argument(
        "--design-document",
        type=Path,
        default=Path(
            "docs/superpowers/specs/2026-07-23-bc-capacity-search-210m-design.md"
        ),
    )
    parser.add_argument(
        "--cxx-runtime-lib",
        type=Path,
        default=Path("/home/dragon_bra/.local/ptcg-cxx-runtime/lib"),
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[3]
    result = run_campaign(
        repository_root,
        args.experiment_root,
        args.dataset,
        args.source_package,
        args.design_document,
        args.cxx_runtime_lib,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
