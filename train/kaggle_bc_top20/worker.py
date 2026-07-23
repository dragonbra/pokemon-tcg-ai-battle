"""Run one frozen single-expert BC job inside a Kaggle Notebook.

The worker reuses the repository's canonical replay downloader, dataset builder,
dataset audit, trainer, and submission builder.  It only owns orchestration and
portable output assembly.  One invocation handles exactly one expert policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


INPUT_SCHEMA = "ptcg_kaggle_bc_input_v1"
JOB_SCHEMA = "ptcg_kaggle_single_expert_job_v1"
RESULT_SCHEMA = "ptcg_kaggle_single_expert_result_v1"
PREPARE_RESULT_SCHEMA = "ptcg_kaggle_single_expert_prepare_result_v1"
VALIDATION_EXACT_GATE = 0.75
VALIDATION_MULTI_GATE = 0.65
VALIDATION_COUNT_GATE = 0.98
ATTEMPT_TIE_WINDOW = 0.003


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    print(f"START {log_path.stem}: {shlex.join(command)}", flush=True)
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(cwd) if not current_pythonpath else f"{cwd}{os.pathsep}{current_pythonpath}"
    )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    output_lines: list[str] = []
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)
    return_code = process.wait()
    stdout = "".join(output_lines)
    result = subprocess.CompletedProcess(
        args=command,
        returncode=return_code,
        stdout=stdout,
        stderr="",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"command: {shlex.join(command)}",
                f"exit_code: {result.returncode}",
                f"wall_seconds: {time.monotonic() - started:.3f}",
                "",
                "stdout:",
                result.stdout,
                "stderr:",
                "stderr was merged into the live stdout stream",
            ]
        ),
        encoding="utf-8",
    )
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); inspect {log_path}")
    print(
        f"DONE {log_path.stem}: exit={result.returncode} "
        f"seconds={time.monotonic() - started:.1f}",
        flush=True,
    )
    return result


def _summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_epoch_metrics") or {}
    test = summary.get("final_test") or {}
    return {
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


def gate_from_summary(
    summary: dict[str, Any],
    *,
    attempt_id: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build the same validation-first gate used by the local Top-20 campaign."""
    metrics = _summary_metrics(summary)
    validation_checks = {
        "validation_exact_action_rate": (
            metrics["validation_exact_action_rate"] >= VALIDATION_EXACT_GATE
        ),
        "validation_multi_action_exact_rate": (
            metrics["validation_multi_action_exact_rate"] >= VALIDATION_MULTI_GATE
        ),
        "validation_selection_count_accuracy": (
            metrics["validation_selection_count_accuracy"] >= VALIDATION_COUNT_GATE
        ),
        "validation_legal_action_rate": metrics["validation_legal_action_rate"] == 1.0,
    }
    test_evaluated = metrics["test_legal_action_rate"] is not None
    checks: dict[str, bool | None] = {
        **validation_checks,
        "test_legal_action_rate": (
            metrics["test_legal_action_rate"] == 1.0 if test_evaluated else None
        ),
    }
    runtime = summary.get("runtime") or {}
    return {
        "schema_version": "ptcg_single_expert_bc_gate_v2",
        "attempt_id": attempt_id,
        "decided_at": _timestamp(),
        "selection_inputs": ["train", "validation"],
        "test_used_for_lr_selection": False,
        "metrics": metrics,
        "checks": checks,
        "validation_gate_passed": all(validation_checks.values()),
        "test_evaluated": test_evaluated,
        "offline_gate_passed": test_evaluated and all(value is True for value in checks.values()),
        "best_epoch": int(summary["best_epoch"]),
        "checkpoint": str(summary["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha256,
        "wall_seconds": float(runtime.get("train_function_seconds", 0.0)),
        "peak_gpu_memory_bytes": int(runtime.get("peak_gpu_memory_bytes", 0)),
        "parameter_count": int(runtime.get("parameter_count", 0)),
    }


def choose_rescue_side(
    gate: dict[str, Any],
    curve: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Choose the first precommitted rescue LR without test/evaluation evidence."""
    metrics = gate["metrics"]
    gap = float(metrics["train_exact_action_rate"]) - float(
        metrics["validation_exact_action_rate"]
    )
    validation = [float(row["validation/exact_action_rate"]) for row in curve]
    train = [float(row["train/exact_action_rate"]) for row in curve]
    late_drop = bool(validation) and max(validation) - validation[-1] > 0.01
    tail = validation[-5:]
    tail_std = statistics.pstdev(tail) if len(tail) >= 2 else 0.0
    oscillation = len(validation) >= 5 and tail_std > 0.004
    improving = (
        len(validation) >= 3
        and validation[-1] > validation[-3]
        and train[-1] > train[-3]
    )
    if gap > 0.12 or late_drop or oscillation:
        side = "lower"
        reason = "overfit gap, late validation drop, or oscillation"
    elif float(metrics["train_exact_action_rate"]) < 0.75 and improving:
        side = "upper"
        reason = "train and validation are both low while the last three epochs improve"
    else:
        side = "lower"
        reason = "evidence is ambiguous; protocol defaults to lower LR"
    return side, {
        "train_validation_gap": gap,
        "validation_late_drop": late_drop,
        "validation_last5_std": tail_std,
        "last3_train_validation_improving": improving,
        "reason": reason,
    }


def select_attempt(gates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select one attempt from validation evidence only."""
    attempts = list(gates)
    if not attempts:
        raise ValueError("cannot select from an empty attempt list")
    eligible = [gate for gate in attempts if gate["validation_gate_passed"]] or attempts
    highest = max(float(gate["metrics"]["validation_exact_action_rate"]) for gate in eligible)
    near = [
        gate
        for gate in eligible
        if highest - float(gate["metrics"]["validation_exact_action_rate"])
        < ATTEMPT_TIE_WINDOW
    ]
    selected = min(
        near,
        key=lambda gate: (
            float(gate["metrics"]["validation_policy_loss"]),
            -float(gate["metrics"]["validation_exact_action_rate"]),
            str(gate["attempt_id"]),
        ),
    )
    return {
        "schema_version": "ptcg_single_expert_attempt_selection_v1",
        "selected_at": _timestamp(),
        "selection_inputs": ["train", "validation"],
        "test_or_evaluation_used": False,
        "tie_window": ATTEMPT_TIE_WINDOW,
        "passing_attempt_preferred": any(gate["validation_gate_passed"] for gate in attempts),
        "attempts": attempts,
        "selected_attempt_id": selected["attempt_id"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "validation_gate_passed": selected["validation_gate_passed"],
    }


def _load_curve(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _training_command(
    *,
    repo_root: Path,
    dataset: Path,
    attempt_root: Path,
    shared: dict[str, Any],
    learning_rate: float,
    device: str,
    storage_path: Path,
    min_free_gib: float,
) -> list[str]:
    return [
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
        device,
        "--storage-path",
        str(storage_path),
        "--min-free-gib",
        str(min_free_gib),
        "--skip-test",
    ]


def _train_attempt(
    *,
    repo_root: Path,
    log_root: Path,
    experiment_root: Path,
    dataset: Path,
    attempt_id: str,
    learning_rate: float,
    shared: dict[str, Any],
    device: str,
    storage_path: Path,
    min_free_gib: float,
) -> dict[str, Any]:
    attempt_root = experiment_root / attempt_id
    if attempt_root.exists():
        raise FileExistsError(f"attempt directory is already used: {attempt_root}")
    _run_command(
        _training_command(
            repo_root=repo_root,
            dataset=dataset,
            attempt_root=attempt_root,
            shared=shared,
            learning_rate=learning_rate,
            device=device,
            storage_path=storage_path,
            min_free_gib=min_free_gib,
        ),
        cwd=repo_root,
        log_path=log_root / "logs" / f"{attempt_id}_train.log",
    )
    summary_path = attempt_root / "training_summary.json"
    summary = _read_json(summary_path)
    checkpoint = Path(str(summary["checkpoint"]))
    gate = gate_from_summary(
        summary,
        attempt_id=attempt_id,
        checkpoint_sha256=_sha256(checkpoint),
    )
    _write_json(attempt_root / "gate_decision.json", gate)
    return gate


def _evaluate_frozen_test(
    *,
    repo_root: Path,
    log_root: Path,
    experiment_root: Path,
    dataset: Path,
    selection: dict[str, Any],
    shared: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    attempt_id = str(selection["selected_attempt_id"])
    checkpoint = Path(str(selection["selected_checkpoint"]))
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
        device,
    ]
    result = _run_command(
        command,
        cwd=repo_root,
        log_path=log_root / "logs" / "frozen_test.log",
    )
    test_metrics = json.loads(result.stdout)
    summary_path = experiment_root / attempt_id / "training_summary.json"
    summary = _read_json(summary_path)
    summary["final_test"] = test_metrics
    summary["test_evaluation_protocol"] = "after validation-only attempt selection"
    _write_json(summary_path, summary)
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != selection["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed before frozen test evaluation")
    gate = gate_from_summary(
        summary,
        attempt_id=attempt_id,
        checkpoint_sha256=checkpoint_hash,
    )
    _write_json(experiment_root / attempt_id / "gate_decision.json", gate)
    _write_json(
        log_root / "frozen_test_evaluation.json",
        {
            "evaluated_at": _timestamp(),
            "attempt_id": attempt_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "metrics": test_metrics,
        },
    )
    return gate


def _load_input(input_root: Path, job_order: int) -> tuple[dict[str, Any], dict[str, Any]]:
    input_manifest = _read_json(input_root / "ptcg_kaggle_bc_input.json")
    if input_manifest.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"unsupported input schema: {input_manifest.get('schema_version')}")
    jobs = _read_json(input_root / str(input_manifest["jobs_file"]))
    rows = jobs.get("jobs") or []
    matches = [row for row in rows if int(row["selected_roster_order"]) == job_order]
    if len(matches) != 1:
        raise ValueError(f"job order {job_order} resolved to {len(matches)} rows")
    job = matches[0]
    if job.get("schema_version") != JOB_SCHEMA:
        raise ValueError(f"unsupported job schema: {job.get('schema_version')}")
    return input_manifest, job


def _verify_bundle(input_root: Path, input_manifest: dict[str, Any]) -> None:
    hashes = input_manifest.get("files_sha256") or {}
    if not hashes:
        raise ValueError("input manifest has no file hashes")
    for relative, expected in hashes.items():
        path = input_root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"frozen input file is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen input hash mismatch: {relative}: {actual} != {expected}")


def _preflight(
    repo_root: Path,
    device: str,
    *,
    require_kaggle_api: bool,
) -> dict[str, Any]:
    if sys.version_info < (3, 11):
        raise RuntimeError(f"Python 3.11+ is required, found {sys.version}")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Kaggle accelerator is not enabled: torch.cuda.is_available() is false")
    if require_kaggle_api:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            KaggleApi().authenticate()
        except (Exception, SystemExit) as exc:
            raise RuntimeError(
                "Kaggle API authentication failed; add KAGGLE_API_TOKEN or legacy "
                "KAGGLE_USERNAME/KAGGLE_KEY as private Notebook secrets"
            ) from exc
    git_commit = None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        git_commit = result.stdout.strip()
    return {
        "captured_at": _timestamp(),
        "python": sys.version,
        "platform": platform.platform(),
        "repo_root": str(repo_root),
        "git_commit": git_commit,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "kaggle_api_authentication_required": require_kaggle_api,
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }


def _discover_mounted_replay_datasets(root: Path | None) -> list[Path]:
    if root is None:
        return []
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"mounted replay input root is missing: {root}")
    candidates = [
        *root.glob("pokemon-tcg-ai-battle-episodes-*"),
        *root.glob("datasets/*/pokemon-tcg-ai-battle-episodes-*"),
        *root.glob("datasets/*/pokemon-tcg-ai-battle-episodes-*/versions/*"),
    ]
    datasets = sorted(
        {
            path.resolve()
            for path in candidates
            if path.is_dir() and next(path.glob("[0-9]*.json"), None) is not None
        }
    )
    if not datasets:
        raise RuntimeError(f"no mounted daily Episode datasets found below {root}")
    return datasets


def discover_prebuilt_prepare_root(root: Path) -> Path:
    """Resolve exactly one completed CPU prepare output below a Kernel mount."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"prebuilt Kernel input root is missing: {root}")
    results = sorted(path for path in root.rglob("PREPARE_RESULT.json") if path.is_file())
    if len(results) != 1:
        raise RuntimeError(
            f"expected one PREPARE_RESULT.json below {root}, found {len(results)}"
        )
    return results[0].parent


def _tar_candidate(candidate: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(candidate.rglob("*")):
            archive.add(path, arcname=path.relative_to(candidate).as_posix(), recursive=False)


def _remove_python_caches(root: Path) -> None:
    for directory in sorted(root.rglob("__pycache__"), reverse=True):
        if directory.is_dir():
            shutil.rmtree(directory)
    for suffix in ("*.pyc", "*.pyo"):
        for path in root.rglob(suffix):
            if path.is_file():
                path.unlink()


def _copy_dataset_evidence(dataset: Path, evidence: Path) -> None:
    suffixes = (
        ".summary.json",
        ".data_manifest.json",
        ".card_metadata.json",
    )
    for suffix in suffixes:
        source = dataset.with_suffix(dataset.suffix + suffix)
        if source.is_file():
            shutil.copy2(source, evidence / source.name)


def _prepared_file_hashes(run_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("dataset", "evidence"):
        root = run_root / name
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            hashes[path.relative_to(run_root).as_posix()] = _sha256(path)
    if not hashes:
        raise ValueError("prepared output has no portable files")
    return hashes


def _load_prebuilt_dataset(
    prebuilt_root: Path,
    *,
    expected_job: dict[str, Any],
    corpus_mode: str = "exact_submission",
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Verify and open one immutable CPU prepare result for GPU training."""
    prebuilt_root = prebuilt_root.resolve()
    result_path = prebuilt_root / "PREPARE_RESULT.json"
    result = _read_json(result_path)
    if result.get("schema_version") != PREPARE_RESULT_SCHEMA:
        raise ValueError(f"unsupported prepare result schema: {result.get('schema_version')}")
    if result.get("status") != "dataset_ready":
        raise ValueError(f"prepare result is not ready: {result.get('status')}")
    if result.get("job") != expected_job:
        raise ValueError("prebuilt dataset job identity does not match frozen input job")

    hashes = result.get("files_sha256") or {}
    if not hashes:
        raise ValueError("prepare result has no file hashes")
    root_resolved = prebuilt_root.resolve()
    for relative, expected_hash in hashes.items():
        path = (prebuilt_root / str(relative)).resolve()
        if root_resolved not in (path, *path.parents):
            raise ValueError(f"unsafe prepared file path: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"prepared file is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"prepared file hash mismatch: {relative}: {actual_hash} != {expected_hash}"
            )

    dataset_info = result.get("dataset") or {}
    dataset = (prebuilt_root / str(dataset_info.get("relative_path", ""))).resolve()
    if root_resolved not in (dataset, *dataset.parents) or not dataset.is_file():
        raise ValueError("prepared dataset path is invalid")
    dataset_hash = _sha256(dataset)
    if dataset_hash != dataset_info.get("sha256"):
        raise ValueError("prepared dataset hash does not match PREPARE_RESULT.json")
    if dataset.stat().st_size != int(dataset_info.get("bytes", -1)):
        raise ValueError("prepared dataset byte size does not match PREPARE_RESULT.json")

    audit = dataset_info.get("audit") or {}
    if audit.get("status") != "passed":
        raise ValueError("prepared dataset audit did not pass")
    if audit.get("dataset_sha256") != dataset_hash:
        raise ValueError("prepared dataset audit hash does not match dataset")
    data_manifest = _read_json(dataset.with_suffix(dataset.suffix + ".data_manifest.json"))
    if data_manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("prepared data manifest hash does not match dataset")
    if (
        corpus_mode == "exact_submission"
        and data_manifest.get("source_identity") != expected_job.get("source_identity")
    ):
        raise ValueError("prepared data manifest source identity does not match frozen job")
    return dataset, audit, result


def _collect_training_artifacts(
    *,
    repo_root: Path,
    run_root: Path,
    experiment_id: str,
    selected: Path | None,
) -> Path | None:
    """Move canonical cloud run artifacts into one portable Kernel output tree."""
    artifact_root = run_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    destination: Path | None = None
    if selected is not None and selected.is_file():
        destination = artifact_root / "selected_checkpoint.pt"
        shutil.copy2(selected, destination)
    sources = (
        (repo_root / "rl_runs" / experiment_id, run_root / "training"),
        (
            repo_root / "rl_runs/tensorboard" / experiment_id,
            run_root / "tensorboard",
        ),
    )
    for source, target in sources:
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    checkpoints = repo_root / "rl_runs/checkpoint" / experiment_id
    if checkpoints.exists():
        if selected is None:
            shutil.move(str(checkpoints), str(run_root / "failed_checkpoints"))
        else:
            shutil.rmtree(checkpoints)
    return destination


def run_job(
    *,
    input_root: Path,
    output_root: Path,
    job_order: int,
    device: str = "cuda",
    retention: str = "dataset",
    min_free_gib: float = 5.0,
    mounted_replay_root: Path | None = None,
    mode: str = "full",
    prebuilt_root: Path | None = None,
    corpus_mode: str = "exact_submission",
) -> dict[str, Any]:
    if mode not in {"full", "prepare", "train"}:
        raise ValueError(f"unsupported worker mode: {mode}")
    if mode == "train" and prebuilt_root is None:
        raise ValueError("train mode requires --prebuilt-root")
    if mode != "train" and prebuilt_root is not None:
        raise ValueError("--prebuilt-root is only valid in train mode")
    if mode == "prepare" and device != "cpu":
        raise ValueError("prepare mode must run on CPU")
    if corpus_mode not in {"exact_submission", "daily_team_winners"}:
        raise ValueError(f"unsupported corpus mode: {corpus_mode}")
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    input_manifest, job = _load_input(input_root, job_order)
    package_name = str(job["package_name"])
    run_root = output_root / f"{job_order:02d}-{package_name}"
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {run_root}")
    run_root.mkdir(parents=True)
    result_path = run_root / (
        "PREPARE_RESULT.json" if mode == "prepare" else "RESULT.json"
    )
    raw_root = run_root / "raw_replays"
    dataset = run_root / "dataset" / "dataset.jsonl"
    evidence = run_root / "evidence"
    evidence.mkdir()
    started = time.monotonic()
    repo_root: Path | None = None
    experiment_root: Path | None = None
    selected_checkpoint: Path | None = None
    try:
        _verify_bundle(input_root, input_manifest)
        repo_root = (input_root / str(input_manifest["repo_dir"])).resolve()
        experiment_id = str(job["experiment_id"])
        experiment_root = repo_root / "rl_runs" / experiment_id
        if experiment_root.exists():
            raise FileExistsError(f"cloud experiment directory is already used: {experiment_root}")
        shared_path = (input_root / str(input_manifest["shared_model_config"])).resolve()
        cg_source = (input_root / str(input_manifest["cg_source"])).resolve()
        source_manifest = (input_root / str(job["source_manifest"])).resolve()
        deck = (input_root / str(job["deck"])).resolve()
        shared = _read_json(shared_path)
        if shared.get("read_only") is not True:
            raise ValueError("shared model config is not frozen read-only")
        if mode in {"full", "prepare"} and corpus_mode == "daily_team_winners":
            if mounted_replay_root is None or not mounted_replay_root.is_dir():
                raise FileNotFoundError("daily winner mounted replay root is missing")
            # Mirror the reference Notebook's Path('/kaggle/input').rglob('*.json').
            # Kaggle has used both flat and version-nested Dataset mount layouts.
            mounted_replay_datasets = [mounted_replay_root.resolve()]
        else:
            mounted_replay_datasets = (
                _discover_mounted_replay_datasets(mounted_replay_root)
                if mode in {"full", "prepare"}
                else []
            )
        environment = _preflight(
            repo_root,
            device,
            require_kaggle_api=(
                mode in {"full", "prepare"} and corpus_mode == "exact_submission"
            ),
        )
        environment["worker_mode"] = mode
        environment["corpus_mode"] = corpus_mode
        environment["mounted_replay_datasets"] = [str(path) for path in mounted_replay_datasets]
        _write_json(evidence / "environment.json", environment)
        shutil.copy2(source_manifest, evidence / "source_manifest.json")
        shutil.copy2(deck, evidence / "source_deck.csv")
        shutil.copy2(shared_path, evidence / "shared_model_config.json")
        prepare_result: dict[str, Any] | None = None
        if mode == "train":
            assert prebuilt_root is not None
            dataset, audit, prepare_result = _load_prebuilt_dataset(
                prebuilt_root,
                expected_job=job,
                corpus_mode=corpus_mode,
            )
            environment["prebuilt_root"] = str(prebuilt_root.resolve())
            environment["prepare_result_sha256"] = _sha256(
                prebuilt_root.resolve() / "PREPARE_RESULT.json"
            )
            _write_json(evidence / "environment.json", environment)
            shutil.copy2(
                prebuilt_root.resolve() / "PREPARE_RESULT.json",
                evidence / "prebuilt_prepare_result.json",
            )
        else:
            dataset.parent.mkdir(parents=True)
            if corpus_mode == "daily_team_winners":
                dataset_command = [
                    sys.executable,
                    "-m",
                    "train.kaggle_bc_top20.training.build_daily_winner_bc_dataset",
                    "--output",
                    str(dataset),
                    "--teacher-team",
                    str(job["team_name"]),
                    "--storage-path",
                    str(output_root),
                    "--min-free-gib",
                    str(min_free_gib),
                ]
                for mounted_dataset in mounted_replay_datasets:
                    dataset_command.extend(["--input-root", str(mounted_dataset)])
            else:
                download_command = [
                    sys.executable,
                    "-m",
                    "train.kaggle_bc_top20.training.download_expert_replays",
                    "--source-manifest",
                    str(source_manifest),
                    "--output",
                    str(raw_root),
                    "--workers",
                    "2",
                    "--retries",
                    "12",
                    "--request-interval",
                    "1.0",
                    "--network-timeout",
                    "90.0",
                ]
                for mounted_dataset in mounted_replay_datasets:
                    download_command.extend(["--seed-root", str(mounted_dataset)])
                _run_command(
                    download_command,
                    cwd=repo_root,
                    log_path=run_root / "logs" / "download.log",
                )
                raw_manifest = raw_root / "manifest.json"
                shutil.copy2(raw_manifest, evidence / "replay_manifest.json")
                dataset_command = [
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
                    "--storage-path",
                    str(output_root),
                    "--min-free-gib",
                    str(min_free_gib),
                ]
            _run_command(
                dataset_command,
                cwd=repo_root,
                log_path=run_root / "logs" / "dataset.log",
            )
            data_manifest = dataset.with_suffix(dataset.suffix + ".data_manifest.json")
            audit_path = evidence / "dataset_audit.json"
            audit_command = [
                    sys.executable,
                    "-m",
                    "train.kaggle_bc_top20.training.audit_kaggle_bc_dataset",
                    str(dataset),
                    "--manifest",
                    str(data_manifest),
                    "--output",
                    str(audit_path),
                ]
            if corpus_mode == "daily_team_winners":
                audit_command.append("--allow-multiple-experts")
            _run_command(
                audit_command,
                cwd=repo_root,
                log_path=run_root / "logs" / "audit.log",
            )
            audit = _read_json(audit_path)
            if audit.get("status") != "passed":
                raise RuntimeError("dataset audit did not pass")
            _copy_dataset_evidence(dataset, evidence)

            if mode == "prepare":
                prepared = {
                    "schema_version": PREPARE_RESULT_SCHEMA,
                    "status": "dataset_ready",
                    "completed_at": _timestamp(),
                    "wall_seconds": time.monotonic() - started,
                    "job": job,
                    "environment": environment,
                    "dataset": {
                        "relative_path": dataset.relative_to(run_root).as_posix(),
                        "sha256": _sha256(dataset),
                        "bytes": dataset.stat().st_size,
                        "audit": audit,
                    },
                    "files_sha256": _prepared_file_hashes(run_root),
                    "training_performed": False,
                    "official_game_evaluation_performed": False,
                }
                _write_json(result_path, prepared)
                return prepared

        gates: list[dict[str, Any]] = []
        v1 = _train_attempt(
            repo_root=repo_root,
            log_root=run_root,
            experiment_root=experiment_root,
            dataset=dataset,
            attempt_id="V1_shared_config",
            learning_rate=float(shared["base_learning_rate"]),
            shared=shared,
            device=device,
            storage_path=output_root,
            min_free_gib=min_free_gib,
        )
        gates.append(v1)
        if (
            not v1["validation_gate_passed"]
            and float(v1["metrics"]["validation_exact_action_rate"])
            < VALIDATION_EXACT_GATE
        ):
            curve = _load_curve(
                experiment_root / "V1_shared_config" / "training_metrics.jsonl"
            )
            side, rescue_evidence = choose_rescue_side(v1, curve)
            _write_json(
                evidence / "rescue_decision.json",
                {"side": side, **rescue_evidence},
            )
            v2 = _train_attempt(
                repo_root=repo_root,
                log_root=run_root,
                experiment_root=experiment_root,
                dataset=dataset,
                attempt_id=f"V2_lr_{side}",
                learning_rate=float(shared[f"{side}_rescue_learning_rate"]),
                shared=shared,
                device=device,
                storage_path=output_root,
                min_free_gib=min_free_gib,
            )
            gates.append(v2)
            if not v2["validation_gate_passed"]:
                other = "upper" if side == "lower" else "lower"
                gates.append(
                    _train_attempt(
                        repo_root=repo_root,
                        log_root=run_root,
                        experiment_root=experiment_root,
                        dataset=dataset,
                        attempt_id="V3_lr_other_side",
                        learning_rate=float(shared[f"{other}_rescue_learning_rate"]),
                        shared=shared,
                        device=device,
                        storage_path=output_root,
                        min_free_gib=min_free_gib,
                    )
                )

        selection = select_attempt(gates)
        _write_json(evidence / "attempt_selection.json", selection)
        dataset_summary = _read_json(dataset.with_suffix(dataset.suffix + ".summary.json"))
        has_test = bool((dataset_summary.get("records_by_split") or {}).get("test", 0))
        if has_test:
            selected_gate = _evaluate_frozen_test(
                repo_root=repo_root,
                log_root=run_root,
                experiment_root=experiment_root,
                dataset=dataset,
                selection=selection,
                shared=shared,
                device=device,
            )
        else:
            selected_gate = next(
                gate
                for gate in gates
                if gate["attempt_id"] == selection["selected_attempt_id"]
            )
        selection["offline_gate_passed"] = selected_gate["offline_gate_passed"]
        selection["test_evaluated_after_selection"] = has_test
        _write_json(evidence / "attempt_selection.json", selection)

        selected_checkpoint = Path(str(selection["selected_checkpoint"]))
        candidate = run_root / "candidate"
        package_deck = deck
        package_source_identity: Path | None = source_manifest
        if corpus_mode == "daily_team_winners":
            try:
                checkpoint_payload = torch.load(
                    selected_checkpoint, map_location="cpu", weights_only=False
                )
            except TypeError:
                checkpoint_payload = torch.load(selected_checkpoint, map_location="cpu")
            checkpoint_deck = [
                int(value)
                for value in ((checkpoint_payload.get("metadata") or {}).get("deck") or [])
            ]
            if len(checkpoint_deck) != 60:
                raise ValueError("daily winner checkpoint has no 60-card runtime deck")
            package_deck = evidence / "runtime_deck.csv"
            package_deck.write_text(
                "\n".join(str(value) for value in checkpoint_deck) + "\n",
                encoding="utf-8",
            )
            package_source_identity = None
        package_command = [
                sys.executable,
                "-m",
                "train.kaggle_bc_top20.training.build_full_action_submission",
                "--checkpoint",
                str(selected_checkpoint),
                "--output",
                str(candidate),
                "--deck",
                str(package_deck),
                "--cg-source",
                str(cg_source),
                "--shared-model-config",
                str(shared_path),
                "--experiment-id",
                str(job["experiment_id"]),
                "--name",
                package_name,
            ]
        if package_source_identity is not None:
            package_command.extend(["--source-identity", str(package_source_identity)])
        _run_command(
            package_command,
            cwd=repo_root,
            log_path=run_root / "logs" / "package.log",
        )
        validation = _run_command(
            [sys.executable, "-m", "evaluation", "validate", str(candidate)],
            cwd=repo_root,
            log_path=run_root / "logs" / "package_validation.log",
            check=False,
        )
        if validation.returncode:
            raise RuntimeError("candidate package failed repository structural validation")
        _remove_python_caches(candidate)

        export = run_root / "exports" / f"{package_name}-candidate.tar.gz"
        _tar_candidate(candidate, export)
        portable_checkpoint = _collect_training_artifacts(
            repo_root=repo_root,
            run_root=run_root,
            experiment_id=experiment_id,
            selected=selected_checkpoint,
        )
        if portable_checkpoint is None:
            raise RuntimeError("selected checkpoint was not collected")
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "candidate_ready",
            "completed_at": _timestamp(),
            "wall_seconds": time.monotonic() - started,
            "job": job,
            "environment": environment,
            "dataset": {
                "path": str(dataset),
                "sha256": _sha256(dataset),
                "bytes": dataset.stat().st_size,
                "retained_in_kernel_output": retention == "dataset" and mode != "train",
                "audit": audit,
                "corpus_mode": corpus_mode,
            },
            "prebuilt_prepare": (
                {
                    "root": str(prebuilt_root.resolve()),
                    "result_sha256": _sha256(
                        prebuilt_root.resolve() / "PREPARE_RESULT.json"
                    ),
                    "completed_at": prepare_result.get("completed_at"),
                }
                if prepare_result is not None and prebuilt_root is not None
                else None
            ),
            "selection": selection,
            "selected_gate": selected_gate,
            "offline_gate_passed": selected_gate["offline_gate_passed"],
            "package_validated": True,
            "official_game_evaluation_performed": False,
            "candidate": {
                "directory": str(candidate),
                "tree_sha256": _tree_hash(candidate),
                "archive": str(export),
                "archive_sha256": _sha256(export),
                "archive_bytes": export.stat().st_size,
            },
            "selected_checkpoint": {
                "path": str(portable_checkpoint),
                "sha256": _sha256(portable_checkpoint),
                "bytes": portable_checkpoint.stat().st_size,
            },
            "promotion_note": (
                "Package validity and offline imitation metrics are not strategy-strength "
                "claims; run the official engine evaluation locally before adding this "
                "candidate to the opponent catalog."
            ),
        }
        _write_json(result_path, result)
        if retention == "package" and mode != "train":
            shutil.rmtree(dataset.parent)
        return result
    except Exception as exc:
        if mode != "prepare" and repo_root is not None and experiment_root is not None:
            _collect_training_artifacts(
                repo_root=repo_root,
                run_root=run_root,
                experiment_id=experiment_root.name,
                selected=None,
            )
        failure = {
            "schema_version": (
                PREPARE_RESULT_SCHEMA if mode == "prepare" else RESULT_SCHEMA
            ),
            "status": "failed",
            "failed_at": _timestamp(),
            "wall_seconds": time.monotonic() - started,
            "job_order": job_order,
            "worker_mode": mode,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_json(result_path, failure)
        raise
    finally:
        if raw_root.exists():
            shutil.rmtree(raw_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job-order", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--retention", choices=("dataset", "package"), default="dataset")
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--mode", choices=("full", "prepare", "train"), default="full")
    parser.add_argument(
        "--prebuilt-root",
        type=Path,
        help="CPU prepare run directory containing PREPARE_RESULT.json",
    )
    parser.add_argument(
        "--mounted-replay-root",
        type=Path,
        help="parent of attached pokemon-tcg-ai-battle-episodes-* datasets",
    )
    parser.add_argument(
        "--corpus-mode",
        choices=("exact_submission", "daily_team_winners"),
        default="exact_submission",
    )
    args = parser.parse_args()
    result = run_job(
        input_root=args.input_root,
        output_root=args.output_root,
        job_order=args.job_order,
        device=args.device,
        retention=args.retention,
        min_free_gib=args.min_free_gib,
        mounted_replay_root=args.mounted_replay_root,
        mode=args.mode,
        prebuilt_root=args.prebuilt_root,
        corpus_mode=args.corpus_mode,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
