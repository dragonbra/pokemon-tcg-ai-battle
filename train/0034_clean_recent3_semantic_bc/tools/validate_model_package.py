"""Validate a downloaded 0034 cleaned-BC model package end to end."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED_DAYS = ["2026-08-01", "2026-08-02", "2026-08-03"]
EXPECTED_MODEL_CONFIG = {
    "d_model": 320,
    "heads": 8,
    "state_layers": 4,
    "option_layers": 3,
    "ffn_multiplier": 3,
    "dropout": 0.1,
    "max_card_id": 2048,
    "max_attack_id": 2048,
    "max_skill_id": 512,
    "max_options": 128,
    "max_action_steps": 64,
}
EXPECTED_PARAMETER_COUNT = 22_595_202
EXPECTED_METRICS = (
    "weighted_loss",
    "token_accuracy",
    "teacher_exact_action",
    "action_type_accuracy",
    "attach_target_token_accuracy",
    "attach_target_exact_action",
    "attack_action_token_accuracy",
    "attack_action_exact",
)
REQUIRED_FILES = (
    "best_model.pt",
    "last_model.pt",
    "model_contract.json",
    "official_public_prototypes_v1.json",
    "official_full_engine_prototypes_v2.json",
    "feature_audit.json",
    "cleaning_report.json",
    "training_report.json",
    "training_config.json",
    "dataset_reference.json",
    "model_package_manifest.json",
    "MODEL_USAGE.md",
    "load_model.py",
    "smoke_input.pt",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch fallback
        payload = torch.load(path, map_location="cpu")
    _require(isinstance(payload, dict), f"checkpoint is not a mapping: {path.name}")
    _require(
        set(payload) == {"schema_version", "state_dict", "metadata"},
        f"checkpoint payload drift: {path.name}",
    )
    _require(
        payload["schema_version"] == "0034_model_only_checkpoint_v1",
        f"checkpoint schema mismatch: {path.name}",
    )
    state = payload["state_dict"]
    _require(isinstance(state, dict) and state, f"checkpoint state is empty: {path.name}")
    _require(
        not any(str(name).startswith("module.") for name in state),
        f"DataParallel prefix leaked into checkpoint: {path.name}",
    )
    _require(
        all(isinstance(value, torch.Tensor) for value in state.values()),
        f"checkpoint state contains non-tensors: {path.name}",
    )
    _require(
        all(bool(torch.isfinite(value).all()) for value in state.values()),
        f"checkpoint state contains non-finite tensors: {path.name}",
    )
    metadata = payload["metadata"]
    _require(metadata.get("architecture") == "SemanticPolicy", "checkpoint architecture mismatch")
    _require(
        metadata.get("actor_schema") == "0034_clean_recent3_semantic_decision_v1",
        "checkpoint actor schema mismatch",
    )
    _require(metadata.get("parameter_count") == EXPECTED_PARAMETER_COUNT, "checkpoint parameter count mismatch")
    _require(metadata.get("model_config") == EXPECTED_MODEL_CONFIG, "checkpoint model config mismatch")
    _require(metadata.get("random_initialization") is True, "checkpoint is not random-initialized")
    _require(metadata.get("initialized_from_checkpoint") is None, "checkpoint has pretrained lineage")
    _require(metadata.get("winner_only") is True, "checkpoint is not winner-only")
    return payload


def _validate_epoch_metrics(metrics: dict[str, Any], *, decisions: int, label: str) -> None:
    _require(int(metrics.get("decisions", -1)) == decisions, f"{label} decision coverage mismatch")
    _require(int(metrics.get("tokens", 0)) > 0, f"{label} has no action tokens")
    for name in EXPECTED_METRICS:
        value = float(metrics.get(name, math.nan))
        _require(math.isfinite(value), f"{label} metric is non-finite: {name}")
        if name != "weighted_loss":
            _require(0.0 <= value <= 1.0, f"{label} accuracy is outside [0, 1]: {name}")


def _load_packaged_model(package_dir: Path, device: torch.device):
    loader_path = package_dir / "load_model.py"
    spec = importlib.util.spec_from_file_location("ptcg_0034_packaged_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load package helper: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_model(package_dir, device=device)


def validate_model_package(
    package_dir: Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    for name in REQUIRED_FILES:
        path = package_dir / name
        _require(path.is_file() and path.stat().st_size > 0, f"missing package file: {name}")

    manifest = _load_json(package_dir / "model_package_manifest.json")
    contract = _load_json(package_dir / "model_contract.json")
    config = _load_json(package_dir / "training_config.json")
    report = _load_json(package_dir / "training_report.json")
    dataset = _load_json(package_dir / "dataset_reference.json")
    cleaning = _load_json(package_dir / "cleaning_report.json")
    feature_audit = _load_json(package_dir / "feature_audit.json")

    _require(manifest.get("schema_version") == "0034_model_package_v1", "package manifest schema mismatch")
    _require(manifest.get("weights") == ["best_model.pt", "last_model.pt"], "package weight list mismatch")
    _require(manifest.get("smoke_input") == "smoke_input.pt", "package smoke input is undeclared")
    _require(contract.get("schema_version") == "0034_audited_model_contract_v1", "model contract schema mismatch")
    _require(contract.get("architecture") == "SemanticPolicy", "model contract architecture mismatch")
    _require(contract.get("actor_schema") == "0034_clean_recent3_semantic_decision_v1", "model contract actor schema mismatch")
    _require(contract.get("model_config") == EXPECTED_MODEL_CONFIG, "model contract config mismatch")
    _require(contract.get("parameter_count") == EXPECTED_PARAMETER_COUNT, "model contract parameter count mismatch")
    _require(contract.get("initialized_from_checkpoint") is None, "model contract has pretrained lineage")
    _require(contract.get("random_initialization") is True, "model contract is not random-initialized")
    _require(contract.get("winner_only") is True, "model contract is not winner-only")
    _require(contract.get("wandb") is None, "model contract contains W&B")
    _require(contract.get("checkpoint_state_dict") == "trainable_parameters_only", "checkpoint state contract mismatch")
    _require(contract.get("sample_weight_field") == "audit.cleaning.sample_weight", "sample-weight contract mismatch")
    device_ids = contract.get("data_parallel_device_ids")
    device_names = contract.get("gpu_device_names")
    _require(device_ids == [0, 1], "training did not use two DataParallel devices")
    _require(
        isinstance(device_names, list)
        and len(device_names) >= 2
        and all("T4" in str(name) for name in device_names[:2]),
        "training devices are not dual Nvidia T4",
    )

    _require(config.get("schema_version") == "0034_cleaned_bc_training_v1", "training config schema mismatch")
    _require(config.get("epoch_definition") == "one_epoch_is_full_merged_recent3_train", "epoch definition mismatch")
    _require(config.get("epochs_requested") == 3, "training did not request three epochs")
    _require(config.get("partition_count") == 3, "training did not use three partitions")
    _require(config.get("random_initialization") is True, "training config is not random-initialized")
    _require(config.get("initialized_from_checkpoint") is None, "training config has pretrained lineage")
    _require(config.get("wandb") is None, "training config contains W&B")
    _require(config.get("data_parallel_device_ids") == [0, 1], "training config is not dual-GPU")
    _require(config.get("gpu_device_names") == device_names, "training/contract GPU evidence differs")
    split_counts = config.get("dataset_split_counts", {})
    train_decisions = int(split_counts.get("train", 0))
    validation_decisions = int(split_counts.get("validation", 0))
    _require(train_decisions > 0 and validation_decisions > 0, "training split is empty")

    _require(report.get("schema_version") == "0034_cleaned_bc_training_report_v1", "training report schema mismatch")
    _require(report.get("state") == "complete", "training report is incomplete")
    _require(report.get("training_config") == config, "training config/report mismatch")
    _require(report.get("epochs_completed") == 3, "training did not finish three epochs")
    _require(report.get("full_train_passes_completed") == 3, "full train passes mismatch")
    _require(report.get("full_validation_passes_completed") == 3, "full validation passes mismatch")
    history = report.get("history", [])
    _require(isinstance(history, list) and len(history) == 3, "training history length mismatch")
    for expected_epoch, row in enumerate(history, 1):
        _require(row.get("epoch") == expected_epoch, "training epoch order mismatch")
        _validate_epoch_metrics(row.get("train", {}), decisions=train_decisions, label=f"epoch {expected_epoch} train")
        _validate_epoch_metrics(
            row.get("validation", {}),
            decisions=validation_decisions,
            label=f"epoch {expected_epoch} validation",
        )

    _require(dataset.get("winner_only") is True, "dataset reference is not winner-only")
    _require(dataset.get("dates") == EXPECTED_DAYS, "dataset dates mismatch")
    inventory = dataset.get("inventory", [])
    _require(isinstance(inventory, list) and len(inventory) == 3, "dataset inventory length mismatch")
    _require(
        sum(int(item["split_counts"]["train"]) for item in inventory) == train_decisions,
        "dataset inventory train count mismatch",
    )
    _require(
        sum(int(item["split_counts"]["validation"]) for item in inventory)
        == validation_decisions,
        "dataset inventory validation count mismatch",
    )
    _require(cleaning.get("status") == "passed", "packaged cleaning report did not pass")
    _require(cleaning.get("dates") == EXPECTED_DAYS, "packaged cleaning dates mismatch")
    _require(feature_audit.get("actor_schema") == contract["actor_schema"], "feature audit schema mismatch")

    checkpoints = {
        name: _load_checkpoint(package_dir / name)
        for name in ("best_model.pt", "last_model.pt")
    }
    best_artifact = report.get("best_artifact", {})
    _require(best_artifact.get("path") == "best_model.pt", "best artifact path mismatch")
    _require(best_artifact.get("sha256") == _sha256(package_dir / "best_model.pt"), "best artifact SHA256 mismatch")
    _require(best_artifact.get("bytes") == (package_dir / "best_model.pt").stat().st_size, "best artifact size mismatch")
    last_artifact = history[-1].get("last_artifact", {})
    _require(last_artifact.get("sha256") == _sha256(package_dir / "last_model.pt"), "last artifact SHA256 mismatch")

    target_device = torch.device(device)
    policy, loaded_contract = _load_packaged_model(package_dir, target_device)
    _require(loaded_contract == contract, "packaged loader returned a different contract")
    parameter_names = {name for name, _ in policy.named_parameters()}
    for name, payload in checkpoints.items():
        _require(set(payload["state_dict"]) == parameter_names, f"checkpoint parameter keys mismatch: {name}")
    _require(sum(parameter.numel() for parameter in policy.parameters()) == EXPECTED_PARAMETER_COUNT, "loaded model parameter count mismatch")
    try:
        smoke = torch.load(package_dir / "smoke_input.pt", map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch fallback
        smoke = torch.load(package_dir / "smoke_input.pt", map_location="cpu")
    _require(isinstance(smoke, dict) and smoke, "smoke input is empty")
    smoke = {name: value.to(target_device) for name, value in smoke.items()}
    with torch.inference_mode():
        logits = policy.teacher_logits(smoke)
    _require(logits.ndim == 3 and bool(torch.isfinite(logits).all()), "package-only forward is non-finite")

    return {
        "status": "passed",
        "package": str(package_dir),
        "epochs_completed": 3,
        "train_decisions_per_epoch": train_decisions,
        "validation_decisions_per_epoch": validation_decisions,
        "best_model": {
            "bytes": (package_dir / "best_model.pt").stat().st_size,
            "sha256": _sha256(package_dir / "best_model.pt"),
        },
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "device_names": device_names,
        "smoke_forward": {
            "shape": list(logits.shape),
            "finite": True,
        },
        "final_validation": history[-1]["validation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_model_package(args.package_dir, device=args.device)
    encoded = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
