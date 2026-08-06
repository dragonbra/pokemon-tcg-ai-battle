"""Build the 0034 merged recent-three-day BC training notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "04_train_recent3_rolling_t4" / "04_train_recent3_rolling_t4.ipynb"
TARGET_DIR = ROOT / "02_train_recent3_cleaned_bc"


MERGED_RUN = r'''
history = []
best_validation_loss = math.inf
best_artifact = None
started = time.time()
training_config = {
    "schema_version": "0034_cleaned_bc_training_v1",
    "experiment": "0034_clean_recent3_semantic_bc",
    "dataset_interval": ["2026-08-01", "2026-08-03"],
    "dataset_manifest_sha256": dataset.manifest_sha256,
    "dataset_split_counts": dataset.split_counts,
    "partition_count": len(daily_partitions),
    "epoch_definition": "one_epoch_is_full_merged_recent3_train",
    "epochs_requested": EPOCHS,
    "winner_only": True,
    "train_passes": EPOCHS,
    "validation_passes": EPOCHS,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "validation_batch_size": VALIDATION_BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "amp_dtype": "float16",
    "grad_scaler": True,
    "gradient_clip": 1.0,
    "data_parallel_device_ids": device_ids,
    "gpu_device_names": device_names,
    "initialized_from_checkpoint": None,
    "random_initialization": True,
    "wandb": None,
    "optimizer_state_saved": False,
    "resumable_training_state_saved": False,
    "sample_weight": {
        "field": "audit.cleaning.sample_weight",
        "loss": "per_token_weighted_mean",
        "ordinary": 1.0,
        "high_confidence_attach": 1.15,
        "low_confidence_special_energy": 0.95,
    },
}
atomic_json(OUTPUT / "training_config.json", training_config)

for epoch in range(1, EPOCHS + 1):
    train_metrics = train_partition_pass(epoch, "combined_recent3", dataset)
    validation_metrics = validate_dataset(epoch, "merged_validation_recent3", dataset)
    artifact = save_model(OUTPUT / "last_model.pt", epoch, validation_metrics)
    if validation_metrics["weighted_loss"] < best_validation_loss:
        best_validation_loss = validation_metrics["weighted_loss"]
        shutil.copy2(OUTPUT / "last_model.pt", OUTPUT / "best_model.pt")
        best_artifact = {
            "path": "best_model.pt",
            "sha256": sha256_file(OUTPUT / "best_model.pt"),
            "bytes": (OUTPUT / "best_model.pt").stat().st_size,
            "epoch": epoch,
            "validation_loss": best_validation_loss,
            "validation_weighted_loss": validation_metrics["weighted_loss"],
        }
    row = {
        "epoch": epoch,
        "train": train_metrics,
        "validation": validation_metrics,
        "last_artifact": artifact,
        "best_artifact": best_artifact,
        "elapsed_seconds": time.time() - started,
    }
    history.append(row)
    report = {
        "schema_version": "0034_cleaned_bc_training_report_v1",
        "state": "training",
        "experiment": "0034_clean_recent3_semantic_bc",
        "model_contract": model_contract,
        "training_config": training_config,
        "dataset_inventory": inventory,
        "history": history,
        "best_artifact": best_artifact,
    }
    atomic_json(OUTPUT / "training_report.json", report)
    atomic_json(OUTPUT / "training_status.json", {
        "state": "training",
        "epoch_completed": epoch,
        "epochs_requested": EPOCHS,
        "artifact": best_artifact,
    })
    print(json.dumps({"event": "0034_cleaned_epoch_complete", **row}, sort_keys=True), flush=True)

if len(history) != EPOCHS or best_artifact is None:
    raise RuntimeError("0034 cleaned training did not complete the requested epochs")
report["state"] = "complete"
report["epochs_completed"] = len(history)
report["full_train_passes_completed"] = len(history)
report["full_validation_passes_completed"] = len(history)
report["elapsed_seconds"] = time.time() - started
atomic_json(OUTPUT / "training_report.json", report)
atomic_json(OUTPUT / "training_status.json", {
    "state": "complete",
    "epoch_completed": len(history),
    "epochs_requested": EPOCHS,
    "artifact": best_artifact,
})
print(json.dumps({
    "event": "0034_cleaned_training_complete",
    "output": str(OUTPUT),
    "epochs_completed": len(history),
    "best_artifact": best_artifact,
    "validation": history[-1]["validation"],
}, sort_keys=True), flush=True)
'''


AUDITED_FUNCTIONS = r'''
def _audit_sample_weight(audit: dict) -> float:
    cleaning = audit.get("cleaning", {}) if isinstance(audit, dict) else {}
    value = cleaning.get("sample_weight", 1.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 1.0
    if not math.isfinite(value) or value <= 0:
        return 1.0
    return value


def _new_action_metrics() -> dict:
    return {
        "action_type_tokens": 0,
        "action_type_correct": 0,
        "attach_target_tokens": 0,
        "attach_target_correct": 0,
        "attach_target_decisions": 0,
        "attach_target_exact": 0,
        "attack_action_tokens": 0,
        "attack_action_correct": 0,
        "attack_action_decisions": 0,
        "attack_action_exact": 0,
    }


def _update_action_metrics(metrics: dict, batch: dict[str, torch.Tensor], logits: torch.Tensor) -> None:
    targets = batch["targets"]
    mask = targets.ne(-100)
    predictions = logits.argmax(-1)
    option_types = batch["option_cat"][..., 0]
    option_count = option_types.size(1)
    safe_targets = targets.clamp(min=0, max=option_count - 1)
    safe_predictions = predictions.clamp(min=0, max=option_count - 1)
    rows = torch.arange(targets.size(0), device=targets.device).unsqueeze(1)
    gathered_target_types = option_types[rows, safe_targets]
    gathered_predicted_types = option_types[rows, safe_predictions]
    stop_types = torch.full_like(gathered_target_types, -1)
    target_types = torch.where(
        mask & targets.ge(0) & targets.lt(option_count),
        gathered_target_types,
        stop_types,
    )
    predicted_types = torch.where(
        predictions.lt(option_count),
        gathered_predicted_types,
        stop_types,
    )
    metrics["action_type_tokens"] += int(mask.sum().item())
    metrics["action_type_correct"] += int((target_types.eq(predicted_types) & mask).sum().item())
    exact = (predictions.eq(targets) | ~mask).all(1)
    for label, action_type in (("attach", 9), ("attack", 14)):
        selected = target_types.eq(action_type) & mask
        selected_decisions = selected.any(1)
        correct_tokens = predictions.eq(targets) & selected
        metrics[f"{label}_target_tokens" if label == "attach" else f"{label}_action_tokens"] += int(selected.sum().item())
        metrics[f"{label}_target_correct" if label == "attach" else f"{label}_action_correct"] += int(correct_tokens.sum().item())
        if label == "attach":
            metrics["attach_target_decisions"] += int(selected_decisions.sum().item())
            metrics["attach_target_exact"] += int((exact & selected_decisions).sum().item())
        else:
            metrics["attack_action_decisions"] += int(selected_decisions.sum().item())
            metrics["attack_action_exact"] += int((exact & selected_decisions).sum().item())


def _finalize_action_metrics(metrics: dict) -> dict:
    return {
        **metrics,
        "action_type_accuracy": metrics["action_type_correct"] / max(1, metrics["action_type_tokens"]),
        "attach_target_token_accuracy": metrics["attach_target_correct"] / max(1, metrics["attach_target_tokens"]),
        "attach_target_exact_action": metrics["attach_target_exact"] / max(1, metrics["attach_target_decisions"]),
        "attack_action_token_accuracy": metrics["attack_action_correct"] / max(1, metrics["attack_action_tokens"]),
        "attack_action_exact": metrics["attack_action_exact"] / max(1, metrics["attack_action_decisions"]),
    }


def _weights_from_audits(audits: list[dict], device: torch.device) -> torch.Tensor:
    return torch.tensor([_audit_sample_weight(audit) for audit in audits], dtype=torch.float32, device=device)


def save_model(path: Path, stage: int, validation: dict) -> dict:
    temporary = path.with_suffix(path.suffix + ".tmp")
    metadata = {
        "architecture": "SemanticPolicy",
        "actor_schema": fields.SCHEMA_VERSION,
        "model_config": model_config.to_dict(),
        "parameter_count": parameter_count,
        "stage": stage,
        "epoch": stage,
        "validation": validation,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "initialized_from_checkpoint": None,
        "random_initialization": True,
        "winner_only": True,
        "training_mode": "merged_recent3_cleaned_bc",
    }
    payload = {
        "schema_version": "0034_model_only_checkpoint_v1",
        "state_dict": {
            name: value.detach().cpu()
            for name, value in parallel.module.policy.named_parameters()
        },
        "metadata": metadata,
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "stage": stage,
    }


def train_partition_pass(stage: int, date: str, partition_dataset) -> dict:
    parallel.train()
    torch.cuda.reset_peak_memory_stats(primary)
    source = partition_dataset.iter_audited_batches(
        "train",
        TRAIN_BATCH_SIZE,
        seed=SEED + stage,
        length_bucketed=True,
    )
    prefetch = prefetch_module.PrefetchIterator(source, depth=PREFETCH_DEPTH)
    loss_sum = 0.0
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    tokens = 0
    correct = 0
    exact = 0
    decisions = 0
    updates = 0
    skipped_updates = 0
    action_metrics = _new_action_metrics()
    started = time.time()
    try:
        for batch_index, item in enumerate(prefetch, 1):
            raw, audits = item
            batch = move_batch(raw)
            weights = _weights_from_audits(audits, primary)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = parallel(batch)
                mask = batch["targets"].ne(-100)
                token_losses = F.cross_entropy(
                    logits[mask], batch["targets"][mask], reduction="none"
                )
                token_weights = weights.unsqueeze(1).expand_as(mask)[mask]
                loss = (token_losses * token_weights).sum() / token_weights.sum().clamp_min(1e-6)
                unweighted_loss = token_losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite 0034 weighted BC loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(parallel.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            token_count = int(mask.sum().item())
            matches = logits.argmax(-1).eq(batch["targets"]) & mask
            decision_count = int(batch["targets"].size(0))
            loss_sum += float(unweighted_loss.detach().cpu()) * token_count
            weighted_loss_sum += float(loss.detach().cpu()) * float(token_weights.sum().detach().cpu())
            weight_sum += float(token_weights.sum().detach().cpu())
            tokens += token_count
            correct += int(matches.sum().item())
            exact += int((matches | ~mask).all(1).sum().item())
            decisions += decision_count
            _update_action_metrics(action_metrics, batch, logits)
            if batch_index == 1 or batch_index % 100 == 0:
                print(json.dumps({
                    "event": "0034_cleaned_train_progress",
                    "stage": stage,
                    "date": date,
                    "batch": batch_index,
                    "decisions": decisions,
                    "expected_decisions": partition_dataset.split_counts["train"],
                    "loss": loss_sum / max(tokens, 1),
                    "weighted_loss": weighted_loss_sum / max(weight_sum, 1e-6),
                }, sort_keys=True), flush=True)
    finally:
        prefetch.close()
    if decisions != partition_dataset.split_counts["train"]:
        raise ValueError(f"0034 cleaned train coverage mismatch: {decisions} != {partition_dataset.split_counts['train']}")
    if not tokens or not updates:
        raise ValueError("0034 cleaned train produced no valid updates")
    return {
        "loss": loss_sum / tokens,
        "weighted_loss": weighted_loss_sum / max(weight_sum, 1e-6),
        "mean_sample_weight": weight_sum / max(tokens, 1),
        "token_accuracy": correct / tokens,
        "teacher_exact_action": exact / decisions,
        "tokens": tokens,
        "decisions": decisions,
        "updates": updates,
        "skipped_updates": skipped_updates,
        "seconds": time.time() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(primary),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(primary),
        **_finalize_action_metrics(action_metrics),
    }


def validate_dataset(stage: int, label: str, eval_dataset) -> dict:
    parallel.eval()
    source = eval_dataset.iter_audited_batches(
        "validation",
        VALIDATION_BATCH_SIZE,
        seed=SEED,
        length_bucketed=False,
    )
    prefetch = prefetch_module.PrefetchIterator(source, depth=PREFETCH_DEPTH)
    loss_sum = 0.0
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    tokens = 0
    correct = 0
    exact = 0
    decisions = 0
    action_metrics = _new_action_metrics()
    started = time.time()
    try:
        with torch.inference_mode():
            for raw, audits in prefetch:
                batch = move_batch(raw)
                weights = _weights_from_audits(audits, primary)
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = parallel(batch)
                    mask = batch["targets"].ne(-100)
                    token_losses = F.cross_entropy(
                        logits[mask], batch["targets"][mask], reduction="none"
                    )
                    token_weights = weights.unsqueeze(1).expand_as(mask)[mask]
                loss_sum += float(token_losses.sum().cpu())
                weighted_loss_sum += float((token_losses * token_weights).sum().cpu())
                weight_sum += float(token_weights.sum().cpu())
                tokens += int(mask.sum().item())
                matches = logits.argmax(-1).eq(batch["targets"]) & mask
                decisions += int(batch["targets"].size(0))
                correct += int(matches.sum().item())
                exact += int((matches | ~mask).all(1).sum().item())
                _update_action_metrics(action_metrics, batch, logits)
    finally:
        prefetch.close()
    if decisions != eval_dataset.split_counts["validation"]:
        raise ValueError(f"0034 cleaned validation coverage mismatch: {decisions} != {eval_dataset.split_counts['validation']}")
    if not tokens or not decisions:
        raise ValueError("0034 cleaned validation is empty")
    return {
        "label": label,
        "loss": loss_sum / tokens,
        "weighted_loss": weighted_loss_sum / max(weight_sum, 1e-6),
        "mean_sample_weight": weight_sum / max(tokens, 1),
        "token_accuracy": correct / tokens,
        "teacher_exact_action": exact / decisions,
        "tokens": tokens,
        "decisions": decisions,
        "seconds": time.time() - started,
        **_finalize_action_metrics(action_metrics),
    }
'''


MODEL_USAGE = r'''"""Load the 0034 cleaned BC model package without hidden training state."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch


def load_model(package_dir: str | Path, *, device: str | torch.device = "cpu"):
    package_dir = Path(package_dir)
    contract = json.loads((package_dir / "model_contract.json").read_text(encoding="utf-8"))
    source_root = package_dir / "model_source"
    if not source_root.is_dir():
        raise FileNotFoundError("model package is missing model_source")
    sys.path.insert(0, str(source_root))
    package = contract.get("package_name", "0034_clean_recent3_semantic_bc")
    domain = importlib.import_module(f"{package}.domain.prototypes")
    config_module = importlib.import_module(f"{package}.model.config")
    policy_module = importlib.import_module(f"{package}.model.policy")
    prototypes = domain.PrototypeIndex.load(
        package_dir / "official_public_prototypes_v1.json",
        package_dir / "official_full_engine_prototypes_v2.json",
    )
    config = config_module.ModelConfig(**contract["model_config"])
    policy = policy_module.SemanticPolicy(config, prototypes).to(device).eval()
    try:
        payload = torch.load(package_dir / "best_model.pt", map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(package_dir / "best_model.pt", map_location=device)
    if set(payload) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("checkpoint payload must contain only schema_version/state_dict/metadata")
    state = payload["state_dict"]
    if any(str(key).startswith("module.") for key in state):
        raise ValueError("checkpoint contains forbidden DataParallel module. prefix")
    missing, unexpected = policy.load_state_dict(state, strict=False)
    unexpected = list(unexpected)
    parameter_names = {name for name, _ in policy.named_parameters()}
    allowed_missing_prefixes = (
        "state_encoder.prototypes.",
        "option_encoder.prototypes.",
        "prototype_encoder.",
    )
    invalid_missing = [
        name for name in missing
        if not str(name).startswith(allowed_missing_prefixes)
    ]
    if unexpected or invalid_missing:
        raise ValueError({"unexpected": unexpected, "invalid_missing": invalid_missing})
    return policy, contract
'''


def _source(cell: nbf.NotebookNode) -> str:
    value = cell.source
    return value if isinstance(value, str) else "".join(value)


def build() -> Path:
    notebook = nbf.reads(BASE.read_text(encoding="utf-8-sig"), as_version=4)
    notebook.cells[0].source = (
        "# PTCG 0034 merged recent-three-day cleaned BC\n\n"
        "Trains three complete epochs over the merged cleaned train split from 2026-08-01 "
        "through 2026-08-03; each epoch includes all three days and keeps d_model=320, "
        "four state layers, and three option layers."
    )

    for cell in notebook.cells:
        source = _source(cell)
        source = source.replace("20_567_042", "22_595_202")
        source = source.replace("EPOCHS = 1", "EPOCHS = 3")
        source = source.replace(
            "model_config = model_module.ModelConfig()",
            "model_config = model_module.ModelConfig(\n"
            "    d_model=320,\n"
            "    state_layers=4,\n"
            "    option_layers=3,\n"
            "    ffn_multiplier=3,\n"
            ")",
        )
        source = source.replace(
            'os.environ[\"WANDB_MODE\"] = \"disabled\"',
            'os.environ[\"WANDB_MODE\"] = \"disabled\"\n'
            'os.environ[\"WANDB_DISABLED\"] = \"true\"',
        )
        source = source.replace(
            'if EPOCHS != 1:\n    raise ValueError("rolling 0034 training uses exactly one train pass per daily partition")',
            'if EPOCHS < 1:\n    raise ValueError("0034 merged training requires at least one complete epoch")',
        )
        source = source.replace(
            "ptcg_0034_recent3_rolling_t4_train",
            "ptcg_0034_recent3_cleaned_bc_train",
        )
        source = source.replace(
            '"0034_audited_bc_rolling_training_report_v1"',
            '"0034_cleaned_bc_training_report_v1"',
        )
        source = source.replace(
            '"rolling_daily_partition_once"',
            '"merged_recent3_cleaned_bc"',
        )
        source = source.replace(
            'for name, value in parallel.module.policy.state_dict().items()',
            'for name, value in parallel.module.policy.named_parameters()',
        )
        source = source.replace(
            '    "data_parallel_device_ids": device_ids,\n    "winner_only": True,',
            '    "data_parallel_device_ids": device_ids,\n'
            '    "gpu_device_names": device_names,\n'
            '    "winner_only": True,',
        )
        source = source.replace(
            'batch_contract.DecisionBatch.from_mapping(sample)\n'
            'sample = {name: value.to(primary) for name, value in sample.items()}',
            'batch_contract.DecisionBatch.from_mapping(sample)\n'
            'smoke_input = {\n'
            '    name: value[:8].detach().cpu()\n'
            '    for name, value in sample.items()\n'
            '}\n'
            'torch.save(smoke_input, OUTPUT / "smoke_input.pt")\n'
            'sample = {name: value.to(primary) for name, value in sample.items()}',
        )
        source = source.replace(
            'del sample, sample_logits',
            'del sample, sample_logits, smoke_input',
        )
        source = source.replace(
            '        shutil.copytree(flat_unique[0], staged)\n        return staged',
            '        shutil.copytree(flat_unique[0], staged)\n'
            '        fields_path = staged / "contracts" / "fields.py"\n'
            '        fields_text = fields_path.read_text(encoding="utf-8")\n'
            '        fields_path.write_text(fields_text.replace(\n'
            '            "0033_effect_summary_semantic_decision_v1",\n'
            '            "0034_clean_recent3_semantic_decision_v1",\n'
            '        ), encoding="utf-8")\n'
            '        return staged',
        )
        source = source.replace(
            'def find_source_root() -> Path:\n'
            '    package_name = "0034_clean_recent3_semantic_bc"\n'
            '    package_candidates = []\n'
            '    flat_candidates = []\n'
            '    for public_asset in INPUT.rglob("official_public_prototypes_v1.json"):\n'
            '        root = public_asset.parent.parent\n'
            '        if (\n'
            '            (root / "features" / "audit.py").is_file()\n'
            '            and (root / "training" / "dataset.py").is_file()\n'
            '        ):\n'
            '            if root.name == package_name:\n'
            '                package_candidates.append(root)\n'
            '            else:\n'
            '                flat_candidates.append(root)\n'
            '    package_unique = sorted(set(package_candidates))\n'
            '    if len(package_unique) == 1:\n'
            '        return package_unique[0]\n'
            '    flat_unique = sorted(set(flat_candidates))\n'
            '    if len(flat_unique) == 1:\n'
            '        staged = WORKING / package_name\n'
            '        if staged.exists():\n'
            '            shutil.rmtree(staged)\n'
            '        shutil.copytree(flat_unique[0], staged)\n'
            '        fields_path = staged / "contracts" / "fields.py"\n'
            '        fields_text = fields_path.read_text(encoding="utf-8")\n'
            '        fields_path.write_text(fields_text.replace(\n'
            '            "0033_effect_summary_semantic_decision_v1",\n'
            '            "0034_clean_recent3_semantic_decision_v1",\n'
            '        ), encoding="utf-8")\n'
            '        return staged\n'
            '    raise FileNotFoundError(\n'
            '        "expected one complete 0034 source root or one flat source dataset, "\n'
            '        f"found package={package_unique}, flat={flat_unique}"\n'
            '    )',
            'def _source_schema(root: Path) -> str | None:\n'
            '    fields_path = root / "contracts" / "fields.py"\n'
            '    if not fields_path.is_file():\n'
            '        return None\n'
            '    for line in fields_path.read_text(encoding="utf-8").splitlines():\n'
            '        line = line.strip()\n'
            '        if line.startswith("SCHEMA_VERSION") and "=" in line:\n'
            '            return line.split("=", 1)[1].strip().strip(chr(34)).strip(chr(39))\n'
            '    return None\n'
            '\n'
            '\n'
            'def _is_0034_source(root: Path) -> bool:\n'
            '    return _source_schema(root) == "0034_clean_recent3_semantic_decision_v1"\n'
            '\n'
            '\n'
            'def _stage_source(root: Path, package_name: str) -> Path:\n'
            '    staged = WORKING / package_name\n'
            '    if staged.exists():\n'
            '        shutil.rmtree(staged)\n'
            '    shutil.copytree(root, staged)\n'
            '    fields_path = staged / "contracts" / "fields.py"\n'
            '    fields_text = fields_path.read_text(encoding="utf-8")\n'
            '    fields_text = fields_text.replace(\n'
            '        "0033_effect_summary_semantic_decision_v1",\n'
            '        "0034_clean_recent3_semantic_decision_v1",\n'
            '    )\n'
            '    fields_path.write_text(fields_text, encoding="utf-8")\n'
            '    feature_audit_path = staged / "contracts" / "feature_audit.json"\n'
            '    if feature_audit_path.is_file():\n'
            '        feature_audit = json.loads(feature_audit_path.read_text(encoding="utf-8"))\n'
            '        feature_audit["actor_schema"] = "0034_clean_recent3_semantic_decision_v1"\n'
            '        feature_audit_path.write_text(json.dumps(feature_audit, indent=2, sort_keys=True) + "\\n", encoding="utf-8")\n'
            '    if _source_schema(staged) != "0034_clean_recent3_semantic_decision_v1":\n'
            '        raise ValueError(f"staged source has unexpected schema: {_source_schema(staged)}")\n'
            '    return staged\n'
            '\n'
            '\n'
            'def find_source_root() -> Path:\n'
            '    package_name = "0034_clean_recent3_semantic_bc"\n'
            '    package_candidates = []\n'
            '    flat_candidates = []\n'
            '    rejected = []\n'
            '    for public_asset in INPUT.rglob("official_public_prototypes_v1.json"):\n'
            '        root = public_asset.parent.parent\n'
            '        if not (\n'
            '            (root / "features" / "audit.py").is_file()\n'
            '            and (root / "training" / "dataset.py").is_file()\n'
            '        ):\n'
            '            continue\n'
            '        schema = _source_schema(root)\n'
            '        if schema not in {\n'
            '            "0034_clean_recent3_semantic_decision_v1",\n'
            '            "0033_effect_summary_semantic_decision_v1",\n'
            '        }:\n'
            '            rejected.append({"root": str(root), "schema": schema})\n'
            '            continue\n'
            '        if root.name == package_name:\n'
            '            package_candidates.append(root)\n'
            '        else:\n'
            '            flat_candidates.append(root)\n'
            '    package_unique = sorted(set(package_candidates))\n'
            '    if len(package_unique) == 1:\n'
            '        return _stage_source(package_unique[0], package_name)\n'
            '    flat_unique = sorted(set(flat_candidates))\n'
            '    if len(flat_unique) == 1:\n'
            '        return _stage_source(flat_unique[0], package_name)\n'
            '    raise FileNotFoundError(\n'
            '        "expected one complete 0034 source root or one flat 0034 source dataset, "\n'
            '        f"found package={package_unique}, flat={flat_unique}, rejected={rejected}"\n'
            '    )',
        )
        cell.source = source

    notebook.cells[6].source = _source(notebook.cells[6]) + r'''
for asset in ("official_public_prototypes_v1.json", "official_full_engine_prototypes_v2.json"):
    source_asset = PROTOTYPE_PATH if asset.startswith("official_public") else PROTOTYPE_PATH.with_name(asset)
    shutil.copy2(source_asset, OUTPUT / asset)
shutil.copy2(SOURCE_ROOT / "contracts" / "feature_audit.json", OUTPUT / "feature_audit.json")
model_source = OUTPUT / "model_source" / PACKAGE
model_source.mkdir(parents=True, exist_ok=True)
for source_name in ("__init__.py", "contracts", "domain", "features", "knowledge", "model"):
    source_path = SOURCE_ROOT / source_name
    target_path = model_source / source_name
    if source_path.is_dir():
        shutil.copytree(source_path, target_path)
    elif source_path.is_file():
        shutil.copy2(source_path, target_path)
atomic_json(OUTPUT / "model_package_manifest.json", {
    "schema_version": "0034_model_package_v1",
    "package_name": PACKAGE,
    "weights": ["best_model.pt", "last_model.pt"],
    "contract": "model_contract.json",
    "prototype_assets": ["official_public_prototypes_v1.json", "official_full_engine_prototypes_v2.json"],
    "feature_audit": "feature_audit.json",
    "cleaning_report": "cleaning_report.json",
    "training_report": "training_report.json",
    "training_config": "training_config.json",
    "dataset_reference": "dataset_reference.json",
    "usage_loader": "load_model.py",
    "usage_readme": "MODEL_USAGE.md",
    "smoke_input": "smoke_input.pt",
})
(OUTPUT / "load_model.py").write_text(MODEL_USAGE, encoding="utf-8")
(OUTPUT / "MODEL_USAGE.md").write_text(
    "# 0034 cleaned BC model package\n\n"
    "Use `load_model.py` with this directory. The contract and both prototype assets are required; "
    "the checkpoint contains trainable parameters only and reconstructs deterministic prototype buffers. "
    "Load `smoke_input.pt` with `torch.load(..., weights_only=True)` to run a package-only finite forward.\n",
    encoding="utf-8",
)
aggregate_cleaning_reports = []
for cleaning_report in INPUT.rglob("cleaning_report.json"):
    try:
        cleaning_payload = json.loads(cleaning_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if (
        cleaning_payload.get("schema_version") == "0034_cleaning_report_v1"
        and set(cleaning_payload.get("dates", [])) == EXPECTED_DATES
        and cleaning_payload.get("status") == "passed"
    ):
        aggregate_cleaning_reports.append(cleaning_report)
if len(aggregate_cleaning_reports) != 1:
    raise RuntimeError(
        f"expected exactly one aggregate 0034 cleaning report, found {aggregate_cleaning_reports}"
    )
shutil.copy2(aggregate_cleaning_reports[0], OUTPUT / "cleaning_report.json")
'''
    notebook.cells[6].source = notebook.cells[6].source.replace(
        "write_text(MODEL_USAGE, encoding=\"utf-8\")",
        f"write_text({json.dumps(MODEL_USAGE)}, encoding=\"utf-8\")",
    )
    notebook.cells[6].source = notebook.cells[6].source.replace(
        '"schema_version": "0034_audited_model_contract_v1",',
        '"schema_version": "0034_audited_model_contract_v1",\n    "package_name": PACKAGE,',
    )
    notebook.cells[6].source = notebook.cells[6].source.replace(
        '    "prototype_assets": ["official_public_prototypes_v1.json", "official_full_engine_prototypes_v2.json"],\n    "package_name": PACKAGE,\n    "feature_audit"',
        '    "prototype_assets": ["official_public_prototypes_v1.json", "official_full_engine_prototypes_v2.json"],\n    "feature_audit"',
    )
    notebook.cells[6].source = notebook.cells[6].source.replace(
        '"wandb": None,\n}',
        '"wandb": None,\n    "checkpoint_state_dict": "trainable_parameters_only",\n    "prototype_assets": ["official_public_prototypes_v1.json", "official_full_engine_prototypes_v2.json"],\n    "sample_weight_field": "audit.cleaning.sample_weight",\n}',
    )
    base_training_functions = _source(notebook.cells[8])
    legacy_marker = "\ndef train_partition_pass("
    if base_training_functions.count(legacy_marker) != 1:
        raise RuntimeError("unexpected base training-function layout")
    notebook.cells[8].source = (
        base_training_functions.split(legacy_marker, 1)[0].rstrip()
        + "\n\n"
        + AUDITED_FUNCTIONS.lstrip()
    )
    notebook.cells[10].source = MERGED_RUN
    notebook.cells[7].source = "## Weighted merged training and validation functions"
    notebook.cells[9].source = "## Run three complete merged-data epochs"
    for cell in notebook.cells:
        source = _source(cell)
        source = source.replace(
            "0034_rolling_training_environment",
            "0034_cleaned_training_environment",
        )
        source = source.replace(
            "0034_rolling_partition_plan",
            "0034_merged_partition_inventory",
        )
        cell.source = source

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    target = TARGET_DIR / "02_train_recent3_cleaned_bc.ipynb"
    nbf.validate(notebook)
    nbf.write(notebook, target)
    metadata = {
        "id": "horizen12/ptcg-0034-02-train-recent3-cleaned-bc",
        "title": "PTCG 0034 02 Train Recent3 Cleaned BC",
        "code_file": target.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["pokemon-tcg", "0034", "cleaned-bc", "dual-t4"],
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": ["horizen12/ptcg-0034-01-clean-recent3-data"],
        "model_sources": [],
    }
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


if __name__ == "__main__":
    print(build())
