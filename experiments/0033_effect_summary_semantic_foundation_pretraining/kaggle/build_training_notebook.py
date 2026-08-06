"""Build the audited dual-T4 training notebook with nbformat."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "05_train_dual_t4" / "05_train_dual_t4.ipynb"


SETUP = dedent(
    """
    from __future__ import annotations

    import hashlib
    import importlib
    import json
    import math
    import os
    from pathlib import Path
    import random
    import shutil
    import sys
    import time

    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F

    INPUT = Path(os.environ.get("PTCG_KAGGLE_INPUT", "/kaggle/input"))
    WORKING = Path(os.environ.get("PTCG_KAGGLE_WORKING", "/kaggle/working"))
    OUTPUT = WORKING / "ptcg_0032_dual_t4_train"
    SMOKE = os.environ.get("PTCG_NOTEBOOK_SMOKE") == "1"
    SEED = 20260804
    EPOCHS = 1 if SMOKE else 5
    TRAIN_BATCH_SIZE = 512 if SMOKE else 192
    VALIDATION_BATCH_SIZE = 512 if SMOKE else 256
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.02
    PREFETCH_DEPTH = 2
    EXPECTED_PARAMETER_COUNT = 20_567_042
    EXPECTED_DATES = {
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
        "2026-07-31",
        "2026-08-01",
    }

    if not SMOKE and EPOCHS != 5:
        raise ValueError("formal 0032 training must use exactly five epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("0032 training requires CUDA")
    gpu_count = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    if SMOKE:
        device_ids = list(range(min(2, gpu_count)))
    else:
        if gpu_count < 2 or any("T4" not in name for name in device_names[:2]):
            raise RuntimeError(f"0032 formal training requires two T4 GPUs: {device_names}")
        device_ids = [0, 1]
    if not device_ids:
        raise RuntimeError("no CUDA device selected")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ["WANDB_MODE"] = "disabled"

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)


    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def atomic_json(path: Path, payload: object) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)


    def find_source_root() -> Path:
        candidates = []
        for public_asset in INPUT.rglob("official_public_prototypes_v1.json"):
            root = public_asset.parent.parent
            if (
                root.name == "0033_effect_summary_semantic_foundation_pretraining"
                and (root / "features" / "audit.py").is_file()
                and (root / "training" / "dataset.py").is_file()
            ):
                candidates.append(root)
        unique = sorted(set(candidates))
        if len(unique) != 1:
            raise FileNotFoundError(f"expected one complete 0032 source root: {unique}")
        return unique[0]


    SOURCE_ROOT = find_source_root()
    sys.path.insert(0, str(SOURCE_ROOT.parent))
    PACKAGE = SOURCE_ROOT.name
    fields = importlib.import_module(f"{PACKAGE}.contracts.fields")
    batch_contract = importlib.import_module(f"{PACKAGE}.contracts.batch")
    dataset_module = importlib.import_module(f"{PACKAGE}.training.dataset")
    prefetch_module = importlib.import_module(f"{PACKAGE}.training.prefetch")
    prototypes_module = importlib.import_module(f"{PACKAGE}.domain.prototypes")
    model_module = importlib.import_module(f"{PACKAGE}.model")

    if not hasattr(dataset_module, "CombinedCanonicalDecisionDataset"):
        raise RuntimeError("0032 source does not contain the combined partition loader")
    if fields.SCHEMA_VERSION != "0033_effect_summary_semantic_decision_v1":
        raise ValueError(f"unexpected actor schema: {fields.SCHEMA_VERSION}")

    print(
        json.dumps(
            {
                "event": "0032_training_environment",
                "smoke": SMOKE,
                "epochs": EPOCHS,
                "gpu_count": gpu_count,
                "device_ids": device_ids,
                "device_names": device_names,
                "wandb": None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    """
)


DATA = dedent(
    """
    daily = {}
    for manifest_path in INPUT.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("schema_version") != "0032_daily_semantic_partition_v1":
            continue
        date = manifest.get("date")
        if date not in EXPECTED_DATES:
            continue
        if date in daily:
            raise RuntimeError(f"duplicate daily partition attached for {date}")
        if manifest.get("status") != "complete" or manifest.get("winner_only") is not True:
            raise ValueError(f"daily partition is not a complete winner-only dataset: {date}")
        counts = manifest.get("split_counts", {})
        if any(int(counts.get(split, 0)) <= 0 for split in ("train", "validation")):
            raise ValueError(f"daily partition contains an empty split: {date}")
        feature_audit = manifest.get("feature_input_audit", {})
        acceptance = manifest.get("acceptance", {})
        if (
            feature_audit.get("status") != "passed"
            or feature_audit.get("audited_decisions") != manifest.get("records")
            or acceptance.get("status") != "passed"
            or acceptance.get("validated_records") != manifest.get("records")
            or acceptance.get("model_forward", {}).get("parameter_count")
            != EXPECTED_PARAMETER_COUNT
        ):
            raise ValueError(f"daily feature/model acceptance is incomplete: {date}")
        canonical_root = manifest_path.parent / manifest.get("canonical_path", "canonical")
        canonical_manifest_path = canonical_root / "manifest.json"
        if (
            not canonical_manifest_path.is_file()
            or sha256_file(canonical_manifest_path)
            != manifest.get("canonical_manifest_sha256")
        ):
            raise ValueError(f"canonical manifest commitment mismatch: {date}")
        daily[date] = {
            "manifest_path": manifest_path,
            "manifest": manifest,
            "canonical_root": canonical_root,
            "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        }

    if set(daily) != EXPECTED_DATES:
        raise FileNotFoundError(
            f"expected all five daily datasets; missing={sorted(EXPECTED_DATES - set(daily))}"
        )
    canonical_hashes = [daily[date]["canonical_manifest_sha256"] for date in sorted(daily)]
    if not SMOKE and len(set(canonical_hashes)) != len(canonical_hashes):
        raise ValueError("formal daily partitions contain duplicate canonical manifests")

    roots = [daily[date]["canonical_root"] for date in sorted(daily)]
    dataset = dataset_module.CombinedCanonicalDecisionDataset(roots)
    expected_counts = {
        split: sum(int(daily[date]["manifest"]["split_counts"][split]) for date in daily)
        for split in ("train", "validation")
    }
    if dataset.split_counts != expected_counts:
        raise ValueError("combined dataset split counts disagree with daily manifests")
    if dataset.manifest.get("partition_count") != 5:
        raise ValueError("combined dataset did not retain all five daily partitions")

    inventory = [
        {
            "date": date,
            "records": daily[date]["manifest"]["records"],
            "split_counts": daily[date]["manifest"]["split_counts"],
            "canonical_manifest_sha256": daily[date]["canonical_manifest_sha256"],
        }
        for date in sorted(daily)
    ]
    atomic_json(
        OUTPUT / "dataset_reference.json",
        {
            "schema_version": "0032_combined_dataset_reference_v1",
            "winner_only": True,
            "dates": sorted(EXPECTED_DATES),
            "combined_manifest_sha256": dataset.manifest_sha256,
            "combined_manifest": dataset.manifest,
            "inventory": inventory,
        },
    )
    print(
        json.dumps(
            {
                "event": "0032_dataset_preflight",
                "dates": sorted(EXPECTED_DATES),
                "partitions": len(roots),
                "split_counts": dataset.split_counts,
                "combined_manifest_sha256": dataset.manifest_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    """
)


MODEL = dedent(
    """
    PROTOTYPE_PATH = SOURCE_ROOT / "assets" / "official_public_prototypes_v1.json"
    prototypes = prototypes_module.PrototypeIndex.load(PROTOTYPE_PATH)
    prototype_counts = {
        "cards": len(prototypes.engine_cards),
        "attacks": len(prototypes.engine_attacks),
        "skills": len(prototypes.skills),
    }
    if prototype_counts != {"cards": 1267, "attacks": 1556, "skills": 433}:
        raise ValueError(f"incomplete full-engine prototypes: {prototype_counts}")

    model_config = model_module.ModelConfig()
    policy = model_module.SemanticPolicy(model_config, prototypes)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"0032 parameter count drift: {parameter_count}")


    class TeacherLogits(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.policy = inner

        def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.policy.teacher_logits(batch)


    primary = torch.device("cuda:0")
    parallel = nn.DataParallel(
        TeacherLogits(policy).to(primary),
        device_ids=device_ids,
        output_device=device_ids[0],
    )
    optimizer = torch.optim.AdamW(
        parallel.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=4096.0,
        growth_interval=10_000,
    )
    amp_dtype = torch.float16

    sample = next(dataset.iter_batches("train", TRAIN_BATCH_SIZE, seed=SEED))
    batch_contract.DecisionBatch.from_mapping(sample)
    sample = {name: value.to(primary) for name, value in sample.items()}
    parallel.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        sample_logits = parallel(sample)
    if sample_logits.ndim != 3 or not torch.isfinite(sample_logits).all():
        raise ValueError("dual-device teacher-logit preflight failed")
    del sample, sample_logits
    torch.cuda.empty_cache()

    model_contract = {
        "schema_version": "0032_audited_model_contract_v1",
        "architecture": "SemanticPolicy",
        "model_config": model_config.to_dict(),
        "parameter_count": parameter_count,
        "actor_schema": fields.SCHEMA_VERSION,
        "action_contract": "ordered_legal_option_pointer_plus_stop",
        "initialized_from_checkpoint": None,
        "random_initialization": True,
        "data_parallel_device_ids": device_ids,
        "winner_only": True,
        "wandb": None,
    }
    atomic_json(OUTPUT / "model_contract.json", model_contract)
    shutil.copy2(PROTOTYPE_PATH, OUTPUT / PROTOTYPE_PATH.name)
    shutil.copy2(
        PROTOTYPE_PATH.with_name("official_full_engine_prototypes_v2.json"),
        OUTPUT / "official_full_engine_prototypes_v2.json",
    )
    print(
        json.dumps(
            {
                "event": "0032_model_preflight",
                "parameter_count": parameter_count,
                "model_config": model_config.to_dict(),
                "initialized_from_checkpoint": None,
                "random_initialization": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    """
)


TRAINING = dedent(
    """
    def move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: value.to(primary, non_blocking=True)
            for name, value in batch.items()
        }


    def train_epoch(epoch: int) -> dict:
        parallel.train()
        torch.cuda.reset_peak_memory_stats(primary)
        source = dataset.iter_batches(
            "train",
            TRAIN_BATCH_SIZE,
            seed=SEED + epoch,
            length_bucketed=True,
        )
        prefetch = prefetch_module.PrefetchIterator(source, depth=PREFETCH_DEPTH)
        loss_sum = 0.0
        tokens = 0
        correct = 0
        exact = 0
        decisions = 0
        updates = 0
        skipped_updates = 0
        started = time.time()
        try:
            for batch_index, raw in enumerate(prefetch, 1):
                batch = move_batch(raw)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = parallel(batch)
                    mask = batch["targets"].ne(-100)
                    loss = F.cross_entropy(logits[mask], batch["targets"][mask])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite 0032 BC loss")
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
                loss_sum += float(loss.detach().cpu()) * token_count
                tokens += token_count
                correct += int(matches.sum().item())
                exact += int((matches | ~mask).all(1).sum().item())
                decisions += decision_count
                if batch_index == 1 or batch_index % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "0032_train_progress",
                                "epoch": epoch,
                                "batch": batch_index,
                                "decisions": decisions,
                                "loss": loss_sum / max(tokens, 1),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        finally:
            prefetch.close()
        if decisions != dataset.split_counts["train"]:
            raise ValueError(
                f"epoch {epoch} train coverage mismatch: "
                f"{decisions} != {dataset.split_counts['train']}"
            )
        if not tokens or not updates:
            raise ValueError("training epoch produced no valid updates")
        return {
            "loss": loss_sum / tokens,
            "token_accuracy": correct / tokens,
            "teacher_exact_action": exact / decisions,
            "tokens": tokens,
            "decisions": decisions,
            "updates": updates,
            "skipped_updates": skipped_updates,
            "seconds": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(primary),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(primary),
        }


    def validate_epoch(epoch: int) -> dict:
        parallel.eval()
        source = dataset.iter_batches(
            "validation",
            VALIDATION_BATCH_SIZE,
            seed=SEED,
            length_bucketed=False,
        )
        prefetch = prefetch_module.PrefetchIterator(source, depth=PREFETCH_DEPTH)
        loss_sum = 0.0
        tokens = 0
        correct = 0
        exact = 0
        decisions = 0
        started = time.time()
        try:
            with torch.inference_mode():
                for raw in prefetch:
                    batch = move_batch(raw)
                    with torch.autocast("cuda", dtype=amp_dtype):
                        logits = parallel(batch)
                        mask = batch["targets"].ne(-100)
                        batch_loss = F.cross_entropy(
                            logits[mask],
                            batch["targets"][mask],
                            reduction="sum",
                        )
                    matches = logits.argmax(-1).eq(batch["targets"]) & mask
                    loss_sum += float(batch_loss.cpu())
                    tokens += int(mask.sum().item())
                    decisions += int(batch["targets"].size(0))
                    correct += int(matches.sum().item())
                    exact += int((matches | ~mask).all(1).sum().item())
        finally:
            prefetch.close()
        if decisions != dataset.split_counts["validation"]:
            raise ValueError(
                f"epoch {epoch} validation coverage mismatch: "
                f"{decisions} != {dataset.split_counts['validation']}"
            )
        if not tokens or not decisions:
            raise ValueError("validation epoch is empty")
        return {
            "loss": loss_sum / tokens,
            "token_accuracy": correct / tokens,
            "teacher_exact_action": exact / decisions,
            "tokens": tokens,
            "decisions": decisions,
            "seconds": time.time() - started,
        }


    def save_model(path: Path, epoch: int, validation: dict) -> dict:
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "schema_version": "0032_audited_bc_model_v1",
            "architecture": "SemanticPolicy",
            "state_dict": {
                name: value.detach().cpu()
                for name, value in parallel.module.policy.state_dict().items()
            },
            "model_config": model_config.to_dict(),
            "parameter_count": parameter_count,
            "epoch": epoch,
            "validation": validation,
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "initialized_from_checkpoint": None,
            "random_initialization": True,
            "winner_only": True,
        }
        torch.save(payload, temporary)
        temporary.replace(path)
        return {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "epoch": epoch,
        }
    """
)


RUN = dedent(
    """
    history = []
    best_validation_loss = math.inf
    best_artifact = None
    started = time.time()
    training_config = {
        "schema_version": "0032_audited_bc_training_v1",
        "experiment": "0033_effect_summary_semantic_foundation_pretraining",
        "dataset_interval": ["2026-07-28", "2026-08-01"],
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_split_counts": dataset.split_counts,
        "winner_only": True,
        "epochs": EPOCHS,
        "full_train_passes": EPOCHS,
        "full_validation_passes": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "amp_dtype": "float16",
        "grad_scaler": True,
        "gradient_clip": 1.0,
        "data_parallel_device_ids": device_ids,
        "initialized_from_checkpoint": None,
        "random_initialization": True,
        "wandb": None,
        "optimizer_state_saved": False,
        "resumable_training_state_saved": False,
    }
    atomic_json(OUTPUT / "training_config.json", training_config)

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch(epoch)
        validation_metrics = validate_epoch(epoch)
        last_artifact = save_model(
            OUTPUT / "last_model.pt",
            epoch,
            validation_metrics,
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            shutil.copy2(OUTPUT / "last_model.pt", OUTPUT / "best_model.pt")
            best_artifact = {
                "path": "best_model.pt",
                "sha256": sha256_file(OUTPUT / "best_model.pt"),
                "bytes": (OUTPUT / "best_model.pt").stat().st_size,
                "epoch": epoch,
                "validation_loss": best_validation_loss,
            }
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "last_artifact": last_artifact,
            "best_artifact": best_artifact,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        report = {
            "schema_version": "0032_audited_bc_training_report_v1",
            "state": "training",
            "experiment": "0033_effect_summary_semantic_foundation_pretraining",
            "model_contract": model_contract,
            "training_config": training_config,
            "dataset_inventory": inventory,
            "history": history,
            "best_artifact": best_artifact,
        }
        atomic_json(OUTPUT / "training_report.json", report)
        atomic_json(
            OUTPUT / "training_status.json",
            {
                "state": "training",
                "epoch_completed": epoch,
                "epochs_requested": EPOCHS,
                "elapsed_seconds": time.time() - started,
                "best_artifact": best_artifact,
            },
        )
        print(
            json.dumps(
                {
                    "event": "0032_epoch_complete",
                    **row,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(history) != EPOCHS or best_artifact is None:
        raise RuntimeError("0032 training did not complete every requested epoch")
    report["state"] = "complete"
    report["elapsed_seconds"] = time.time() - started
    report["epochs_completed"] = len(history)
    report["full_train_passes_completed"] = len(history)
    report["full_validation_passes_completed"] = len(history)
    atomic_json(OUTPUT / "training_report.json", report)
    atomic_json(
        OUTPUT / "training_status.json",
        {
            "state": "complete",
            "epoch_completed": len(history),
            "epochs_requested": EPOCHS,
            "elapsed_seconds": time.time() - started,
            "best_artifact": best_artifact,
        },
    )
    print(
        json.dumps(
            {
                "event": "0032_training_complete",
                "output": str(OUTPUT),
                "epochs_completed": len(history),
                "best_artifact": best_artifact,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    """
)


def main() -> None:
    notebook = nbf.read(OUTPUT, as_version=4)
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# PTCG 0032 dual-T4 full five-epoch BC\n\n"
            "Randomly initializes the audited 0032 SemanticPolicy and performs five complete "
            "training and validation passes over the five winner-only daily partitions."
        ),
        nbf.v4.new_markdown_cell("## Environment and immutable training contract"),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_markdown_cell("## Five-day dataset acceptance"),
        nbf.v4.new_code_cell(DATA),
        nbf.v4.new_markdown_cell("## Exact model and dual-GPU preflight"),
        nbf.v4.new_code_cell(MODEL),
        nbf.v4.new_markdown_cell("## Full-pass training and validation functions"),
        nbf.v4.new_code_cell(TRAINING),
        nbf.v4.new_markdown_cell("## Run exactly five full epochs"),
        nbf.v4.new_code_cell(RUN),
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    nbf.validate(notebook)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
