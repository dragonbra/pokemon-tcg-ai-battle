"""Build 0034 recent-three-day Kaggle data and quick-look BC notebooks."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
RECENT_DAYS = (
    ("01_day_0801", "01_day_0801.ipynb", "2026-08-01", "PTCG 0034 01 Day 0801"),
    ("02_day_0802", "02_day_0802.ipynb", "2026-08-02", "PTCG 0034 02 Day 0802"),
    ("03_day_0803", "03_day_0803.ipynb", "2026-08-03", "PTCG 0034 03 Day 0803"),
)
TRAIN_DIR = "04_train_recent3_rolling_t4"
TRAIN_FILE = "04_train_recent3_rolling_t4.ipynb"
SOURCE_DATASET = "horizen12/ptcg-0034-source"
PARAMETER_COUNT = "22_595_202"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _text_transform(source: str) -> str:
    source = source.replace("0032", "0034")
    source = source.replace(
        "0034_audited_semantic_foundation_pretraining",
        "0034_clean_recent3_semantic_bc",
    )
    source = source.replace(
        "0034_audited_semantic_decision_v2",
        "0034_clean_recent3_semantic_decision_v1",
    )
    source = source.replace("20_567_042", PARAMETER_COUNT)
    source = source.replace("20,567,042", "22,595,202")
    source = source.replace("ptcg_0034_{PART_NAME}", "ptcg_0034_{PART_NAME}")
    source = source.replace("horizen12/ptcg-0034-source", SOURCE_DATASET)
    source = source.replace("expected all five daily datasets", "expected all three daily datasets")
    return source


ATTACH_LABEL_AUDIT = r'''
from collections import Counter


ATTACH_OPTION_CAT_VALUE = 9  # raw engine action type 8 plus one-based categorical encoding.


def attach_label_audit(day_manifest: dict) -> dict:
    day = day_manifest["date"]
    day_root = OUTPUT_ROOT / f"date={day}"
    shard_root = day_root / "canonical"
    counts = Counter()
    selected_source_cards = Counter()
    selected_target_cards = Counter()
    available_source_cards = Counter()
    available_target_cards = Counter()

    for split in ("train", "validation"):
        for shard in day_manifest["shards"][split]:
            with gzip.open(shard_root / shard["path"], "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    actor = record["actor"]
                    options = actor["option_cat"]
                    sources = actor["option_source"]
                    targets = actor["option_target"]
                    action = record["target"]["ordered_action"]
                    attach_options = [
                        index for index, option in enumerate(options)
                        if int(option[0]) == ATTACH_OPTION_CAT_VALUE
                    ]
                    if attach_options:
                        counts[f"{split}_decisions_with_attach_available"] += 1
                        counts[f"{split}_attach_options_available"] += len(attach_options)
                    for index in attach_options:
                        if sources[index] <= 0 or targets[index] <= 0:
                            raise ValueError("Attach option relation is missing source or target")
                        available_source_cards[str(options[index][5])] += 1
                        available_target_cards[str(options[index][6])] += 1
                    selected = [
                        index for index in action
                        if 0 <= index < len(options)
                        and int(options[index][0]) == ATTACH_OPTION_CAT_VALUE
                    ]
                    if selected:
                        counts[f"{split}_decisions_with_attach_label"] += 1
                        counts[f"{split}_attach_label_tokens"] += len(selected)
                    for index in selected:
                        if sources[index] <= 0 or targets[index] <= 0:
                            raise ValueError("Selected Attach label relation is missing source or target")
                        selected_source_cards[str(options[index][5])] += 1
                        selected_target_cards[str(options[index][6])] += 1

    total_available = (
        counts["train_attach_options_available"]
        + counts["validation_attach_options_available"]
    )
    total_labels = (
        counts["train_attach_label_tokens"]
        + counts["validation_attach_label_tokens"]
    )
    return {
        "schema_version": "0034_attach_label_audit_v1",
        "date": day,
        "status": "passed",
        "attach_option_cat_value": ATTACH_OPTION_CAT_VALUE,
        "counts": dict(sorted(counts.items())),
        "total_attach_options_available": total_available,
        "total_attach_label_tokens": total_labels,
        "selected_source_card_top20": selected_source_cards.most_common(20),
        "selected_target_card_top20": selected_target_cards.most_common(20),
        "available_source_card_top20": available_source_cards.most_common(20),
        "available_target_card_top20": available_target_cards.most_common(20),
        "note": (
            "This verifies Attach source/target relation completeness and label distribution. "
            "It does not treat energy-deficit simulation as an actor input."
        ),
    }


attach_audits = []
for daily_manifest in daily_manifests:
    audit = attach_label_audit(daily_manifest)
    day = daily_manifest["date"]
    day_root = OUTPUT_ROOT / f"date={day}"
    manifest_path = day_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["attach_label_audit"] = audit
    write_json(manifest_path, manifest)
    write_json(day_root / "attach_label_audit.json", audit)
    attach_audits.append(audit)

partition_manifest_path = OUTPUT_ROOT / "partition_manifest.json"
partition_manifest = json.loads(partition_manifest_path.read_text(encoding="utf-8"))
partition_manifest["attach_label_audits"] = attach_audits
partition_manifest["attach_label_totals"] = {
    "total_attach_options_available": sum(
        item["total_attach_options_available"] for item in attach_audits
    ),
    "total_attach_label_tokens": sum(
        item["total_attach_label_tokens"] for item in attach_audits
    ),
}
write_json(partition_manifest_path, partition_manifest)
print(json.dumps(partition_manifest["attach_label_totals"], sort_keys=True), flush=True)
'''


ATTACH_METRIC_HELPERS = r'''
ATTACH_OPTION_CAT_VALUE = 9  # raw engine action type 8 plus one-based categorical encoding.


def empty_attach_counts() -> dict:
    return {
        "attach_available_decisions": 0,
        "attach_available_options": 0,
        "attach_target_tokens": 0,
        "attach_target_correct": 0,
        "attach_target_decisions": 0,
        "attach_target_exact": 0,
    }


def update_attach_counts(counts: dict, batch: dict[str, torch.Tensor], logits: torch.Tensor) -> None:
    targets = batch["targets"]
    target_mask = targets.ne(-100)
    termination = int(batch["option_mask"].shape[1])
    if termination < 1:
        return
    predictions = logits.argmax(-1)
    option_action_type = batch["option_cat"][..., 0]
    available_attach = batch["option_mask"] & option_action_type.eq(ATTACH_OPTION_CAT_VALUE)
    counts["attach_available_decisions"] += int(available_attach.any(1).sum().item())
    counts["attach_available_options"] += int(available_attach.sum().item())

    safe_targets = targets.clamp(min=0, max=termination - 1)
    batch_indices = torch.arange(targets.size(0), device=targets.device).unsqueeze(1).expand_as(targets)
    target_action_type = option_action_type[batch_indices, safe_targets]
    attach_target = target_mask & targets.lt(termination) & target_action_type.eq(ATTACH_OPTION_CAT_VALUE)
    if not bool(attach_target.any()):
        return
    target_correct = predictions.eq(targets) & attach_target
    decision_has_attach = attach_target.any(1)
    counts["attach_target_tokens"] += int(attach_target.sum().item())
    counts["attach_target_correct"] += int(target_correct.sum().item())
    counts["attach_target_decisions"] += int(decision_has_attach.sum().item())
    counts["attach_target_exact"] += int(((predictions.eq(targets) | ~attach_target).all(1) & decision_has_attach).sum().item())


def finalize_attach_counts(counts: dict) -> dict:
    return {
        **counts,
        "attach_target_token_accuracy": counts["attach_target_correct"] / max(1, counts["attach_target_tokens"]),
        "attach_target_exact_action": counts["attach_target_exact"] / max(1, counts["attach_target_decisions"]),
    }
'''


TRAINING_FIND_SOURCE_ROOT = r'''def find_source_root() -> Path:
    package_name = "0034_clean_recent3_semantic_bc"
    package_candidates = []
    flat_candidates = []
    for public_asset in INPUT.rglob("official_public_prototypes_v1.json"):
        root = public_asset.parent.parent
        if (
            (root / "features" / "audit.py").is_file()
            and (root / "training" / "dataset.py").is_file()
        ):
            if root.name == package_name:
                package_candidates.append(root)
            else:
                flat_candidates.append(root)
    package_unique = sorted(set(package_candidates))
    if len(package_unique) == 1:
        return package_unique[0]
    flat_unique = sorted(set(flat_candidates))
    if len(flat_unique) == 1:
        staged = WORKING / package_name
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(flat_unique[0], staged)
        return staged
    raise FileNotFoundError(
        "expected one complete 0034 source root or one flat source dataset, "
        f"found package={package_unique}, flat={flat_unique}"
    )'''


def _replace_expected_dates(source: str) -> str:
    dates = "{\n        " + ",\n        ".join(repr(day) for _, _, day, _ in RECENT_DAYS) + ",\n    }"
    return re.sub(r"EXPECTED_DATES = \{.*?\}", f"EXPECTED_DATES = {dates}", source, flags=re.S)


def _transform_daily_notebook(notebook: nbf.NotebookNode) -> nbf.NotebookNode:
    for cell in notebook.cells:
        cell.source = _text_transform(cell.source)
    notebook.cells.append(nbf.v4.new_markdown_cell("## Attach label audit"))
    notebook.cells.append(nbf.v4.new_code_cell(ATTACH_LABEL_AUDIT))
    nbf.validate(notebook)
    return notebook


def build_daily_notebooks() -> list[str]:
    daily_builder = _load_module(ROOT / "build_daily_notebooks.py", "_0034_daily_builder")
    created: list[str] = []
    for directory, filename, day, title in RECENT_DAYS:
        target_dir = ROOT / directory
        target_dir.mkdir(parents=True, exist_ok=True)
        notebook = daily_builder.build_notebook(directory, [day])
        notebook = _transform_daily_notebook(notebook)
        notebook_path = target_dir / filename
        nbf.write(notebook, notebook_path)
        metadata = {
            "id": f"horizen12/ptcg-0034-{directory.replace('_', '-')}",
            "title": title,
            "code_file": filename,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "machine_shape": "None",
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["pokemon-tcg", "0034", "effect-summary", "winner-bc"],
            "dataset_sources": [
                f"kaggle/pokemon-tcg-ai-battle-episodes-{day}",
                SOURCE_DATASET,
            ],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (target_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        created.append(str(notebook_path))
    return created


def _transform_training_cell(source: str) -> str:
    source = _text_transform(source)
    source = _replace_expected_dates(source)
    if "def find_source_root() -> Path:" in source and "training.dataset" in source:
        source = re.sub(
            r"def find_source_root\(\) -> Path:\n.*?\n\n\nSOURCE_ROOT = find_source_root\(\)",
            TRAINING_FIND_SOURCE_ROOT + "\n\n\nSOURCE_ROOT = find_source_root()",
            source,
            flags=re.S,
        )
    source = source.replace("EPOCHS = 1 if SMOKE else 5", "EPOCHS = 1")
    source = source.replace("TRAIN_BATCH_SIZE = 512 if SMOKE else 192", "TRAIN_BATCH_SIZE = 512 if SMOKE else 192")
    source = source.replace("VALIDATION_BATCH_SIZE = 512 if SMOKE else 256", "VALIDATION_BATCH_SIZE = 512 if SMOKE else 256")
    source = source.replace("ptcg_0034_rolling_t4_train", "ptcg_0034_recent3_rolling_t4_train")
    source = source.replace(
        "if not SMOKE and EPOCHS != 5:\n        raise ValueError(\"formal 0034 training must use exactly five epochs\")",
        "if EPOCHS != 1:\n        raise ValueError(\"recent-three-day quick-look training must use exactly one rolling pass\")",
    )
    source = source.replace(
        "if dataset.manifest.get(\"partition_count\") != 5:\n    raise ValueError(\"combined dataset did not retain all five daily partitions\")",
        "if dataset.manifest.get(\"partition_count\") != 3:\n    raise ValueError(\"combined dataset did not retain all three daily partitions\")",
    )
    source = source.replace(
        "\"dataset_interval\": [\"2026-07-28\", \"2026-08-01\"],",
        "\"dataset_interval\": [\"2026-08-01\", \"2026-08-03\"],",
    )
    if "def validate_dataset" in source and "ATTACH_OPTION_CAT_VALUE" not in source:
        source = source.replace("def validate_dataset", ATTACH_METRIC_HELPERS + "\n\ndef validate_dataset")
        source = source.replace(
            "    decisions = 0\n    started = time.time()",
            "    decisions = 0\n    attach_counts = empty_attach_counts()\n    started = time.time()",
        )
        source = source.replace(
            "                matches = logits.argmax(-1).eq(batch[\"targets\"]) & mask\n                loss_sum",
            "                update_attach_counts(attach_counts, batch, logits)\n                matches = logits.argmax(-1).eq(batch[\"targets\"]) & mask\n                loss_sum",
        )
        source = source.replace(
            "\"seconds\": time.time() - started,\n    }",
            "\"seconds\": time.time() - started,\n        \"attach\": finalize_attach_counts(attach_counts),\n    }",
            1,
        )
    return source


def build_training_notebook() -> str:
    source_path = ROOT / "06_train_rolling_t4" / "06_train_rolling_t4.ipynb"
    notebook = nbf.read(source_path, as_version=4)
    notebook.cells[0].source = (
        "# PTCG 0034 recent-three-day rolling BC\n\n"
        "Randomly initializes the 0034 effect-summary SemanticPolicy and rolls once "
        "through the three latest complete winner-only daily partitions, reporting "
        "overall BC metrics plus Attach-only source/target decision slices."
    )
    for cell in notebook.cells:
        cell.source = _transform_training_cell(cell.source)
        cell.source = cell.source.replace("five accepted", "three latest complete")
        cell.source = cell.source.replace("Five-day", "Three-day")
        cell.source = cell.source.replace("five-day", "three-day")
        cell.source = cell.source.replace("five full", "one rolling")
        cell.source = cell.source.replace("five winner-only", "three winner-only")
    target_dir = ROOT / TRAIN_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / TRAIN_FILE
    nbf.validate(notebook)
    nbf.write(notebook, target_path)
    metadata = {
        "id": "horizen12/ptcg-0034-04-train-recent3-rolling-t4",
        "title": "PTCG 0034 04 Train Recent3 Rolling T4",
        "code_file": TRAIN_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["pokemon-tcg", "0034", "rolling", "attach-bc"],
        "dataset_sources": [SOURCE_DATASET],
        "competition_sources": [],
        "kernel_sources": [
            "horizen12/ptcg-0034-01-day-0801",
            "horizen12/ptcg-0034-02-day-0802",
            "horizen12/ptcg-0034-03-day-0803",
        ],
        "model_sources": [],
    }
    (target_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(target_path)


def main() -> None:
    created = build_daily_notebooks()
    created.append(build_training_notebook())
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
