"""Build the 0034 merged recent-three-day cleaning notebook."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
DATES = ["2026-08-01", "2026-08-02", "2026-08-03"]
SOURCE_DATASET = "horizen12/ptcg-0033-source"


def _load_builder():
    path = ROOT / "build_daily_notebooks.py"
    spec = importlib.util.spec_from_file_location("daily_builder_0034", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLEANING_CELL = r'''
import io

ATTACH_OPTION_CAT_VALUE = 9
PLAY_OPTION_CAT_VALUE = 8
EVOLVE_OPTION_CAT_VALUE = 10
ABILITY_OPTION_CAT_VALUE = 11
DISCARD_OPTION_CAT_VALUE = 12
RETREAT_OPTION_CAT_VALUE = 13
ATTACK_OPTION_CAT_VALUE = 14
SKILL_OPTION_CAT_VALUE = 16
SWITCH_CONTEXT_CAT_VALUE = 4  # raw SelectContext.SWITCH + one-based encoding
OUTPUT_SCHEMA_VERSION = "0034_clean_recent3_semantic_decision_v1"
PROTOTYPE_ASSETS = (
    "official_public_prototypes_v1.json",
    "official_full_engine_prototypes_v2.json",
)
RELATION_FIELDS = (
    "card_parent",
    "event_source",
    "event_target",
    "option_source",
    "option_target",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_gzip_rows(path: Path, rows: list[dict]) -> dict:
    temporary = path.with_name(path.name + ".clean.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
    temporary.replace(path)
    return {
        "bytes": path.stat().st_size,
        "count": len(rows),
        "path": path.name,
        "sha256": sha256_file(path),
    }


def _append_gzip_rows(path: Path, rows: list[dict]) -> None:
    """Append one deterministic gzip member so audits are streamed, not held in RAM."""
    with path.open("ab") as raw_handle:
        compressed = gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0)
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
                handle.write("\n")


def _actor_relation_counts(actor: dict) -> Counter:
    return Counter({
        name: sum(int(value) > 0 for value in actor[name])
        for name in RELATION_FIELDS
    })


OPTION_AREA_TO_ZONE = {
    2: 7,   # deck
    3: 3,   # hand
    4: 4,   # discard
    5: 1,   # active
    6: 2,   # bench
    8: 5,   # stadium
    13: 6,  # looking
}
CHILD_ZONES = frozenset({8, 9, 10})
TRAINER_CARD_TYPES = frozenset({1, 2, 3, 4})


def _effective_zone(actor: dict, relation: int) -> int:
    if relation <= 0 or relation > len(actor["card_cat"]):
        return 0
    card = actor["card_cat"][relation - 1]
    zone = int(card[2])
    if zone in CHILD_ZONES:
        parent = int(actor["card_parent"][relation - 1])
        if parent > 0 and parent <= len(actor["card_cat"]):
            return int(actor["card_cat"][parent - 1][2])
    return zone


def _relation_status(actor: dict, option: list[int], relation: int, side: str) -> tuple[bool, str]:
    relation = int(relation)
    identity_column = 5 if side == "source" else 6
    owner_column = 1 if side == "source" else 3
    area_column = 2 if side == "source" else 4
    expected_id = int(option[identity_column])
    if expected_id <= 0:
        return relation <= 0, "not_required" if relation <= 0 else "unexpected_relation"
    if relation <= 0 or relation > len(actor["card_cat"]):
        return False, "missing_instance"
    card = actor["card_cat"][relation - 1]
    if int(card[0]) != expected_id:
        return False, "identity_mismatch"
    expected_owner = int(option[owner_column])
    if expected_owner > 0 and int(card[1]) != expected_owner:
        return False, "owner_mismatch"
    expected_zone = OPTION_AREA_TO_ZONE.get(int(option[area_column]))
    if expected_zone is not None and _effective_zone(actor, relation) != expected_zone:
        return False, "area_mismatch"
    return True, "valid"


def _energy_facts(card: dict | None) -> dict:
    if not isinstance(card, dict) or int(card.get("card_type", -1)) not in {5, 6}:
        return {
            "card_type": None if not isinstance(card, dict) else int(card.get("card_type", -1)),
            "prototype_type_mask": None,
            "prototype_units": None,
            "attach_provides_type_mask": None,
            "attach_provides_units": None,
            "confidence": "not_energy",
            "resolution": "not_an_energy_card",
        }
    card_type = int(card.get("card_type", -1))
    static_mask = int(card.get("energy_type_mask", 0))
    static_units = int(card.get("energy_count", 0))
    if card_type == 5:
        return {
            "card_type": card_type,
            "prototype_type_mask": static_mask,
            "prototype_units": static_units,
            "attach_provides_type_mask": static_mask,
            "attach_provides_units": static_units,
            "confidence": "high",
            "resolution": "basic_energy_engine_fact",
        }
    return {
        "card_type": card_type,
        "prototype_type_mask": static_mask,
        "prototype_units": static_units,
        # Conditional Special Energy is deliberately not presented as a resolved answer.
        "attach_provides_type_mask": None,
        "attach_provides_units": None,
        "confidence": "low",
        "resolution": "special_energy_requires_target_engine_resolution",
    }


def _action_audit(record: dict) -> dict:
    actor = record["actor"]
    target = record["target"]
    options = actor["option_cat"]
    selected = [int(value) for value in target["ordered_action"]]
    context = int(actor["global_cat"][1]) if len(actor["global_cat"]) > 1 else 0
    classes = Counter()
    relation_errors = Counter()
    selected_actions = []
    high_confidence_attach = 0
    low_confidence_attach = 0
    for index in selected:
        option = [int(value) for value in options[index]]
        action_type = int(option[0])
        source_ok, source_reason = _relation_status(
            actor, option, int(actor["option_source"][index]), "source"
        )
        target_ok, target_reason = _relation_status(
            actor, option, int(actor["option_target"][index]), "target"
        )
        if not source_ok:
            relation_errors[f"source_{source_reason}"] += 1
        if not target_ok:
            relation_errors[f"target_{target_reason}"] += 1
        source_id = int(option[5])
        source_relation = int(actor["option_source"][index])
        if source_relation > 0 and source_relation <= len(actor["card_cat"]):
            source_id = int(actor["card_cat"][source_relation - 1][0])
        source_card = prototypes.engine_cards.get(source_id)
        if action_type == ATTACH_OPTION_CAT_VALUE:
            action_class = "attach"
            energy = _energy_facts(source_card)
            if energy["confidence"] == "high":
                high_confidence_attach += 1
            elif energy["confidence"] == "low":
                low_confidence_attach += 1
        elif action_type == ATTACK_OPTION_CAT_VALUE:
            action_class = "attack"
        elif context == SWITCH_CONTEXT_CAT_VALUE:
            action_class = "switch"
        elif action_type == PLAY_OPTION_CAT_VALUE and isinstance(source_card, dict) and int(source_card.get("card_type", -1)) in TRAINER_CARD_TYPES:
            action_class = "trainer"
        else:
            action_class = "other"
        classes[action_class] += 1
        entry = {
            "option_index": index,
            "action_type": action_type,
            "action_class": action_class,
            "source_exists": source_ok,
            "source_reason": source_reason,
            "target_exists": target_ok,
            "target_reason": target_reason,
            "execution_failed": None,
            "execution_status": "unknown_not_observable_from_actor_sample",
        }
        if action_class == "attach":
            entry["energy"] = energy
        elif action_class == "attack":
            entry.update({
                "attack_id": int(option[7]),
                "obviously_ineffective": None,
                "effect_result_status": "unknown_requires_post_action_engine_trace",
            })
        elif action_class == "switch":
            entry["ko_avoided_by_switch"] = None
            entry["counterfactual_status"] = "unknown_requires_post_action_engine_trace"
        elif action_class == "trainer":
            entry["trainer_card_type"] = int(source_card.get("card_type", -1))
        selected_actions.append(entry)
    if relation_errors:
        classes["relation_error"] += sum(relation_errors.values())
    if high_confidence_attach:
        sample_weight = 1.15
        weight_reason = "high_confidence_basic_energy_attach"
    elif low_confidence_attach:
        sample_weight = 0.95
        weight_reason = "conditional_special_energy_attach"
    else:
        sample_weight = 1.0
        weight_reason = "ordinary_or_unresolved_action"
    return {
        "schema_version": "0034_action_audit_v1",
        "selected_actions": selected_actions,
        "selected_action_count": len(selected_actions),
        "action_class_counts": dict(sorted(classes.items())),
        "relation_errors": dict(sorted(relation_errors.items())),
        "sample_weight": sample_weight,
        "weight_reason": weight_reason,
        "execution_failure_observable": False,
        "execution_failure_note": "Actor samples contain pre-action state and selected legal options; post-action immunity/failure is not fabricated.",
    }


def _validate_record(record: dict) -> tuple[bool, str, dict]:
    actor = record.get("actor")
    target = record.get("target")
    if not isinstance(actor, dict) or set(actor) != fields.ACTOR_KEYS:
        return False, "actor_schema", {}
    if not isinstance(target, dict) or not isinstance(target.get("ordered_action"), list):
        return False, "target_schema", {}
    options = actor["option_cat"]
    sources = actor["option_source"]
    targets = actor["option_target"]
    cards = actor["card_cat"]
    if not options or len(options) != len(sources) or len(options) != len(targets):
        return False, "empty_or_misaligned_options", {}
    if not int(actor["min_count"]) <= len(target["ordered_action"]) <= int(actor["max_count"]):
        return False, "target_count_bounds", {}
    selected = [int(value) for value in target["ordered_action"]]
    if len(set(selected)) != len(selected) or any(value < 0 or value >= len(options) for value in selected):
        return False, "illegal_target", {}
    for index, option in enumerate(options):
        if len(option) != fields.WIDTHS.option_cat:
            return False, "option_width", {}
        if any(int(value) < 0 or int(value) >= int(vocab) for value, vocab in zip(option, fields.OPTION_CAT_VOCABS, strict=True)):
            return False, "option_category_range", {}
        for relation_name, relation_value in (("source", sources[index]), ("target", targets[index])):
            if int(relation_value) < 0 or int(relation_value) > len(cards):
                return False, f"{relation_name}_relation_range", {}
        if int(option[0]) == ATTACH_OPTION_CAT_VALUE and (int(sources[index]) <= 0 or int(targets[index]) <= 0):
            return False, "attach_relation_missing", {}
    action_audit = _action_audit(record)
    selected_actions = action_audit["selected_actions"]
    for action in selected_actions:
        if not action["source_exists"] and int(action["action_type"]) in {
            PLAY_OPTION_CAT_VALUE,
            ATTACH_OPTION_CAT_VALUE,
            EVOLVE_OPTION_CAT_VALUE,
            ABILITY_OPTION_CAT_VALUE,
            DISCARD_OPTION_CAT_VALUE,
            RETREAT_OPTION_CAT_VALUE,
            ATTACK_OPTION_CAT_VALUE,
            SKILL_OPTION_CAT_VALUE,
        }:
            return False, "selected_source_relation_invalid", {}
        if not action["target_exists"] and int(action["action_type"]) in {
            ATTACH_OPTION_CAT_VALUE,
            EVOLVE_OPTION_CAT_VALUE,
            ATTACK_OPTION_CAT_VALUE,
        }:
            return False, "selected_target_relation_invalid", {}
    selected_attach = [index for index in selected if int(options[index][0]) == ATTACH_OPTION_CAT_VALUE]
    attach_valid = all(int(sources[index]) > 0 and int(targets[index]) > 0 for index in selected_attach)
    return True, "accepted", {
        "selected_attach_count": len(selected_attach),
        "attach_relation_valid": attach_valid,
        "sample_weight": action_audit["sample_weight"],
        "action_audit": action_audit,
    }


def clean_partition(day: str) -> dict:
    day_root = OUTPUT_ROOT / f"date={day}"
    daily_path = day_root / "manifest.json"
    daily_manifest = _read_json(daily_path)
    canonical_root = day_root / daily_manifest.get("canonical_path", "canonical")
    canonical_path = canonical_root / "manifest.json"
    canonical_manifest = _read_json(canonical_path)
    report = Counter()
    split_counts = {}
    new_shards = {}
    samples = []
    action_audit_path = day_root / "action_audit.jsonl.gz"
    if action_audit_path.exists():
        action_audit_path.unlink()
    action_audit_count = 0
    sample_weight_counts = Counter()
    relation_counts = Counter({name: 0 for name in RELATION_FIELDS})
    for split in ("train", "validation"):
        kept = 0
        new_shards[split] = []
        for shard in canonical_manifest["shards"][split]:
            source = canonical_root / shard["path"]
            rows = []
            audit_rows = []
            with gzip.open(source, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    valid, reason, quality = _validate_record(record)
                    if not valid:
                        report[f"rejected_{reason}"] += 1
                        continue
                    actor = record["actor"]
                    # Full event history is intentionally not part of this ablation.
                    # Current board, status, relation, and rule prototype facts remain.
                    for key in ("event_cat", "event_num", "event_state", "event_source", "event_target"):
                        actor[key] = []
                    relation_counts.update(_actor_relation_counts(actor))
                    audit = dict(record.get("audit", {}))
                    action_audit = quality["action_audit"]
                    audit["cleaning"] = {
                        "schema_valid": True,
                        "event_history_compacted": True,
                        **{key: value for key, value in quality.items() if key != "action_audit"},
                    }
                    audit["action_audit"] = action_audit
                    record["schema_version"] = OUTPUT_SCHEMA_VERSION
                    record["audit"] = audit
                    record["sample_weight"] = float(action_audit["sample_weight"])
                    sample_weight_counts[str(action_audit["sample_weight"])] += 1
                    rows.append(record)
                    action_audit_row = {
                        "identity": audit.get("identity", {}),
                        "split": split,
                        "schema_version": "0034_action_audit_v1",
                        "action_audit": action_audit,
                        "sample_weight": float(action_audit["sample_weight"]),
                    }
                    audit_rows.append(action_audit_row)
                    for action_class, count in action_audit["action_class_counts"].items():
                        report[f"selected_{action_class}_actions"] += int(count)
                    for relation_reason, count in action_audit["relation_errors"].items():
                        report[f"relation_error_{relation_reason}"] += int(count)
                    kept += 1
                    if len(samples) < 4:
                        samples.append(record)
            _append_gzip_rows(action_audit_path, audit_rows)
            action_audit_count += len(audit_rows)
            new_shards[split].append(_write_gzip_rows(source, rows))
        split_counts[split] = kept
    action_audit_file = {
        "bytes": action_audit_path.stat().st_size,
        "count": action_audit_count,
        "path": action_audit_path.name,
        "sha256": sha256_file(action_audit_path),
    }
    report["input_records"] = sum(int(value) for value in canonical_manifest["split_counts"].values())
    report["kept_records"] = sum(split_counts.values())
    report["rejected_records"] = report["input_records"] - report["kept_records"]
    report["event_history_compacted_records"] = report["kept_records"]
    report["sample_weight_distribution"] = dict(sorted(sample_weight_counts.items()))
    report["relation_counts"] = {
        name: int(relation_counts[name]) for name in RELATION_FIELDS
    }
    report["status"] = "passed"
    report["schema_version"] = "0034_cleaning_report_v1"
    report["date"] = day
    report = dict(sorted(report.items()))

    canonical_manifest["schema_version"] = OUTPUT_SCHEMA_VERSION
    canonical_manifest["shards"] = new_shards
    canonical_manifest["decisions"] = sum(split_counts.values())
    canonical_manifest["split_counts"] = split_counts
    canonical_manifest["feature_input_audit"] = {
        **canonical_manifest.get("feature_input_audit", {}),
        "status": "passed",
        "audited_decisions": sum(split_counts.values()),
        "exact_event_relations": 0,
        "scope": "all cleaned canonical decisions before shard write",
    }
    canonical_manifest["cleaning"] = report
    write_json(canonical_path, canonical_manifest)

    daily_manifest["schema_version"] = "0034_daily_semantic_partition_v1"
    daily_manifest["actor_schema"] = OUTPUT_SCHEMA_VERSION
    daily_manifest["records"] = canonical_manifest["decisions"]
    daily_manifest["split_counts"] = split_counts
    daily_manifest["shards"] = new_shards
    daily_manifest["canonical_manifest_sha256"] = sha256_file(canonical_path)
    daily_manifest["feature_input_audit"] = canonical_manifest["feature_input_audit"]
    daily_manifest["cleaning"] = report
    daily_manifest["action_audit"] = action_audit_file
    daily_manifest["acceptance"] = {
        **daily_manifest.get("acceptance", {}),
        "status": "passed",
        "validated_records": canonical_manifest["decisions"],
        "cleaned": True,
        "relation_counts": {
            name: int(relation_counts[name]) for name in RELATION_FIELDS
        },
    }
    write_json(daily_path, daily_manifest)
    write_json(day_root / "cleaning_report.json", report)
    return {
        "date": day,
        "records": daily_manifest["records"],
        "split_counts": split_counts,
        "cleaning": report,
        "action_audit": action_audit_file,
        "forward_samples": samples,
    }


cleaned_manifests = [clean_partition(day) for day in TARGET_DATES]
forward_samples = [
    record
    for item in cleaned_manifests
    for record in item["forward_samples"]
]
partition_path = OUTPUT_ROOT / "partition_manifest.json"
partition_manifest = _read_json(partition_path)
partition_manifest["schema_version"] = "0034_clean_recent3_partition_v1"
partition_manifest["experiment"] = "0034_clean_recent3_semantic_bc"
partition_manifest["cleaning"] = {
    "schema_version": "0034_cleaning_report_v1",
    "status": "passed",
    "event_history_compacted": True,
    "dates": TARGET_DATES,
    "input_records": sum(item["cleaning"]["input_records"] for item in cleaned_manifests),
    "kept_records": sum(item["records"] for item in cleaned_manifests),
    "rejected_records": sum(item["cleaning"]["rejected_records"] for item in cleaned_manifests),
}
partition_manifest["total_records"] = sum(item["records"] for item in cleaned_manifests)
partition_manifest["day_manifests"] = [
    {
        "date": item["date"],
        "records": item["records"],
        "split_counts": item["split_counts"],
        "manifest_path": f"date={item['date']}/manifest.json",
        "manifest_sha256": sha256_file(OUTPUT_ROOT / f"date={item['date']}" / "manifest.json"),
        "action_audit": item["action_audit"],
    }
    for item in cleaned_manifests
]
write_json(partition_path, partition_manifest)
write_json(OUTPUT_ROOT / "cleaning_report.json", partition_manifest["cleaning"])
for asset in PROTOTYPE_ASSETS:
    shutil.copy2(SOURCE_ROOT / "assets" / asset, OUTPUT_ROOT / asset)
feature_audit = _read_json(SOURCE_ROOT / "contracts" / "feature_audit.json")
feature_audit["actor_schema"] = OUTPUT_SCHEMA_VERSION
write_json(OUTPUT_ROOT / "feature_audit.json", feature_audit)

forward_batch = batch_contract.DecisionBatch.from_mapping(
    collate_module.collate_canonical_records(forward_samples)
)
prototypes = prototypes_module.PrototypeIndex.load(PROTOTYPE_PATH)
model = model_module.SemanticPolicy(model_module.ModelConfig(), prototypes).eval()
with torch.inference_mode():
    logits = model.teacher_logits(forward_batch)
if not torch.isfinite(logits).all():
    raise ValueError("cleaned model-forward validation produced non-finite logits")
print(json.dumps({
    "event": "0034_clean_recent3_complete",
    "dates": TARGET_DATES,
    "total_records": partition_manifest["total_records"],
    "cleaning": partition_manifest["cleaning"],
    "forward": {"rows": len(forward_samples), "finite": True, "logits_shape": list(logits.shape)},
}, sort_keys=True), flush=True)
'''


def _source(cell: nbf.NotebookNode) -> str:
    value = cell.source
    return value if isinstance(value, str) else "".join(value)


def build() -> Path:
    daily_builder = _load_builder()
    notebook = daily_builder.build_notebook("01_clean_recent3_data", DATES)
    notebook.cells[0].source = (
        "# PTCG 0034 clean recent-three-day winner-only data\n\n"
        "Reads the latest three Top battle partitions, performs conservative schema/relation "
        "cleaning, compacts full event history, and preserves prototype/feature audit assets."
    )
    for cell in notebook.cells:
        source = _source(cell)
        source = source.replace("20_567_042", "22_595_202")
        source = source.replace("0032 model", "0034 model")
        source = source.replace("0032 field", "0034 field")
        source = source.replace(
            'SCHEMA_VERSION = fields.SCHEMA_VERSION\n',
            'INPUT_SCHEMA_VERSION = fields.SCHEMA_VERSION\nSCHEMA_VERSION = INPUT_SCHEMA_VERSION\nOUTPUT_SCHEMA_VERSION = "0034_clean_recent3_semantic_decision_v1"\n',
        )
        source = source.replace(
            'if SCHEMA_VERSION != "0034_clean_recent3_semantic_decision_v1":\n    raise ValueError(f"unexpected actor schema: {SCHEMA_VERSION}")',
            'if INPUT_SCHEMA_VERSION not in {"0033_effect_summary_semantic_decision_v1", OUTPUT_SCHEMA_VERSION}:\n    raise ValueError(f"unexpected actor schema: {INPUT_SCHEMA_VERSION}")',
        )
        cell.source = source
    notebook.cells.append(nbf.v4.new_markdown_cell("## Conservative cleaning and package assets"))
    notebook.cells.append(nbf.v4.new_code_cell(CLEANING_CELL))
    target_dir = ROOT / "01_clean_recent3_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "01_clean_recent3_data.ipynb"
    nbf.validate(notebook)
    nbf.write(notebook, target)
    metadata = {
        "id": "horizen12/ptcg-0034-01-clean-recent3-data",
        "title": "PTCG 0034 01 Clean Recent3 Data",
        "code_file": target.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "machine_shape": "None",
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["pokemon-tcg", "0034", "cleaning", "winner-bc"],
        "dataset_sources": [
            "kaggle/pokemon-tcg-ai-battle-episodes-2026-08-01",
            "kaggle/pokemon-tcg-ai-battle-episodes-2026-08-02",
            "kaggle/pokemon-tcg-ai-battle-episodes-2026-08-03",
            SOURCE_DATASET,
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (target_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


if __name__ == "__main__":
    print(build())
