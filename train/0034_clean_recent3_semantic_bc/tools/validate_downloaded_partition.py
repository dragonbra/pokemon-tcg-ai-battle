"""Validate every record in a downloaded 0032 Kaggle daily partition."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import time
from typing import Any

import torch

try:
    import ujson as fast_json
except ImportError:  # pragma: no cover - optional local acceleration
    fast_json = json


BASE = "train.0034_clean_recent3_semantic_bc"
EXPECTED_PARAMETER_COUNT = 22_595_202
IN_PLAY_ZONES = frozenset({1, 2})
CHILD_ZONES = frozenset({8, 9, 10})
ENERGY_ZONE = 8
ATTACH_ACTION = 9
EXPECTED_DAYS = ("2026-08-01", "2026-08-02", "2026-08-03")
EXPECTED_SAMPLE_WEIGHTS = frozenset({0.95, 1.0, 1.15})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_cleaning_report(
    cleaning: dict[str, Any],
    *,
    expected_day: str,
    records: int,
) -> Counter[float]:
    _require(
        cleaning.get("schema_version") == "0034_cleaning_report_v1",
        "cleaning schema mismatch",
    )
    _require(cleaning.get("status") == "passed", "cleaning report did not pass")
    _require(cleaning.get("date") == expected_day, "cleaning report date mismatch")
    input_records = int(cleaning.get("input_records", -1))
    kept_records = int(cleaning.get("kept_records", -1))
    rejected_records = int(cleaning.get("rejected_records", -1))
    _require(kept_records == records, "cleaning kept-record count mismatch")
    _require(
        input_records == kept_records + rejected_records,
        "cleaning record accounting mismatch",
    )
    _require(
        int(cleaning.get("event_history_compacted_records", -1)) == kept_records,
        "event-history compaction count mismatch",
    )
    rejected_by_reason = sum(
        int(value)
        for key, value in cleaning.items()
        if key.startswith("rejected_") and key != "rejected_records"
    )
    _require(
        rejected_by_reason == rejected_records,
        "rejected-sample reasons do not reconcile",
    )
    raw_distribution = cleaning.get("sample_weight_distribution", {})
    _require(
        isinstance(raw_distribution, dict),
        "sample-weight distribution is missing",
    )
    distribution = Counter(
        {float(weight): int(count) for weight, count in raw_distribution.items()}
    )
    _require(
        set(distribution) == EXPECTED_SAMPLE_WEIGHTS,
        "sample-weight classes are incomplete",
    )
    _require(
        all(count > 0 for count in distribution.values()),
        "sample-weight class is empty",
    )
    _require(
        sum(distribution.values()) == kept_records,
        "sample-weight count mismatch",
    )
    return distribution


def validate_partition(
    day_root: Path,
    *,
    expected_day: str,
    progress_every: int = 10,
) -> dict[str, Any]:
    contracts = importlib.import_module(f"{BASE}.contracts")
    fields = importlib.import_module(f"{BASE}.contracts.fields")
    collate = importlib.import_module(f"{BASE}.features.collate")
    model_module = importlib.import_module(f"{BASE}.model")
    prototypes_module = importlib.import_module(f"{BASE}.domain.prototypes")

    canonical_root = day_root / "canonical"
    daily = _load_json(day_root / "manifest.json")
    canonical = _load_json(canonical_root / "manifest.json")
    _require(
        daily.get("schema_version") == "0034_daily_semantic_partition_v1",
        "daily manifest schema mismatch",
    )
    _require(daily.get("date") == expected_day, "daily manifest date mismatch")
    _require(daily.get("status") == "complete", "daily partition is incomplete")
    _require(daily.get("winner_only") is True, "daily partition is not winner-only")
    _require(
        canonical.get("schema_version") == fields.SCHEMA_VERSION,
        "canonical manifest schema mismatch",
    )
    _require(canonical.get("status") == "complete", "canonical partition is incomplete")
    _require(
        daily.get("canonical_manifest_sha256")
        == _sha256(canonical_root / "manifest.json"),
        "canonical manifest commitment mismatch",
    )
    _require(daily.get("records") == canonical.get("decisions"), "record count drift")
    _require(
        daily.get("split_counts") == canonical.get("split_counts"),
        "split count drift",
    )
    records = int(daily["records"])
    cleaning = daily.get("cleaning", {})
    _require(
        cleaning == canonical.get("cleaning"),
        "daily/canonical cleaning report mismatch",
    )
    declared_sample_weights = _validate_cleaning_report(
        cleaning,
        expected_day=expected_day,
        records=records,
    )
    split_counts = daily["split_counts"]
    _require(
        all(int(split_counts.get(split, 0)) > 0 for split in ("train", "validation")),
        "daily partition has an empty split",
    )
    action_audit = daily.get("action_audit", {})
    action_audit_name = action_audit.get("path")
    _require(isinstance(action_audit_name, str), "action audit manifest is missing")
    action_audit_path = day_root / action_audit_name
    _require(
        action_audit_path.is_file() and action_audit_path.stat().st_size > 0,
        "action audit file is missing/empty",
    )
    _require(
        action_audit_path.stat().st_size == int(action_audit["bytes"]),
        "action audit byte mismatch",
    )
    _require(_sha256(action_audit_path) == action_audit["sha256"], "action audit SHA256 mismatch")
    external_action_audits = 0
    external_audit_splits: Counter[str] = Counter()
    external_sample_weights: Counter[float] = Counter()
    with gzip.open(action_audit_path, "rt", encoding="utf-8") as audit_handle:
        for line in audit_handle:
            item = fast_json.loads(line)
            _require(item.get("schema_version") == "0034_action_audit_v1", "action audit schema mismatch")
            _require(item.get("split") in {"train", "validation"}, "action audit split mismatch")
            _require(
                item.get("identity", {}).get("date") == expected_day,
                "action audit date mismatch",
            )
            nested_action = item.get("action_audit", {})
            _require(
                nested_action.get("schema_version") == "0034_action_audit_v1",
                "nested action audit schema mismatch",
            )
            weight = float(item.get("sample_weight", -1))
            _require(
                weight in EXPECTED_SAMPLE_WEIGHTS,
                "external action audit has invalid weight",
            )
            _require(
                weight == float(nested_action.get("sample_weight", -1)),
                "external action audit weight mismatch",
            )
            external_audit_splits[str(item["split"])] += 1
            external_sample_weights[weight] += 1
            external_action_audits += 1
    _require(external_action_audits == daily["records"], "action audit line count mismatch")
    _require(
        dict(external_audit_splits) == split_counts,
        "action audit split counts mismatch",
    )
    _require(
        external_sample_weights == declared_sample_weights,
        "external action audit weights mismatch",
    )
    feature_audit = daily.get("feature_input_audit", {})
    acceptance = daily.get("acceptance", {})
    _require(feature_audit.get("status") == "passed", "feature audit did not pass")
    _require(
        feature_audit.get("audited_decisions") == daily.get("records"),
        "feature audit did not cover every decision",
    )
    _require(acceptance.get("status") == "passed", "Kaggle acceptance did not pass")
    _require(
        acceptance.get("validated_records") == daily.get("records"),
        "Kaggle acceptance did not cover every record",
    )

    counts: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    observed_action_audits = 0
    sample_weights: Counter[float] = Counter()
    shard_total = sum(
        len(canonical["shards"][split]) for split in ("train", "validation")
    )
    shard_number = 0
    started = time.perf_counter()

    for split in ("train", "validation"):
        for item in canonical["shards"][split]:
            shard_number += 1
            name = item.get("path")
            _require(
                isinstance(name, str) and Path(name).name == name,
                "unsafe canonical shard path",
            )
            path = canonical_root / name
            _require(path.is_file(), f"missing canonical shard: {path}")
            _require(path.stat().st_size == int(item["bytes"]), "shard byte mismatch")
            _require(_sha256(path) == item["sha256"], "shard SHA256 mismatch")
            observed = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = fast_json.loads(line)
                    observed += 1
                    _require(
                        record.get("schema_version") == fields.SCHEMA_VERSION,
                        "record schema mismatch",
                    )
                    audit = record.get("audit", {})
                    identity = audit.get("identity", {})
                    _require(identity.get("date") == expected_day, "cross-date record")
                    _require(audit.get("split") == split, "record split mismatch")
                    actor = record.get("actor", {})
                    _require(set(actor) == fields.ACTOR_KEYS, "actor key set mismatch")
                    _require(
                        all(
                            not actor[name]
                            for name in (
                                "event_cat",
                                "event_num",
                                "event_state",
                                "event_source",
                                "event_target",
                            )
                        ),
                        "event history was not compacted",
                    )
                    cleaning = audit.get("cleaning", {})
                    _require(cleaning.get("event_history_compacted") is True, "cleaning audit missing")
                    weight = float(record.get("sample_weight", cleaning.get("sample_weight", 1.0)))
                    action_audit_record = audit.get("action_audit", {})
                    _require(
                        action_audit_record.get("schema_version")
                        == "0034_action_audit_v1",
                        "embedded action audit schema mismatch",
                    )
                    _require(
                        weight in EXPECTED_SAMPLE_WEIGHTS,
                        "record has invalid sample weight",
                    )
                    _require(
                        weight == float(cleaning.get("sample_weight", -1)),
                        "cleaning weight mismatch",
                    )
                    _require(
                        weight == float(action_audit_record.get("sample_weight", -1)),
                        "embedded action audit weight mismatch",
                    )
                    sample_weights[weight] += 1
                    observed_action_audits += 1

                    current_parent = 0
                    for index, card in enumerate(actor["card_cat"]):
                        zone = int(card[2])
                        parent = int(actor["card_parent"][index])
                        if zone in IN_PLAY_ZONES:
                            current_parent = index + 1
                            _require(parent == 0, "in-play Pokemon has a parent")
                            _require(
                                actor["card_state"][index][2:] == [1] * 12,
                                "in-play Energy histogram is not PRESENT",
                            )
                        elif zone in CHILD_ZONES:
                            _require(
                                current_parent > 0 and parent == current_parent,
                                "attached card points to the wrong Pokemon",
                            )
                            _require(
                                actor["card_cat"][parent - 1][1] == card[1],
                                "attached card owner differs from parent",
                            )
                            if zone == ENERGY_ZONE:
                                _require(
                                    actor["card_state"][index][2:] == [3] * 12,
                                    "Energy child has an applicable Pokemon histogram",
                                )
                        else:
                            current_parent = 0

                    for relation_name, identity_column in (
                        ("option_source", 5),
                        ("option_target", 6),
                    ):
                        for option_index, relation in enumerate(actor[relation_name]):
                            relation = int(relation)
                            if relation <= 0:
                                continue
                            _require(
                                relation <= len(actor["card_cat"]),
                                "option relation is outside the card sequence",
                            )
                            _require(
                                actor["card_cat"][relation - 1][0]
                                == actor["option_cat"][option_index][identity_column],
                                "option relation identity mismatch",
                            )
                            relations[relation_name] += 1
                    for relation_name in ("card_parent", "event_source", "event_target"):
                        relations[relation_name] += sum(
                            int(value) > 0 for value in actor[relation_name]
                        )
                    for option_index, option in enumerate(actor["option_cat"]):
                        if int(option[0]) != ATTACH_ACTION:
                            continue
                        _require(
                            actor["option_source"][option_index] > 0,
                            "Attach option has no source Energy instance",
                        )
                        _require(
                            actor["option_target"][option_index] > 0,
                            "Attach option has no target Pokemon instance",
                        )
                        counts["attach_options"] += 1
                    counts[split] += 1
                    if observed <= 2:
                        samples.append(record)
            _require(observed == int(item["count"]), "shard record count mismatch")
            if shard_number % progress_every == 0 or shard_number == shard_total:
                print(
                    json.dumps(
                        {
                            "event": "local_partition_audit_progress",
                            "date": expected_day,
                            "shards": shard_number,
                            "shards_total": shard_total,
                            "records": counts["train"] + counts["validation"],
                            "seconds": round(time.perf_counter() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    _require(counts["train"] == split_counts["train"], "train count mismatch")
    _require(
        counts["validation"] == split_counts["validation"],
        "validation count mismatch",
    )
    _require(observed_action_audits == daily["records"], "action audit record count mismatch")
    _require(int(action_audit["count"]) == daily["records"], "action audit manifest count mismatch")
    _require(
        sample_weights == declared_sample_weights,
        "embedded sample weights disagree with cleaning report",
    )
    _require(
        sample_weights == external_sample_weights,
        "embedded/external action audit weights disagree",
    )
    expected_relations = {
        name: int(value) for name, value in acceptance["relation_counts"].items()
    }
    _require(dict(relations) == expected_relations, "relation count mismatch")
    _require(
        counts["attach_options"] == feature_audit["attach_options"],
        "Attach count mismatch",
    )

    sample_batch = contracts.DecisionBatch.from_mapping(
        collate.collate_canonical_records(samples)
    )
    _require(sample_batch.batch_size == len(samples), "sample batch size mismatch")
    prototypes = prototypes_module.PrototypeIndex.load(canonical_root / "prototypes.json")
    model = model_module.SemanticPolicy(model_module.ModelConfig(), prototypes).eval()
    forward_batch = contracts.DecisionBatch.from_mapping(
        collate.collate_canonical_records(samples[:8])
    )
    with torch.inference_mode():
        logits = model.teacher_logits(forward_batch)
    _require(bool(torch.isfinite(logits).all()), "model forward is non-finite")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(parameter_count == EXPECTED_PARAMETER_COUNT, "model parameter count drift")

    return {
        "status": "passed",
        "date": expected_day,
        "winner_only": True,
        "records": daily["records"],
        "split_counts": split_counts,
        "shards": {
            split: len(canonical["shards"][split])
            for split in ("train", "validation")
        },
        "sampled_batch_records": len(samples),
        "relation_counts": dict(sorted(relations.items())),
        "attach_options": counts["attach_options"],
        "sample_weight_distribution": {
            str(weight): count for weight, count in sorted(sample_weights.items())
        },
        "model_forward": {
            "rows": 8,
            "shape": list(logits.shape),
            "finite": True,
            "parameters": parameter_count,
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


def validate_output_root(
    output_root: Path,
    *,
    progress_every: int = 10,
) -> dict[str, Any]:
    partition_path = output_root / "partition_manifest.json"
    partition = _load_json(partition_path)
    _require(
        partition.get("schema_version") == "0034_clean_recent3_partition_v1",
        "partition manifest schema mismatch",
    )
    _require(
        partition.get("experiment") == "0034_clean_recent3_semantic_bc",
        "partition experiment mismatch",
    )
    entries = partition.get("day_manifests", [])
    _require(
        isinstance(entries, list) and len(entries) == len(EXPECTED_DAYS),
        "partition day count mismatch",
    )
    by_day = {str(entry.get("date")): entry for entry in entries}
    _require(set(by_day) == set(EXPECTED_DAYS), "partition dates mismatch")

    reports = []
    for day in EXPECTED_DAYS:
        entry = by_day[day]
        relative_manifest = entry.get("manifest_path")
        _require(
            isinstance(relative_manifest, str),
            "day manifest path is missing",
        )
        manifest_path = output_root / relative_manifest
        _require(
            manifest_path.is_file(),
            f"missing day manifest: {manifest_path}",
        )
        _require(
            _sha256(manifest_path) == entry.get("manifest_sha256"),
            "day manifest SHA256 mismatch",
        )
        report = validate_partition(
            manifest_path.parent,
            expected_day=day,
            progress_every=progress_every,
        )
        _require(
            report["records"] == int(entry.get("records", -1)),
            "partition day record mismatch",
        )
        _require(
            report["split_counts"] == entry.get("split_counts"),
            "partition day split mismatch",
        )
        reports.append(report)

    total_records = sum(int(report["records"]) for report in reports)
    combined_splits = {
        split: sum(int(report["split_counts"][split]) for report in reports)
        for split in ("train", "validation")
    }
    _require(
        total_records == int(partition.get("total_records", -1)),
        "partition total-record mismatch",
    )
    aggregate = _load_json(output_root / "cleaning_report.json")
    _require(
        aggregate == partition.get("cleaning"),
        "aggregate cleaning report mismatch",
    )
    _require(
        aggregate.get("schema_version") == "0034_cleaning_report_v1",
        "aggregate cleaning schema mismatch",
    )
    _require(
        aggregate.get("status") == "passed",
        "aggregate cleaning did not pass",
    )
    _require(
        aggregate.get("dates") == list(EXPECTED_DAYS),
        "aggregate cleaning dates mismatch",
    )
    _require(
        int(aggregate.get("kept_records", -1)) == total_records,
        "aggregate kept-record mismatch",
    )
    _require(
        int(aggregate.get("input_records", -1))
        == total_records + int(aggregate.get("rejected_records", -1)),
        "aggregate cleaning record accounting mismatch",
    )
    for name in (
        "official_public_prototypes_v1.json",
        "official_full_engine_prototypes_v2.json",
        "feature_audit.json",
    ):
        path = output_root / name
        _require(
            path.is_file() and path.stat().st_size > 0,
            f"missing output asset: {name}",
        )
    feature_audit = _load_json(output_root / "feature_audit.json")
    _require(
        feature_audit.get("actor_schema")
        == "0034_clean_recent3_semantic_decision_v1",
        "output feature audit schema mismatch",
    )
    return {
        "status": "passed",
        "dates": list(EXPECTED_DAYS),
        "records": total_records,
        "split_counts": combined_splits,
        "daily_reports": reports,
        "partition_manifest_sha256": _sha256(partition_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--day")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.day is None:
        report = validate_output_root(
            args.root,
            progress_every=args.progress_every,
        )
    else:
        report = validate_partition(
            args.root,
            expected_day=args.day,
            progress_every=args.progress_every,
        )
    encoded = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
