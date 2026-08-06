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


BASE = "train.0033_effect_summary_semantic_foundation_pretraining"
EXPECTED_PARAMETER_COUNT = 22_595_202
IN_PLAY_ZONES = frozenset({1, 2})
CHILD_ZONES = frozenset({8, 9, 10})
ENERGY_ZONE = 8
ATTACH_ACTION = 9


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
        daily.get("schema_version") == "0032_daily_semantic_partition_v1",
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
    split_counts = daily["split_counts"]
    _require(
        all(int(split_counts.get(split, 0)) > 0 for split in ("train", "validation")),
        "daily partition has an empty split",
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
        "model_forward": {
            "rows": 8,
            "shape": list(logits.shape),
            "finite": True,
            "parameters": parameter_count,
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("day_root", type=Path)
    parser.add_argument("--day", required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    report = validate_partition(
        args.day_root,
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
