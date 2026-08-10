"""Freeze deterministic pre-fix semantic-parity evidence as regression fixtures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .canonical import (
    canonical_legal_set,
    canonical_public_observation,
    stable_hash,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    ROOT / "train/0042_full_model_design/tests/fixtures/semantic_parity_v1"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _canonical_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    observation = row["actor_observation"]
    return {
        "seed": int(row["seed"]),
        "decision": int(row["decision"]),
        "observation": canonical_public_observation(observation),
        "observation_hash": stable_hash(canonical_public_observation(observation)),
        "legal": canonical_legal_set(observation["select"]),
        "ordered_action": [int(value) for value in row["ordered_action"]],
    }


def build_fixtures(
    *,
    fixed_trace: Path,
    gate_c_summary: Path,
    zoroark_trace: Path,
    zoroark_failure_log: Path,
    package_manifest: Path,
    load_report: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()
    fixed_rows = _read_jsonl(fixed_trace)
    zoroark_rows = _read_jsonl(zoroark_trace)
    if len(fixed_rows) != 283 or [row["decision"] for row in fixed_rows] != list(range(283)):
        raise ValueError("fixed primitive trace is not the exact 283-decision fixture")
    if len(zoroark_rows) <= 128:
        raise ValueError("Zoroark trace does not reach pre-divergence decision 128")
    pre_row = zoroark_rows[128]
    prefix_actions = [row["ordered_action"] for row in zoroark_rows[:129]]
    gate_a = {
        "schema_version": "0038_gate_a_decision129_prefx_fixture_v1",
        "case": "007_dragapult_vs_zoroark_munkidori",
        "seed": 1,
        "decision_count_after_failing_action": 129,
        "pre_action_trace_decision": 128,
        "pre_action": _canonical_trace_row(pre_row),
        "prefix_actions": prefix_actions,
        "prefix_actions_sha256": stable_hash(prefix_actions),
        "expected_failure_line": zoroark_failure_log.read_text(
            encoding="utf-8"
        ).splitlines()[-1],
        "expected": {
            "official_terminal": False,
            "official_continuation": "0:79:0:1:0:0:0:0",
            "cuda_pod_error": 20,
            "cuda_pod_detail": 403,
            "cuda_pod_continuations": [],
        },
    }

    summary = json.loads(gate_c_summary.read_text(encoding="utf-8"))
    decision0 = [
        row for row in summary["tensor_mismatch_details"]
        if int(row["decision"]) == 0
    ]
    if not decision0:
        raise ValueError("Gate C summary has no decision-0 field mismatch")
    gate_c_field = {
        "schema_version": "0038_gate_c_decision0_field_diff_prefx_v1",
        "trace_decision": 0,
        "snapshot": _canonical_trace_row(fixed_rows[0]),
        "diffs": [
            {
                key: row.get(key)
                for key in (
                    "field", "flat_index", "expected", "actual",
                    "expected_shape", "actual_shape",
                )
            }
            for row in decision0
        ],
    }
    greedy = dict(summary["model"]["first_greedy_divergence"])
    engine_decision = int(greedy["engine_decision"])
    gate_c_greedy = {
        "schema_version": "0038_gate_c_decision60_greedy_prefx_v1",
        "snapshot": _canonical_trace_row(fixed_rows[engine_decision]),
        "policy": greedy,
        "checkpoint_sha256": (
            "a70b41d8b979a72ed3d79d8e739def40345efaf99dbf0ac49c9a08b81f9f8ab8"
        ),
    }
    focal_engine = [
        index for index, row in enumerate(fixed_rows)
        if row["actor_observation"]["current"]["yourIndex"] == 0
    ]
    value = summary["value_model"]
    maximum = dict(value["maximum_divergence"])
    maximum["engine_decision"] = focal_engine[int(maximum["decision"])]
    signs = []
    for row in value["sign_divergence_samples"]:
        item = dict(row)
        item["engine_decision"] = focal_engine[int(item["decision"])]
        signs.append(item)
    gate_c_value = {
        "schema_version": "0038_gate_c_value_divergence_prefx_v1",
        "maximum": {
            **maximum,
            "snapshot": _canonical_trace_row(fixed_rows[maximum["engine_decision"]]),
        },
        "sign_divergences": [
            {
                **row,
                "snapshot": _canonical_trace_row(fixed_rows[row["engine_decision"]]),
            }
            for row in signs
        ],
        "total_sign_divergences": int(value["value_sign_divergences"]),
    }

    phantom = {
        "schema_version": "0038_phantom_expanded_bench_protocol_fixture_v1",
        "total_counters": 6,
        "cases": [],
    }
    counts = {6: 462, 7: 924, 8: 1716}
    for target_count in range(6, 9):
        counters = [0] * (target_count - 6) + [1] * 6
        targets = [
            {
                "player_index": 1,
                "serial": 1000 + index,
                "card_id": 600 + index,
                "initial_bench_slot": index,
            }
            for index in range(target_count)
        ]
        primitive = [
            target["serial"]
            for target, counter in zip(targets, counters, strict=True)
            for _ in range(counter)
        ]
        phantom["cases"].append({
            "n": target_count,
            "canonical_allocation_count": counts[target_count],
            "target_ids": targets,
            "counters": counters,
            "primitive_target_serials": primitive,
            "expected_policy_forwards": 1,
            "expected_value_forwards": 1,
            "expected_policy_transitions": 1,
            "expected_official_primitive_callbacks": 6,
        })

    package = {
        "schema_version": "0038_u230_package_load_prefx_fixture_v1",
        "package_manifest": json.loads(package_manifest.read_text(encoding="utf-8")),
        "load_report": json.loads(load_report.read_text(encoding="utf-8")),
    }

    entries: list[dict[str, Any]] = []
    entries.append(_write(output / "gate_a_zoroark_decision129.json", _json_bytes(gate_a)))
    compressed_trace = gzip.compress(fixed_trace.read_bytes(), compresslevel=9, mtime=0)
    entries.append(_write(output / "fixed_283_primitive_trace.jsonl.gz", compressed_trace))
    entries.append(_write(output / "gate_c_decision0_field_diff.json", _json_bytes(gate_c_field)))
    entries.append(_write(output / "gate_c_decision60_greedy.json", _json_bytes(gate_c_greedy)))
    entries.append(_write(output / "gate_c_value_divergences.json", _json_bytes(gate_c_value)))
    entries.append(_write(output / "phantom_n6_n8_protocol.json", _json_bytes(phantom)))
    entries.append(_write(output / "u230_package_load_report.json", _json_bytes(package)))
    manifest = {
        "schema_version": "0038_semantic_parity_prefx_fixtures_v1",
        "immutable": True,
        "purpose": "release-blocking pre-fix regression evidence; never update in place",
        "sources": {
            "fixed_trace_sha256": _sha256(fixed_trace),
            "gate_c_summary_sha256": _sha256(gate_c_summary),
            "zoroark_trace_sha256": _sha256(zoroark_trace),
            "zoroark_failure_log_sha256": _sha256(zoroark_failure_log),
            "package_manifest_sha256": _sha256(package_manifest),
            "load_report_sha256": _sha256(load_report),
        },
        "entries": entries,
    }
    manifest_bytes = _json_bytes(manifest)
    _write(output / "manifest.json", manifest_bytes)
    _write(
        output / "manifest.json.sha256",
        (_sha256_bytes(manifest_bytes) + "\n").encode("ascii"),
    )
    return manifest


def validate_fixtures(root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = (root / "manifest.json.sha256").read_text(encoding="ascii").strip()
    if _sha256(manifest_path) != sidecar:
        raise ValueError("semantic parity fixture manifest hash mismatch")
    if manifest.get("schema_version") != "0038_semantic_parity_prefx_fixtures_v1":
        raise ValueError("semantic parity fixture schema mismatch")
    for entry in manifest.get("entries", []):
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"semantic parity fixture size mismatch: {path.name}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"semantic parity fixture hash mismatch: {path.name}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    audit = ROOT / ".tmp/evaluation/0038_semantic_parity_audit"
    parser.add_argument("--fixed-trace", type=Path, default=audit / "gate_c/semantic_trace.jsonl")
    parser.add_argument("--gate-c-summary", type=Path, default=audit / "gate_c/summary_detailed_prefx.json")
    parser.add_argument("--zoroark-trace", type=Path, default=audit / "gate_a/zoroark_seed1_trace.jsonl")
    parser.add_argument("--zoroark-failure-log", type=Path, default=audit / "gate_a/munkidori_seed1_repro.log")
    parser.add_argument(
        "--package-manifest", type=Path,
        default=ROOT / "archive/submission/0038_dragapult_ex_rl_update230/manifest.json",
    )
    parser.add_argument("--load-report", type=Path, default=audit / "runtime_inventory_archived.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = build_fixtures(
        fixed_trace=args.fixed_trace,
        gate_c_summary=args.gate_c_summary,
        zoroark_trace=args.zoroark_trace,
        zoroark_failure_log=args.zoroark_failure_log,
        package_manifest=args.package_manifest,
        load_report=args.load_report,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_fixtures", "validate_fixtures"]
