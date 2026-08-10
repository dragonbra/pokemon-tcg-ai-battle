"""Parse model-free official-CPU/CUDA rule differential evidence.

The paired CUDA executable intentionally emits either one terminal ``CASE``
JSON record or stops at the first textual mismatch.  This module turns that
stream into a small, immutable release-gate record without attempting to
reinterpret later, consequential differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FAILURE_PREFIXES = (
    "CPU status/official terminal mismatch ",
    "CPU/GPU status mismatch ",
    "CPU/POD state mismatch ",
    "CPU/GPU state mismatch ",
    "outcome mismatch ",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_paired_output(text: str) -> dict[str, Any]:
    """Return exactly the first rule result or divergence in one process log."""

    active_case: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("START\t"):
            active_case = int(line.split("\t", 1)[1])
            continue
        if line.startswith("CASE\t"):
            _, ordinal_text, payload_text = line.split("\t", 2)
            payload = json.loads(payload_text)
            passed = bool(payload.get("passed"))
            return {
                "status": "PASS" if passed else "FAIL",
                "case_ordinal": int(ordinal_text),
                "first_divergence": None if passed else {
                    "category": "CUDA rules",
                    "message": payload.get("first_failure") or "paired rule case failed",
                },
                "metrics": payload,
            }
        for prefix in FAILURE_PREFIXES:
            if line.startswith(prefix):
                return {
                    "status": "FAIL",
                    "case_ordinal": active_case,
                    "first_divergence": {
                        "category": "CUDA rules",
                        "kind": prefix.strip(),
                        "message": line,
                    },
                    "metrics": None,
                }
    return {
        "status": "INCOMPLETE",
        "case_ordinal": active_case,
        "first_divergence": None,
        "metrics": None,
        "reason": "paired executable emitted neither CASE nor a recognized mismatch",
    }


def build_summary(cases: Iterable[tuple[str, Path]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for name, raw_path in cases:
        path = raw_path.resolve()
        parsed = parse_paired_output(path.read_text(encoding="utf-8"))
        records.append({
            "name": name,
            "log": str(path),
            "log_sha256": _sha256(path),
            **parsed,
        })
    status = (
        "FAIL" if any(row["status"] == "FAIL" for row in records)
        else "INCOMPLETE" if any(row["status"] == "INCOMPLETE" for row in records)
        else "PASS"
    )
    return {
        "schema_version": "0038_gate_a_rule_differential_v1",
        "gate": "A",
        "status": status,
        "model_involved": False,
        "comparison": "official seeded CPU state vs CUDA/POD state after identical primitive selections",
        "cases": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", required=True, metavar="NAME=LOG",
        help="Named paired-executable log; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cases: list[tuple[str, Path]] = []
    for item in args.case:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --case value: {item!r}")
        cases.append((name, Path(raw_path)))
    summary = build_summary(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_summary", "parse_paired_output"]
