"""Build a concise Phantom Dive macro release-gate record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..action_boundary.dragapult import PHANTOM_DIVE_MAX_TARGETS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_contract(package_root: Path) -> dict[str, Any]:
    if package_root.is_file():
        payload = json.loads(package_root.read_text(encoding="utf-8"))
        required = {
            "maximum_macro_targets",
            "macro_protocol_fail_closed",
            "package_manifest_sha256",
            "source_evidence",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"immutable package contract missing fields: {missing}")
        return {
            "root": str(package_root.resolve()),
            "evidence_file_sha256": _sha256(package_root),
            **payload,
        }
    source_path = package_root / "strategy/deployment/compound_inference.py"
    source = source_path.read_text(encoding="utf-8")
    limit = 8 if "1 <= len(raw_bench) <= 8" in source else (
        5 if "1 <= len(raw_bench) <= 5" in source else None
    )
    fallback_handler = ""
    if "except MacroProtocolError:" in source:
        fallback_handler = source.split(
            "except MacroProtocolError:", 1
        )[1].split("else:", 1)[0]
    return {
        "root": str(package_root.resolve()),
        "source_sha256": _sha256(source_path),
        "maximum_macro_targets": limit,
        "macro_protocol_fail_closed": "raise" in fallback_handler,
    }


def build_summary(
    *, official_parity: Path, archived_package: Path, fixed_package: Path
) -> dict[str, Any]:
    parity_path = official_parity.resolve()
    official = json.loads(parity_path.read_text(encoding="utf-8"))
    counts = {int(row["n"]): int(row["allocations"]) for row in official["results"]}
    expected_1_8 = {n: math.comb(n + 5, 6) for n in range(1, 9)}
    official_1_5_pass = (
        official.get("parity_failures") == 0
        and counts == {n: expected_1_8[n] for n in range(1, 6)}
    )
    archived = _package_contract(archived_package.resolve())
    fixed = _package_contract(fixed_package.resolve())
    archived_pass = (
        archived["maximum_macro_targets"] == PHANTOM_DIVE_MAX_TARGETS
        and archived["macro_protocol_fail_closed"]
    )
    fixed_source_pass = (
        fixed["maximum_macro_targets"] == PHANTOM_DIVE_MAX_TARGETS
        and fixed["macro_protocol_fail_closed"]
    )
    # The candidate under audit is the immutable archived U230 package.  The
    # fixed diagnostic package proves the proposed source contract, but cannot
    # retroactively make the archived artifact releasable.
    status = "PASS" if official_1_5_pass and archived_pass else "FAIL"
    return {
        "schema_version": "0038_gate_b_phantom_macro_v1",
        "gate": "B",
        "status": status,
        "canonical_allocation_counts": expected_1_8,
        "official_exhaustive_n1_n5": {
            "status": "PASS" if official_1_5_pass else "FAIL",
            "total_allocations": official.get("total_allocations"),
            "parity_failures": official.get("parity_failures"),
            "source": str(parity_path),
            "source_sha256": _sha256(parity_path),
        },
        "expanded_bench_n6_n8": {
            "enumeration_and_source_contract": (
                "PASS" if PHANTOM_DIVE_MAX_TARGETS == 8 and fixed_source_pass else "FAIL"
            ),
            "official_state_fixture": "NOT_COVERED",
            "risk": "no real official Area Zero callback fixture exhaustively covers n=6..8",
        },
        "archived_u230_package": archived,
        "fixed_diagnostic_package": fixed,
        "first_divergence": None if status == "PASS" else {
            "category": "packaging/lifecycle",
            "stage": "archived U230 macro contract",
            "evidence": (
                "archive supports only n<=5 and clears MacroProtocolError into a fresh "
                "policy decision instead of failing closed"
            ),
        },
        "execution": (
            "one policy macro plan followed by six official primitive selections; "
            "no fused direct rule mutation"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-parity", type=Path, required=True)
    parser.add_argument("--archived-package", type=Path, required=True)
    parser.add_argument("--fixed-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_summary(
        official_parity=args.official_parity,
        archived_package=args.archived_package,
        fixed_package=args.fixed_package,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_summary"]
