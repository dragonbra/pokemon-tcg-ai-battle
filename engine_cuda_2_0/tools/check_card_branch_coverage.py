from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.official_coverage import (  # noqa: E402
    validate_official_coverage_inventory,
)
from ptcg_cuda_engine.official_ir import load_and_validate_official_ir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate complete card/attack/skill/branch coverage inventory."
    )
    parser.add_argument("--input", type=Path, required=True, help="private typed IR JSON")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require every entity and branch to be CUDA paired",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, _ = load_and_validate_official_ir(args.input)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    summary = validate_official_coverage_inventory(payload, matrix)
    statuses = [
        row["status"]
        for name in ("cards", "skills", "attacks", "branches")
        for row in matrix["domains"][name]
    ]
    status_counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    if args.require_complete and any(status != "cuda_paired" for status in statuses):
        raise SystemExit(
            json.dumps(
                {
                    "passed": False,
                    "reason": "coverage is not fully CUDA paired",
                    "status_counts": status_counts,
                    **summary,
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {
                "passed": True,
                "complete": all(status == "cuda_paired" for status in statuses),
                "status_counts": status_counts,
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
