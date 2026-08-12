from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.official_coverage import build_official_semantic_coverage  # noqa: E402
from ptcg_cuda_engine.official_ir import load_and_validate_official_ir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-text-free semantic coverage matrix from private IR."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oracle-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, _ = load_and_validate_official_ir(args.input)
    matrix, summary = build_official_semantic_coverage(payload, args.oracle_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(matrix, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **summary.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
